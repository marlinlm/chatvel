import sys
import os
import traceback
import base64
import numpy as np
import platform
import cv2
import re
import time
from os.path import dirname
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm.llm_for_openai_api import OpenAILLM
# from langchain.schema import Document
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

sys.path.append(dirname(dirname(__file__)))
from config.model_config import VECTOR_SEARCH_TOP_K, CHUNK_SIZE, VECTOR_SEARCH_SCORE_THRESHOLD, \
    PROMPT_TEMPLATE, STREAMING
from db.mysql.mysql_client import KnowledgeBaseManager
from db.faiss.faiss_client import FaissClient
from rerank.rerank_backend import RerankBackend
from embedding.embedding_backend import EmbeddingBackend
from utils.logger import debug_logger, qa_logger
from utils.general_utils import num_tokens, get_time, async_run
from search.web_search_tool import duckduckgo_search
from loader.chatvel_loader import ChatvelLoader
from service_context import ServiceContext


class DocumentService:
    def __init__(self, context: ServiceContext):
        self._context: ServiceContext = context
        if context is None:
            raise Exception('The service context should not be None!')

    def deduplicate_documents(self, source_docs):
        unique_docs = set()
        deduplicated_docs = []
        for doc in source_docs:
            if doc.page_content not in unique_docs:
                unique_docs.add(doc.page_content)
                deduplicated_docs.append(doc)
        return deduplicated_docs

    def local_doc_search(self, query, kb_ids, merge=True):
        source_documents = async_run(self.get_source_documents(query, kb_ids, merge=merge))
        deduplicated_docs = self.deduplicate_documents(source_documents)
        retrieval_documents = sorted(deduplicated_docs, key=lambda x: x.metadata['score'], reverse=False)
        debug_logger.info(f"local doc search retrieval_documents: {retrieval_documents}")
        return retrieval_documents

    @get_time
    def get_web_search(self, queries, top_k=None):
        if not top_k:
            top_k = self.top_k
        query = queries[0]
        web_content, web_documents = duckduckgo_search(query)
        source_documents = []
        for doc in web_documents:
            doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
            file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', doc.metadata['title'])
            doc.metadata['file_name'] = file_name + '.web'
            doc.metadata['file_url'] = doc.metadata['source']
            doc.metadata['embed_version'] = self.embeddings.embed_version
            source_documents.append(doc)
            if 'description' in doc.metadata:
                desc_doc = Document(page_content=doc.metadata['description'], metadata=doc.metadata)
                source_documents.append(desc_doc)
        source_documents = self.web_splitter.split_documents(source_documents)
        return web_content, source_documents

    def web_page_search(self, query, top_k=None):
        # 防止get_web_search调用失败，需要try catch
        try:
            web_content, source_documents = self.get_web_search([query], top_k)
        except Exception as e:
            debug_logger.error(f"web search error: {e}")
            return []

        return source_documents

    

    async def get_source_documents(self, query, kb_ids, cosine_thresh=None, top_k=None, merge=True):
        if not top_k:
            top_k = self.top_k
        source_documents = []
        t1 = time.time()
        filter = lambda metadata: metadata['kb_id'] in kb_ids
        # filter = None
        debug_logger.info(f"query: {query}")
        docs = await self.faiss_client.search(kb_ids, query, filter=filter, top_k=top_k, merge=merge)
        debug_logger.info(f"query_docs: {len(docs)}")
        t2 = time.time()
        debug_logger.info(f"faiss search time: {t2 - t1}")
        for idx, doc in enumerate(docs):
            if doc.metadata['file_name'].endswith('.faq'):
                faq_dict = doc.metadata['faq_dict']
                doc.page_content = f"{faq_dict['question']}：{faq_dict['answer']}"
                nos_keys = faq_dict.get('nos_keys')
                doc.metadata['nos_keys'] = nos_keys
            doc.metadata['retrieval_query'] = query  # 添加查询到文档的元数据中
            doc.metadata['embed_version'] = self.embeddings.getModelVersion
            source_documents.append(doc)
        if cosine_thresh:
            source_documents = [item for item in source_documents if float(item.metadata['score']) < cosine_thresh]

        return source_documents

    def reprocess_source_documents(self, query: str,
                                   source_docs: List[Document],
                                   history: List[str],
                                   prompt_template: str) -> List[Document]:
        # 组装prompt,根据max_token
        query_token_num = self.llm.num_tokens_from_messages([query])
        history_token_num = self.llm.num_tokens_from_messages([x for sublist in history for x in sublist])
        template_token_num = self.llm.num_tokens_from_messages([prompt_template])

        limited_token_nums = self.llm.token_window - self.llm.max_token - self.llm.offcut_token - query_token_num - history_token_num - template_token_num
        new_source_docs = []
        total_token_num = 0
        for doc in source_docs:
            doc_token_num = self.llm.num_tokens_from_docs([doc])
            if total_token_num + doc_token_num <= limited_token_nums:
                new_source_docs.append(doc)
                total_token_num += doc_token_num
            else:
                remaining_token_num = limited_token_nums - total_token_num
                doc_content = doc.page_content
                doc_content_token_num = self.llm.num_tokens_from_messages([doc_content])
                while doc_content_token_num > remaining_token_num:
                    # Truncate the doc content to fit the remaining tokens
                    if len(doc_content) > 2 * self.llm.truncate_len:
                        doc_content = doc_content[self.llm.truncate_len: -self.llm.truncate_len]
                    else:  # 如果最后不够truncate_len长度的2倍，说明不够切了，直接赋值为空
                        doc_content = ""
                        break
                    doc_content_token_num = self.llm.num_tokens_from_messages([doc_content])
                doc.page_content = doc_content
                new_source_docs.append(doc)
                break

        debug_logger.info(f"limited token nums: {limited_token_nums}")
        debug_logger.info(f"template token nums: {template_token_num}")
        debug_logger.info(f"query token nums: {query_token_num}")
        debug_logger.info(f"history token nums: {history_token_num}")
        debug_logger.info(f"new_source_docs token nums: {self.llm.num_tokens_from_docs(new_source_docs)}")
        return new_source_docs

    def generate_prompt(self, query, source_docs, prompt_template):
        context = "\n".join([doc.page_content for doc in source_docs])
        prompt = prompt_template.replace("{question}", query).replace("{context}", context)
        return prompt

    def rerank_documents(self, query, source_documents):
        if num_tokens(query) > 300:  # tokens数量超过300时不使用local rerank
            return source_documents

        scores = self.local_rerank_backend.get_rerank(query, [doc.page_content for doc in source_documents])
        debug_logger.info(f"rerank scores: {scores}")
        for idx, score in enumerate(scores):
                source_documents[idx].metadata['score'] = score
        source_documents = sorted(source_documents, key=lambda x: x.metadata['score'], reverse=True)
        return source_documents

    def retrieve(self, query, kb_ids, need_web_search=False, score_threshold=0.35):
        retrieval_documents = self.local_doc_search(query, kb_ids)
        if need_web_search:
            retrieval_documents.extend(self.web_page_search(query, top_k=3))
        debug_logger.info(f"retrieval_documents: {retrieval_documents}")
        if len(retrieval_documents) > 1:
            retrieval_documents = self.rerank_documents(query, retrieval_documents)
            # 删除掉分数低于阈值的文档
            tmp_documents = [item for item in retrieval_documents if float(item.metadata['score']) > score_threshold]
            if tmp_documents:
                retrieval_documents = tmp_documents

        retrieval_documents = retrieval_documents[: self.rerank_top_k]

        debug_logger.info(f"reranked retrieval_documents: {retrieval_documents}")
        return retrieval_documents

    



    def get_knowledge_based_answer(self, query, kb_ids, custom_prompt=None, chat_history=None, 
                                         streaming: bool = STREAMING,
                                         rerank: bool = False,
                                         need_web_search: bool = False):
        if chat_history is None:
            chat_history = []

        retrieval_documents = self.retrieve(query, kb_ids, need_web_search=need_web_search)

        if custom_prompt is None:
            prompt_template = PROMPT_TEMPLATE
        else:
            prompt_template = custom_prompt + '\n' + PROMPT_TEMPLATE

        source_documents = self.reprocess_source_documents(query=query,
                                                           source_docs=retrieval_documents,
                                                           history=chat_history,
                                                           prompt_template=prompt_template)
        prompt = self.generate_prompt(query=query,
                                      source_docs=source_documents,
                                      prompt_template=prompt_template)
        t1 = time.time()
        async def async_iter():
            results = []
            async for answer_result in self.llm.generatorAnswer(prompt=prompt,
                                                      history=chat_history,
                                                      streaming=streaming):

                history = answer_result.history

                history[-1][0] = query
                results.append((answer_result))
            return results
        return async_run(async_iter())            
