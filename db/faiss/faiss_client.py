from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document

from typing import Optional, Union, Callable, Dict, Any, List, Tuple
from langchain_community.vectorstores.faiss import dependable_faiss_import
from functools import lru_cache
import shutil
import stat
import os
import platform
import asyncio

from utils.general_utils import num_tokens
from utils.logger import debug_logger
from config.model_config import VECTOR_SEARCH_TOP_K, FAISS_LOCATION, FAISS_CACHE_SIZE
from db.mysql.mysql_client import KnowledgeBaseManager

os_system = platform.system()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 可能是由于是MacOS系统的原因


class SelfInMemoryDocstore(InMemoryDocstore):
    def add(self, texts: Dict[str, Document]) -> None:
        """Add texts to in memory dictionary.

        Args:
            texts: dictionary of id -> document.

        Returns:
            None
        """
        # overlapping = set(texts).intersection(self._dict)
        # if overlapping:
        #     raise ValueError(f"Tried to add ids that already exist: {overlapping}")
        # self._dict = {**self._dict, **texts}
        self._dict.update(texts)


@lru_cache(FAISS_CACHE_SIZE)
def load_vector_store(faiss_index_path, embeddings):
    debug_logger.info(f'load faiss index: {faiss_index_path}')
    return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)


class FaissClient:
    def __init__(self, mysql_client: KnowledgeBaseManager, embeddings):
        self.mysql_client: KnowledgeBaseManager = mysql_client
        self.embeddings = embeddings
        self.faiss_client: FAISS = None
        self.kb_ids: List[str] = []

    def _load_kb_to_memory(self, kb_ids):
        self.kb_ids = kb_ids
        self.faiss_client = None
        for kb_id in kb_ids:
            faiss_index_path = os.path.join(FAISS_LOCATION, kb_id, 'faiss_index')
            if os.path.exists(faiss_index_path):
                faiss_client: FAISS = load_vector_store(faiss_index_path, self.embeddings)
            else:
                faiss = dependable_faiss_import()
                index = faiss.IndexFlatL2(768)
                docstore = SelfInMemoryDocstore()
                debug_logger.info(f'init FAISS kb_id: {kb_id}')
                faiss_client: FAISS = FAISS(self.embeddings, index, docstore, index_to_docstore_id={})
            if self.faiss_client is None:
                self.faiss_client = faiss_client
            else:
                try:
                    self.faiss_client.merge_from(faiss_client)
                    debug_logger.info(f'merge FAISS kb_id: {kb_id}')
                except ValueError:
                    raise ValueError(f'遗留数据与新版本不匹配，请删除{os.path.dirname(FAISS_LOCATION)}文件夹（清空所有知识库）后重新启动服务并重新创建知识库')
        # debug_logger.info(f'FAISS load kb_ids: {kb_ids}')

#  filter: Optional[Union[Callable, Dict[str, Any]]] = None,
    async def search_document(self, kb_ids, sources, query, 
                    top_k=VECTOR_SEARCH_TOP_K, 
                    merge = True,
                    score_thread = None,
                    descending = False):
        
        if self.faiss_client is None or self.kb_ids != kb_ids:
            self._load_kb_to_memory(kb_ids)

        filter = lambda metadata: metadata['_kb_id'] in kb_ids and metadata['_source'] in sources
        docs_with_score = await self.faiss_client.asimilarity_search_with_score(query, k=top_k, filter=filter,
                                                                                fetch_k=200)
        
        # make a hard copy of the searched doc in case the doc will be updated
        docs_with_score = [(Document(str(doc.page_content), metadata=dict(doc.metadata)),score) for doc, score in docs_with_score]

        # debug_logger.info(f'FAISS search result number: {len(docs_with_score)}')
        docs = []
        for doc, score in docs_with_score:
            if score_thread and float(score) > score_thread:
                continue
            doc.metadata['_score'] = score
            docs.append(doc)
            doc.metadata['_retrieval_query'] = query  # 添加查询到文档的元数据中

        if merge:
            docs = self.merge_docs(docs)
        
        sorted(docs, key=lambda x: x.metadata['_score'], reverse=descending)
        
        return docs

    def merge_docs(self, docs):
        # 把docs按照file_id进行合并，但是需要对所有file_id相同的doc根据chunk_id先排序，chunk_id相邻的doc合并
        merged_docs = []
        docs = sorted(docs, key=lambda x: (x.metadata['file_id'], x.metadata['chunk_id']))
        for doc in docs:
            if not merged_docs or merged_docs[-1].metadata['file_id'] != doc.metadata['file_id']:
                merged_docs.append(doc)
            else:
                if merged_docs[-1].metadata['chunk_id'] == doc.metadata['chunk_id'] - 1:
                    if num_tokens(merged_docs[-1].page_content + doc.page_content) <= 800:
                        # print('MERGE:', merged_docs[-1].metadata['chunk_id'], doc.metadata['chunk_id'])
                        merged_docs[-1].page_content += '\n' + doc.page_content
                        merged_docs[-1].metadata['chunk_id'] = doc.metadata['chunk_id']
                        if(merged_docs[-1].metadata['score'] > doc.metadata['score']):
                            merged_docs[-1].metadata['score'] = doc.metadata['score']
                    else:
                        # print('NOT MERGE:', merged_docs[-1].metadata['chunk_id'], doc.metadata['chunk_id'])
                        merged_docs.append(doc)
                else:
                    # print('NOT MERGE:', merged_docs[-1].metadata['chunk_id'], doc.metadata['chunk_id'])
                    merged_docs.append(doc)
        return merged_docs

    async def add_document(self, docs, user_id, kb_id, doc_id, doc_name, source):
        
        if self.faiss_client is None or self.kb_ids != [kb_id]:
            self._load_kb_to_memory([kb_id])

        for idx, doc in enumerate(docs):
            doc.metadata["_user_id"] = user_id
            doc.metadata["_kb_id"] = kb_id
            doc.metadata["_doc_id"] = doc_id
            doc.metadata["_doc_name"] = doc_name
            doc.metadata["_chunk_id"] = idx
            doc.metadata["_source"] = source

        add_ids = await self.faiss_client.aadd_documents(docs)
        # doc带上id存入Document表中
        for doc, add_id in zip(docs, add_ids):
            self.mysql_client.add_document(add_id, doc.metadata['_chunk_id'], doc_id, doc_name,
                                           kb_id)
        faiss_index_path = os.path.join(FAISS_LOCATION, kb_id, 'faiss_index')
        self.faiss_client.save_local(faiss_index_path)
        os.chmod(os.path.dirname(faiss_index_path), stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        return add_ids

    def delete_documents(self, kb_id, file_ids=None):
        if self.faiss_client is None or self.kb_ids != [kb_id]:
            self._load_kb_to_memory([kb_id])
        doc_ids = []
        if file_ids is None:
            kb_index_path = os.path.join(FAISS_LOCATION, kb_id)
            if os.path.exists(kb_index_path):
                shutil.rmtree(kb_index_path)
                self.faiss_client = None
                debug_logger.info(f'delete kb_id: {kb_id}, {kb_index_path}')
                return
        elif file_ids:
            doc_ids = self.mysql_client.get_documents_by_file_ids(file_ids)
        doc_ids = [doc_id[0] for doc_id in doc_ids]
        if not doc_ids:
            debug_logger.info(f'no documents to delete')
            return
        try:
            res = self.faiss_client.delete(doc_ids)
            debug_logger.info(f'delete documents: {res}')
            faiss_index_path = os.path.join(FAISS_LOCATION, kb_id, 'faiss_index')
            self.faiss_client.save_local(faiss_index_path)
            debug_logger.info(f'save faiss index: {faiss_index_path}')
            os.chmod(os.path.dirname(faiss_index_path), stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        except ValueError as e:
            debug_logger.warning(f'delete documents not find docs')

