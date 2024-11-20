import re
from service.service_context import ServiceContext
from langchain.docstore.document import Document

from utils.general_utils import async_run

POI_STATUS_LOCKED = 'LOCKED'
POI_STATUS_NORMAL = 'NORMAL'

class POIService:
    def __init__(self, 
                 context:ServiceContext,
                 ):
        self._context:ServiceContext = context
        self._kb_checked = False
        self.try_add_kb()
        
    def insert_poi(self, poi, source:str):
        user_id, _ = self._context.session_manager.get_current_user()
        kb_id = self._context.config.kb_id_poi_name

        poi_name = poi['name']
        poi_id = poi['id']
        lat = poi['lat'] if 'lat' in poi else None
        lon = poi['lon'] if 'lon' in poi else None
        address = poi['address'] if 'address' in poi else None
        desc = poi['desc'] if 'desc' in poi else None

        # insert poi with locked status
        self._context.mysql_client.insert_poi(poi_id = poi_id, user_id = user_id, kb_id = kb_id, name = poi_name, lat=lat, lon = lon, address=address, desc = desc, source = source, status = POI_STATUS_LOCKED)

        # handle alias
        docs = self.insert_poi_names(poi['alias'] if 'alias' in poi else None,
                                     poi_id,
                                     poi_name,
                                     source)
        
        # handle descriptions
        self.insert_poi_desc(poi_id, poi_name, desc, source)
        
        # set poi status to normal
        self.unlock_poi(poi_id = poi_id)
        
        return docs
    
    def try_add_kb(self):
        if not self._kb_checked:
            kb_ids = {
                self._context.config.kb_id_poi_desc:self._context.config.kb_name_poi_desc,
                self._context.config.kb_id_poi_name:self._context.config.kb_name_poi_name
            }
            user_id, user_name = self._context.session_manager.get_current_user()
            unavailable_kbs = self._context.mysql_client.check_kb_exist(user_id, list(kb_ids.keys()))
            if unavailable_kbs and len(unavailable_kbs) > 0:
                for kb_id in unavailable_kbs:
                    self._context.mysql_client.new_knowledge_base(kb_id, user_id, kb_ids[kb_id], user_name)
            self._kb_checked = True

    
    def retrieve_poi_by_name(self, name, addr = None, sources:list = []):
        retrieval_documents = async_run(self._context.faiss_client.search_document(kb_ids=[self._context.config.kb_id_poi_name], 
                                                                                   sources = sources,
                                                                                   query = name, 
                                                                                   top_k = 10, 
                                                                                   merge=False, 
                                                                                   score_thread = 0.2, 
                                                                                   descending=False))
        merged = {}
        for d in retrieval_documents:
            key = d.metadata['id']
            if key in merged:
                merged[key].page_content = merged[key].page_content + ';' + d.page_content
            else:
                merged[key] = d
        
        retrieval_documents = sorted(merged.values(), key=lambda x: x.metadata['_score'], reverse=False)
        return retrieval_documents
    
    def rescore_poi(self, pois_with_desc:list[dict], desc:str):
        rerank_score = self._context.local_rerank_backend.get_rerank(desc, [poi_with_desc['desc'] for poi_with_desc in pois_with_desc])
        for poi_with_desc, score in zip(pois_with_desc, rerank_score):
            poi_with_desc['_rerank_score'] = score
        return pois_with_desc
    
    def retrieve_poi_by_requirement(self, positive_desc, negative_desc, sources:list = []):
        retrieval_documents = async_run(self._context.faiss_client.search_document(kb_ids=[self._context.config.kb_id_poi_desc, 
                                                                                        #    self._context.config.kb_id_poi_name
                                                                                           ], 
                                                                                   sources = sources,
                                                                                   query = positive_desc if positive_desc else negative_desc, 
                                                                                   top_k = 100, 
                                                                                   merge=False, 
                                                                                   score_thread = 10, 
                                                                                   descending=False))
        retrieved_pois =    [{
                                        'desc':doc.page_content, 
                                        'id':doc.metadata['id'], 
                                        'name':doc.metadata['name']
                            } for doc in retrieval_documents]
        
        unique_pois = {}
        for retrieved_poi in retrieved_pois:
            if not (retrieved_poi['id'] in unique_pois):
                unique_pois[retrieved_poi['id']] = retrieved_poi
        unique_pois = unique_pois.values()
        pos_scores = [0 for i in range(len(unique_pois))]
        neg_scores = list(pos_scores)
        if not positive_desc is None:
            self.rescore_poi(unique_pois, positive_desc)
            pos_scores = [doc['_rerank_score'] for doc in unique_pois]
        if not negative_desc is None:
            self.rescore_poi(unique_pois, negative_desc)
            neg_scores = [doc['_rerank_score'] for doc in unique_pois]
            
        reranked_scores = [ pos_score - neg_score 
                                for pos_score, neg_score in zip(pos_scores, neg_scores)]
        
        for poi, reranked_score in zip(unique_pois, reranked_scores):
            poi['_rerank_score'] = reranked_score

            
        reranked_poi = sorted(unique_pois, key=lambda x: x['_rerank_score'], reverse=True)
        
        return reranked_poi
        

    # get poi info from mysql
    def get_poi_info(self, id:str):
        poi = {}
        query_result = self._context.mysql_client.get_poi_by_id(id)
        if query_result is None or len(query_result) == 0:
            return None
        
        poi_raw = query_result[0]
        
        poi_status = poi_raw[8]
        if poi_status != POI_STATUS_NORMAL:
            return None
        
        poi['id'] = poi_raw[0]
        poi['name'] = poi_raw[3]
        poi['lat'] = poi_raw[4]
        poi['lon'] = poi_raw[5]
        poi['address'] = poi_raw[6]
        poi['desc'] = poi_raw[7]
        return poi

    def update_poi_desc(self, poi_id:str, poi_name:str, desc:str, source:str):
        self.lock_poi(poi_id = poi_id)
        
        # delete old poi description in vector store
        self.delete_poi_desc(poi_id = poi_id)
        
        # insert new poi desc in vector store
        self.insert_poi_desc(poi_id = poi_id, poi_name = poi_name, desc = desc, source = source)
        
        # update poi entry in mysql
        self._context.mysql_client.update_poi(poi_id=poi_id, desc=desc)
        
        self.unlock_poi(poi_id = poi_id)
    
    def insert_poi_desc(self, poi_id:str, poi_name:str, desc:str, source:str):
        if desc is None or desc == '':
            return
        
        user_id, _ = self._context.session_manager.get_current_user()
        # insert new poi desc in vector store
        kb_id = self._context.config.kb_id_poi_desc
        docs = [Document(page_content = desc, metadata = {'id':poi_id, 'name':poi_name})]
        _ = async_run(self._context.faiss_client.add_document(docs, user_id, kb_id, poi_id, poi_name, source))

    def delete_poi_desc(self, poi_id:str):
        self._context.faiss_client.delete_documents(self._context.config.kb_id_poi_desc, [poi_id])

    def insert_poi_names(self, alias:list, poi_id:str, poi_name:str, source:str):
        if alias is None or len(alias) == 0:
            return 
        
        user_id, _ = self._context.session_manager.get_current_user()
        kb_id = self._context.config.kb_id_poi_name
        docs = []
        for alia in alias:
            content = re.sub(r'[\n\t]+', '\n', alia).strip()
            docs.append(Document(content, metadata={'id':poi_id, 'name':poi_name}))    
        _ = async_run(self._context.faiss_client.add_document(docs, user_id, kb_id, poi_id, poi_name, source))
        return docs
    
    def lock_poi(self, poi_id:str):
        self._context.mysql_client.update_poi_status(poi_id=poi_id, status=POI_STATUS_LOCKED)
        
    def unlock_poi(self, poi_id:str):
        self._context.mysql_client.update_poi_status(poi_id=poi_id, status=POI_STATUS_NORMAL)
