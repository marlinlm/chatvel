from config.dataset_config import POI_SOURCE_OSM
from poi.poi_service import POIService
from query.query_decompose import QueryDecomposer
from service.service_context import ServiceContext
from utils.logger import debug_logger

class TravelQaService:
    def __init__(self, context:ServiceContext):
        self._context = context
        self._poi_service = POIService(context = context)
        self._query_service = QueryDecomposer(llm = context.llm)

    def qa(self, query, history:list[list[str]] = []):
        requirements,slot = self._query_service.decompose_query(query)
        matched_pois = self.match_poi_with_requirement(requirements)
        sorted_pois = sorted(matched_pois.values(), key=lambda x: float(x['_total_score']),reverse=True)
        debug_logger.info("最终POI:")
        for poi in sorted_pois:
            debug_logger.info(poi['id'] + ":" + poi['name'] + ":" + str(poi['_total_score']))
        return sorted_pois


    def match_poi_with_requirement(self, requirements:list[dict] = []):
        if requirements is None or len(requirements) == 0:
            return []
        

        
        pois = {}
        for type, pos_neg_req in requirements.items():
            pos_req = pos_neg_req['pos'] if 'pos' in pos_neg_req else None
            neg_req = pos_neg_req['neg'] if 'neg' in pos_neg_req else None
            debug_logger.info("提取POI，请求：" + str(pos_neg_req))
            retrieved_pois = self._poi_service.retrieve_poi_by_requirement(pos_req, neg_req, sources=[POI_SOURCE_OSM])
            debug_logger.info("提取到的POI：")
            for poi in retrieved_pois:
                debug_logger.info(poi['id'] + ":" + poi['name'] + ":" + str(poi['_rerank_score']))
                if poi['id'] in pois:
                    pois[poi['id']]['_total_score'] = pois[poi['id']]['_total_score'] + poi['_rerank_score']
                else:
                    pois[poi['id']] = dict(poi)
                    pois[poi['id']]['_total_score'] = poi['_rerank_score']
        
        return pois
        