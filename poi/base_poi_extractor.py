from typing import Callable

from service.service_context import ServiceContext

class BasePoiExtractor:
    def __init__(self, context:ServiceContext, poi_retriever: Callable):
        self._poi_retriever = poi_retriever
        self._context = context
        pass
    
    def retrieve_poi(self, pois:list) -> list:
        poi_retrieved = {}
        for my_poi_key in pois.keys():
            name = my_poi_key
            addr = None
            if 'address' in pois[name] and pois[name]['address'] != 'null':
                addr = pois[name]['address']
            searched_pois = self._poi_retriever(name, addr, sources=self._context.config.poi_retrieve_sources)
            if searched_pois and len(searched_pois) > 0 and searched_pois[0].metadata['score'] < 0.3:
                # poi_retrieved[my_poi_key] = [searched_pois[0].page_content, str(searched_pois[0].metadata['id']), str(searched_pois[0].metadata['_score'])]
                poi_retrieved[my_poi_key]  = searched_pois[0]
        return poi_retrieved    