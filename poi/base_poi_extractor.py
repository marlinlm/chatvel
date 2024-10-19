from typing import Callable

class BasePoiExtractor:
    def __init__(self, poi_retriever: Callable):
        self.poi_retriever = poi_retriever
        pass
    
    def retrieve_poi(self, pois:list) -> list:
        poi_retrieved = {}
        for my_poi in pois:
            searched_pois = self.poi_retriever(my_poi)
            if searched_pois[0].metadata['score'] < 0.3:
                poi_retrieved[my_poi] = [searched_pois[0].page_content, str(searched_pois[0].metadata['id']), str(searched_pois[0].metadata['score'])]
    
        return poi_retrieved    