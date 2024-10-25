from typing import Callable, Iterator, List
from langchain.docstore.document import Document

from config.dataset_config import POI_SOURCE_OSM
from loader.chatvel_loader import ChatvelLoader
from poi import osmpoi
from poi.poi_service import POIService
from service.service_context import ServiceContext

class OSMPoiDatasetLoader(ChatvelLoader):
    def __init__(self, 
                 context:ServiceContext, 
                 data_name:str, 
                 data_dir:str,
                 ):
        super().__init__()
        self._data_name = data_name
        self._data_dir = data_dir
        
        self._poi_service = POIService(context=context)
        
    def on_load_dataset(self) -> Iterator:
        for poi in osmpoi.get_pois(data_dir=self._data_dir, dataset_name=self._data_name):
            yield poi
        
    def data_handlers(self) -> List[Callable]:
        return[self.save_poi]

    def save_poi(self, poi:dict) -> Iterator[Document]:
        docs = self._poi_service.insert_poi(poi, source = POI_SOURCE_OSM)
        for doc in docs:
            yield doc

