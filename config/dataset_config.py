import os.path
from os.path import dirname
DATA_ROOT = os.path.join(dirname(dirname(__file__)),'.data')
POI_DATASET_DIR_OSM_BEIJING = os.path.join(DATA_ROOT,'in','osm','beijing')
POI_DATASET_NAME_OSM_BEIJING = 'beijing'
POI_SOURCE_OSM = 'OSM'
POI_SOURCE_XHS = 'XHS'

KB_ID_POI_NAME = 'beijing_poi'
KB_NAME_POI_NAME = 'beijing_poi'

XHS_DATASET_DIR = os.path.join(DATA_ROOT,'in','xhs')
KB_ID_POI_DESC = 'desc_poi'
KB_NAME_POI_DESC = 'desc_poi'

XHS_HEADERS={'cookie':'web_session=040069b3fca2d9d40d070584c2344b819836aa;', 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'}

class DatasetConfig:
    def __init__(self):
        self.kb_id_poi_name = KB_ID_POI_NAME
        self.kb_name_poi_name = KB_NAME_POI_NAME
        self.kb_id_poi_desc = KB_ID_POI_DESC
        self.kb_name_poi_desc = KB_NAME_POI_DESC
        self.xhs_claw_header = XHS_HEADERS
        self.poi_retrieve_sources = [POI_SOURCE_OSM]
