from typing import Iterator
from pyrosm import get_data
from pyrosm import OSM
import numpy as np
import tqdm
from tqdm import trange
import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))
from utils.logger import debug_logger
import json
# from pyrosm.data import sources
# import matplotlib.pyplot as plt
import pandas as pd


def update_china():
    # df = pd.read_csv('/data/linmao/poi/china-latest.osm.csv', sep='|')
    fp = get_data('beijing',directory='./.db/poi/')
    osm = OSM(fp)
    # custom_filter = {'amenity': True, "shop": True}
    pois = osm.get_pois()
    
    pois['amenity'] = pois["amenity"].fillna(pois["shop"])
    ret = []
    for i in trange(0,len(pois)):
        poi = pois.loc[i]
        content = {}
        
        content['id'] = poi['id']
        content['lon'] = poi['lon'] if not np.isnan(poi['lon']) else poi['geometry'].centroid.x
        content['lat'] = poi['lat'] if not np.isnan(poi['lat']) else poi['geometry'].centroid.y

        drop_col = ['timestamp','visible','version','changeset','osm_type','geometry','id','lon','lat']
        dropped = poi.drop(drop_col)
        dropped = dropped.dropna()
        
        if 'lat' in dropped:
            debug_logger("WRONG:", dropped)
            return
        
        content['doc_id'] = poi['amenity'] + '_' + str(poi['id'])
        if 'name' in dropped.keys():
            content['doc_id'] = content['doc_id'] + '_' + poi['name']
            
        if 'tags' in dropped.keys():
            tags = dropped['tags']
            tags = json.loads(tags)
            dropped['tags'] = tags

        content['page'] = {}
        for k in dropped.keys():
            content['page'][k] = dropped[k]
        content['page'] = str(content['page'])
        
        ret.append(content)
    
    return ret

def get_pois(dataset_name:str, data_dir:str) -> Iterator[dict]:
    fp = get_data(dataset_name, directory=data_dir)
    osm = OSM(fp)
    
    pois = osm.get_pois()
    pois['amenity'] = pois["amenity"].fillna(pois["shop"])
    pois['amenity'] = pois['amenity'].fillna(pois["tourism"])
    for i in trange(0,len(pois)):
        poi = pois.loc[i]
        meta = {}
        
        meta['id'] = '_'.join(['OSM', dataset_name, str(poi['id'])])
        meta['lon'] = poi['lon'] if not np.isnan(poi['lon']) else poi['geometry'].centroid.x
        meta['lat'] = poi['lat'] if not np.isnan(poi['lat']) else poi['geometry'].centroid.y

        drop_col = ['timestamp','visible','version','changeset','osm_type','geometry','id','lon','lat']
        dropped = poi.drop(drop_col)
        dropped = dropped.dropna()
        
        alias = []
        if 'name' in dropped:
            alias.append(poi['name'])
        if 'amenity' in dropped:
            meta['amenity'] = poi['amenity']

        if 'tags' in dropped:
            tags = dropped['tags']
            tags = json.loads(tags)
            dropped['tags'] = tags
            
            meta['tags'] = tags
            
            if 'name' in tags:
                alias.append(tags['name'])
            if 'name:zh' in tags:
                alias.append(tags['name:zh'])
            if 'name:zh-Hans' in tags:
                alias.append(tags['name:zh-Hans'])
            if 'alt_name:zh' in tags:
                alias.append(tags['alt_name:zh'])
            if 'alt_name' in tags:
                alias.append(tags['alt_name'])
            if 'alt_name:zh-Hans' in tags:
                alias.append(tags['alt_name:zh-Hans'])
            if 'name:en' in tags:
                alias.append(tags['name:en'])
        
        if len(alias) == 0:
            alias.append(poi['amenity'])
            
        meta['name'] = alias[0]
        meta['alias'] = set(alias)

        yield meta
    
        
    
    
