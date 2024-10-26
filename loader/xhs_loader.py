import logging
from typing import Iterator, List, Optional, Set, Callable
from urllib.parse import urljoin, urldefrag
from os.path import dirname
import uuid
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import cv2

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

import json
import time
import random
import os
import base64

from config import prompt_template_loader
from config.dataset_config import POI_SOURCE_OSM, POI_SOURCE_XHS
from llm.prompt import prompt_generator
from loader.chatvel_loader import ChatvelLoader
from poi.poi_service import POIService
from service.service_context import ServiceContext
from utils.general_utils import async_run

XHS_PREFIX = 'https://www.xiaohongshu.com/explore/'
TEMPLATE_NAME_DESC_MERGE = 'poi_desc_merge'
def load_desc_merge_prompt_template():
    return prompt_template_loader.load_prompt_template(TEMPLATE_NAME_DESC_MERGE)
POI_DESC_MERGE_PROMPT = load_desc_merge_prompt_template()

def generate_desc_merge_prompt(desc:str):
    prompt = {
        'description':desc
    }
    return prompt_generator.generate_prompt(POI_DESC_MERGE_PROMPT, prompt)

class XhsLoader(ChatvelLoader):
    """Loads all child links from a given url."""

    def __init__(
        self,
        context:ServiceContext, 
        file_path: str,
        ):

        super().__init__()

        self._context = context
        self._file_path = file_path
        self._headers = context.config.xhs_claw_header
        self._file_dir = dirname(file_path)
        self._poi_extract_dir = os.path.join(self._file_dir, 'out', 'poi_extract')
        self._parsed_dir = os.path.join(self._file_dir, 'out', 'parsed')
        self._raw_dir = os.path.join(self._file_dir, 'out', 'raw')
        self._img_dir = os.path.join(self._file_dir, 'out', 'img')
        self._llm = context.llm
        
        self._poi_service = POIService(context = context)
        
        from poi.llm_poi_extractor import LlmPoiExtractor
        self._poi_extractor = LlmPoiExtractor(context = context, poi_retriever = self._poi_service.retrieve_poi, save_poi_extraction = False, poi_extraction_save_dir = self._poi_extract_dir)
        # self.poi_extractor = PosPoiExtractor(poi_retriever = self.poi_retriever, save_poi_extraction = True, poi_extraction_save_dir = self.poi_extract_dir)
        
        os.makedirs(self._poi_extract_dir, exist_ok=True)
        os.makedirs(self._parsed_dir, exist_ok=True)
        os.makedirs(self._raw_dir, exist_ok=True)
        os.makedirs(self._img_dir, exist_ok=True)

    
    def on_load_dataset(self) -> Iterator:
        
        with open(self._file_path, 'rb') as f:
            xhs_content = f.read()
        
        content_json = json.loads(xhs_content)
        if content_json['code'] == 0 and 'items' in content_json['data'] and len(content_json['data']['items']) > 0:
            for item in tqdm(content_json['data']['items']):
                id = item['id']
                if('note_card' in item):
                    url = XHS_PREFIX + id
                    response = requests.request(url=url,method='get',headers=self._headers)
                    if response.ok:
                        yield (id, response.text)
                        
                time.sleep(random.randint(1,6))

        
    def data_handlers(self) -> List[Callable]:
        return[self.save_raw, self.save_note]

    def save_raw(self, raw_text_with_id) -> Iterator[Document]:
        note_id, raw_text = raw_text_with_id
        with open(os.path.join(self._raw_dir, note_id + ".xhs.xml"),'w') as f:
            f.write(raw_text)

    def save_note(self, raw_text_with_id) -> Iterator[Document]:
        note_id, raw_text = raw_text_with_id
        note = self.parse_note(note_id, raw_text)
        if note:
            pois, poi_retrieved = self._poi_extractor.extract_poi(  text = note['desc'], 
                                                                    title = note['title'],
                                                                    from_image = note['image_desc'], 
                                                                    from_video=None)
            note['pois'] = pois
            note['retrieved_pois'] = poi_retrieved

            text_in_image = ''
            if note['image_desc']:
                text_in_image = ''.join([''.join([t for t in segment]) for segment in note['image_desc']])
                
            with open(os.path.join(self._parsed_dir, note_id + ".txt"),'w') as f:
                f.write(note['title'])
                f.write('\n\n')
                f.write(note['desc'])
                f.write('\n\n')
                f.write(text_in_image)
                f.write('\n\n')
                f.write(str(pois))
                f.write('\n\n')
                f.write(str(poi_retrieved))
                
            self.post_process(note)
            

    def recognize_images(self, note_id, images:list[str]):
        ocr_results = []
        if images and len(images) > 0:
            for idx, img in enumerate(images):
                r = requests.get(img)
                img_name = str(idx) + '.jpg'
                img_dir = os.path.join(self._img_dir, note_id)
                img_path = os.path.join(img_dir, img_name)
                os.makedirs(img_dir,exist_ok=True)
                with open(img_path,'wb') as f:
                    f.write(r.content)
                img_np = cv2.imread(img_path)
                h, w, c = img_np.shape
                img_data = {"img64": base64.b64encode(img_np).decode("utf-8"), "height": h, "width": w, "channels": c}
                # result = self._context.ocr_engine(img_data)
                result = self._context.ocr_engine(img_np)
                ocr_results.append(result)
        return ocr_results
                

    def parse_note(self, note_id:str, raw:str):
        meta = {}
        soup = BeautifulSoup(raw, "html.parser")
        desc = soup.find(name='meta',attrs={'name':'description'})
        if desc:
            desc = desc.attrs['content']
        else:
            return None
        
        meta['desc'] = desc
        
        keywords = soup.find(name='meta',attrs={'name':'keywords'})
        if keywords:
            meta['keywords'] = keywords.attrs['content']
        
        title = soup.find(name='meta', attrs={'name':'og:title'})
        if title:
            meta['title'] = title.attrs['content']
        else:
            meta['title'] = None
        
        comments = soup.find(name='meta',attrs={'name':'og:xhs:note_comment'})
        if comments:
            meta['comments'] = comments.attrs['content']
            
        likes = soup.find(name='meta',attrs={'name':'og:xhs:note_like'})
        if likes:
            meta['likes'] = likes.attrs['content']
            
        collects = soup.find(name='meta',attrs={'name':'og:xhs:note_collect'})
        if collects:
            meta['collects'] = collects.attrs['content']
        
        imgs = soup.find_all(name='meta', attrs={'name':'og:image'})
        meta['images'] = []
        if imgs and len(imgs)>0:
            for img in imgs:
                img_url = img.attrs['content'].replace('!','\u0021')
                meta['images'].append(img_url)

        ocr_results = self.recognize_images(note_id, meta['images'])
        if ocr_results and len(ocr_results) > 0:
            meta['image_desc'] = ocr_results
        else:
            meta['image_desc'] = None
                
        return meta

    def post_process(self, note) -> Iterator[Document]:
        pois = note['pois'] 
        retrieved_pois = note['retrieved_pois']
        
        for poi_name in pois.keys():
            desc = None if pois[poi_name]['desc'] == 'null' else pois[poi_name]['desc'] 
            addr = None if pois[poi_name]['address'] == 'null' else pois[poi_name]['address']
            
            if poi_name in retrieved_pois:
                score = float(retrieved_pois[poi_name].metadata['_score'])
                poi_id = retrieved_pois[poi_name].metadata['id']
                poi_source = retrieved_pois[poi_name].metadata['_source']
                poi = self._poi_service.get_poi_info(poi_id)
                if poi is None:
                    # data inconsistence!
                    # poi data is in vector store but missing in mysql, 
                    # put a log.
                    logging.warning(f"POI Data {poi_id} missing in mysql. Please check!")
                    continue
                
                new_desc = self.merge_poi_desc(desc_old=poi['desc'], desc_new = desc)
                if new_desc:
                    self._poi_service.update_poi_desc(poi_id = poi_id, poi_name = poi['name'], desc = new_desc, source=poi_source)
                if score >= 0.01:
                    # add alias name for the retrieved poi
                    self._poi_service.insert_poi_names(alias=[poi_name], poi_id = poi_id, poi_name=poi['name'], source=poi_source)
            else:
                # create xhs poi entity
                xhs_poi = {
                    'name': poi_name,
                    'id': '_'.join([POI_SOURCE_XHS, uuid.uuid4().hex]),
                    'address': addr,
                    'desc':desc,
                    'alias':[poi_name]
                }
                self._poi_service.insert_poi(poi = xhs_poi, source = POI_SOURCE_XHS)
            
    def merge_poi_desc(self, desc_old, desc_new) -> str:
            
        if desc_new is None or len(desc_new) == 0:
            return None
        
        if desc_old is None or len(desc_old) == 0:
            return desc_new
        
        desc = '\n'.join([desc_old, desc_new])
        prompt = generate_desc_merge_prompt(desc)
        async def async_iter():
            results = []
            async for answer_result in self._llm.generatorAnswer(prompt=prompt,
                                                    history=[],
                                                    streaming=False):
                results.append((answer_result))
            return results
        
        merged = None
        answer = async_run(async_iter())
        if answer and len(answer) > 0:
            raw_answer = answer[0].llm_output['answer'][6:]
            parsed = json.loads(raw_answer)
            merged = parsed['answer']
        if merged is None or len(merged) == 0:
            merged = desc
            
        return merged
