from typing import Iterator, List, Optional, Set, Callable
from urllib.parse import urljoin, urldefrag
from os.path import dirname
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



from poi.poi_extractor import PosPoiExtractor


XHS_PREFIX = 'https://www.xiaohongshu.com/explore/'



class XhsLoader(BaseLoader):
    """Loads all child links from a given url."""

    def __init__(
        self,
        ocr_engine: Callable,
        poi_retriever: Callable,
        llm: object,
        file_path: str,
        headers: dict,
        save_raw = False,
        save_parsed = False,
        save_poi_extraction = False,
    ) -> None:
        """claws the urls from xhs search results.

        Args:
            file_path: The file path of the xhs search result.
            header: A dict of headers.
        """
        self.ocr_engine = ocr_engine
        self.poi_retriever = poi_retriever
        
        self.save_raw = save_raw
        self.save_parsed = save_parsed
        self.save_poi_extraction = save_poi_extraction
        self.file_path = file_path
        self.headers = headers
        self.file_dir = dirname(file_path)
        self.poi_extract_dir = os.path.join(self.file_dir, 'out', 'poi_extract')
        self.parsed_dir = os.path.join(self.file_dir, 'out', 'parsed')
        self.raw_dir = os.path.join(self.file_dir, 'out', 'raw')
        self.img_dir = os.path.join(self.file_dir, 'out', 'img')
        self.llm = llm
        
        from poi.llm_poi_extractor import LlmPoiExtractor
        self.poi_extractor = LlmPoiExtractor(poi_retriever = self.poi_retriever, llm = self.llm, save_poi_extraction = False, poi_extraction_save_dir = self.poi_extract_dir)
        # self.poi_extractor = PosPoiExtractor(poi_retriever = self.poi_retriever, save_poi_extraction = True, poi_extraction_save_dir = self.poi_extract_dir)
        
        os.makedirs(self.poi_extract_dir, exist_ok=True)
        os.makedirs(self.parsed_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)




    def load_images(self, id, images:list[str]):
        ocr_results = []
        if images and len(images) > 0:
            for idx, img in enumerate(images):
                r = requests.get(img)
                img_name = str(idx) + '.jpg'
                img_dir = os.path.join(self.img_dir, id)
                img_path = os.path.join(img_dir, img_name)
                os.makedirs(img_dir,exist_ok=True)
                with open(img_path,'wb') as f:
                    f.write(r.content)
                img_np = cv2.imread(img_path)
                h, w, c = img_np.shape
                img_data = {"img64": base64.b64encode(img_np).decode("utf-8"), "height": h, "width": w, "channels": c}
                result = self.ocr_engine(img_data)
                ocr_results.append(result)
        return ocr_results
                

    def parse(self, meta_in:dict, raw:str):
        meta = dict(meta_in)
        soup = BeautifulSoup(raw, "html.parser")
        desc = soup.find(name='meta',attrs={'name':'description'})
        if desc:
            desc = desc.attrs['content']
        else:
            return None
        
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

        ocr_results = self.load_images(meta['id'], meta['images'])
        if ocr_results and len(ocr_results) > 0:
            meta['image_desc'] = ocr_results
        else:
            meta['image_desc'] = None
                
        return Document(page_content=desc, metadata=meta)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load web pages."""
        
        with open(self.file_path, 'rb') as f:
            xhs_content = f.read()
        
        content_json = json.loads(xhs_content)
        if content_json['code'] == 0 and 'items' in content_json['data'] and len(content_json['data']['items']) > 0:
            for item in tqdm(content_json['data']['items']):
                id = item['id']
                if('note_card' in item):
                    card = item['note_card']
                    title = card['display_title']
                    url = XHS_PREFIX + id
                    meta = {'id':id, 'title':title, 'url':url}
                    response = requests.request(url=url,method='get',headers=self.headers)
                    if self.save_raw:
                        with open(os.path.join(self.raw_dir, id + ".xhs.xml"),'w') as f:
                            f.write(response.text)
                    document = None
                    if response.ok:
                        document = self.parse(meta, response.text)
                                                    
                    time.sleep(random.randint(1,6))
                    if document:
                        pois, poi_retrieved = self.poi_extractor.extract_poi(id = id, 
                                                                             text = document.page_content, 
                                                                             title = document.metadata['title'],
                                                                             from_image = document.metadata['image_desc'], 
                                                                             from_video=None)
                        document.metadata['pois'] = pois
                        document.metadata['retrieved_pois'] = poi_retrieved
                        if self.save_parsed:
                            text_in_image = ''
                            if document.metadata['image_desc']:
                                text_in_image = ''.join([''.join([t for t in segment]) for segment in document.metadata['image_desc']])
                            with open(os.path.join(self.parsed_dir, id + ".txt"),'w') as f:
                                f.write(document.metadata['title'])
                                f.write('\n\n')
                                f.write(document.page_content)
                                f.write('\n\n')
                                f.write(text_in_image)
                                f.write('\n\n')
                                f.write(str(pois))
                                f.write('\n\n')
                                f.write(str(poi_retrieved))
                            

                        yield document
                    
    # def load_from_raw(self) -> Iterator[Document]:
    #     """Lazy load web pages."""
    #     with open(self.file_path, 'rb') as f:
    #         xhs_content = f.read()
            
    #     content_json = json.loads(xhs_content)
    #     if content_json['code'] == 0 and 'items' in content_json['data'] and len(content_json['data']['items']) > 0:
    #         for item in tqdm(content_json['data']['items']):
    #             id = item['id']
    #             if('note_card' in item):
    #                 card = item['note_card']
    #                 title = card['display_title']
    #                 url = XHS_PREFIX + id
    #                 meta = {'id':id, 'title':title, 'url':url}

    #                 document = None
    #                 for ff in os.listdir(self.raw_dir):
    #                     if ff.startswith(id):
    #                         with open(os.path.join(self.raw_dir, ff)) as f:
    #                             raw = f.read()
    #                             document = self.parse(meta, raw)
    #                 if document:
    #                     document.metadata['pois'] = self.extract_poi(document.page_content)
    #                     yield document
                        
    def load(self) -> List[Document]:
        """Load web pages."""
        return list(self.lazy_load())
