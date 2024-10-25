import datetime
from typing import list
import re
from poi import osmpoi
import tqdm
import urllib.parse

from service.base_service import BaseService
from local_doc_qa import LocalDocQA
from utils.logger import debug_logger
from utils.general_utils import truncate_filename, check_and_transform_excel
from service.data_source import LocalFile

POI_NAME_KB_ID = 'beijing_poi'
POI_NAME_KB_NAME = 'beijing_poi'

class PoiService(BaseService):
    def __init__(self, local_doc_qa:LocalDocQA):
        super().__init__(local_doc_qa=local_doc_qa)
    
        
    def save_pois(self, user_id, kb_id = POI_NAME_KB_ID, pois:list = [], files:list = None):
        debug_logger.info("saving pois %s", user_id)
        file_status = {}
        if pois is None:
            pois = []
            for file in files:
                debug_logger.info('ori name: %s', file.name)
                file_name = urllib.parse.unquote(file.name, encoding='UTF-8')
                debug_logger.info('decode name: %s', file_name)
                # 删除掉全角字符
                file_name = re.sub(r'[\uFF01-\uFF5E\u3000-\u303F]', '', file_name)
                file_name = file_name.replace("/", "_")
                debug_logger.info('cleaned name: %s', file_name)
                file_name = truncate_filename(file_name)
                file_faqs = check_and_transform_excel(file.body)
                if isinstance(file_faqs, str):
                    file_status[file_name] = file_faqs
                else:
                    pois.extend(file_faqs)
                    file_status[file_name] = "success"

        not_exist_kb_ids = self.local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
        if not_exist_kb_ids:
            msg = "invalid kb_id: {}, please check...".format(not_exist_kb_ids)
            raise Exception(msg)

        data = []
        now = datetime.now()
        local_files = []
        timestamp = now.strftime("%Y%m%d%H%M")
        debug_logger.info(f"start insert {len(pois)} pois to mysql, user_id: {user_id}, kb_id: {kb_id}")
        exist_pois = []
        for poi in tqdm(pois):
            doc_id = poi['meta']['doc_id']
            poi_name = poi['meta']['name']
            if doc_id not in exist_pois:
                exist_pois.append(doc_id)
            else:
                debug_logger.info(f"pois {doc_id} already exists, skip it")
                continue
            file_name = f"POI_{doc_id}.poi"
            file_name = file_name.replace("/", "_").replace(":", "_")  # 文件名中的/和：会导致写入时出错
            file_id, msg = self.local_doc_qa.mysql_client.add_file(user_id, kb_id, poi_name, timestamp, status='green')
            local_file = LocalFile(user_id, kb_id, poi, file_id, file_name, self.local_doc_qa.embeddings)
            self.local_doc_qa.mysql_client.update_file_path(file_id, local_file.file_path)
            local_files.append(local_file)
            self.local_doc_qa.mysql_client.add_poi(file_id, user_id, kb_id, doc_id, str(poi))
            # debug_logger.info(f"{file_name}, {file_id}, {msg}, {faq}")
        debug_logger.info(f"end insert {len(pois)} faqs to mysql, user_id: {user_id}, kb_id: {kb_id}")
        self.local_doc_qa.insert_files_to_faiss(user_id, kb_id, local_files)
