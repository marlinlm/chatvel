import numpy as np
import base64

from ocr.ocr import ChatvelOCR
from service.local_doc_qa import LocalDocQA
from utils.logger import debug_logger

class BaseService:
    def __init__(self, local_doc_qa:LocalDocQA):
        self.local_doc_qa:LocalDocQA = local_doc_qa


    def list_docs(self, user_id, kb_id):
        debug_logger.info("list_docs %s", user_id)
        debug_logger.info("kb_id: {}".format(kb_id))
        data = []
        file_infos = self.local_doc_qa.mysql_client.get_files(user_id, kb_id)
        status_count = {}
        msg_map = {'gray': "正在上传中，请耐心等待",
                'red': "split或embedding失败，请检查文件类型，仅支持[md,txt,pdf,jpg,png,jpeg,docx,xlsx,pptx,eml,csv]",
                'yellow': "faiss插入失败，请稍后再试", 'green': "上传成功"}
        for file_info in file_infos:
            status = file_info[2]
            if status not in status_count:
                status_count[status] = 1
            else:
                status_count[status] += 1
            data.append({"file_id": file_info[0], "file_name": file_info[1], "status": file_info[2], "bytes": file_info[3],
                        "content_length": file_info[4], "timestamp": file_info[5], "msg": file_info[6]})
            file_name = file_info[1]
            file_id = file_info[0]
            if file_name.endswith('.faq'):
                faq_info = self.local_doc_qa.mysql_client.get_faq(file_id)
                if faq_info:
                    data[-1]['question'] = faq_info[2]
                    data[-1]['answer'] = faq_info[3]

        return status_count, data

    def search(self, kb_ids, query):
        return self.local_doc_qa.retrieve(query=query, kb_ids=kb_ids)

    def qa(self, kb_ids, query):
        return self.local_doc_qa.get_knowledge_based_answer(query=query, kb_ids=kb_ids)
    
    def get_ocr_result(self, input: dict):
        img_file = input['img64']
        height = input['height']
        width = input['width']
        channels = input['channels']
        binary_data = base64.b64decode(img_file)
        img_array = np.frombuffer(binary_data, dtype=np.uint8).reshape((height, width, channels))
        ocr_res = self.ocr_reader(img_array)
        res = [line for line in ocr_res if line]
        return res
