from typing import List

from loader.xhs_loader import XhsLoader
from service.data_load_service import DataLoadService
from service.data_source import LocalFile
from service.service_context import ServiceContext

class XhsHandler:
    def __init__(self, 
                 context:ServiceContext,
                 data_load_service:DataLoadService,
                 xhs_headers,):
        self._xhs_headers = xhs_headers
        self._data_load_service = data_load_service
        self._context = context
    

    def claw_xhs(self, dir_in:str, kb_id:str, kb_name:str, user_id:int, user_name:str):
        user_id = 1
        kb_id = 'xhs_beijing'
        kb_name = 'xhs_beijing'
        
        xhs_loader = XhsLoader(
            context = self._context,
            file_path = dir_in,
            headers = self._xhs_headers,
            user_id = user_id,
            user_name = user_name,
            kb_id = kb_id,
            kb_name = kb_name)
        
        self._data_load_service.load_data([xhs_loader])
        
        
        
        
        
        
        
        
        not_exist_kb_ids = self.local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
        if not_exist_kb_ids:
            self.add_kb(user_id=user_id, user_name='linmao', kb_id=kb_id, kb_name=kb_name)
        
        dir_out = os.path.join(dir_in, 'out')
        fs = os.listdir(dir_in)
        for f in fs:
            if(not f.endswith('.xhs')):
                continue
            f_in = os.path.join(dir_in, f)
            fn = os.path.basename(f_in)
            local_file = LocalFile(user_id, kb_id, f_in, None, fn, self.local_doc_qa.embeddings,headers=self.xhs_header)
            local_doc_qa.save_xhs_files_to_local(dir=dir_out, local_files=[local_file])
            
            if not local_file.docs or len(local_file.docs) == 0:
                continue
            
            pois = {}
            extracted_pois = {}
            for doc in local_file.docs:
                my_pois = doc.metadata['pois'] 
                my_extracted_pois = doc.metadata['extracted_pois']
                if my_extracted_pois:
                    extracted_pois.update(my_extracted_pois)
                
                if my_pois and len(my_pois.keys()) > 0:
                    for poi_name in my_pois.keys():
                        desc = [] if pois[poi_name]['desc'] == 'null' else [pois[poi_name]['desc']]
                        addr = [] if pois[poi_name]['address'] == 'null' else [pois[poi_name]['address']]
                        if not (poi_name in pois):
                            pois[poi_name]={}
                            pois[poi_name]['desc']=[]
                            pois[poi_name]['address'] = []
                        pois[poi_name]['desc'].expand(desc)
                        pois[poi_name]['address'].expand(addr)
                        
                
            for poi_name in pois.keys():
                desc = '\n'.join(pois[poi_name]['desc'])
                addr = '\n'.join(pois[poi_name]['address'])
                
                if poi_name in extracted_pois:
                    score = float(extracted_pois[poi_name][2])
                    poi_id = extracted_pois[poi_name][1]
                    # use existing poi entity
                    
                else:
                    # create virtual poi entity