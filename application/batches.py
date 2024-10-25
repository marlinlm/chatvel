
from os.path import dirname
import sys
sys.path.append(dirname(dirname(__file__)))
from config.dataset_config import POI_DATASET_DIR_OSM_BEIJING, POI_DATASET_NAME_OSM_BEIJING
from poi.poi_loader import OSMPoiDatasetLoader
from service.data_load_service import DataLoadService
from service.service_context import ServiceContext
from utils.logger import debug_logger
from datetime import datetime
from arguments import parse_arg

# def upload_weblink(user_id, kb_id, local_doc_qa:LocalDocQA, url, headers):
#     debug_logger.info("upload_weblink %s", url)

#     timestamp = datetime.now().strftime("%Y%m%d%H%M")
#     file_id, _ = local_doc_qa.mysql_client.add_file(user_id, kb_id, url, timestamp)
#     local_file = LocalFile(user_id=user_id, kb_id=kb_id, file=url, file_id=file_id, file_name=url, embedding=local_doc_qa.embeddings, is_url=True, headers=headers)
#     # asyncio.create_task(local_doc_qa.insert_files_to_faiss(1, 1, [local_file]))
#     local_doc_qa.insert_files_to_faiss(user_id, kb_id, [local_file])
#     return 


    
# def upload_link(local_doc_qa:LocalDocQA, link):
#     user_id=1
#     kb_id='weblink'
#     kb_name='weblink'
#     not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
#     if not_exist_kb_ids:
#         add_kb(user_id=user_id, user_name='linmao', kb_id=kb_id, kb_name=kb_name, local_doc_qa=local_doc_qa)
#     upload_weblink(user_id=user_id,
#                 kb_id=kb_id,
#                 local_doc_qa=local_doc_qa,
#                 url=link,
#                 headers=XHS_HEADERS)



if __name__ == "__main__":
    args = parse_arg()
    context = ServiceContext()
    context.init_cfg(args)

    poi_loader = OSMPoiDatasetLoader(context = context,
                                     data_name = POI_DATASET_NAME_OSM_BEIJING,
                                     data_dir = POI_DATASET_DIR_OSM_BEIJING,
                                     )
    loader_service = DataLoadService(context = context)
    loader_service.load_data([poi_loader])
    

    # dir_in = os.path.join(dirname(dirname(__file__)),'.data','beijing')
    # claw_xhs(local_doc_qa, dir_in)

    
    # upload_link(local_doc_qa, 'https://www.xiaohongshu.com/explore/6673c834000000001d016bad')
    
    # upload_weblink(user_id=user_id,
    #                kb_id=kb_id,
    #                local_doc_qa=local_doc_qa,
    #                url="https://www.xiaohongshu.com/explore/66f6d084000000002a034b91",
    #                headers={'cookie':'web_session=040069b3fca2d9d40d070584c2344b819836aa;', 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'})
    # print(list_docs(user_id=user_id, kb_id=kb_id, local_doc_qa=local_doc_qa))
    # print(search(local_doc_qa=local_doc_qa, kb_ids=['beijing_poi'], query='北京大学'))
    # for answer in qa(local_doc_qa=local_doc_qa, query='南沙', kb_ids=['1']):
    #     print(answer.llm_output["answer"])
    # update_beijing_poi(local_doc_qa)