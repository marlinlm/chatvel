import os
from os.path import dirname
import sys


sys.path.append(dirname(dirname(__file__)))
from service.travel_qa_service import TravelQaService
# from loader.xhs_loader import XhsLoader
# from poi.poi_loader import OSMPoiDatasetLoader
# from service.data_load_service import DataLoadService
from config.dataset_config import POI_DATASET_DIR_OSM_BEIJING, POI_DATASET_NAME_OSM_BEIJING, XHS_DATASET_DIR
from service.service_context import ServiceContext
from arguments import parse_arg

from query.query_decompose import QueryDecomposer

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

# def get_xhs_files(file_dir:str):
#     loaders = []
#     for file_name in os.listdir(file_dir):
#         if file_name.endswith('.xhs'):
#             path = os.path.join(XHS_DATASET_DIR, file_name)
#             xhs_loader = XhsLoader(context = context, file_path=path)
#             loaders.append(xhs_loader)
#     return loaders

if __name__ == "__main__":
    args = parse_arg()
    context = ServiceContext()
    context.init_cfg(args)
    
    qa = TravelQaService(context = context)
    query = '游玩北京的园林，前门大街，体验豆汁等北京传统美食，北京旅游，不要太拥挤'
    pois = qa.qa(query)
    
    
    
    
    loader_service = DataLoadService(context = context)

    # poi_loader = OSMPoiDatasetLoader(context = context,
    #                                  data_name = POI_DATASET_NAME_OSM_BEIJING,
    #                                  data_dir = POI_DATASET_DIR_OSM_BEIJING,
    #                                  )
    # loader_service.load_data([poi_loader])
    # loader_service.load_data(get_xhs_files(file_dir = XHS_DATASET_DIR))
