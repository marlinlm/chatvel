
from os.path import dirname
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
from tqdm import tqdm
import os
sys.path.append(dirname(dirname(__file__)))
from service.local_doc_qa import LocalDocQA, LocalFile
from utils.logger import debug_logger
from datetime import datetime
from arguments import parse_arg
from poi import osmpoi

XHS_HEADERS={'cookie':'web_session=040069b3fca2d9d40d070584c2344b819836aa;', 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'}

def upload_weblink(user_id, kb_id, local_doc_qa:LocalDocQA, url, headers):
    debug_logger.info("upload_weblink %s", url)

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    file_id, _ = local_doc_qa.mysql_client.add_file(user_id, kb_id, url, timestamp)
    local_file = LocalFile(user_id=user_id, kb_id=kb_id, file=url, file_id=file_id, file_name=url, embedding=local_doc_qa.embeddings, is_url=True, headers=headers)
    # asyncio.create_task(local_doc_qa.insert_files_to_faiss(1, 1, [local_file]))
    local_doc_qa.insert_files_to_faiss(user_id, kb_id, [local_file])
    return 


def upload_pois(user_id, kb_id, local_doc_qa:LocalDocQA, pois:list, files:list = None):
    debug_logger.info("upload_pois %s", user_id)
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

    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
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
        file_id, msg = local_doc_qa.mysql_client.add_file(user_id, kb_id, poi_name, timestamp, status='green')
        local_file = LocalFile(user_id, kb_id, poi, file_id, file_name, local_doc_qa.embeddings)
        local_doc_qa.mysql_client.update_file_path(file_id, local_file.file_path)
        local_files.append(local_file)
        local_doc_qa.mysql_client.add_poi(file_id, user_id, kb_id, doc_id, str(poi))
        # debug_logger.info(f"{file_name}, {file_id}, {msg}, {faq}")
    debug_logger.info(f"end insert {len(pois)} faqs to mysql, user_id: {user_id}, kb_id: {kb_id}")
    local_doc_qa.insert_files_to_faiss(user_id, kb_id, local_files)


def add_kb(user_id, user_name, kb_id, kb_name, local_doc_qa:LocalDocQA):
    local_doc_qa.mysql_client.new_knowledge_base(kb_id, user_id, kb_name, user_name)
    return

def list_docs(user_id, kb_id, local_doc_qa:LocalDocQA):
    debug_logger.info("list_docs %s", user_id)
    debug_logger.info("kb_id: {}".format(kb_id))
    data = []
    file_infos = local_doc_qa.mysql_client.get_files(user_id, kb_id)
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
            faq_info = local_doc_qa.mysql_client.get_faq(file_id)
            if faq_info:
                data[-1]['question'] = faq_info[2]
                data[-1]['answer'] = faq_info[3]

    return status_count, data

def search(local_doc_qa:LocalDocQA, kb_ids, query):
    return local_doc_qa.retrieve(query=query, kb_ids=kb_ids)

def qa(local_doc_qa:LocalDocQA, kb_ids, query):
    return local_doc_qa.get_knowledge_based_answer(query=query, kb_ids=kb_ids)

def update_beijing_poi(local_doc_qa:LocalDocQA):
    user_id=1
    kb_id='beijing_poi'
    kb_name='beijing_poi'
    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        add_kb(user_id=user_id, user_name='linmao', kb_id=kb_id, kb_name=kb_name, local_doc_qa=local_doc_qa)
    pois = osmpoi.update_beijing()
    upload_pois(user_id, kb_id, local_doc_qa, pois)
    
def upload_link(local_doc_qa:LocalDocQA, link):
    user_id=1
    kb_id='weblink'
    kb_name='weblink'
    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        add_kb(user_id=user_id, user_name='linmao', kb_id=kb_id, kb_name=kb_name, local_doc_qa=local_doc_qa)
    upload_weblink(user_id=user_id,
                kb_id=kb_id,
                local_doc_qa=local_doc_qa,
                url=link,
                headers=XHS_HEADERS)

def claw_xhs(local_doc_qa:LocalDocQA, dir_in:str):
    user_id = 1
    kb_id = 'xhs_beijing'
    kb_name = 'xhs_beijing'
    
    not_exist_kb_ids = local_doc_qa.mysql_client.check_kb_exist(user_id, [kb_id])
    if not_exist_kb_ids:
        add_kb(user_id=user_id, user_name='linmao', kb_id=kb_id, kb_name=kb_name, local_doc_qa=local_doc_qa)
    
    dir_out = os.path.join(dir_in, 'out')
    fs = os.listdir(dir_in)
    for f in fs:
        if(not f.endswith('.xhs')):
            continue
        f_in = os.path.join(dir_in, f)
        fn = os.path.basename(f_in)
        local_file = LocalFile(user_id, kb_id, f_in, None, fn, local_doc_qa.embeddings,headers=XHS_HEADERS)
        local_doc_qa.save_xhs_files_to_local(dir=dir_out, local_files=[local_file])

if __name__ == "__main__":
    args = parse_arg()
    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(args)
    
    dir_in = os.path.join(dirname(dirname(__file__)),'.data','beijing')
    claw_xhs(local_doc_qa, dir_in)

    
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