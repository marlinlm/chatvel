import sys
import traceback
from os.path import dirname
from typing import List

sys.path.append(dirname(dirname(__file__)))
from utils.logger import debug_logger
from loader.chatvel_loader import ChatvelLoader
from service.service_context import ServiceContext


class DataLoadService:
    def __init__(self, context: ServiceContext):
        self._context: ServiceContext = context
        if context is None:
            raise Exception('The service context should not be None!')

    def load_data(self, loaders: List[ChatvelLoader]):
        success_list = []
        failed_list = []

        for loader in loaders:
            try:
                docs = loader.load_and_split()
            except Exception as e:
                error_info = f'error: {traceback.format_exc()}'
                debug_logger.error(error_info)
                self._context.mysql_client.update_loading_status(loader.id, status='red', reason=error_info)
                failed_list.append(loader)
                continue
            if docs is None or len(docs) == 0:
                self._context.mysql_client.update_loading_status(loader.id, status='red', reason='上传文件内容为空，请检查文件内容')
                debug_logger.info(f'上传文件内容为空，请检查文件内容')
                continue
            self._context.mysql_client.update_loading_status(loader.id, status='green', reason=" ")
            success_list.append(loader)
        debug_logger.info(
            f"insert_to_faiss: success num: {len(success_list)}, failed num: {len(failed_list)}")

    