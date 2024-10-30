import sys
import platform
from os.path import dirname
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.dataset_config import DatasetConfig
from llm.llm_for_openai_api import OpenAILLM

sys.path.append(dirname(dirname(__file__)))
from config.model_config import VECTOR_SEARCH_TOP_K, CHUNK_SIZE, VECTOR_SEARCH_SCORE_THRESHOLD, OCR_MODEL_PATH
from db.mysql.mysql_client import KnowledgeBaseManager
from db.faiss.faiss_client import FaissClient
from rerank.rerank_backend import RerankBackend
from embedding.embedding_backend import EmbeddingBackend
from utils.general_utils import num_tokens
from ocr.ocr import ChatvelOCR

class SessionManager:
    def __init__(self):
        pass
        
    def get_current_user(self):
        # get current session info for each user session
        user_id:int = 1
        user_name:str = 'linmao'
        return user_id, user_name

class ServiceContext:
    def __init__(self):
        self.faiss_client: FaissClient = None
        self.mysql_client: KnowledgeBaseManager = None

        self.llm: object = None
        self.embeddings: EmbeddingBackend = None
        self.ocr_engine: ChatvelOCR = None
        self.local_rerank_backend: RerankBackend = None
        
        self.top_k: int = VECTOR_SEARCH_TOP_K
        self.chunk_size: int = CHUNK_SIZE
        self.score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD
        self.mode: str = None
        self.use_cpu: bool = True
        self.llm_model_name: str = None
        self.web_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "!", "！", "?", "？", "；", ";", "……", "…", "、", "，", ",", " ", ""],
            chunk_size=800,
            chunk_overlap=200,
            length_function=num_tokens,
        )
        
        self.config = DatasetConfig()
        self.session_manager = SessionManager()

    def init_cfg(self, args=None):
        self.rerank_top_k = int(args.model_size[0])
        self.use_cpu = args.use_cpu
        if args.use_openai_api:
            self.llm_model_name = args.openai_api_model_name
        else:
            self.llm_model_name = args.model.split('/')[-1]
        if platform.system() == 'Linux':
            if args.use_openai_api:
                self.llm: OpenAILLM = OpenAILLM(args)
            else:
                from llm.llm_for_fastchat import OpenAICustomLLM
                self.llm: OpenAICustomLLM = OpenAICustomLLM(args)
        #     from rerank.rerank_onnx_backend import RerankOnnxBackend
        #     # from embedding.embedding_onnx_backend import EmbeddingOnnxBackend
        #     from embedding.embedding_torch_backend import EmbeddingTorchBackend
        #     self.local_rerank_backend: RerankOnnxBackend = RerankOnnxBackend(self.use_cpu)
        #     # self.embeddings: EmbeddingOnnxBackend = EmbeddingOnnxBackend(self.use_cpu)
        #     self.embeddings: EmbeddingTorchBackend = EmbeddingTorchBackend(self.use_cpu)
        # else:
        #     if args.use_openai_api:
        #         self.llm: OpenAILLM = OpenAILLM(args)
        #     else:
        #         from llm.llm_for_llamacpp import LlamaCPPCustomLLM
        #         self.llm: LlamaCPPCustomLLM = LlamaCPPCustomLLM(args)
        #     from rerank.rerank_torch_backend import RerankTorchBackend
        #     from embedding.embedding_torch_backend import EmbeddingTorchBackend
        #     self.local_rerank_backend: RerankTorchBackend = RerankTorchBackend(self.use_cpu)
        #     self.embeddings: EmbeddingTorchBackend = EmbeddingTorchBackend(self.use_cpu)
            
        # self.mysql_client = KnowledgeBaseManager()
        # self.faiss_client = FaissClient(self.mysql_client, self.embeddings)
        # self.ocr_engine = ChatvelOCR(model_dir=OCR_MODEL_PATH, device="cpu")