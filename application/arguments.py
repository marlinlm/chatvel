import os
import platform
from argparse import ArgumentParser
import torch
from utils.general_utils import download_file, get_gpu_memory_utilization, check_package_version
from utils.logger import debug_logger
from config import model_config 
from config.model_config import DT_7B_MODEL_PATH, \
    DT_7B_DOWNLOAD_PARAMS, DT_3B_MODEL_PATH, DT_3B_DOWNLOAD_PARAMS, PDF_MODEL_PATH
from modelscope import snapshot_download
    
def parse_arg():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os_system = platform.system()
    parser = ArgumentParser()
    parser.add_argument('--host', dest='host', default='0.0.0.0', help='set host for qanything server')
    parser.add_argument('--port', dest='port', default=8777, type=int, help='set port for qanything server')
    parser.add_argument('--workers', dest='workers', default=4, type=int, help='sanic server workers number')
    # 是否使用GPU
    parser.add_argument('--use_cpu', dest='use_cpu', action='store_true', help='use gpu')
    # 是否使用Openai API
    parser.add_argument('--use_openai_api', dest='use_openai_api', action='store_true', help='use openai api')
    # parser.add_argument('--openai_api_base', dest='openai_api_base', default='https://api.openai.com/v1', type=str,
    #                     help='openai api base url')
    parser.add_argument('--openai_api_base', dest='openai_api_base', default='https://gpt-api.hkust-gz.edu.cn/v1', type=str,
                        help='openai api base url')
    parser.add_argument('--openai_api_key', dest='openai_api_key', default='99ec7acbd5014b37bf5c5c1e399fc771b4503a8a223e490f875c1c90c908dbf0', type=str,
                        help='openai api key')
    parser.add_argument('--openai_api_model_name', dest='openai_api_model_name', default='gpt-3.5-turbo-1106', type=str,
                        help='openai api model name')
    parser.add_argument('--openai_api_context_length', dest='openai_api_context_length', default='4096', type=str,
                        help='openai api content length')
    #  必填参数
    parser.add_argument('--model_size', dest='model_size', default='7B', help='set LLM model size for qanything server')
    parser.add_argument('--device_id', dest='device_id', default='0', help='cuda device id for qanything server')
    args = parser.parse_args()
    
    print('use_cpu:', args.use_cpu, flush=True)
    print('use_openai_api:', args.use_openai_api, flush=True)

    # 输出用户启动的端口
    print(f"The server is starting on port: {args.port}")

    if os_system != 'Darwin':
        if not args.use_cpu:
            cuda_version = torch.version.cuda
            if cuda_version is None:
                raise ValueError("CUDA is not installed.")
            elif float(cuda_version) < 12:
                raise ValueError("CUDA version must be 12.0 or higher.")

        python_version = platform.python_version()
        python3_version = python_version.split('.')[1]
        os_system = platform.system()
        if os_system == "Windows":
            raise ValueError("The project must be run in the WSL environment on Windows system.")
        if os_system != "Linux":
            raise ValueError(f"Unsupported system: {os_system}")
        system_name = 'manylinux_2_28_x86_64'
        glibc_info = platform.libc_ver()
        if glibc_info[0] != 'glibc':
            raise ValueError(f"Unsupported libc: {glibc_info[0]}, 请确认系统是否为Linux系统。")
        glibc_version = float(glibc_info[1])
        if glibc_version < 2.28:
            if not check_package_version("onnxruntime", "1.16.3"):
                print(f"当前系统glibc版本为{glibc_version}<2.28，无法使用onnxruntime-gpu(cuda12.x)，将安装onnxruntime来代替", flush=True)
                os.system("pip install onnxruntime")
        elif not args.use_cpu:
            # 官方发布的1.17.1不支持cuda12以上的系统，需要根据官方文档:https://onnxruntime.ai/docs/install/里提到的地址手动下载whl
            if not check_package_version("onnxruntime-gpu", "1.17.1"):
                download_url = f"https://aiinfra.pkgs.visualstudio.com/PublicPackages/_apis/packaging/feeds/9387c3aa-d9ad-4513-968c-383f6f7f53b8/pypi/packages/onnxruntime-gpu/versions/1.17.1/onnxruntime_gpu-1.17.1-cp3{python3_version}-cp3{python3_version}-{system_name}.whl/content"
                debug_logger.info(f'开始从{download_url}下载onnxruntime，也可以手动下载并通过pip install *.whl安装')
                whl_name = f'onnxruntime_gpu-1.17.1-cp3{python3_version}-cp3{python3_version}-{system_name}.whl'
                download_file(download_url, whl_name)
                exit_status = os.system(f"pip install {whl_name}")
                if exit_status != 0:
                    # raise ValueError(f"安装onnxruntime失败，请手动安装{whl_name}")
                    debug_logger.warning(f"安装onnxruntime-gpu失败，将安装onnxruntime来代替")
                    print(f"安装onnxruntime-gpu失败，将安装onnxruntime来代替", flush=True)
                    os.system("pip install onnxruntime")
        if not args.use_openai_api:
            if not check_package_version("vllm", "0.2.7"):
                os.system(f"pip install vllm==0.2.7 -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn")

            from vllm.engine.arg_utils import AsyncEngineArgs

            parser = AsyncEngineArgs.add_cli_args(parser)
            args = parser.parse_args()

    else:
        # mac下ocr依赖onnxruntime
        if not check_package_version("onnxruntime", "1.17.1"):
            os.system("pip install onnxruntime -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn")
        # torch==2.1.2
        # torchvision==0.16.2
        # os.system("pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn")
        # os.system("pip install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn")
        if not args.use_openai_api:
            if not check_package_version("llama_cpp_python", "0.2.60"):
                os.system(f'CMAKE_ARGS="-DLLAMA_METAL_EMBED_LIBRARY=ON -DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn')

    model_download_params = None
    if not args.use_openai_api:
        model_size = args.model_size
        if os_system == "Linux" and not args.use_cpu:
            model_config.CUDA_DEVICE = args.device_id
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
            args.gpu_memory_utilization = get_gpu_memory_utilization(model_size, args.device_id)
            debug_logger.info(f"GPU memory utilization: {args.gpu_memory_utilization}")
        if model_size == '3B':
            args.model = DT_3B_MODEL_PATH
            model_download_params = DT_3B_DOWNLOAD_PARAMS
        elif model_size == '7B':
            args.model = DT_7B_MODEL_PATH
            model_download_params = DT_7B_DOWNLOAD_PARAMS
        else:
            raise ValueError(f"Unsupported model size: {model_size}, supported model size: 3B, 7B")

    # 如果模型不存在, 下载模型
    if args.use_openai_api:
        debug_logger.info(f'使用openai api {args.openai_api_model_name} 无需下载大模型')
    elif not os.path.exists(args.model):
        debug_logger.info(f'开始下载大模型：{model_download_params}')
        # if os_system == 'Darwin':
        #     cache_dir = model_file_download(**model_download_params)
        #     debug_logger.info(f'模型下载完毕！{cache_dir}')
        # else:
        #     cache_dir = snapshot_download(**model_download_params)
        cache_dir = snapshot_download(**model_download_params)
        debug_logger.info(f'模型下载完毕！{cache_dir}')
        # output = subprocess.check_output(['ln', '-s', cache_dir, args.model], text=True)
        # debug_logger.info(f'模型下载完毕！cache地址：{cache_dir}, 软链接地址：{args.model}')
        debug_logger.info(f"CUDA_DEVICE: {model_config.CUDA_DEVICE}")
    else:
        debug_logger.info(f'{args.model}路径已存在，不再重复下载大模型（如果下载出错可手动删除此目录）')
        debug_logger.info(f"CUDA_DEVICE: {model_config.CUDA_DEVICE}")
        
    return args

    
