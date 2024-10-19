FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# ARG username
# ARG password

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt update --fix-missing && \
        apt install python3.11 python3.11-distutils -y && \
        apt install python3-pip -y && \
        ln -sf /usr/bin/python3.11 /usr/bin/python && \
        ln -sf /usr/bin/pip3 /usr/bin/pip

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN apt-get install git -y


WORKDIR /root


COPY ./id_rsa ./.ssh/id_rsa
# SHELL ["/bin/bash", "-c"]
# RUN git clone git@github.com:marlinlm/chatvel.git
# RUN git clone https://${username}:${password}@github.com/marlinlm/chatvel.git
RUN git clone https://github.com/marlinlm/chatvel.git
# RUN pip install -r ./chatvel/requirements.txt

# CMD ["/bin/echo", "git clone git@github.com:marlinlm/chatvel.git"]
# CMD ["git", "pull", "git@github.com:marlinlm/chatvel.git"]
CMD ["pip", "install", "-r", "/root/chatvel/requirements.txt"]