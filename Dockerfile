# modified from https://github.com/xfbs/docker-openpcdet
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

RUN apt update

RUN apt install -y python3.6 python3-pip apt-transport-https ca-certificates gnupg software-properties-common wget git ninja-build libboost-dev build-essential

RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html


# Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - > /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update && apt install -y cmake

# Install spconv
COPY spconv /code/spconv
WORKDIR /code/spconv
ENV SPCONV_FORCE_BUILD_CUDA=1
RUN python3 setup.py bdist_wheel
RUN pip3 install dist/*.whl

# Install LLVM 10
WORKDIR /code
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 10

# OpenPCDet dependencies fail to install unless LLVM 10 exists on the system
# and there is a llvm-config binary available, so we have to symlink it here.
RUN ln -s /usr/bin/llvm-config-10 /usr/bin/llvm-config

RUN pip3 install --upgrade pip

ARG TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX"

# Install CenterPoint
COPY CenterPoint-dev /code/CenterPoint-dev
WORKDIR /code/CenterPoint-dev
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 0
RUN pip3 install -r requirements.txt
RUN pip3 uninstall opencv-python  --yes
RUN pip3 install opencv-python-headless 
RUN bash setup.sh

RUN chmod -R +777 /code 

WORKDIR /code/CenterPoint-dev
ENV PYTHONPATH "${PYTHONPATH}:/code/CenterPoint-dev"
