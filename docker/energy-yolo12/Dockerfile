    # Ubuntu22.04環境ベースのCUDA Toolkit 11.8
#docker-compose up -d --build
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home/hiromu/energy

RUN apt update && \
    apt install -y \
    build-essential \
    libbz2-dev \
    libdb-dev \
    libreadline-dev \
    libffi-dev \
    libgdbm-dev \
    liblzma-dev \
    libncursesw5-dev \
    libsqlite3-dev \
    libssl-dev \
    zlib1g-dev \
    uuid-dev \
    tk-dev \
    wget \
    bzip2 \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1

RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tar.xz && \
    tar xJf Python-3.11.9.tar.xz && \
    cd Python-3.11.9 && \
    ./configure && \
    make && \
    make install


RUN pip3 install --no-cache-dir -U pip

RUN pip3 install --no-cache-dir torch==2.2.2+cu118 torchvision==0.17.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install timm

RUN pip install opencv-python

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt install -y nodejs


RUN pip install scipy \
    matplotlib \
    japanize_matplotlib \
    pandas \
    cupy \
    ultralytics \
    "ocrd-fork-pylsd == 0.0.3" \
    optuna \
    scikit-learn

RUN pip install --upgrade torch torchvision
