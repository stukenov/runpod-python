ARG BASE_IMAGE=nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMAGE} as dev-base

ARG MODEL_URL
ENV MODEL_URL=${MODEL_URL}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        wget \
        bash \
        openssh-server \
        software-properties-common \
        git \
        build-essential \
        python3 \
        python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY . /app
RUN pip3 install -e .

CMD ["python3", "-u", "/app/my_worker.py"]