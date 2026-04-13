FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel ninja && \
    pip install tqdm scikit-learn torchvision numpy==1.26.4 opencv-python-headless==4.10.0.84

RUN pip install -v --no-build-isolation -e .