FROM python:3.10-slim
LABEL maintainer="KageRyo"

# 安裝基本工具
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /workspace

# 複製 requirements.txt 並安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 設置默認 shell 為 bash
SHELL ["/bin/bash", "-c"]
