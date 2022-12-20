# use torch 1.11, as 1.12 & 1.13 for jetson are not built with distributed support
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.11-py3

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY . .
