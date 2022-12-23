# use torch 1.11, as 1.12 & 1.13 for jetson are not built with distributed support
FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.11-py3

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# execute python command to get model from huggingface
RUN python3 -c "import huggingface_hub; huggingface_hub.hf_hub_download(repo_id='facebook/opt-1.3b', filename='pytorch_model.bin')"

COPY . .
RUN python3 -c "import utils; utils.download_checkpoint('facebook/opt-1.3b')"
