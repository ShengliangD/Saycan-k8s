import os

def download_checkpoint(model_name):
  import huggingface_hub
  for fname in ['config.json', 'merges.txt', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json', 'pytorch_model.bin']:
    huggingface_hub.hf_hub_download(repo_id=model_name, filename=fname)

def find_checkpoint_path(model_name):
  parts = model_name.split('/')
  try:
    dirs = os.listdir(f'/root/.cache/huggingface/hub/models--{parts[0]}--{parts[1]}/snapshots')
  except FileNotFoundError:
    dirs = []
  if len(dirs) == 0:
    download_checkpoint(model_name)
    dirs = os.listdir(f'/root/.cache/huggingface/hub/models--{parts[0]}--{parts[1]}/snapshots')
  return f'/root/.cache/huggingface/hub/models--{parts[0]}--{parts[1]}/snapshots/{dirs[0]}/pytorch_model.bin'
