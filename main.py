import numpy as np
import torch
torch.set_grad_enabled(False)
import torch.distributed.rpc as rpc
from torch.distributed.nn.api.remote_module import RemoteModule
RemoteModule._backward_hooks = {}
RemoteModule._forward_hooks = {}
RemoteModule._forward_pre_hooks = {}
import time
import os
import accelerate
import yaml
from transformers import AutoConfig
from transformers import GPT2Tokenizer, OPTForCausalLM
from typing import Optional, Tuple
import argparse
import socket

import logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)

# we have to wrap a library function to make it ignore Identity layers
import accelerate.utils.modeling
orig_set_module_tensor_to_device = accelerate.utils.modeling.set_module_tensor_to_device
def permisive_set_module_tensor_to_device(*args, **kwargs):
  try:
    return orig_set_module_tensor_to_device(*args, **kwargs)
  except AttributeError:
    pass
accelerate.utils.modeling.set_module_tensor_to_device = permisive_set_module_tensor_to_device

DEVICE='cuda'

overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}

# argument parser
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--leader-ip', type=str, default='127.0.0.1')
arg_parser.add_argument('--leader-port', type=str, default='29500')
arg_parser.add_argument('--task-file', type=str)
arg_parser.add_argument('--world-size', type=int)
arg_parser.add_argument('--rank', type=int)
# available: 125m, 350m, 1.3b, 2.7b, 6.7b, 13b, 30b, 66b
arg_parser.add_argument('--model-name', type=str, default='facebook/opt-1.3b')
args = arg_parser.parse_args()
# we may use a hostname instead of an IP address
while True:
  try:
    args.leader_ip = socket.gethostbyname(args.leader_ip)
    break
  except:
    logger.debug(f'failed to resolve {args.leader_ip}, retrying...')
    time.sleep(0.1)
logger.debug(f'resolved leader_ip {args.leader_ip}')

LLM = OPTForCausalLM
from utils import find_checkpoint_path
checkpoint_path = find_checkpoint_path(args.model_name)

os.environ['MASTER_ADDR'] = args.leader_ip
os.environ['MASTER_PORT'] = args.leader_port
if not ('GLOO_SOCKET_IFNAME' in os.environ and 'TP_SOCKET_IFNAME' in os.environ and os.environ['GLOO_SOCKET_IFNAME'] == os.environ['TP_SOCKET_IFNAME']):
  logger.warning('you may need to set GLOO_SOCKET_IFNAME and TP_SOCKET_IFNAME to make torch.distributed.rpc init on multiple machines')


def opt_call(prompt, options):
  rets = {'choices': [{'logprobs': {'tokens': [], 'token_logprobs': []}} for _ in range(len(options))]}
  prompt_tokens = tokenizer(prompt, padding=True, return_tensors="pt")
  prompt_tokens_len = len(prompt_tokens.input_ids[0])

  options_tokens = tokenizer(options, padding=True, return_tensors="pt")
  options_tokens.input_ids = options_tokens.input_ids[:, 1:].to(DEVICE) # skip <s/>
  options_tokens.attention_mask = options_tokens.attention_mask[:, 1:].to(DEVICE) # skip <s/>
  # set pad tokens to another special token to avoid detected as right padding
  # does not affect the result as we will stop at the end of each option
  options_tokens.input_ids[options_tokens.attention_mask == 0] = 2
  options_tokens_lens = [sum(options_tokens.attention_mask[i]) for i in range(len(options))]

  # concat prompt and options for later batch input
  prompt_tokens.input_ids = prompt_tokens.input_ids.repeat(len(options), 1).to(DEVICE)
  prompt_tokens.attention_mask = prompt_tokens.attention_mask.repeat(len(options), 1).to(DEVICE)
  input_tokens = torch.concat([prompt_tokens.input_ids, options_tokens.input_ids], dim=1)
  input_masks = torch.concat([prompt_tokens.attention_mask, options_tokens.attention_mask], dim=1)

  for i in range(prompt_tokens_len, len(input_tokens[0])):
    ret = model.generate(input_tokens[:, :i], attention_mask=input_masks[:, :i], max_new_tokens=1, output_scores=True, return_dict_in_generate=True, renormalize_logits=True, temperature=0.0, use_cache=False)
    if DEVICE == 'cuda':
      torch.cuda.empty_cache()
    for j in range(len(options)):
      if i - prompt_tokens_len >= options_tokens_lens[j]:
        continue
      rets['choices'][j]['logprobs']['tokens'].append(tokenizer.decode(input_tokens[j][i]))
      rets['choices'][j]['logprobs']['token_logprobs'].append(ret['scores'][0][j][input_tokens[j][i]].item())

  return rets


def gpt3_scoring(query, options, limit_num_options=None, verbose=False, print_tokens=False):
  if limit_num_options:
    options = options[:limit_num_options]
  verbose and print("Scoring", len(options), "options")
  response = opt_call(query, options)

  scores = {}
  for option, choice in zip(options, response["choices"]):
    tokens = choice["logprobs"]["tokens"]
    token_logprobs = choice["logprobs"]["token_logprobs"]

    total_logprob = 0
    for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
      print_tokens and print(token, token_logprob)
      total_logprob += token_logprob
    scores[option] = total_logprob

  for i, option in enumerate(sorted(scores.items(), key=lambda x : -x[1])):
    verbose and print(option[1], "\t", option[0])
    if i >= 10:
      break

  return scores, response


def normalize_scores(scores):
  max_score = max(scores.values())  
  normed_scores = {key: np.clip(scores[key] / max_score, 0, 1) for key in scores}
  return normed_scores


def make_plan(context, command, options, terminate_string, affordance_scores, max_tasks=5, verbose=False, engine="text-ada-001", print_tokens=False):
  gpt3_prompt = context + "\n" + command + "\n"

  all_llm_scores = []
  all_affordance_scores = []
  all_combined_scores = []
  num_tasks = 0
  selected_task = ""
  steps_text = []
  while not selected_task == terminate_string:
    num_tasks += 1
    if num_tasks > max_tasks:
      break

    llm_scores, _ = gpt3_scoring(gpt3_prompt, options, verbose=True, print_tokens=False)
    combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options if num_tasks > 1 or option != terminate_string}
    combined_scores = normalize_scores(combined_scores)
    selected_task = max(combined_scores, key=combined_scores.get)
    steps_text.append(selected_task)
    print(num_tasks, "Selecting: ", selected_task)
    gpt3_prompt += selected_task + "\n"

    all_llm_scores.append(llm_scores)
    all_affordance_scores.append(affordance_scores)
    all_combined_scores.append(combined_scores)

  print('**** Solution ****')
  print(command)
  for i, step in enumerate(steps_text):
    if step == '' or step == terminate_string:
      break
    print('Step ' + str(i) + ': ' + step)
  return steps_text


def convert_meta_to_device(module, device):
  """Replace all meta tensors in the module with real tensors, recursively."""
  for name, param in module._parameters.items():
    if param is not None and param.is_meta:
      module._parameters[name] = param.new_empty(param.size(),device=device)
  for name, buf in module._buffers.items():
    if buf is not None and buf.is_meta():
      module._buffers[name] = buf.new_empty(buf.size(),device='device')
  for name, module in module._modules.items():
    if module is not None:
      convert_meta_to_device(module, device)


class SequentialDecoders(torch.nn.Sequential):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def forward(self, hidden_states: torch.Tensor,
              attention_mask: Optional[torch.Tensor] = None,
              layer_head_mask: Optional[torch.Tensor] = None,
              output_attentions: Optional[bool] = False,
              use_cache: Optional[bool] = False,
              past_key_value: Optional[Tuple[torch.Tensor]] = None):
    # doesn't work for the following params, early fail if used
    assert layer_head_mask is None
    assert output_attentions is False
    assert past_key_value is None
    assert not self.training

    for idx, decoder_layer in enumerate(self):
      layer_outputs = decoder_layer(
          hidden_states,
          attention_mask=attention_mask,
          layer_head_mask=layer_head_mask,
          past_key_value=past_key_value,
          output_attentions=output_attentions,
          use_cache=use_cache,
      )

      hidden_states = layer_outputs[0]

    if use_cache:
      layer_outputs += (past_key_value,)

    if DEVICE == 'cuda':
      torch.cuda.empty_cache()

    return layer_outputs


def load_part(model_name, start_idx, end_idx, device):
  """Load layers [start_idx:end_idx], then return as a module list."""
  config = AutoConfig.from_pretrained(model_name)
  with accelerate.init_empty_weights():
    model = LLM._from_config(config)
    model.eval()
    model.half()
  # skip loading other layers by replacing them with torch.nn.Identity
  for i in range(0, start_idx):
    model.model.decoder.layers[i] = torch.nn.Identity()
  for i in range(end_idx, len(model.model.decoder.layers)):
    model.model.decoder.layers[i] = torch.nn.Identity()

  for i in range(start_idx, end_idx):
    convert_meta_to_device(model.model.decoder.layers[i], device)

  # accelerate.load_and_dispatch() causes pickling error, so use torch.load() instead
  state_dict = torch.load(checkpoint_path)
  model.load_state_dict(state_dict, strict=False)

  layers = model.model.decoder.layers[start_idx:end_idx]
  ret = SequentialDecoders(*layers)
  ret.eval()
  return ret


def estimate_size(m):
  """Estimate size of a model in Mbytes."""
  return sum(p.numel() * p.element_size() for p in m.parameters())*2/1024/1024


if __name__ == "__main__":
  options = rpc.TensorPipeRpcBackendOptions()
  for i in range(args.world_size):
    for j in range(torch.cuda.device_count()):
      options.set_device_map(f'worker{i}', {f'cuda:{j}': f'cuda:{j}'})
  logger.debug('initializing rpc')
  torch.distributed.rpc.init_rpc(f'worker{args.rank}', rank=args.rank, world_size=args.world_size, rpc_backend_options=options)
  logger.debug('rpc initialized')

  if args.rank == 0:
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name, device_map='sequential')
    config = AutoConfig.from_pretrained(args.model_name)
    logger.debug('loading model structure')
    with accelerate.init_empty_weights():
      model = LLM._from_config(config)
      model.eval()
      model.half()

    # replace all layers with torch.nn.Identity
    num_layers = len(model.model.decoder.layers)
    model.model.decoder.layers = torch.nn.ModuleList([])

    # then load the model weights
    logger.debug('loading model weights excluding decoder layers')
    accelerate.load_checkpoint_and_dispatch(model, checkpoint_path, device_map='sequential')

    remote_layers = torch.nn.ModuleList()
    remote_layers.eval()
    for r, arr in enumerate(np.array_split(range(num_layers), args.world_size)):
      logger.debug(f'loading model decoder layers for worker {r}/{args.world_size-1}')
      start = arr[0]
      end = arr[-1] + 1

      rref = rpc.remote(f'worker{r}', load_part, args=(args.model_name, start, end, DEVICE))
      module = RemoteModule.init_from_module_rref(f'worker{r}', rref)
      remote_layers.append(module)
    model.model.decoder.layers = remote_layers

    logger.debug('start')
    while True:
        logger.debug('===========================')
        tbegin = time.time()
        task_def = yaml.load(open(args.task_file, 'r'), Loader=yaml.FullLoader)
        context = task_def['context']
        command = task_def['command']
        options = {opt['text']: opt['affordance'] for opt in task_def['options']}
        termination_string = task_def['termination_string']
        max_steps = task_def['max_steps']
        options[termination_string] = 1/(max_steps+1)
        make_plan(context, command, list(options.keys()), termination_string, options, max_steps, verbose=True, print_tokens=False)
        tend = time.time()
        logger.debug(f'time spent: {tend-tbegin}')

  rpc.shutdown()
