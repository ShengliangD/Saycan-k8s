import numpy as np
import torch
import time

DEVICE='cuda'

overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}

from transformers import GPT2Tokenizer, OPTForCausalLM
# available: 125m, 350m, 1.3b, 2.7b, 6.7b, 13b, 30b, 66b
model = OPTForCausalLM.from_pretrained("facebook/opt-125m", device_map="auto")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m", device_map='auto')

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
    ret = model.generate(input_tokens[:, :i], attention_mask=input_masks[:, :i], max_new_tokens=1, output_scores=True, return_dict_in_generate=True, renormalize_logits=True, temperature=0.0)
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

if __name__ == "__main__":
  import yaml
  import sys
  while True:
      tbegin = time.time()
      print('===================')
      task_def = yaml.load(open(sys.argv[1], 'r'), Loader=yaml.FullLoader)
      context = task_def['context']
      command = task_def['command']
      options = {opt['text']: opt['affordance'] for opt in task_def['options']}
      termination_string = task_def['termination_string']
      max_steps = task_def['max_steps']
      options[termination_string] = 1/(max_steps+1)
      make_plan(context, command, list(options.keys()), termination_string, options, max_steps, verbose=True, print_tokens=False)
      tend = time.time()
      print(f':: time {tend-tbegin}')
