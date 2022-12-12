import numpy as np

openai_api_key = "" 
ENGINE = "text-davinci-002"  # "text-ada-001"
# Note for scoring model, due to limitations of the GPT-3 api, each option 
# requires a separate call and can be expensive. Recommend iterating with ada.

overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}

from transformers import GPT2Tokenizer, OPTForCausalLM
model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")

prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def gpt3_call(engine="text-ada-001", prompt="", max_tokens=128, temperature=0, 
              logprobs=1, echo=False):
  # return a dummy response
  import math
  import random
  resp = {
    "choices": [
        {
            "logprobs": {
                "tokens": [
                    "this",
                    "is",
                    "a",
                    "test",
                    "of",
                ],
                "token_logprobs": [
                    math.log(random.random(), 2)
                ]*5
            }
        } for _ in range(len(prompt))
    ]
  }
  return resp
  import openai
  full_query = ""
  for p in prompt:
    full_query += p
  id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
  if id in LLM_CACHE.keys():
    print('cache hit, returning')
    response = LLM_CACHE[id]
  else:
    response = openai.Completion.create(engine=engine, 
                                        prompt=prompt, 
                                        max_tokens=max_tokens, 
                                        temperature=temperature,
                                        logprobs=logprobs,
                                        echo=echo)
    LLM_CACHE[id] = response
  return response

def gpt3_scoring(query, options, engine="text-ada-001", limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
  if limit_num_options:
    options = options[:limit_num_options]
  verbose and print("Scoring", len(options), "options")
  gpt3_prompt_options = [query + option for option in options]
  response = gpt3_call(
      engine=engine, 
      prompt=gpt3_prompt_options, 
      max_tokens=0,
      logprobs=1, 
      temperature=0,
      echo=True,)

  scores = {}
  for option, choice in zip(options, response["choices"]):
    tokens = choice["logprobs"]["tokens"]
    token_logprobs = choice["logprobs"]["token_logprobs"]

    total_logprob = 0
    for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
      print_tokens and print(token, token_logprob)
      if option_start is None and not token in option:
        break
      if token == option_start:
        break
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
  gpt3_context_lines = context.split("\n")
  gpt3_context_lines_keep = []
  for gpt3_context_line in gpt3_context_lines:
    gpt3_context_lines_keep.append(gpt3_context_line)

  context = "\n".join(gpt3_context_lines_keep)

  gpt3_prompt = context
  gpt3_prompt += "\n# " + command + "\n"

  all_llm_scores = []
  all_affordance_scores = []
  all_combined_scores = []
  affordance_scores = affordance_scores
  num_tasks = 0
  selected_task = ""
  steps_text = []
  while not selected_task == terminate_string:
    num_tasks += 1
    if num_tasks > max_tasks:
      break

    llm_scores, _ = gpt3_scoring(gpt3_prompt, options, verbose=True, engine=ENGINE, print_tokens=False)
    combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
    combined_scores = normalize_scores(combined_scores)
    selected_task = max(combined_scores, key=combined_scores.get)
    steps_text.append(selected_task)
    print(num_tasks, "Selecting: ", selected_task)
    gpt3_prompt += selected_task + "\n"

    all_llm_scores.append(llm_scores)
    all_affordance_scores.append(affordance_scores)
    all_combined_scores.append(combined_scores)

  print('**** Solution ****')
  print('# ' + command)
  for i, step in enumerate(steps_text):
    if step == '' or step == terminate_string:
      break
    print('Step ' + str(i) + ': ' + step)
  return steps_text

if __name__ == "__main__":
  import yaml
  import sys
  task_def = yaml.load(open(sys.argv[1], 'r'), Loader=yaml.FullLoader)
  context = task_def['context']
  command = task_def['command']
  options = {opt['text']: opt['affordance'] for opt in task_def['options']}
  termination_string = task_def['termination_string']
  max_steps = task_def['max_steps']
  options[termination_string] = 1/max_steps
  make_plan(context, command, options, termination_string, options, max_steps, verbose=True, print_tokens=False)
