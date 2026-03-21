# %%
from datasets import load_dataset
trace_responses = load_dataset("jacob-helwig/math_dataset_Qwen3-4B_chunk1024_ntokens4096_greedy_sft", split="train").to_pandas()
import json
with open("/data2/jacob/verl/val_gens_temp0/0.jsonl") as f:
    verl_responses = [json.loads(line) for line in f]
len(verl_responses), len(trace_responses)
# %%
verl_responses[0]
# %%
trace_response_ls = []
i = 0
verl_pos = 0
verl_suffix = " Let's think step by step and output the final answer within \\boxed{}.\nassistant\n<think>\n\n</think>\n\n"
trace_suffix = " \nPlease reason step by step, and put your final answer within \\boxed{}."
while i < len(trace_responses):
    verl_response = verl_responses[verl_pos]
    verl_prompt = verl_response["input"][len("user\n"):-len(verl_suffix)]
    trace_response = trace_responses.iloc[i]
    trace_prompt = trace_response["prompt"][:-len(trace_suffix)]
    if trace_prompt == verl_prompt:
        trace_response_ls.append(trace_response)
        verl_pos += 1
    i += 1

    
verl_prompt, trace_prompt
# %%
len(trace_response_ls), len(verl_responses)
# %%
for trace_response, verl_response in zip(trace_response_ls, verl_responses):
    if not (trace_response["is_correct"] == verl_response["acc"]):
        break
# %%
print(trace_response["response"])
# %%
print(verl_response["output"])