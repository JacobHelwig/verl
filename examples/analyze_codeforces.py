# %%
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import os
os.environ["HF_HOME"] = "/nvme-data/jacob/verlData"
PATH = f"{os.environ['HF_HOME']}/codeforces"
# %%
ds = load_dataset("parquet", data_files={"train": f"{PATH}/train.parquet", "test": f"{PATH}/test.parquet"})
# %%

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
prompt_key = "prompt"
def doc2len(doc) -> int:
    return len(
        tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)
    )

full_data = concatenate_datasets([ds["train"], ds["test"]])

# %%
input_lens = []
for example in tqdm(full_data):
    input_lens.append(doc2len(example))

# %%
import numpy as np
# %%
input_lens_np = np.array(input_lens)
# %%
(input_lens_np <= 1024).sum()
# %%
max(input_lens)

# %%
plt.hist(input_lens, bins=100)
# %%
raw_ds = load_dataset("open-r1/codeforces-cots", "solutions_py_decontaminated")
# %%
full_raw_data = raw_ds["train"]
# %%
resp_lens = []
for example in tqdm(full_raw_data):
    response = example['generation']
    resp_lens.append(len(tokenizer.tokenize(response)))
# %%
resp_lens_np = np.array(resp_lens)
# %%
plt.hist(resp_lens, bins=100)
# %%
(resp_lens_np < 4096).sum()