# %%
from datasets import load_dataset
path = "/home/jacob.a.helwig/verl/../verlData/numina_math_cot_subset_10000/"
train_path = path + "train.parquet"
test_path = path + "test.parquet"
dataset = load_dataset("parquet", data_files={"train": train_path, "test": test_path})
print(dataset)
# %%

print(dataset["train"][0])
# %%
