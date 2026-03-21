# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the open-r1/DAPO-Math-17k-Processed dataset to parquet format.
"""

import argparse
import json
import os

import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/dapo_math_17k_processed",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "open-r1/DAPO-Math-17k-Processed"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "en")
    else:
        dataset = datasets.load_dataset(data_source, "en")

    instruction_following = '\nPlease reason step by step, and put your final answer within \\boxed{}.'

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.get("prompt", "")
            question = question_raw + " " + instruction_following

            answer_raw = example.get("solution")
            reward_model = example.get("reward_model") or {}
            ground_truth = reward_model.get("ground_truth", answer_raw)

            source_prompt = example.get("source_prompt")
            extra_info = example.get("extra_info") or {}

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {
                    "style": reward_model.get("style", "rule"),
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "source_prompt": source_prompt,
                    "source_index": extra_info.get("index"),
                    "source_data_source": example.get("data_source"),
                    "source_ability": example.get("ability"),
                },
            }
            return data

        return process_fn

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    for split_name, split_dataset in dataset.items():
        processed_dataset = split_dataset.map(
            function=make_map_fn(split_name),
            with_indices=True,
            remove_columns=split_dataset.column_names,
        )
        processed_dataset.to_parquet(os.path.join(local_dir, f"{split_name}.parquet"))

        example = processed_dataset[0]
        with open(os.path.join(local_dir, f"{split_name}_example.json"), "w") as f:
            json.dump(example, f, indent=2)
