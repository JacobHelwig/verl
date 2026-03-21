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
Preprocess the AI-MO/NuminaMath-CoT dataset to parquet format.
"""

import argparse
import json
import os
import random

import datasets


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    if "\\boxed{" in s:
        left = "\\boxed{"
        assert s[: len(left)] == left
        return s[len(left) : -1]
    
    if "\\boxed" in s:
        left = "\\boxed"
        assert s[: len(left)] == left
        return s[len(left) : ]

    raise ValueError(s)


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def extract_solution(solution_str):
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return solution_str
    return remove_boxed(boxed)


def parse_sample_splits(sample_splits: str) -> set[str]:
    return {split.strip() for split in sample_splits.split(",") if split.strip()}


def get_split_sample_seed(base_seed: int, split_name: str) -> int:
    return base_seed + sum(ord(ch) for ch in split_name)


def sample_split_dataset(split_dataset, split_name: str, sample_size: int, sample_seed: int):
    total_rows = len(split_dataset)
    if sample_size is None or sample_size >= total_rows:
        return split_dataset, None

    split_seed = get_split_sample_seed(sample_seed, split_name)
    rng = random.Random(split_seed)
    sampled_indices = sorted(rng.sample(range(total_rows), sample_size))
    sampled_dataset = split_dataset.select(sampled_indices)
    return sampled_dataset, sampled_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/numina_math_cot", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help=(
            "If set, randomly sample up to this many rows from each split listed in --sample_splits "
            "before preprocessing."
        ),
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Random seed used for reproducible subset sampling.",
    )
    parser.add_argument(
        "--sample_splits",
        default="train",
        help="Comma-separated split names to sample when --sample_size is set. Default: train",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path
    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample_size must be > 0 when provided")

    data_source = "AI-MO/NuminaMath-CoT"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    instruction_following = '\nPlease reason step by step, and put your final answer within \\boxed{}.'

    def make_map_fn(split, sampled_indices=None):
        def process_fn(example, idx):
            original_index = sampled_indices[idx] if sampled_indices is not None else idx
            question_raw = example.get("problem", "")
            question = question_raw + " " + instruction_following

            answer_raw = example.get("solution")
            solution = extract_solution(answer_raw)

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": original_index,
                    "answer": answer_raw,
                    "question": question_raw,
                    "source": example.get("source"),
                    "source_messages": example.get("messages"),
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
    sampled_split_names = parse_sample_splits(args.sample_splits)
    subset_manifest = {
        "data_source": data_source,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "sample_splits": sorted(sampled_split_names),
        "splits": {},
    }

    for split_name, split_dataset in dataset.items():
        sampled_indices = None
        dataset_to_process = split_dataset
        if args.sample_size is not None and split_name in sampled_split_names:
            dataset_to_process, sampled_indices = sample_split_dataset(
                split_dataset=split_dataset,
                split_name=split_name,
                sample_size=args.sample_size,
                sample_seed=args.sample_seed,
            )
            print(
                f"Sampling split '{split_name}': selected {len(dataset_to_process)} / {len(split_dataset)} rows "
                f"with seed {get_split_sample_seed(args.sample_seed, split_name)}",
                flush=True,
            )

        subset_manifest["splits"][split_name] = {
            "original_num_rows": len(split_dataset),
            "saved_num_rows": len(dataset_to_process),
            "sampled": sampled_indices is not None,
        }

        if sampled_indices is not None:
            with open(os.path.join(local_dir, f"{split_name}_sampled_indices.json"), "w") as f:
                json.dump(sampled_indices, f)

        processed_dataset = dataset_to_process.map(
            function=make_map_fn(split_name, sampled_indices=sampled_indices),
            with_indices=True,
            remove_columns=dataset_to_process.column_names,
        )
        processed_dataset.to_parquet(os.path.join(local_dir, f"{split_name}.parquet"))

        example = processed_dataset[0]
        with open(os.path.join(local_dir, f"{split_name}_example.json"), "w") as f:
            json.dump(example, f, indent=2)

    with open(os.path.join(local_dir, "subset_manifest.json"), "w") as f:
        json.dump(subset_manifest, f, indent=2)


"""
eval "$(conda shell.bash hook)"
conda activate verl
export PATH=$CONDA_PREFIX/bin:$PATH
export DATA_PATH=$PWD/../verlData


N=10000
SAVE_DIR=$DATA_PATH/numina_math_cot_subset_10000

python examples/data_preprocess/numina.py --local_save_dir $SAVE_DIR --sample_size $N --sample_seed 42
"""

