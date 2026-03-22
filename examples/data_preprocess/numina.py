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
from verl.utils.reward_score.math_verify import compute_score as math_verify_compute_score


def remove_boxed(s):
    s = s.strip()
    for left in ("\\boxed ", "\\boxed{", "\\boxed", "\\fbox{", "\\fbox"):
        if not s.startswith(left):
            continue

        content = s[len(left) :]
        if left.endswith("{") and content.endswith("}"):
            content = content[:-1]
        return content.strip()

    return s


def last_boxed_only_string(string):
    boxed_idx = string.rfind("\\boxed")
    fbox_idx = string.rfind("\\fbox")
    idx = max(boxed_idx, fbox_idx)
    if idx < 0:
        return None

    token = "\\boxed" if boxed_idx >= fbox_idx else "\\fbox"
    token_end = idx + len(token)
    suffix = string[token_end:]

    if suffix.startswith(" "):
        return (token + " " + suffix[1:].split("$", 1)[0]).rstrip()

    if suffix.startswith("{"):
        depth = 0
        for offset, ch in enumerate(suffix):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return string[idx : token_end + offset + 1]

    return string[idx:].split("$", 1)[0].rstrip()


def extract_solution(solution_str):
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None, None, "missing_boxed_answer"

    candidate = remove_boxed(boxed)
    if math_verify_compute_score(solution_str, candidate):
        return candidate, candidate, None
    return None, candidate, "unverifiable_boxed_answer"


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
            solution, parsed_candidate, skip_reason = extract_solution(answer_raw)

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
                "parsed_candidate": parsed_candidate,
                "skip_reason": skip_reason,
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
    unverifiable_path = os.path.join(local_dir, "unverifiable")
    sampled_split_names = parse_sample_splits(args.sample_splits)
    subset_manifest = {
        "data_source": data_source,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
        "sample_splits": sorted(sampled_split_names),
        "splits": {},
    }
    total_retained = 0
    total_skipped = 0

    with open(unverifiable_path, "w") as unverifiable_file:
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

            if sampled_indices is not None:
                with open(os.path.join(local_dir, f"{split_name}_sampled_indices.json"), "w") as f:
                    json.dump(sampled_indices, f)

            processed_dataset = dataset_to_process.map(
                function=make_map_fn(split_name, sampled_indices=sampled_indices),
                with_indices=True,
                remove_columns=dataset_to_process.column_names,
            )

            retained_indices = []
            split_retained = 0
            split_skipped = 0
            for idx, example in enumerate(processed_dataset):
                if example["skip_reason"] is None:
                    retained_indices.append(idx)
                    split_retained += 1
                    continue

                split_skipped += 1
                json.dump(
                    {
                        "split": split_name,
                        "index": example["extra_info"]["index"],
                        "skip_reason": example["skip_reason"],
                        "parsed_candidate": example["parsed_candidate"],
                        "answer": example["extra_info"]["answer"],
                        "question": example["extra_info"]["question"],
                        "source": example["extra_info"]["source"],
                        "source_messages": example["extra_info"]["source_messages"],
                    },
                    unverifiable_file,
                )
                unverifiable_file.write("\n")

            retained_dataset = processed_dataset.select(retained_indices).remove_columns(
                ["parsed_candidate", "skip_reason"]
            )
            retained_dataset.to_parquet(os.path.join(local_dir, f"{split_name}.parquet"))

            subset_manifest["splits"][split_name] = {
                "original_num_rows": len(split_dataset),
                "saved_num_rows": split_retained,
                "sampled": sampled_indices is not None,
                "skipped_num_rows": split_skipped,
            }
            total_retained += split_retained
            total_skipped += split_skipped

            print(
                f"Split '{split_name}': retained {split_retained} examples, skipped {split_skipped} unverifiable examples",
                flush=True,
            )

            if split_retained > 0:
                example = retained_dataset[0]
                with open(os.path.join(local_dir, f"{split_name}_example.json"), "w") as f:
                    json.dump(example, f, indent=2)

    with open(os.path.join(local_dir, "subset_manifest.json"), "w") as f:
        json.dump(subset_manifest, f, indent=2)

    print(f"Retained examples: {total_retained}", flush=True)
    print(f"Skipped examples: {total_skipped}", flush=True)
    print(f"Unverifiable log: {unverifiable_path}", flush=True)


"""
eval "$(conda shell.bash hook)"
conda activate verl
export PATH=$CONDA_PREFIX/bin:$PATH
export DATA_PATH=$PWD/../verlData


N=11000
SAVE_DIR=$DATA_PATH/numina_math_cot_subset_$N

python examples/data_preprocess/numina.py --local_save_dir $SAVE_DIR --sample_size $N --sample_seed 42
"""
