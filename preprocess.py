# preprocess.py — tokenize / pack Hub or local text datasets for main.py
import os
import argparse

import torch
from functools import partial
from transformers.trainer_utils import set_seed
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset


def eff_group_texts(examples, max_length, pad_token_id):
    grouped = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    for i in range(len(examples["input_ids"])):
        for j in range(0, len(examples["input_ids"][i]), max_length):
            if j + max_length > len(examples["input_ids"][i]):
                pad_len = max_length - (len(examples["input_ids"][i]) - j)
                grouped["input_ids"].append(
                    examples["input_ids"][i][j:] + [pad_token_id] * pad_len
                )
                grouped["attention_mask"].append(
                    examples["attention_mask"][i][j:] + [0] * pad_len
                )
                grouped["labels"].append(
                    examples["labels"][i][j:] + [pad_token_id] * pad_len
                )
            else:
                grouped["input_ids"].append(examples["input_ids"][i][j : j + max_length])
                grouped["attention_mask"].append(
                    examples["attention_mask"][i][j : j + max_length]
                )
                grouped["labels"].append(examples["labels"][i][j : j + max_length])

    return grouped


def tokenize(examples, tokenizer, input_field):
    encoded = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for i in range(len(examples[input_field])):
        src = examples[input_field][i]
        src_tok = (
            [tokenizer.bos_token_id]
            + tokenizer.encode(src, add_special_tokens=False)
            + [tokenizer.eos_token_id]
        )
        encoded["input_ids"].append(src_tok[:-1])
        encoded["labels"].append(src_tok[1:])
        encoded["attention_mask"].append([1] * (len(src_tok) - 1))
    return encoded


def count_split_tokens(dataset_split, grouped=False):
    """Return total tokens in a split (validation count for BPB reporting)."""
    if grouped:
        return sum(sum(mask) for mask in dataset_split["attention_mask"])
    return sum(len(ids) for ids in dataset_split["input_ids"])


def _resolve_tokenizer(name: str):
    if name == "byt5":
        name = "google/byt5-small"
    elif name == "gpt2":
        name = "openai-community/gpt2"

    tokenizer = AutoTokenizer.from_pretrained(name)
    if name == "google/byt5-small":
        tokenizer.bos_token_id = 383
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
    elif name == "openai-community/gpt2":
        tokenizer.pad_token_id = tokenizer.vocab_size - 1
    return name, tokenizer


def _load_dataset(dataset: str, cache_dir: str):
    if dataset == "pg19":
        return load_dataset("deepmind/pg19", cache_dir=cache_dir or None)
    try:
        return load_dataset(dataset, cache_dir=cache_dir or None)
    except Exception:
        return load_from_disk(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocessing for hierarchical pretraining")
    parser.add_argument("--tokenizer_name", type=str, default="openai-community/gpt2")
    parser.add_argument(
        "--dataset",
        type=str,
        default="pg19",
        help="HF Hub id (e.g. leukas/climbmix-1b-shuffle), local path, or pg19.",
    )
    parser.add_argument("--cache_dir", type=str, default="", help="HF datasets cache dir.")
    parser.add_argument("--output", type=str, default="", help="Output path for save_to_disk.")
    parser.add_argument("--tokenize", action="store_true", help="Tokenize text → input_ids/labels.")
    parser.add_argument("--group", action="store_true", help="Pack into fixed-length chunks.")
    parser.add_argument(
        "--count_tokens",
        action="store_true",
        help="Write validation token count to {output}/num_tokens.txt.",
    )
    parser.add_argument("--data_lim", type=int, default=0, help="Limit train examples, or 0 to disable.")
    parser.add_argument("--no_valid", action="store_true", help="Do not create a validation split.")
    parser.add_argument("--num_proc", type=int, default=30, help="Number of map workers.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for train/valid split.")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Packed sequence length.")
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Column name containing raw text.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    args.tokenizer_name, tokenizer = _resolve_tokenizer(args.tokenizer_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    dataset = _load_dataset(args.dataset, args.cache_dir)

    if not args.no_valid:
        if isinstance(dataset, Dataset) or "validation" not in dataset.keys():
            split_src = dataset if isinstance(dataset, Dataset) else dataset["train"]
            dataset = split_src.train_test_split(test_size=0.01, seed=args.seed)
            dataset["validation"] = dataset["test"]
            del dataset["test"]
    elif isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    if args.data_lim > 0 and "train" in dataset:
        dataset["train"] = dataset["train"].select(range(min(args.data_lim, len(dataset["train"]))))

    group_fn = partial(eff_group_texts, max_length=args.max_seq_len, pad_token_id=pad_id)
    tok_fn = partial(tokenize, tokenizer=tokenizer, input_field=args.text_field)

    needs_input_ids = args.group or args.count_tokens
    if needs_input_ids and not args.tokenize:
        sample_split = next(iter(dataset.keys()))
        if "input_ids" not in dataset[sample_split].column_names:
            raise ValueError(
                "Dataset has no input_ids; pass --tokenize or load an already-tokenized dataset."
            )

    token_counts = {}
    count_after_group = args.count_tokens and args.group

    for split in dataset.keys():
        old_columns = list(dataset[split].features.keys())
        if args.tokenize:
            dataset[split] = dataset[split].map(
                tok_fn,
                batched=True,
                batch_size=16,
                num_proc=args.num_proc,
                remove_columns=old_columns,
                load_from_cache_file=False,
            )
        if args.count_tokens and not count_after_group:
            token_counts[split] = count_split_tokens(dataset[split], grouped=False)

        if args.group:
            dataset[split] = dataset[split].map(
                group_fn,
                batched=True,
                batch_size=16,
                num_proc=args.num_proc,
                load_from_cache_file=False,
                remove_columns=dataset[split].column_names,
            )

        if args.count_tokens and count_after_group:
            token_counts[split] = count_split_tokens(dataset[split], grouped=True)

    if args.count_tokens:
        count_split = "validation" if "validation" in token_counts else next(iter(token_counts))
        sum_n_tokens = token_counts[count_split]
        print(f"Number of tokens ({count_split}): {sum_n_tokens}")
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            with open(f"{args.output}/num_tokens.txt", "w") as f:
                f.write(str(sum_n_tokens))

    if args.group or args.tokenize:
        if not args.output:
            raise ValueError("--output is required when saving a processed dataset")
        os.makedirs(args.output, exist_ok=True)
        dataset.save_to_disk(args.output)
