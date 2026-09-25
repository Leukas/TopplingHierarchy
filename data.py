# data.py - HuggingFace version of ptl_data.py (PyTorch Lightning removed)
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
import torch


def simple_collate_fn(batch):
    padded_batch = {}
    for key in batch[0]:
        try:
            padded_batch[key] = torch.stack([torch.LongTensor(x[key]) for x in batch])
        except RuntimeError:
            print(key)
    return padded_batch


class BBT:
    def __init__(
        self,
        dataset_path="data/pg19",
        seq_len=2048,
        train_batch_size=32,
        eval_batch_size=32,
        pad_idx=0,
        bytes_per_token=None,
        **kwargs,
    ):
        self.dataset_path = dataset_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.seq_len = seq_len
        self.pad_idx = pad_idx

        if bytes_per_token is not None:
            self.bytes_per_token = bytes_per_token
        else:
            total_bytes = 17397485  # from the original validation file as raw text
            total_tokens = int(open(f"{dataset_path}/num_tokens.txt", "r").read().strip())
            self.bytes_per_token = total_bytes / total_tokens

    def setup(self, stage=None):
        self.dataset = datasets.load_from_disk(self.dataset_path)
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_from_disk(self.dataset_path)

    def get_data_collator(self):
        return simple_collate_fn
