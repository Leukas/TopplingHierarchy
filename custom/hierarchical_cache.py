"""Shared cache for hierarchical models (char-level DynamicCache + word-level buffers)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from transformers.cache_utils import DynamicCache


class HierarchicalDynamicCache(DynamicCache):
    """DynamicCache with an auxiliary per-layer word-level KV cache.

    Char-level layers use the parent ``DynamicCache``. Word-level attention
    only materializes K/V at end-of-word positions; batch rows can differ in
    word count. Each word layer keeps a padded
    ``[batch, max_words, num_heads, head_dim]`` buffer and a ``[batch]`` word
    count vector.
    """

    def __init__(self, *args, first_char_layer_idx=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_char_layer_idx = first_char_layer_idx
        self.word_keys: dict[int, torch.Tensor] = {}
        self.word_values: dict[int, torch.Tensor] = {}
        self.word_counts: dict[int, torch.Tensor] = {}
        # Per-batch raw boundary mask of the most recently processed token.
        # Needed so that during single-token decode we can apply the
        # "suppress consecutive boundaries" filter relative to the prefill /
        # previous decode token, which would otherwise be invisible.
        self.last_raw_boundary: Optional[torch.Tensor] = None

    def get_seq_length(self, layer_idx=0):
        return super().get_seq_length(self.first_char_layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> Tuple[int, int]:
        return super().get_mask_sizes(query_length, self.first_char_layer_idx)

    def get_word_count(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self.word_counts.get(layer_idx, None)

    def get_last_raw_boundary(self) -> Optional[torch.Tensor]:
        return self.last_raw_boundary

    def set_last_raw_boundary(self, mask: Optional[torch.Tensor]) -> None:
        self.last_raw_boundary = mask

    def update_word(
        self,
        layer_idx: int,
        new_k: Optional[torch.Tensor],
        new_v: Optional[torch.Tensor],
        new_words_per_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = new_words_per_seq.shape[0]
        device = new_words_per_seq.device

        if layer_idx not in self.word_keys:
            if new_k is None or new_k.shape[0] == 0:
                zeros_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
                empty = torch.zeros(0, 0, 0, device=device)
                return empty, empty, zeros_counts
            max_w = int(new_words_per_seq.max().item())
            num_heads = new_k.shape[1]
            head_dim = new_k.shape[2]
            self.word_keys[layer_idx] = torch.zeros(
                batch_size,
                max_w,
                num_heads,
                head_dim,
                device=new_k.device,
                dtype=new_k.dtype,
            )
            self.word_values[layer_idx] = torch.zeros_like(self.word_keys[layer_idx])
            self.word_counts[layer_idx] = torch.zeros(batch_size, dtype=torch.long, device=device)

        prev_counts = self.word_counts[layer_idx]
        total_counts = prev_counts + new_words_per_seq
        max_total = int(total_counts.max().item()) if total_counts.numel() > 0 else 0

        cur_max = self.word_keys[layer_idx].shape[1]
        if max_total > cur_max:
            pad_cols = max_total - cur_max
            k_buf = self.word_keys[layer_idx]
            v_buf = self.word_values[layer_idx]
            pad_shape = (k_buf.shape[0], pad_cols, k_buf.shape[2], k_buf.shape[3])
            self.word_keys[layer_idx] = torch.cat(
                [k_buf, torch.zeros(pad_shape, device=k_buf.device, dtype=k_buf.dtype)],
                dim=1,
            )
            self.word_values[layer_idx] = torch.cat(
                [v_buf, torch.zeros(pad_shape, device=v_buf.device, dtype=v_buf.dtype)],
                dim=1,
            )

        if new_k is not None and new_k.shape[0] > 0:
            total_new = new_k.shape[0]
            batch_row_ids = torch.repeat_interleave(
                torch.arange(batch_size, device=device), new_words_per_seq
            )
            new_offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
            new_offsets[1:] = new_words_per_seq.cumsum(0)
            within_row = torch.arange(total_new, device=device) - new_offsets[batch_row_ids]
            col_idx = prev_counts[batch_row_ids] + within_row
            self.word_keys[layer_idx][batch_row_ids, col_idx] = new_k.to(self.word_keys[layer_idx].dtype)
            self.word_values[layer_idx][batch_row_ids, col_idx] = new_v.to(self.word_values[layer_idx].dtype)

        self.word_counts[layer_idx] = total_counts

        max_w = self.word_keys[layer_idx].shape[1]
        gather_mask = torch.arange(max_w, device=device).unsqueeze(0) < total_counts.unsqueeze(1)
        k_full = self.word_keys[layer_idx][gather_mask]
        v_full = self.word_values[layer_idx][gather_mask]
        return k_full, v_full, total_counts
