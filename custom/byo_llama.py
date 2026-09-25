# byo_llama.py

import importlib.util
import re
import warnings
from functools import lru_cache
import torch
import torch.nn as nn
from transformers.modeling_flash_attention_utils import is_flash_attn_available
from torch.nn.attention.varlen import varlen_attn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    create_causal_mask,
)
from typing import Optional, Tuple, Union

from custom.hierarchical_cache import HierarchicalDynamicCache


_BYO_LAYOUT_CHARS = frozenset("WCwc")


def _is_byo_pattern_layout(layout: str) -> bool:
    """True if ``layout`` uses only BYO layer letters: W/w (word attn/MLP), C/c (char attn/MLP)."""
    return bool(layout) and all(ch in _BYO_LAYOUT_CHARS for ch in layout)


def _expand_byo_repeat_syntax(layout: str) -> str:
    """
    Expand repeat groups like ``3(Cc)3(Ww)`` → ``CcCcCcWwWwWw``.

    Each group is ``<count>(<substring>)`` with a positive integer count. Count is applied
    as string repetition of the inner substring (not character-wise).
    """
    s = layout.strip()
    if "(" not in s:
        return s

    out: list[str] = []
    pos = 0
    while pos < len(s):
        m = re.match(r"(\d+)\(", s[pos:])
        if m:
            count = int(m.group(1))
            if count < 1:
                raise ValueError(f"Repeat count must be positive in byo_layout: {layout!r}")
            start_inner = pos + m.end()
            depth = 1
            j = start_inner
            while j < len(s) and depth:
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                j += 1
            if depth != 0:
                raise ValueError(f"Unbalanced parentheses in byo_layout: {layout!r}")
            inner = s[start_inner : j - 1]
            if not inner:
                raise ValueError(f"Empty repeat group in byo_layout: {layout!r}")
            out.append(inner * count)
            pos = j
            continue

        next_m = re.search(r"\d+\(", s[pos:])
        if next_m is None:
            out.append(s[pos:])
            break
        if next_m.start() > 0:
            out.append(s[pos : pos + next_m.start()])
        pos = pos + next_m.start()

    return "".join(out)

def _flash_attn_varlen_func(*args, **kwargs):
    try:
        from flash_attn.cute import flash_attn_varlen_func as _fn
    except ModuleNotFoundError:
        from flash_attn_interface import flash_attn_varlen_func as _fn
    return _fn(*args, **kwargs)


def _pytorch_varlen_attn_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
):
    return varlen_attn(
        query_states,
        key_states,
        value_states,
        cu_seq_q=cu_seqlens_q,
        cu_seq_k=cu_seqlens_k,
        max_q=max_seqlen_q,
        max_k=max_seqlen_k,
        scale=softmax_scale,
    )


class LlamaByoForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaByoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        word_idx: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_idx=word_idx,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaByoModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if config.group_method == "bpe":
            self.byte_tokenizer = config.byte_tokenizer
            self.subword_tokenizer = config.subword_tokenizer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self._utf8_token_offset: Optional[int] = None

        n = config.num_hidden_layers * 2
        if self.config.byo_layout == "middle":
            self.config.byo_layout = f"{n // 8}(Cc){n // 4}(Ww){n // 8}(Cc)"
        elif self.config.byo_layout == "bottom":
            if n % 2 != 0:
                raise ValueError(
                    f"byo_layout 'bottom' requires num_hidden_layers even (pairs of Ww), got {n}"
                )
            self.config.byo_layout = f"{n // 2}(Ww)"
        elif self.config.byo_layout == "top":
            if n % 2 != 0:
                raise ValueError(
                    f"byo_layout 'top' requires num_hidden_layers even (pairs of Cc), got {n}"
                )
            self.config.byo_layout = f"{n // 2}(Cc)"
        elif self.config.byo_layout == "interleaved":
            if n % 4 != 0:
                raise ValueError(
                    f"byo_layout 'interleaved' requires num_hidden_layers divisible by 4, got {n}"
                )
            self.config.byo_layout = f"{n // 4}(WwCc)"

        if "(" in self.config.byo_layout:
            self.config.byo_layout = _expand_byo_repeat_syntax(self.config.byo_layout)

        if _is_byo_pattern_layout(self.config.byo_layout):
            pattern = self.config.byo_layout
            num_modules = len(pattern)
            if num_modules != n:
                warnings.warn(
                    f"byo_layout has {num_modules} module slot(s), but config.num_hidden_layers="
                    f"{config.num_hidden_layers} implies {n} (= 2 × num_hidden_layers). "
                    f"Proceeding with the {num_modules} module(s) defined by the layout string.",
                    UserWarning,
                    stacklevel=2,
                )
            self.layers = nn.ModuleList(
                [
                    LlamaByoCustomLayer(config=config, layer_idx=layer_idx, layer_type=pattern[layer_idx])
                    for layer_idx in range(num_modules)
                ]
            )
        else:
            raise ValueError(f"Invalid byo_layout: {self.config.byo_layout}")
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

    def _get_utf8_token_offset(self, input_ids: torch.LongTensor) -> int:
        """Infer the constant offset between token IDs and UTF-8 byte values."""
        if self._utf8_token_offset is not None:
            return self._utf8_token_offset

        token_min = int(input_ids.min().item())
        token_max = int(input_ids.max().item())
        max_offset = max(0, min(token_min, token_max - 255))

        best_offset = 0
        best_score = -1
        for offset in range(max_offset + 1):
            shifted = input_ids - offset
            score = ((shifted >= 0) & (shifted <= 255)).sum().item()
            if score > best_score:
                best_score = score
                best_offset = offset

        self._utf8_token_offset = best_offset
        return best_offset

    def _compute_end_of_word_idx(
        self,
        input_ids: torch.LongTensor,
        prev_raw_boundary: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        if self.config.group_method == "spacebyte":
            return self._compute_end_of_word_idx_spacebyte(input_ids, prev_raw_boundary)
        elif self.config.group_method == "bpe":
            return self._compute_end_of_word_idx_bpe(input_ids, prev_raw_boundary, attention_mask)
        else:
            raise ValueError(f"Invalid group method: {self.config.group_method}")

    def _compute_end_of_word_idx_spacebyte(
        self,
        input_ids: torch.LongTensor,
        prev_raw_boundary: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        """SpaceByte UTF-8 patch–style BPE boundaries, with token-id/byte offset.

        Returns ``(end_of_word_idx, last_raw_boundary)`` for decode continuity.
        """
        offset = self._get_utf8_token_offset(input_ids)
        tokens = input_ids - offset

        raw_boundary = (
            (tokens < ord("0"))
            | ((ord("9") < tokens) & (tokens < ord("A")))
            | ((ord("Z") < tokens) & (tokens < ord("a")))
            | ((ord("z") < tokens) & (tokens < 0b1000_0000))
            | (0b1100_0000 <= tokens)
        )
        end_of_word_idx = raw_boundary.clone()
        end_of_word_idx[:, 1:] = end_of_word_idx[:, 1:] & ~raw_boundary[:, :-1]
        if prev_raw_boundary is not None and end_of_word_idx.shape[1] > 0:
            prev = prev_raw_boundary.to(device=end_of_word_idx.device).bool()
            end_of_word_idx[:, 0] = end_of_word_idx[:, 0] & ~prev
        if self.config.bos_token_id is not None:
            end_of_word_idx = end_of_word_idx | (input_ids == self.config.bos_token_id)
        last_raw = raw_boundary[:, -1].clone() if raw_boundary.shape[1] > 0 else None
        return end_of_word_idx.to(torch.long), last_raw

    @staticmethod
    def _char_last_byte_indices(text: str) -> list[int]:
        """Per character in ``text``, index of the last UTF-8 byte of that character in ``text.encode('utf-8')``."""
        out: list[int] = []
        pos = 0
        for ch in text:
            pos += len(ch.encode("utf-8"))
            out.append(pos - 1)
        return out

    def _compute_end_of_word_idx_bpe(
        self,
        input_ids: torch.LongTensor,
        prev_raw_boundary: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        """Decode bytes with ``byte_tokenizer``, retokenize with ``subword_tokenizer``, mark last UTF-8 byte of each BPE piece.

        ``prev_raw_boundary`` is unused (no causal-cache tie-in); ``last_raw_boundary`` is always ``None``.
        """
        del prev_raw_boundary
        byte_tok = self.byte_tokenizer
        sub_tok = self.subword_tokenizer
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        end_of_word = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        pad_id = self.padding_idx

        for b in range(batch_size):
            row = input_ids[b]
            if attention_mask is not None:
                valid = int(attention_mask[b].sum().item())
            else:
                valid = seq_len
                if pad_id is not None:
                    nonpad = row != pad_id
                    if not bool(nonpad.all()):
                        valid = int(nonpad.nonzero(as_tuple=False)[-1].item()) + 1 if nonpad.any() else 0
            if valid <= 0:
                continue

            ids_list = row[:valid].tolist()
            text = byte_tok.decode(ids_list, skip_special_tokens=False)
            utf8_n = len(text.encode("utf-8"))
            if utf8_n != len(ids_list):
                # Fallback: tokenizer may add/remove bytes vs raw UTF-8 length; clamp into range.
                n = min(utf8_n, len(ids_list))
            else:
                n = utf8_n

            char_last_byte = self._char_last_byte_indices(text)
            if not char_last_byte:
                continue

            enc = sub_tok(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            offsets = enc.get("offset_mapping")
            if offsets is None:
                raise ValueError("subword_tokenizer must support return_offsets_mapping (use a fast tokenizer).")

            for start_c, end_c in offsets:
                if end_c <= start_c:
                    continue
                last_char = end_c - 1
                if last_char < 0 or last_char >= len(char_last_byte):
                    continue
                byte_i = char_last_byte[last_char]
                if byte_i >= n:
                    byte_i = n - 1
                if byte_i < valid:
                    end_of_word[b, byte_i] = 1

            if self.config.bos_token_id is not None:
                bos = (row[:valid] == self.config.bos_token_id).nonzero(as_tuple=True)[0]
                if bos.numel() > 0:
                    end_of_word[b, bos] = 1

        return end_of_word, None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        word_idx: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, HierarchicalDynamicCache):
            try:
                first_char_layer_idx = next(
                    i
                    for i, layer in enumerate(self.layers)
                    if isinstance(layer.module, LlamaByoAttn) and layer.layer_type == "C"
                )
            except StopIteration as e:
                raise ValueError(
                    "BYO with use_cache requires at least one char attention layer ('C') in byo_layout."
                ) from e
            past_key_values = HierarchicalDynamicCache(
                config=self.config, first_char_layer_idx=first_char_layer_idx
            )

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        if causal_mask is not None:
            causal_mask = causal_mask.contiguous()

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if word_idx is None:
            prev_raw_boundary: Optional[torch.Tensor] = None
            if isinstance(past_key_values, HierarchicalDynamicCache):
                prev_raw_boundary = past_key_values.get_last_raw_boundary()
            end_of_word_idx, last_raw_boundary = self._compute_end_of_word_idx(
                input_ids, prev_raw_boundary=prev_raw_boundary, attention_mask=attention_mask
            )
            if isinstance(past_key_values, HierarchicalDynamicCache) and last_raw_boundary is not None:
                past_key_values.set_last_raw_boundary(last_raw_boundary)
        else:
            end_of_word_idx = (word_idx[:, :-1] != word_idx[:, 1:]).to(torch.long)
            end_of_word_idx = torch.cat([end_of_word_idx, torch.ones_like(end_of_word_idx[:, :1])], dim=1)

        new_words_per_seq = end_of_word_idx.bool().sum(dim=1)
        word_layer_idx = next(
            (
                i
                for i, layer in enumerate(self.layers)
                if isinstance(layer.module, LlamaByoAttn) and layer.layer_type == "W"
            ),
            None,
        )
        if isinstance(past_key_values, HierarchicalDynamicCache) and word_layer_idx is not None:
            prev_words_per_seq = past_key_values.get_word_count(word_layer_idx)
            if prev_words_per_seq is None:
                prev_words_per_seq = torch.zeros_like(new_words_per_seq)
        else:
            prev_words_per_seq = torch.zeros_like(new_words_per_seq)

        max_word_pos = int((prev_words_per_seq + new_words_per_seq).max().item())
        if max_word_pos > 0:
            word_position_ids = torch.arange(max_word_pos, device=hidden_states.device).unsqueeze(0)
            word_position_embeddings = self.rotary_emb(hidden_states, word_position_ids)
        else:
            word_position_embeddings = None

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                end_of_word_idx=end_of_word_idx,
                prev_words_per_seq=prev_words_per_seq,
                word_position_embeddings=word_position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

class LlamaByoMLP(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, word_level: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.word_level = word_level
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, hidden_states: torch.Tensor, end_of_word_idx: torch.Tensor):
        residual = hidden_states
        if self.word_level:
            end_mask = end_of_word_idx.bool()
            hidden_states_word = hidden_states[end_mask]  # (tokens, dim)

            hidden_states_word = self.post_attention_layernorm(hidden_states_word)
            hidden_states_word = self.mlp(hidden_states_word)

            flat_indices = end_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            flat_updated_hidden_states = torch.zeros(
                (hidden_states.shape[0]*hidden_states.shape[1], hidden_states.shape[2]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            flat_updated_hidden_states = flat_updated_hidden_states.index_copy(0, flat_indices, hidden_states_word.to(hidden_states.dtype))
            hidden_states_word = flat_updated_hidden_states.view(*hidden_states.shape)
            hidden_states = residual + hidden_states_word
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states

class LlamaByoCustomLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, layer_type: str):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        if layer_type == "W":
            self.module = LlamaByoAttn(config=config, layer_idx=layer_idx, word_level=True)
        elif layer_type == "C":
            self.module = LlamaByoAttn(config=config, layer_idx=layer_idx, word_level=False)
        elif layer_type == "w":
            self.module = LlamaByoMLP(config=config, layer_idx=layer_idx, word_level=True)
        elif layer_type == "c":
            self.module = LlamaByoMLP(config=config, layer_idx=layer_idx, word_level=False)
        else: 
            raise ValueError(f"Invalid layer type: {layer_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        end_of_word_idx: Optional[torch.LongTensor] = None,
        prev_words_per_seq: Optional[torch.Tensor] = None,
        word_position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if isinstance(self.module, LlamaByoAttn):
            return self.module(
                hidden_states,
                end_of_word_idx=end_of_word_idx,
                prev_words_per_seq=prev_words_per_seq,
                word_position_embeddings=word_position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs,
            )
        return self.module(hidden_states, end_of_word_idx)


class LlamaByoAttn(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int, word_level: bool = False):
        super().__init__(config, layer_idx)
        self.word_level = word_level
        self.layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        end_of_word_idx: Optional[torch.LongTensor] = None,
        prev_words_per_seq: Optional[torch.Tensor] = None,
        word_position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        if not self.word_level:
            hidden_states = self.layernorm(hidden_states)
            hidden_states, _ = super().forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            return residual + hidden_states

        assert end_of_word_idx is not None
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        end_mask = end_of_word_idx.bool()
        new_words_per_seq = end_mask.sum(dim=1)
        hidden_states_word = hidden_states[end_mask]
        total_new_words = hidden_states_word.shape[0]

        if prev_words_per_seq is None:
            prev_words_per_seq = torch.zeros_like(new_words_per_seq)

        if total_new_words == 0:
            return residual

        hidden_states_word = self.layernorm(hidden_states_word)

        q_shape = (total_new_words, -1, self.head_dim)
        query_states = self.q_proj(hidden_states_word).view(q_shape)
        key_states_new = self.k_proj(hidden_states_word).view(q_shape)
        value_states_new = self.v_proj(hidden_states_word).view(q_shape)

        max_new = int(new_words_per_seq.max().item())
        local_offsets = torch.arange(max_new, device=device).unsqueeze(0)
        local_mask = local_offsets < new_words_per_seq.unsqueeze(1)
        local_positions = local_offsets.expand(batch_size, -1)[local_mask]
        prev_expanded = prev_words_per_seq.unsqueeze(1).expand(-1, max_new)[local_mask]
        new_positions = (local_positions + prev_expanded).to(torch.long)

        assert word_position_embeddings is not None, (
            "word_position_embeddings must be provided when word-level attention has new keys"
        )
        cos_full, sin_full = word_position_embeddings
        cos_tok = cos_full[0, new_positions]
        sin_tok = sin_full[0, new_positions]
        query_states, key_states_new = apply_rotary_pos_emb(
            query_states, key_states_new, cos_tok, sin_tok, unsqueeze_dim=1
        )

        if past_key_values is not None and isinstance(past_key_values, HierarchicalDynamicCache):
            k_full, v_full, total_words_per_seq = past_key_values.update_word(
                self.layer_idx, key_states_new, value_states_new, new_words_per_seq
            )
        else:
            k_full = key_states_new
            v_full = value_states_new
            total_words_per_seq = new_words_per_seq

        cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        cu_seqlens_q[1:] = new_words_per_seq.cumsum(0).to(torch.int32)
        cu_seqlens_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        cu_seqlens_k[1:] = total_words_per_seq.cumsum(0).to(torch.int32)
        max_seqlen_q = int(new_words_per_seq.max().item())
        max_seqlen_k = int(total_words_per_seq.max().item())

        if is_flash_attn_available() and query_states.is_cuda:
            out = _flash_attn_varlen_func(
                query_states.to(torch.bfloat16),
                k_full.to(torch.bfloat16),
                v_full.to(torch.bfloat16),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                softmax_scale=self.scaling,
            )
        else:
            out = _pytorch_varlen_attn_func(
                query_states.to(torch.bfloat16),
                k_full.to(torch.bfloat16),
                v_full.to(torch.bfloat16),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scaling,
            )
        if isinstance(out, tuple):
            out = out[0]

        out = out.reshape(-1, hidden_dim).to(hidden_states.dtype)
        out = self.o_proj(out).to(hidden_states.dtype)

        flat_indices = end_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        flat_updated = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            device=device,
            dtype=hidden_states.dtype,
        )
        flat_updated = flat_updated.index_copy(0, flat_indices, out)
        updated_hidden_states = flat_updated.view(batch_size, seq_len, hidden_dim)

        return residual + updated_hidden_states