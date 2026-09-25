# model.py - HuggingFace version of ptl_model.py (PyTorch Lightning removed)
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoConfig, GPT2LMHeadModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM

from custom.byo_llama import LlamaByoForCausalLM
from custom.byo_qwen3 import Qwen3ByoForCausalLM
from custom.byo_olmo import OlmoByoForCausalLM

logger = logging.getLogger(__name__)


def compute_metric(predictions, labels):
    mask = labels != -100
    masked_labels = labels[mask]
    masked_predictions = predictions[mask]
    correct = masked_labels == masked_predictions
    accuracy = correct.sum() / float(len(correct))
    return {"acc": float(accuracy)}


def build_olmo2_model(
    model_name: str,
    vocab_size: int,
    max_seq_len: int,
    attn_implementation: str = "flash_attention_3",
    byo_layout: str = None,
) -> nn.Module:
    """Build a Olmo2-based causal LM from a size name (small/base/large)."""
    model_name = model_name.lower().replace("olmo2_", "")

    model_sizes = {
        "small": (768, 12, 12),
        "base": (1024, 24, 16),
        "large": (1536, 24, 16),
    }

    hidden_size, n_layers, n_heads = model_sizes[model_name]
    seq_len, base = max_seq_len, 10000
    imd_size = 4 * hidden_size
    config = Olmo2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=imd_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        max_position_embeddings=seq_len,
        rope_theta=base,
        use_cache=True,
        _attn_implementation=attn_implementation,
    )

    logger.info(f"Building Olmo2 model: hidden={hidden_size}, layers={n_layers}, heads={n_heads}")

    if byo_layout:
        config.byo_layout = byo_layout
        config.group_method = "spacebyte"
        return OlmoByoForCausalLM(config)
    return Olmo2ForCausalLM(config)


def build_llama_model(
    model_name: str,
    vocab_size: int,
    max_seq_len: int,
    byo_layout: str = None,
    attn_implementation: str = "flash_attention_3",
    **kwargs,
) -> nn.Module:
    """Build a Llama-based causal LM from a size name (small/base/large)."""
    del kwargs
    model_name = model_name.lower().replace("llama_", "")

    model_sizes = {
        "small": (768, 12, 12),
        "base": (1024, 24, 16),
        "base2": (1024, 24, 4),
        "large": (1536, 24, 16),
    }
    hidden_size, n_layers, n_heads = model_sizes[model_name]
    seq_len, base = max_seq_len, 10000
    imd_size = 4 * hidden_size
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=imd_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        max_position_embeddings=seq_len,
        rope_theta=base,
        use_cache=True,
        _attn_implementation=attn_implementation,
    )

    logger.info(f"Building Llama model: hidden={hidden_size}, layers={n_layers}, heads={n_heads}")

    if byo_layout:
        config.byo_layout = byo_layout
        config.group_method = "spacebyte"
        return LlamaByoForCausalLM(config)
    return AutoModelForCausalLM.from_config(config)


def build_qwen3_model(
    model_name: str,
    vocab_size: int,
    max_seq_len: int,
    byo_layout: str = None,
    **kwargs,
) -> nn.Module:
    """Build a Qwen3-based causal LM from a size name (small/base/large)."""
    del kwargs
    model_name = model_name.lower().replace("qwen3_", "")

    model_sizes = {
        "small": (768, 12, 12),
        "base": (1024, 24, 16),
        "large": (1536, 24, 16),
    }
    hidden_size, n_layers, n_heads = model_sizes[model_name]

    seq_len, base = max_seq_len, 10000
    imd_size = 4 * hidden_size
    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=imd_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        # Keep MHA shape valid for custom tiny configs; FA3 requires q_heads % kv_heads == 0.
        num_key_value_heads=n_heads,
        max_position_embeddings=seq_len,
        rope_theta=base,
        use_cache=True,
        _attn_implementation="flash_attention_3",
    )

    logger.info(f"Building Qwen3 model: hidden={hidden_size}, layers={n_layers}, heads={n_heads}")

    if byo_layout:
        config.byo_layout = byo_layout
        config.group_method = "spacebyte"
        return Qwen3ByoForCausalLM(config)
    return AutoModelForCausalLM.from_config(config)


def build_gpt2_model(vocab_size: int, max_seq_len: int) -> nn.Module:
    """Build a GPT-2 causal LM."""
    config = AutoConfig.from_pretrained("openai-community/gpt2")
    config.n_positions = max_seq_len
    config.vocab_size = vocab_size
    logger.info(f"Building GPT-2 model: vocab={vocab_size}, seq_len={max_seq_len}")
    return GPT2LMHeadModel(config)
