# main.py - HuggingFace Trainer version of ptl_main.py
import os
import sys
import math
import logging
import argparse
import json

# Path setup: allow imports from the hf/ folder and from the parent directory
_hf_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_hf_dir)
sys.path.insert(0, _hf_dir)
sys.path.insert(0, _parent_dir)

import datasets  # noqa: E402
import torch  # noqa: E402

from transformers.trainer_utils import set_seed, get_last_checkpoint  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    Trainer, TrainingArguments,
)
from transformers.trainer_callback import PrinterCallback, TrainerCallback  # noqa: E402

from data import BBT  # noqa: E402
from model import (  # noqa: E402
    build_llama_model, build_gpt2_model, build_olmo2_model, build_qwen3_model,
    compute_metric,
)
from logging_utils import load_logger, LogFlushCallback  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = load_logger(logger)
datasets.logging.get_verbosity = lambda: logging.NOTSET

# Model args
parser = argparse.ArgumentParser("Pretraining low-resource models")
parser.add_argument("--model_name", type=str, default="", help="The model type to use.")
parser.add_argument("--tokenizer_name", type=str, default="openai-community/gpt2", help="The tokenizer to use.")
parser.add_argument("--output_path", type=str, default="", help="Output path to store model checkpoints.")
parser.add_argument(
    "--save_total_limit",
    type=int,
    default=2,
    help="Max checkpoint folders to keep (latest is always kept for --resume).",
)
# Data args
parser.add_argument("--dataset", type=str, default="data/bbt", help="Path to the BBT dataset.")
# Training args
parser.add_argument("--seed", type=int, default=0, help="Seed.")
parser.add_argument("--debug", action="store_true", help="Activates debug mode, disabling wandb logging.")
parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length.")
parser.add_argument("--grad_acc", type=int, default=1, help="Accumulate gradients.")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps.")
parser.add_argument("--epochs", type=int, default=1, help="Max epochs.")
parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (-1 to use epochs instead).")
parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluation.")
parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between saving.")
parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging.")
parser.add_argument("--eval_only", action="store_true", help="Only evaluate.")
parser.add_argument("--push_to_hub", type=str, default="", help="Push to hub. Specify repo name.")
parser.add_argument("--push_only", action="store_true", help="Only push to hub.")
parser.add_argument("--eval_split", type=str, default="validation", help="Split to evaluate on.")
parser.add_argument("--resume", action="store_true", help="Resume training.")
parser.add_argument(
    "--reload",
    action="store_true",
    help="Load model weights from a checkpoint but start training from step 0 "
    "(no optimizer/scheduler resume). Architecture settings (byo_layout, "
    "max_seq_len, etc.) are read from the checkpoint config.json "
    "unless overridden on the CLI.",
)
parser.add_argument(
    "--reload_checkpoint",
    type=str,
    default="",
    help="Checkpoint directory for --reload. Defaults to the last checkpoint under --output_path.",
)
parser.add_argument(
    "--no_reload_config",
    action="store_true",
    help="With --reload, do not read architecture settings from the checkpoint config.json.",
)
# Architecture args
parser.add_argument("--byo_layout", type=str, default=None, help="BYO layout.")
parser.add_argument(
    "--attn_implementation",
    type=str,
    default="flash_attention_3",
    help='Attention backend, e.g. "sdpa" or "flash_attention_3".',
)


def _argv_has(flag: str) -> bool:
    """True if ``flag`` (e.g. ``--byo_layout``) was passed on the command line."""
    for arg in sys.argv[1:]:
        if arg == flag or arg.startswith(flag + "="):
            return True
    return False


def _resolve_checkpoint(explicit_path: str, output_path: str, flag_name: str) -> str:
    if explicit_path:
        if not os.path.isdir(explicit_path):
            raise ValueError(f"{flag_name} checkpoint not found: {explicit_path!r}")
        return explicit_path
    if not output_path:
        raise ValueError(f"{flag_name} requires --output_path or --reload_checkpoint.")
    ckpt = get_last_checkpoint(output_path)
    if ckpt is None:
        raise ValueError(
            f"No checkpoint found under {output_path!r}. "
            "Train a run first, or pass --reload_checkpoint."
        )
    return ckpt


def _load_checkpoint_config(checkpoint_dir: str) -> dict:
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"Checkpoint missing config.json: {checkpoint_dir!r}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _hydrate_args_from_checkpoint(args, ckpt_config: dict) -> None:
    """Fill architecture-related CLI args from checkpoint config when not set on the CLI."""
    if ckpt_config.get("byo_layout"):
        if _argv_has("--byo_layout"):
            if args.byo_layout != ckpt_config["byo_layout"]:
                logger.warning(
                    "CLI --byo_layout differs from checkpoint; using CLI value."
                )
        else:
            args.byo_layout = ckpt_config["byo_layout"]
            logger.info("Loaded byo_layout from checkpoint config.")

    max_pos = ckpt_config.get("max_position_embeddings")
    if max_pos is not None and not _argv_has("--max_seq_len"):
        if args.max_seq_len != max_pos:
            logger.info(
                "Using max_seq_len=%s from checkpoint (max_position_embeddings).",
                max_pos,
            )
        args.max_seq_len = max_pos


def preprocess_logits_for_metrics(logits, labels):
    """Reduces memory by storing argmax predictions instead of full logits."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def _compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return compute_metric(predictions, labels)


def compute_loss(outputs, labels, num_items_in_batch=None):
    if getattr(outputs, "loss", None) is not None:
        return outputs.loss
    if getattr(outputs, "logits", None) is None:
        raise ValueError(
            "Model returned neither loss nor logits. "
            "Ensure the model forward returns loss when labels are provided."
        )
    logits = outputs.logits.view(-1, outputs.logits.size(-1))
    labels = labels.view(-1)
    return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)


class BPBCallback(TrainerCallback):
    """Logs bits-per-byte alongside eval loss."""
    def __init__(self, bytes_per_token, use_wandb=True):
        self.bytes_per_token = bytes_per_token
        self.use_wandb = use_wandb

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            bpb = metrics["eval_loss"] / (self.bytes_per_token * math.log(2))
            metrics["eval_bpb"] = bpb
            if state.is_local_process_zero:
                logger.info(f"Eval BPB: {bpb:.4f}")
            if self.use_wandb and state.is_local_process_zero:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"eval_bpb": bpb}, step=state.global_step)
                except ImportError:
                    pass


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    args = parser.parse_args()
    args.model_name = args.model_name.lower()

    if args.resume and args.reload:
        raise ValueError("Cannot use --resume and --reload together.")

    reload_ckpt = None
    reload_ckpt_config = None
    if args.reload:
        reload_ckpt = _resolve_checkpoint(
            args.reload_checkpoint, args.output_path, "--reload"
        )
        reload_ckpt_config = _load_checkpoint_config(reload_ckpt)
        if not args.no_reload_config:
            _hydrate_args_from_checkpoint(args, reload_ckpt_config)
        logger.info(
            "Reloading model weights from %s (training will start from step 0).",
            reload_ckpt,
        )

    set_seed(args.seed)

    logger.info(args)

    if args.debug or args.disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    if not args.disable_wandb:
        import wandb
        output_dir = os.path.basename(os.path.normpath(args.output_path))
        wandb.init(
            project='pretraining',
            name=output_dir if args.output_path else None,
            config=vars(args),
        )

    # ---- Tokenizer ----
    if args.tokenizer_name == "byt5":
        args.tokenizer_name = "google/byt5-small"
    elif args.tokenizer_name == "gpt2":
        args.tokenizer_name = "openai-community/gpt2"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # `tokenizer.vocab_size` can exclude added/special tokens; model embeddings
    # must cover the full emitted id range, i.e. `[0, len(tokenizer)-1]`.
    args.vocab_size = len(tokenizer)
    if args.tokenizer_name == "openai-community/gpt2":
        tokenizer.pad_token_id = tokenizer.vocab_size - 1
        logger.info(f"Pad token ID set to: {tokenizer.pad_token_id}")
    elif args.tokenizer_name == "google/byt5-small":
        tokenizer.bos_token_id = 383
        logger.info(f"BOS token ID set to: {tokenizer.bos_token_id}")
        args.vocab_size = 384

    if reload_ckpt_config is not None:
        ckpt_vocab = reload_ckpt_config.get("vocab_size")
        if ckpt_vocab is not None and args.vocab_size != ckpt_vocab:
            logger.warning(
                "Tokenizer vocab_size (%s) differs from checkpoint (%s).",
                args.vocab_size,
                ckpt_vocab,
            )

    # ---- Dataset ----
    datamodule = BBT(
        seq_len=args.max_seq_len,
        dataset_path=args.dataset,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        pad_idx=tokenizer.pad_token_id,
    )

    datamodule.setup("fit")

    # ---- Model ----
    if args.model_name.startswith("olmo"):
        model = build_olmo2_model(
            model_name=args.model_name,
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            byo_layout=args.byo_layout,
            attn_implementation=args.attn_implementation,
        )
    elif args.model_name.startswith("qwen3"):
        model = build_qwen3_model(
            model_name=args.model_name,
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            byo_layout=args.byo_layout,
        )
    elif "llama" in args.model_name:
        print("LLAMA")
        model = build_llama_model(
            model_name=args.model_name,
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            byo_layout=args.byo_layout,
            attn_implementation=args.attn_implementation,
        )
    else:
        model = build_gpt2_model(args.vocab_size, args.max_seq_len)


    # ---- Print model param count ----
    print(f"Model parameters: {model.num_parameters()/1e6:.2f}M")

    resume_ckpt = None
    resume_global_step = None
    if args.resume:
        resume_ckpt = _resolve_checkpoint("", args.output_path, "--resume")
        state_path = os.path.join(resume_ckpt, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                resume_state = json.load(f)
            resume_global_step = resume_state.get("global_step")
            logger.info(
                "Resuming from checkpoint %s (global_step=%s).",
                resume_ckpt,
                resume_global_step,
            )
        else:
            logger.warning(
                "Checkpoint %s does not include trainer_state.json; "
                "optimizer/scheduler resume may be incomplete.",
                resume_ckpt,
            )
    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=args.output_path if args.output_path else None,
        remove_unused_columns=False,
        label_names=["labels"],
        eval_strategy="steps" if args.eval_steps > 0 else "no",
        save_strategy="steps",
        do_eval=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        log_level="error",
        save_total_limit=args.save_total_limit,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        eval_accumulation_steps=1,
        bf16=True,
        bf16_full_eval=True,
        hub_model_id=args.push_to_hub if args.push_to_hub else None,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        adam_beta1=0.9,
        adam_beta2=0.98,
        weight_decay=0.01,
        max_grad_norm=1.0,
        torch_compile=True,
        ignore_data_skip=False,
        disable_tqdm=not args.debug,
        ddp_find_unused_parameters=False,
        report_to="none" if (args.debug or args.disable_wandb) else "wandb",
    )

    use_wandb = not (args.debug or args.disable_wandb)
    callbacks = [
        LogFlushCallback(logger),
        BPBCallback(datamodule.bytes_per_token, use_wandb=use_wandb),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datamodule.dataset["train"],
        eval_dataset=datamodule.dataset.get(args.eval_split) if args.eval_steps > 0 else None,
        data_collator=datamodule.get_data_collator(),
        compute_metrics=_compute_metrics,
        compute_loss_func=compute_loss, # KEEP!
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )

    trainer.pop_callback(PrinterCallback)

    weight_ckpt = resume_ckpt or reload_ckpt
    if weight_ckpt is not None and (args.eval_only or reload_ckpt is not None):
        logger.info("Loading model weights from %s.", weight_ckpt)
        trainer._load_from_checkpoint(weight_ckpt, model=trainer.model)

    if not args.eval_only:
        trainer.train(resume_from_checkpoint=resume_ckpt)
        if resume_global_step is not None and trainer.state.global_step <= resume_global_step:
            logger.warning(
                "Resume did not advance global_step (start=%s, end=%s). "
                "Check checkpoint validity and output_path.",
                resume_global_step,
                trainer.state.global_step,
            )

    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub)

    if args.push_only:
        exit()

    trainer.evaluate()
