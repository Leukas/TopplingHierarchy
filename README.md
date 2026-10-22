# Toppling Hierarchy

This repo contains our code for training hierarchical byte-level language models with a Build-Your-Own (BYO) char/word layer layout. [Our work](https://arxiv.org/abs/2609.00463) was accepted to Findings of EMNLP 2026. Our main finding is that having more character-level attention leads to better performance on character-manipulation tasks.
Layouts mix character-level (`C`/`c`) and word-level (`W`/`w`) attention and MLP blocks over SpaceByte-style word boundaries.

## Prerequisites
The scripts have been tested on:
- Python 3.12
- Transformers 5.14.1
- [flash-attn 3](https://github.com/Dao-AILab/flash-attention) (recommended)

Training can run with `--attn_implementation sdpa` instead of flash attention 3, but runtime and memory will differ.

`--dataset` in `main.py` expects a local `datasets.save_to_disk` folder (with `train` / `validation` and `num_tokens.txt` for BPB). Raw Hub text datasets must be preprocessed first.

## Preprocessing
Tokenize and pack a Hub (or local) text dataset into the on-disk format used by training:
```bash
python preprocess.py \
  --dataset leukas/climbmix-1b-shuffle \
  --tokenizer_name byt5 \
  --tokenize \
  --group \
  --count_tokens \
  --max_seq_len 2048 \
  --output ./data/climbmix-1b-byt5-2048
```

Then point training at that folder with `--dataset ./data/climbmix-1b-byt5-2048`.

## Training
To train a small Olmo2 BYO model with a middle layout, for example, run:
```bash
python main.py \
  --model_name olmo2_small \
  --tokenizer_name byt5 \
  --dataset ./data/climbmix-1b-byt5-2048 \
  --output_path ./models/olmo2-small-middle \
  --byo_layout middle \
  --batch_size 64 \
  --max_seq_len 2048 \
  --attn_implementation flash_attention_3
```

Supported `--model_name` prefixes:
- `olmo2_{small,base,large}`
- `llama_{small,base,base2,large}`
- `qwen3_{small,base,large}`
- otherwise falls back to GPT-2

All flags are explained here:
```
--model_name                Model family and size, e.g. olmo2_small or llama_base. Default: ""
--tokenizer_name            Tokenizer name or path. Shortcuts: gpt2, byt5. Default: openai-community/gpt2
--output_path               Output directory for model checkpoints and logs. Default: ""
--save_total_limit          Max checkpoint folders to keep (latest always kept for --resume). Default: 2
--dataset                   Path to a datasets.load_from_disk folder. Default: data/bbt
--seed                      Random seed. Default: 0
--debug                     Debug mode (disables wandb, enables tqdm).
--disable_wandb             Disable Weights & Biases logging.
--batch_size                Per-device train/eval batch size. Default: 64
--max_seq_len               Maximum sequence length. Default: 2048
--grad_acc                  Gradient accumulation steps. Default: 1
--lr                        Learning rate. Default: 2e-4
--warmup_steps              Warmup steps for the cosine schedule. Default: 1000
--epochs                    Number of training epochs. Default: 1
--max_steps                 Max optimizer steps (-1 to use epochs). Default: -1
--eval_steps                Run evaluation every N steps (0 to disable). Default: 1000
--save_steps                Save a checkpoint every N steps. Default: 1000
--logging_steps             Log training metrics every N steps. Default: 100
--eval_only                 Skip training and only evaluate.
--eval_split                Dataset split used for evaluation. Default: validation
--push_to_hub               Hugging Face repo id to push the model to.
--push_only                 Push then exit without evaluating.
--resume                    Resume optimizer/scheduler from the latest checkpoint under --output_path.
--reload                    Load model weights from a checkpoint but start training from step 0.
--reload_checkpoint         Checkpoint directory for --reload (defaults to last under --output_path).
--no_reload_config          With --reload, do not read architecture settings from checkpoint config.json.
--byo_layout                BYO layout string or preset (see below). Default: None (standard LM)
--attn_implementation       Attention backend, e.g. flash_attention_3 or sdpa. Default: flash_attention_3
```

## BYO layout
`--byo_layout` selects which layers run at word level vs character level.

Presets (expanded from `num_hidden_layers = n`):
```
middle       →  n/8(Cc) n/4(Ww) n/8(Cc)
bottom       →  n/2(Ww)
top          →  n/2(Cc)
interleaved  →  n/4(WwCc)
```

Or pass an explicit pattern / repeat syntax:
```
--byo_layout WwCcWwCc
--byo_layout 3(Cc)6(Ww)3(Cc)
```

Letters:
- `C` / `c` — character attention / MLP
- `W` / `w` — word attention / MLP

Word boundaries use SpaceByte-style grouping (`group_method=spacebyte`).