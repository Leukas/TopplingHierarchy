# logging_utils.py
import os
import sys
import time
import datetime
import logging
import numpy as np
import torch
from transformers.trainer_callback import TrainerCallback
import pynvml

logger = logging.getLogger(__name__)


# def get_gpu_utilization():
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     return f"[GPU: {info.used/1024**3:.2f}/{info.total/1024**3:.2f} GB]"

def get_gpu_utilization():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return f"[GPU: {info.used/1024**3:.2f}/{info.total/1024**3:.2f} GB]"
    except pynvml.NVMLError:
        return "[GPU: N/A]"


class LogFlushCallback(TrainerCallback):
    """ Like printer callback, but with logger and flushes the logs every call """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            formatted = {
                k: (f"{v:.4e}" if k == "learning_rate" else f"{v:.4f}") if isinstance(v, float) else v
                for k, v in logs.items()
            }
            self.logger.info(f"[step {state.global_step}] {formatted}")
            sys.stdout.flush()


def load_logger(logger):
    class LogFormatter():
        def __init__(self):
            self.start_time = time.time()

        def format(self, msg):
            time_passed = round(msg.created - self.start_time)
            gpu_usage = get_gpu_utilization()
            prefix = "[%s - %s]" % (
                time.strftime('%x %X'),
                datetime.timedelta(seconds=time_passed)
            )
            prefix = prefix + gpu_usage
            msg_text = msg.getMessage()
            return "%s %s" % (prefix, msg_text) if msg_text else ''

    logger.setLevel(logging.INFO)
    # Remove any existing handlers to prevent duplicates
    logger.handlers.clear()
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(LogFormatter())
    logger.addHandler(log_handler)
    return logger


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def write_output_to_file(eval_pred, tokenizer, output_path):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    if hasattr(eval_pred, 'input_ids'):
        inputs = eval_pred.input_ids
    else:
        inputs = None

    preds = [] # detokenized outputs
    labs = [] # detokenized labels
    inps = [] # detokenized inputs
    for b in range(len(predictions)):

        l_eos_idx = np.argmax(labels[b]==tokenizer.eos_token_id)
        lab = labels[b][:l_eos_idx]
        p_eos_idx = np.argmax(predictions[b]==tokenizer.eos_token_id)
        if p_eos_idx == 0:
            p_eos_idx = len(predictions[b])
        pred = predictions[b][1:p_eos_idx]

        # remove right and left padding
        lab = lab[lab != -100]
        pred = pred[pred != -100]
        pred = pred[pred != 0]
        
        if inputs is not None:
            inp = inputs[b]
            inp = inp[inp != 0]
            # remove prompt from pred
            pred = pred[len(inp):]
            dt_inp = tokenizer.decode(inp)
            inps.append(dt_inp)


        dt_lab = tokenizer.decode(lab)
        dt_pred = tokenizer.decode(pred)

        labs.append(dt_lab)
        preds.append(dt_pred)


    with open(output_path + "hyp.txt", 'w', encoding='utf8') as file:
        for line in preds:
            file.write(f"{line}\n")

    with open(output_path + "ref.txt", 'w', encoding='utf8') as file:
        for line in labs:
            file.write(f"{line}\n")

    if inputs is not None:
        with open(output_path + "src.txt", 'w', encoding='utf8') as file:
            for line in inps:
                file.write(f"{line}\n")

