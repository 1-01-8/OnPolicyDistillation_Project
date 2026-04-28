"""SFT baseline，与 OPD 等 FLOPs 对照。

与 opd_train.py 对齐：
- 同 LoRA config (r=64, alpha=128)
- 同 effective batch (per_device_batch × grad_accum = 8)
- 同 max_steps / lr / warmup / scheduler
- 同 MetricsLoggerCallback 把指标实时落盘到 src/training_logs/<run_name>/
唯一差别：用 ground-truth 答案做 SFT，不蒸馏 teacher logits。
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

DEFAULT_STUDENT = "models/student"

LOGS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_logs")


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.jsonl_path = os.path.join(log_dir, "metrics.jsonl")
        self.start_time = time.time()
        open(self.jsonl_path, "w").close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {
            "step": state.global_step,
            "epoch": state.epoch,
            "wall_seconds": round(time.time() - self.start_time, 2),
            **logs,
        }
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        summary = {
            "global_step": state.global_step,
            "best_metric": state.best_metric,
            "log_history_tail": state.log_history[-5:] if state.log_history else [],
            "wall_seconds_total": round(time.time() - self.start_time, 2),
        }
        with open(os.path.join(self.log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--per_device_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=1024,
                   help="prompt+response 整段长度上限；GSM8K 单条 ~400-600 tokens")
    p.add_argument("--student", default=DEFAULT_STUDENT,
                   help="student 模型路径；默认 models/student。base 用 models/student-base")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--output_dir", default="runs/sft-qwen3-1.7b")
    p.add_argument("--run_name", default="sft-qwen3-1.7b-gsm8k")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    return p.parse_args()


def main():
    args = parse_args()

    print(
        f"[SFT] CUDA devices visible: {torch.cuda.device_count()} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','-')})"
    )
    assert torch.cuda.device_count() >= 1

    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    _orig_apply = tok.apply_chat_template
    def _no_think_apply(*a, **kw):
        kw.setdefault("enable_thinking", False)
        return _orig_apply(*a, **kw)
    tok.apply_chat_template = _no_think_apply
    print("[SFT] patched tokenizer.apply_chat_template: enable_thinking=False")

    print(f"[SFT] loading student ({args.student}) → logical cuda:0 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=args.attn,
    )

    raw = load_dataset("openai/gsm8k", "main", split="train")

    def fmt(ex):
        return {
            "messages": [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ]
        }

    ds = raw.map(fmt, remove_columns=raw.column_names)
    print(f"[SFT] train dataset size: {len(ds)}")

    lora = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    cfg = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        disable_tqdm=False,
        max_length=args.max_length,
        report_to=("none" if args.no_wandb else "wandb"),
        run_name=args.run_name,
    )

    log_dir = os.path.join(LOGS_ROOT, args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    metrics_cb = MetricsLoggerCallback(log_dir)

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
        callbacks=[metrics_cb],
    )

    trainer.model.enable_input_require_grads()

    print(f"[SFT] start training; metrics → {log_dir}/metrics.jsonl")
    trainer.train()
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    print(
        f"[SFT] done. final adapter → {final_dir}; "
        f"summary → {log_dir}/summary.json"
    )


if __name__ == "__main__":
    main()
