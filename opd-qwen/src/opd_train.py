import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import GKDConfig, GKDTrainer

TEACHER = "models/teacher"
DEFAULT_STUDENT = "models/student"

# src/training_logs/<run_name>/{metrics.jsonl, summary.json, config.json}
LOGS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_logs")


class MetricsLoggerCallback(TrainerCallback):
    """实时把 trainer 的 logs 追加到 JSONL，结束时写 summary.json。

    与 wandb / tqdm 不冲突——它只 hook on_log / on_train_end。
    """

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.jsonl_path = os.path.join(log_dir, "metrics.jsonl")
        self.start_time = time.time()
        open(self.jsonl_path, "w").close()  # 清空旧记录

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
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--per_device_batch", type=int, default=2,
                   help="单卡 batch；bf16_split 下显存富余，建议 2-4")
    p.add_argument("--grad_accum", type=int, default=4,
                   help="grad accumulation；effective batch = per_device_batch × grad_accum")
    p.add_argument("--output_dir", default="runs/opd-qwen3-1.7b")
    p.add_argument("--run_name", default="opd-qwen3-1.7b-gsm8k")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"],
                   help="A5000 sm_86 + flash-attn 2.7+ 推荐用 flash_attention_2")
    p.add_argument("--student", default=DEFAULT_STUDENT,
                   help="student 模型路径；默认 models/student（Instruct）。"
                        "切 base：models/student-base")
    p.add_argument("--lr", type=float, default=1e-5,
                   help="学习率；base student 建议 2e-5")
    p.add_argument(
        "--teacher_bf16",
        action="store_true",
        help="单卡 bf16 teacher（实测 24G 不够，仅供 ≥ 40G 卡用）",
    )
    p.add_argument(
        "--teacher_bf16_split",
        action="store_true",
        help="bf16 teacher 跨 2 张卡（logical cuda:0 + cuda:2）；"
             "用法：CUDA_VISIBLE_DEVICES=0,2,3 + 此 flag",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(
        f"[OPD] CUDA devices visible: {torch.cuda.device_count()} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','-')})"
    )
    if args.teacher_bf16_split:
        assert torch.cuda.device_count() >= 2, (
            "--teacher_bf16_split 需要至少 2 张可见 GPU。"
            "2 卡紧凑：`CUDA_VISIBLE_DEVICES=0,3`；3 卡宽松：`CUDA_VISIBLE_DEVICES=0,2,3`"
        )
    else:
        assert torch.cuda.device_count() >= 2, (
            "需要至少 2 张可见 GPU。先 `export CUDA_VISIBLE_DEVICES=0,3`"
        )

    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    _orig_apply = tok.apply_chat_template
    def _no_think_apply(*a, **kw):
        kw.setdefault("enable_thinking", False)
        return _orig_apply(*a, **kw)
    tok.apply_chat_template = _no_think_apply
    print("[OPD] patched tokenizer.apply_chat_template: enable_thinking=False")

    # 三种 teacher 加载模式：
    #   默认（4-bit）：CUDA_VISIBLE_DEVICES=0,3，teacher 4-bit → logical 0，student bf16 → logical 1
    #   bf16_split 3-GPU 宽松：CUDA_VISIBLE_DEVICES=0,2,3，teacher 跨 logical 0+2，student 独占 logical 1
    #   bf16_split 2-GPU 紧凑：CUDA_VISIBLE_DEVICES=0,3，teacher 偏 logical 0，shard 2 + student 共享 logical 1
    if args.teacher_bf16_split:
        n_gpu = torch.cuda.device_count()
        if n_gpu >= 3:
            # 3 卡宽松：teacher shard 各 22 GiB；student 独占 logical 1
            max_mem = {0: "22GiB", 2: "22GiB"}
            print(
                "[OPD] bf16_split (3-GPU): teacher → logical 0+2 sharded, "
                "student → logical 1 ..."
            )
        else:
            # 2 卡紧凑：让 teacher shard 1 偏 logical 0，shard 2 留小给 student 共享 logical 1
            # 物理 0：teacher shard 1 (~12 GiB) + activations
            # 物理 3：teacher shard 2 (~8 GiB) + student bf16+LoRA + ckpt acts + JSD stack ≈ 18-20 GiB
            max_mem = {0: "12GiB", 1: "8GiB"}
            print(
                "[OPD] bf16_split (2-GPU compact): teacher shard 1 → logical 0, "
                "teacher shard 2 + student SHARED → logical 1.\n"
                "      显存吃紧，建议 --max_new_tokens ≤ 512, --per_device_batch ≤ 2"
            )
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_mem,
            attn_implementation=args.attn,
        )
    elif args.teacher_bf16:
        print("[OPD] loading teacher (bf16, single GPU) → logical cuda:0 ...")
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation=args.attn,
        )
    else:
        # 4-bit nf4：8B 权重 ~16G → ~5G，给 activations / logits / GKD stack 留出足够空间
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print(f"[OPD] loading teacher (4-bit nf4, attn={args.attn}) → logical cuda:0 ...")
        teacher = AutoModelForCausalLM.from_pretrained(
            TEACHER,
            quantization_config=bnb,
            device_map={"": 0},
            attn_implementation=args.attn,
        )
    teacher.eval()

    print(f"[OPD] loading student ({args.student}) → logical cuda:1 ...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=torch.bfloat16,
        device_map={"": 1},
        attn_implementation=args.attn,
    )

    # 数据：GKDTrainer 吃 messages 格式
    # Qwen3 chat template 默认开 thinking，会让 student rollout 偏离 GT 分布；
    # GKDTrainer 在 tokenizer 上调 apply_chat_template，关 thinking 由 trainer 内部
    # 走的 chat_template_kwargs 控制，必要时可在此修改 tokenizer 的默认行为。
    raw = load_dataset("openai/gsm8k", "main", split="train")

    def fmt(ex):
        return {
            "messages": [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ]
        }

    ds = raw.map(fmt, remove_columns=raw.column_names)
    print(f"[OPD] train dataset size: {len(ds)}")

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

    cfg = GKDConfig(
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
        save_total_limit=2,                   # 防磁盘爆（本机 / 仅剩 ~130G）
        disable_tqdm=False,                   # 显式启用进度条
        # GKD 专属
        lmbda=args.lmbda,
        beta=args.beta,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        seq_kd=False,
        # logging
        report_to=("none" if args.no_wandb else "wandb"),
        run_name=args.run_name,
    )

    log_dir = os.path.join(LOGS_ROOT, args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    metrics_cb = MetricsLoggerCallback(log_dir)

    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
        callbacks=[metrics_cb],
    )

    # gradient checkpointing + LoRA：必须在 wrap 后的 PeftModel 上调用，
    # 否则 forward 走 PeftModel 时 hook 注册位置错，依旧拿不到 grad
    trainer.model.enable_input_require_grads()

    print(f"[OPD] start training; metrics → {log_dir}/metrics.jsonl")
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    print(
        f"[OPD] done. final adapter → {final_dir}; "
        f"summary → {log_dir}/summary.json"
    )


if __name__ == "__main__":
    main()
