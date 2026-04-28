# Qwen3-8B → Qwen3-1.7B On-Policy Distillation 技术路线

> **⚠️ 原始指南** — 本机实测后的最终配置见 [`opd-qwen/EXPERIMENT_PLAN.md`](opd-qwen/EXPERIMENT_PLAN.md) 与 [`opd-qwen/BENCHMARK.md`](opd-qwen/BENCHMARK.md)。
> 关键更正（实测得出）：
> - GPU 3 被外部进程占用，本项目实际只用 GPU 0+2
> - TRL `GKDTrainer` 单卡化（强制把 student 拉回 GPU 0），故 student device_map 设置无效
> - Qwen3 必须显式 `enable_thinking=False`，否则单步 159 s（已在 `src/opd_train.py` / `sft_baseline.py` 修复）
> - bf16_split 模式不推荐（跨 PCIe 慢且不稳定），生产配置 = 4-bit nf4 单卡 teacher

---

**目标硬件**：三种部署
- **2× A5000 + 4-bit teacher**（默认）：teacher 4-bit + student bf16，各占 1 卡
- **2× A5000 + bf16 紧凑**：teacher bf16 sharded 跨 2 卡，shard 2 与 student **共享** 1 张卡（显存边界，max_new_tokens≤512）
- **3× A5000 + bf16 宽松**：teacher bf16 sharded 跨 2 卡 + student 独占 1 卡

**任务**：GSM8K 数学推理
**算法**：On-Policy Distillation (Lu et al., Thinking Machines, Oct 2025) via TRL `GKDTrainer`

> **本机适配修订（2026-04-26，详细推导见 [feasibility-report.md](feasibility-report.md)）**
> 1. 版本固定为 `trl==0.21.* / transformers==4.46.* / peft==0.13.* / accelerate==1.0.*`，避免半年内接口漂移
> 2. flash-attn 直接装 wheel：`flash-attn==2.7.4.post1 --no-build-isolation`，不再源码编译
> 3. **Qwen3 thinking 模式必须关闭**（eval/训练 chat template 都加 `enable_thinking=False`），否则 baseline 数字被低估
> 4. `huggingface-cli download --local-dir` 加 `--local-dir-use-symlinks False`，避免 HF cache 与 local-dir 双份占用（本机 / 仅剩 132 GB，偏紧）
> 5. **卡分配（用户指定方案：bf16 OPD 用 0+3，SFT 用 2）**：`CUDA_VISIBLE_DEVICES=0,3 python src/opd_train.py --teacher_bf16_split`，触发 **2-GPU 紧凑模式**：teacher shard 1 → 物理 0，teacher shard 2 + student **共享** 物理 3；SFT 占物理 2；eval 用物理 0（训练完成后）或物理 2（SFT 完成后）。**物理 GPU 1 已损坏，禁止用于任何计算**。显存吃紧，必须 `--max_new_tokens ≤ 512 --per_device_batch ≤ 2`。**bf16 teacher 单卡 24G 装不下**（实测 OOM）；若 OOM 退路：去掉 `--teacher_bf16_split` 切回 4-bit teacher。
> 6. **`gradient_checkpointing_kwargs={"use_reentrant": False}`** 必须配上，否则 PEFT + ckpt 会触发 `None of the inputs have requires_grad=True`，梯度断链
> 7. **`trainer.model.enable_input_require_grads()`**（不是 raw `student`），因为 trl 内部把 student wrap 成 PeftModel，hook 必须打在 wrap 后的对象上
> 8. **batch 配置**：bf16_split 下显存富余，用 `--per_device_batch 2 --grad_accum 4`（effective batch = 8 不变，wall time ~0.55×）
> 9. 启动前必加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，减少显存碎片
> 10. **关键指标实时落盘到 `src/training_logs/<run_name>/metrics.jsonl`**（`MetricsLoggerCallback`），崩溃也不丢；训练结束写 `summary.json`

---

## 0. 环境

A5000 是 Ampere (sm_86)，CUDA 12.1+ 即可。

```bash
# 安装 uv（比 pip 快）
curl -LsSf https://astral.sh/uv/install.sh | sh

mkdir opd-qwen && cd opd-qwen
uv venv --python 3.11 && source .venv/bin/activate

uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# 固版（修订）：避免几个月后 minor 接口漂移
uv pip install "transformers==4.46.*" "trl==0.21.*" "peft==0.13.*" \
    "accelerate==1.0.*" "datasets>=3.0" bitsandbytes wandb hf_transfer
# 直接装 wheel，无需源码编译（修订）
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

避坑：
- `flash-attn` 2.7.4.post1 已有 cu121/torch2.5 官方 wheel，无需源码编译。装失败再退回 `attn_implementation="sdpa"`，吞吐 -30%
- A5000 没有 fp8，所有 fp8 相关库（如 transformer_engine）不要装
- TRL 0.21 仍兼容 transformers 4.46–4.50；不要单独升级 transformers 到 4.50+，chat template 行为可能变

---

## 1. HF 配置（日本 IP 直连）

日本节点连 HF 不需要镜像，直接开 `hf_transfer` 加速即可。

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
# 不要同时设 HF_HOME 和用 --local-dir，否则会双份占用磁盘（本机 / 偏紧）。
# 选一种：(a) 用 --local-dir 显式拉到 models/teacher 等 → 不设 HF_HOME；
#         (b) 走 HF cache → 不传 --local-dir，代码里用 model_id "Qwen/Qwen3-8B"
# 本指南选 (a)。
huggingface-cli login   # 粘贴 token，避免限流
```

---

## 2. 下载模型和数据

```bash
# 修订：加 --local-dir-use-symlinks False，避免 HF cache 与 local-dir 双份占用
# Teacher: Qwen3-8B (instruct, ~16GB)
huggingface-cli download Qwen/Qwen3-8B \
    --local-dir models/teacher --local-dir-use-symlinks False

# Student: Qwen3-1.7B (instruct, ~3.4GB) — 用 instruct 版直接起，省一道 SFT
huggingface-cli download Qwen/Qwen3-1.7B \
    --local-dir models/student --local-dir-use-symlinks False

# Dataset
huggingface-cli download openai/gsm8k --repo-type dataset \
    --local-dir data/gsm8k --local-dir-use-symlinks False
```

避坑：
- 模型名 `Qwen/Qwen3-1.7B` 已经是 instruct 版；`-Base` 后缀的是 pretraining 模型，需要先 SFT 才能用
- 别下 `Qwen3-1.7B-Chat`，那是老版本

---

## 3. 项目结构

```
OnPolicyDistillation_Project/
├── opd-qwen3-guide.md          # 本指南
├── feasibility-report.md       # 环境检验报告
├── disk-cleanup-report.md      # 磁盘清理建议
└── opd-qwen/                   # 实际工作区
    ├── .venv/                  # uv 创建的 python 3.11 env
    ├── models/{teacher,student}/
    ├── data/gsm8k/
    ├── src/
    │   ├── eval_gsm8k.py       # 含 enable_thinking=False 修订
    │   ├── opd_train.py        # OPD 训练，含 MetricsLoggerCallback
    │   ├── sft_baseline.py     # 同 FLOPs 对照
    │   └── training_logs/
    │       └── <run_name>/
    │           ├── metrics.jsonl   # 实时落盘的 step-level 指标
    │           ├── summary.json    # 训练结束的总览
    │           └── config.json     # 该 run 的 CLI 参数
    └── runs/                   # ckpt 输出（save_total_limit=2）
```

---

## 4. Eval 脚本

```python
# src/eval_gsm8k.py
import re, torch, argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def extract_answer(text):
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m: return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{args.n}]")
    correct = 0
    for ex in ds:
        msgs = [{"role": "user", "content": ex["question"]}]
        # 修订：Qwen3 默认会注入 <think> 段，必须显式关闭，否则抽数会抽到中间步
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.input_ids.shape[1]:],
                          skip_special_tokens=True)
        # 兜底：若意外开了 thinking，剥掉 <think>...</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if extract_answer(text) == extract_answer(ex["answer"]):
            correct += 1
    print(f"Acc: {correct/len(ds):.3f} ({correct}/{len(ds)})")

if __name__ == "__main__":
    main()
```

跑 baseline（**先做这步**，没数字就没 ablation）：

```bash
# 卡分配：OPD 训练占 0/3（bf16 紧凑或 4-bit），SFT 占 2，GPU 1 坏掉禁用
# Student baseline（一张卡就够，训练前在任意空闲卡上跑）
CUDA_VISIBLE_DEVICES=0 python src/eval_gsm8k.py \
    --model models/student --n 200

# Teacher upper bound（bf16 单卡装不下 8B，eval 时也要跨卡）
# 在 OPD 训练前一次性跑完（OPD 启动后这两张卡就被占了）
CUDA_VISIBLE_DEVICES=0,3 python src/eval_gsm8k.py \
    --model models/teacher --n 200
```

> 注：student baseline (bf16 单卡) 与训练后 student (bf16 + LoRA) **eval 条件一致**，可直接对比；teacher upper bound 跨 2 卡 bf16 forward 推理一次约 30-40 分钟（200 条），如果赶时间可用 4-bit teacher eval 但要在简历里标注。

期望数字：student ~70%，teacher ~88-92%（200 条子集会有波动，最终用全量 1319 条）。

避坑：
- `do_sample=False` 才能让 eval 数字稳定
- Qwen3-1.7B 的 baseline 已经不低，所以项目的卖点要变成 **"训练效率"** 而不是 "提点"——同样 FLOPs 下 OPD vs SFT 的差距才是关键

---

## 5. 核心：on-policy distillation 训练脚本

`GKDTrainer` 关键参数：

| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `lmbda` | on-policy 比例。1.0=纯 student rollout，0.0=纯 ground-truth | `0.5` 起步 |
| `beta` | generalized JSD 插值。0=forward KL，1=reverse KL，0.5=对称 JSD | `0.5` |
| `temperature` | rollout 采样温度 | `0.9` |
| `max_new_tokens` | student 单次 rollout 长度 | `512` 默认（4-bit）/ `1024`（bf16_split 富余下） |
| `seq_kd` | True=序列级 KD（每整段一个 loss），False=token 级 | `False`（信号更密） |
| `per_device_train_batch_size` | 单卡 batch | `2`（bf16_split 富余）|
| `gradient_accumulation_steps` | grad 累积 | `4`（保持 effective batch = 8）|

完整脚本见 [src/opd_train.py](opd-qwen/src/opd_train.py)，关键设计点：

- **三种 teacher 加载模式**（`--teacher_bf16_split` / `--teacher_bf16` / 默认 4-bit nf4），按显存预算切换
- **`MetricsLoggerCallback`** 把 `loss / kl / grad_norm / lr / student_response_length` 等指标按 step 实时追加到 `src/training_logs/<run_name>/metrics.jsonl`，崩溃也能复盘
- **必备的两条**：`gradient_checkpointing_kwargs={"use_reentrant": False}` + `trainer.model.enable_input_require_grads()`（在 wrap 后的 PeftModel 上调用），少一条都会触发 grad=None
- **CLI 参数化** `lmbda / beta / max_steps / max_new_tokens / per_device_batch / grad_accum / output_dir / run_name`，§7 ablation 直接 for 循环复用
- **断言** `CUDA_VISIBLE_DEVICES` 数量 ≥ 模式所需

启动：

```bash
wandb login   # 粘 wandb token

# === 用户当前选择：bf16 OPD on 0+3，SFT on 2（2-GPU 紧凑模式）===
# 物理 0：teacher shard 1 (~12 GB) + activations
# 物理 3：teacher shard 2 (~8 GB) + student bf16+LoRA + ckpt acts + JSD stack ≈ 18-20 GB
# 物理 2：SFT baseline 单独跑（独立 process）
# 物理 1：eval 用
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,3 python src/opd_train.py \
    --teacher_bf16_split \
    --max_steps 300 --max_new_tokens 512 \
    --per_device_batch 2 --grad_accum 4

# === 备选 1：3-GPU 宽松模式（多一张卡，student 独占一张，最稳定）===
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,2,3 python src/opd_train.py \
    --teacher_bf16_split \
    --max_steps 300 --max_new_tokens 512 \
    --per_device_batch 2 --grad_accum 4
# 注：此模式 OPD 占 0/2/3，GPU 1 已损坏，SFT 无法与 OPD 并行运行；需 OPD 跑完再串行跑 SFT

# === 备选 2：4-bit teacher 退路（紧凑模式 OOM 时使用）===
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,3 python src/opd_train.py \
    --max_steps 300 --max_new_tokens 512 \
    --per_device_batch 2 --grad_accum 4
```

> 冒烟测试（**先做这步**，2-GPU 紧凑显存边界，必须先验证 5 步不 OOM）：
> ```bash
> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
> CUDA_VISIBLE_DEVICES=0,3 python src/opd_train.py \
>     --teacher_bf16_split \
>     --max_steps 5 --max_new_tokens 256 \
>     --per_device_batch 1 --grad_accum 8 --no_wandb
> ```
> 冒烟用 batch=1 + max_new_tokens=256 最保守，跑通后再加大到 batch=2 + 512。

**OOM 应急**（紧凑模式踩边界）：
1. `--per_device_batch 1 --grad_accum 8`（保 effective batch=8 不变）
2. `--max_new_tokens 256`
3. 仍 OOM → 去掉 `--teacher_bf16_split` 切回 4-bit

避坑（**这一节最容易翻车，仔细看**）：

1. **bf16 teacher 单卡 24G 必 OOM**（实测两次：`max_new_tokens=1024` 和 `=512` 都炸）。三种活法：
   - `--teacher_bf16_split`：2 卡紧凑（共享 student 卡）或 3 卡宽松（student 独占），按 device_count 自适应
   - 默认 4-bit nf4 单卡（省显存，1-3 acc 点损失）
   - 单卡 bf16 仅在 ≥40G 卡（A100/H100）时可行

2. **single GPU `device_map={"": N}` 不要写成 `"auto"`**。`auto` 会按可用显存自动切分，把 student 也切到 teacher 卡上 → 双 OOM。

3. **Gradient checkpointing + LoRA 的两条命脉**（少一条 grad=None）：
   - `gradient_checkpointing_kwargs={"use_reentrant": False}` 写进 `GKDConfig` / `SFTConfig`
   - **`trainer.model.enable_input_require_grads()`**（不是 raw `student`），因为 trl 内部 wrap 成 PeftModel，hook 必须打在 wrap 后

4. **OOM 三层应对**（按顺序试）：
   - 启动加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（碎片省 1-2 GB）
   - `max_new_tokens` 1024 → 768 → 512
   - `per_device_batch` 4 → 2 → 1（同时调大 `grad_accum` 保 effective batch 不变）
   - Teacher 切 4-bit（去掉 `--teacher_bf16_split`）

5. **Tokenizer 一致性**：teacher 和 student 必须共用 tokenizer。Qwen3 系列内部共用，没事；要是混搭家族（Llama teacher + Qwen student）GKDTrainer 直接拒绝跑。

6. **`lmbda` 选择**：纯 on-policy（`lmbda=1.0`）在 student baseline 弱时学不起来——student 自己 rollout 的轨迹和 teacher 期望的差太远，KL 爆炸。`0.5` 是稳健起点，跑通后再做 ablation。

7. **长度膨胀**：训 100+ 步后若发现 student 输出越来越长 + 重复。临时缓解：在 generation_config 里加 `repetition_penalty=1.05`。

8. **wandb / metrics.jsonl 看什么**：重点盯 `loss`、`kl`、`grad_norm`、`student_response_length`。后两个曲线异常（KL 不降 / 长度暴涨 / grad_norm=0）就是训崩前兆，立刻停下检查 lmbda。

9. **磁盘**：本机 / 仅剩 132 GB。`save_total_limit=2` 已固化在 `GKDConfig`/`SFTConfig`，避免 ckpt 累积撑爆。

10. **训练并行不等于 GPU 并行**：accelerate `device_map="auto"` 是 **pipeline parallel**（顺序），不是 tensor parallel。teacher 跨 2 卡 forward 永远是 shard 1 → shard 2 串行，nvidia-smi 上看到的"一张卡 80% util、另一张 0%"是正常的，不是 bug。想压榨 wall time，加大 `per_device_batch` 比追求"两卡同时跑"更直接。

---

## 5b. SFT baseline（用于 ablation §7-A）

完整脚本见 [src/sft_baseline.py](opd-qwen/src/sft_baseline.py)，与 OPD 完全对齐：

| 项 | OPD | SFT baseline |
|------|------|------|
| 模型 | Qwen3-1.7B + LoRA r=64/alpha=128/dropout=0.05 | **同上** |
| `max_steps` | 300 | **300（同 FLOPs）** |
| `per_device_batch × grad_accum` | 2 × 4 = 8 | **2 × 4 = 8** |
| `learning_rate / warmup_ratio` | 1e-5 / 0.05 | **同上** |
| `gradient_checkpointing` + `use_reentrant=False` | ✅ | ✅ |
| 信号源 | teacher logits (JSD) | ground-truth answer (CE) |
| Metrics 落盘 | `src/training_logs/opd-qwen3-1.7b-gsm8k/` | `src/training_logs/sft-qwen3-1.7b-gsm8k/` |

启动（与 OPD 训练**并行**；OPD 占 0/3 → SFT 占 2）：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=2 python src/sft_baseline.py \
    --max_steps 300 --per_device_batch 2 --grad_accum 4
```

> 冒烟测试：
> ```bash
> CUDA_VISIBLE_DEVICES=2 python src/sft_baseline.py \
>     --max_steps 5 --per_device_batch 2 --grad_accum 4 --no_wandb
> ```
>
> 若用 3-GPU 宽松模式（OPD 占 0/2/3），GPU 1 已损坏，SFT 无法与 OPD 并行；需 OPD 训完后再串行跑 SFT

---

## 6. 训完 eval

```bash
# OPD 训完后的 eval（GPU 0 训练结束后空闲）
CUDA_VISIBLE_DEVICES=0 python src/eval_gsm8k.py \
    --model models/student \
    --lora runs/opd-qwen3-1.7b/final \
    --n 1319   # 全量 test set

# SFT baseline 的 eval（GPU 2 SFT 结束后空闲）
CUDA_VISIBLE_DEVICES=2 python src/eval_gsm8k.py \
    --model models/student \
    --lora runs/sft-qwen3-1.7b/final \
    --n 1319
```

---

## 7. Ablation（简历价值的核心）

光有一个 OPD 数字不够，需要对比线。**至少跑这两个**：

**A. 纯 SFT 同 FLOPs baseline**：用 [src/sft_baseline.py](opd-qwen/src/sft_baseline.py)，相同 `max_steps=300` + 相同 batch + 相同 LoRA config，在 GSM8K train 的 ground-truth 答案上做 SFT（已写好，见 §5b）。这是回答"为什么不直接 SFT"的关键对照。

**B. lmbda 扫描**：`lmbda ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` 各跑一次（每个 200 steps 即可），画 GSM8K acc 曲线：

```bash
# 用户路径：bf16 紧凑模式（5 × ~3.5 小时 ≈ 17 小时，一夜挂完）
for L in 0.0 0.25 0.5 0.75 1.0; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,3 python src/opd_train.py \
      --teacher_bf16_split \
      --lmbda $L --max_steps 200 --max_new_tokens 512 \
      --per_device_batch 2 --grad_accum 4 \
      --output_dir runs/abl/lmbda_$L \
      --run_name opd-lmbda-$L
done

# 备选：4-bit 退路（去掉 --teacher_bf16_split，~14-16 小时）
```

每个 run 的 metrics 自动落到 `src/training_logs/opd-lmbda-{L}/metrics.jsonl`，画图直接读。

可选 C（更深入）：`beta ∈ {0.1, 0.5, 0.9}` 看 forward / 对称 / reverse KL 的差异。

---

## 8. 时间和资源估算（实测口径，per_device_batch=2 / grad_accum=4）

冒烟测试实测：5 步 = 506 秒 ≈ **101 秒/step**（max_new_tokens=512, bf16_split 3-GPU）。
**rollout 才是大头**（每 step 4 次 student.generate），不是 teacher forward。

| 阶段 | 4-bit 退路（0/3） | **bf16 紧凑（0/3）** | bf16 宽松（0/2/3） |
|------|-------------------|---------------------|---------------------|
| 模型 + 数据下载 | 15-20 分钟 | 同 | 同 |
| Baseline eval：student 单卡（200 条） | ~5-8 分钟 | 同 | 同 |
| Baseline eval：teacher 跨卡 bf16（200 条）| ~25-35 分钟 | 同 | 同 |
| **OPD 训练（300 steps, max_new_tokens=512）** | **~4-5 小时**（每步 ~50-60s）| **~5-6 小时**（每步 ~70s，紧凑共享有 PCIe + 显存交换开销）| ~8-10 小时（每步 ~100s）|
| SFT baseline（300 steps，并行）| ~40-60 分钟（GPU 2）| ~40-60 分钟（GPU 2）| 无法并行（GPU 1 坏，OPD 占 0/2/3）|
| 全量 eval（1319 条 × N 个 ckpt）| 每次 30-40 分钟 | 同 | 同 |
| Ablation（5 个 lmbda × 200 steps）| ~14-16 小时 | **~17-20 小时**（一夜挂完） | ~25 小时 |

**总计（用户路径 bf16 紧凑）**：约 1.5 天纯训练时间。建议晚上挂 ablation。

---

## 9. 简历产出 checklist

- [ ] **wandb 公开报告链接**（项目设为 public，附上）
- [ ] **关键数字表**：

| Setting | GSM8K Acc | Train FLOPs (rel) |
|---------|-----------|-------------------|
| Teacher Qwen3-8B (bf16, upper bound) | xx.x% | — |
| Teacher Qwen3-8B (4-bit nf4, used if 4-bit teacher run) | yy.y% | — |
| Student baseline (Qwen3-1.7B) | xx.x% | 0 |
| SFT (300 steps, batch 8) | xx.x% | 1× |
| **OPD bf16_split (300 steps, batch 8)** | **xx.x%** | ~1.5× |

> 用 `src/training_logs/<run_name>/metrics.jsonl` 直接读 loss/kl/grad_norm/student_response_length 画图。

- [ ] **训练曲线截图**：loss + KL + response_length（从 metrics.jsonl 画或截 wandb）
- [ ] **lmbda ablation 折线图**
- [ ] **README** 含：算法引用 (Lu et al., 2025)、硬件 (3× RTX A5000：teacher sharded + student)、复现命令、所有超参

**简历 bullet 模板**：

> Implemented on-policy distillation (Lu et al., 2025) on 3× RTX A5000:
> distilled Qwen3-8B (bf16, sharded across 2 GPUs) reasoning into Qwen3-1.7B
> via LoRA, achieving X% on GSM8K (vs. Y% off-policy SFT baseline at equal
> FLOPs, run in parallel on the 4th GPU); ablated lmbda ∈ {0, 0.25, 0.5,
> 0.75, 1.0} and reverse-KL vs JSD losses.

---

## 10. 如果想往论文方向延伸

跑通 baseline + ablation 后，可以考虑这几个方向（任选其一就够写一篇 workshop paper）：

- **Stable-OPD 复现 + 改进**：实现 reference-based KL 约束，看能否缓解长度膨胀
- **OPSD 简化版**：让 1.7B 同时当 teacher 和 student（teacher 看 ground-truth，student 不看），不需要 8B
- **跨 domain 持续学习**：先在 code（如 MBPP）上 SFT 让 IF-eval 掉点，再用 OPD 恢复指令遵循能力——这是 Thinking Machines 博客后半段的实验
