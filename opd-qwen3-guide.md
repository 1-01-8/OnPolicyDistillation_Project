# Qwen3-8B → Qwen3-1.7B On-Policy Distillation 技术路线

**目标硬件**：2× RTX A5000 (24GB each)
**任务**：GSM8K 数学推理
**算法**：On-Policy Distillation (Lu et al., Thinking Machines, Oct 2025) via TRL `GKDTrainer`

---

## 0. 环境

A5000 是 Ampere (sm_86)，CUDA 12.1+ 即可。

```bash
# 安装 uv（比 pip 快）
curl -LsSf https://astral.sh/uv/install.sh | sh

mkdir opd-qwen && cd opd-qwen
uv venv --python 3.11 && source .venv/bin/activate

uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install "transformers>=4.46" "trl>=0.13" "peft>=0.13" \
    "accelerate>=1.0" datasets bitsandbytes wandb hf_transfer
uv pip install flash-attn --no-build-isolation
```

避坑：
- `flash-attn` 编译要 5-10 分钟。失败就跳过，代码里把 `attn_implementation="sdpa"` 顶上，性能差不多
- A5000 没有 fp8，所有 fp8 相关库（如 transformer_engine）不要装
- TRL < 0.13 的 GKDTrainer 接口和这里写的不一样，务必装到 0.13+

---

## 1. HF 配置（日本 IP 直连）

日本节点连 HF 不需要镜像，直接开 `hf_transfer` 加速即可。

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=$PWD/hf_cache
huggingface-cli login   # 粘贴 token，避免限流
```

---

## 2. 下载模型和数据

```bash
# Teacher: Qwen3-8B (instruct, ~16GB)
huggingface-cli download Qwen/Qwen3-8B --local-dir models/teacher

# Student: Qwen3-1.7B (instruct, ~3.4GB) — 用 instruct 版直接起，省一道 SFT
huggingface-cli download Qwen/Qwen3-1.7B --local-dir models/student

# Dataset
huggingface-cli download openai/gsm8k --repo-type dataset \
    --local-dir data/gsm8k
```

避坑：
- 模型名 `Qwen/Qwen3-1.7B` 已经是 instruct 版；`-Base` 后缀的是 pretraining 模型，需要先 SFT 才能用
- 别下 `Qwen3-1.7B-Chat`，那是老版本

---

## 3. 项目结构

```
opd-qwen/
├── models/{teacher,student}/
├── data/gsm8k/
├── src/
│   ├── eval_gsm8k.py
│   └── opd_train.py
└── runs/
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
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=512, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.input_ids.shape[1]:],
                          skip_special_tokens=True)
        if extract_answer(text) == extract_answer(ex["answer"]):
            correct += 1
    print(f"Acc: {correct/len(ds):.3f} ({correct}/{len(ds)})")

if __name__ == "__main__":
    main()
```

跑 baseline（**先做这步**，没数字就没 ablation）：

```bash
# Student baseline（一张卡就够）
CUDA_VISIBLE_DEVICES=1 python src/eval_gsm8k.py \
    --model models/student --n 200

# Teacher upper bound
CUDA_VISIBLE_DEVICES=0 python src/eval_gsm8k.py \
    --model models/teacher --n 200
```

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
| `max_new_tokens` | student 单次 rollout 长度 | `1024` |
| `seq_kd` | True=序列级 KD（每整段一个 loss），False=token 级 | `False`（信号更密） |

```python
# src/opd_train.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GKDConfig, GKDTrainer

TEACHER = "models/teacher"
STUDENT = "models/student"

def main():
    tok = AutoTokenizer.from_pretrained(STUDENT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # GPU 0: teacher (frozen, eval mode)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    teacher.eval()

    # GPU 1: student + LoRA
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16,
        device_map={"": 1}, attn_implementation="sdpa")

    # 数据：GKDTrainer 吃 messages 格式
    raw = load_dataset("openai/gsm8k", "main", split="train")
    def fmt(ex):
        return {"messages": [
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]}
    ds = raw.map(fmt, remove_columns=raw.column_names)

    lora = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    cfg = GKDConfig(
        output_dir="runs/opd-qwen3-1.7b",
        max_steps=300,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,        # effective batch = 8
        learning_rate=1e-5,
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=100,
        # GKD 专属
        lmbda=0.5,
        beta=0.5,
        temperature=0.9,
        max_new_tokens=1024,
        seq_kd=False,
        # logging
        report_to="wandb",
        run_name="opd-qwen3-1.7b-gsm8k",
    )

    trainer = GKDTrainer(
        model=student,
        teacher_model=teacher,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
    )

    # gradient checkpointing + LoRA 必须的一行
    student.enable_input_require_grads()

    trainer.train()
    trainer.save_model("runs/opd-qwen3-1.7b/final")

if __name__ == "__main__":
    main()
```

启动：

```bash
wandb login   # 粘 wandb token
python src/opd_train.py
```

避坑（**这一节最容易翻车，仔细看**）：

1. **`device_map={"": N}` 必须显式指定**，不要用 `"auto"`。`auto` 会按可用显存自动切分，会把两个模型都切到两张卡上，结果两个都 OOM。

2. **Gradient checkpointing + LoRA = "element 0 of tensors does not require grad"**：必须调 `student.enable_input_require_grads()`，已写在上面。

3. **OOM 三层应对**（按顺序试）：
   - `max_new_tokens` 1024 → 768 → 512
   - `gradient_accumulation_steps` 8 → 16（保持 effective batch 不变）
   - Teacher 改 4-bit：加 `quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`，省到 ~5GB；代价是 logits 有数值噪声，效果掉 1-2 个点

4. **Tokenizer 一致性**：teacher 和 student 必须共用 tokenizer。Qwen3 系列内部共用，没事；要是混搭家族（Llama teacher + Qwen student）GKDTrainer 直接拒绝跑。

5. **`lmbda` 选择**：纯 on-policy（`lmbda=1.0`）在 student baseline 弱时学不起来——student 自己 rollout 的轨迹和 teacher 期望的差太远，KL 爆炸。`0.5` 是稳健起点，跑通后再做 ablation。

6. **长度膨胀**：训 100+ 步后若发现 student 输出越来越长 + 重复，这是 Stable-OPD 论文（arXiv 2604.08527）点出的问题。临时缓解：在 generation_config 里加 `repetition_penalty=1.05`。

7. **wandb 看什么**：重点盯 `loss`、`kl`、`student_response_length`。后两个曲线异常（KL 不降 / 长度暴涨）就是训崩前兆，立刻停下检查 lmbda。

---

## 6. 训完 eval

```bash
CUDA_VISIBLE_DEVICES=1 python src/eval_gsm8k.py \
    --model models/student \
    --lora runs/opd-qwen3-1.7b/final \
    --n 1319   # 全量 test set
```

---

## 7. Ablation（简历价值的核心）

光有一个 OPD 数字不够，需要对比线。**至少跑这两个**：

**A. 纯 SFT 同 FLOPs baseline**：用 TRL `SFTTrainer`，相同 `max_steps=300` + 相同 batch + 相同 LoRA config，在 GSM8K train 的 ground-truth 答案上做 SFT。这是回答"为什么不直接 SFT"的关键对照。

**B. lmbda 扫描**：`lmbda ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` 各跑一次（每个 200 steps 即可），画 GSM8K acc 曲线。这能直接展示 on-policy 的价值。

可选 C（更深入）：`beta ∈ {0.1, 0.5, 0.9}` 看 forward / 对称 / reverse KL 的差异。

---

## 8. 时间和资源估算

| 阶段 | 时长 |
|------|------|
| 模型 + 数据下载 | 15-20 分钟 |
| Baseline eval（200 条 ×2） | 10 分钟 |
| OPD 训练（300 steps） | 1.5-2.5 小时 |
| 全量 eval（1319 条 ×N 个 ckpt） | 每次 30-40 分钟 |
| Ablation（5 个 lmbda × 200 steps） | 6-8 小时 |

**总计**：一个完整可写简历的项目大约 1.5-2 天纯训练时间。建议晚上挂 ablation。

---

## 9. 简历产出 checklist

- [ ] **wandb 公开报告链接**（项目设为 public，附上）
- [ ] **关键数字表**：

| Setting | GSM8K Acc | Train FLOPs (rel) |
|---------|-----------|-------------------|
| Teacher (Qwen3-8B) | xx.x% | — |
| Student baseline | xx.x% | 0 |
| SFT (300 steps) | xx.x% | 1× |
| **OPD (300 steps)** | **xx.x%** | ~1.5× |

- [ ] **训练曲线截图**：loss + KL + response_length
- [ ] **lmbda ablation 折线图**
- [ ] **README** 含：算法引用 (Lu et al., 2025)、硬件 (2× A5000)、复现命令、所有超参

**简历 bullet 模板**：

> Implemented on-policy distillation (Lu et al., 2025) on 2× RTX A5000:
> distilled Qwen3-8B reasoning into Qwen3-1.7B via LoRA, achieving X% on
> GSM8K (vs. Y% off-policy SFT baseline at equal FLOPs); ablated lmbda
> ∈ {0, 0.25, 0.5, 0.75, 1.0} and reverse-KL vs JSD losses.

---

## 10. 如果想往论文方向延伸

跑通 baseline + ablation 后，可以考虑这几个方向（任选其一就够写一篇 workshop paper）：

- **Stable-OPD 复现 + 改进**：实现 reference-based KL 约束，看能否缓解长度膨胀
- **OPSD 简化版**：让 1.7B 同时当 teacher 和 student（teacher 看 ground-truth，student 不看），不需要 8B
- **跨 domain 持续学习**：先在 code（如 MBPP）上 SFT 让 IF-eval 掉点，再用 OPD 恢复指令遵循能力——这是 Thinking Machines 博客后半段的实验
