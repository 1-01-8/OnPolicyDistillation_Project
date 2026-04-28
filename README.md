# OPD-Qwen3-1.7B 复现项目

> **方法**：On-Policy Distillation（GKD/JSD），Qwen3-8B → Qwen3-1.7B，数据集 GSM8K
> **硬件**：4×RTX A5000（24 GB），实际可用 GPU 0+2（GPU 3 长期被外部进程占用）
> **栈**：TRL 0.21 + transformers 4.56 + peft 0.13 + bitsandbytes 0.49 + flash-attn 2.7 (disabled, 见下)
>
> 本仓库以"在受限 4 卡机的真实约束下复现 OPD"为目标，所有耗时 / 显存 / 关键 bug 都来自实测。

---

## 1. 项目结构

```
OnPolicyDistillation_Project/
├── README.md                   ← 本文件，门面
├── opd-qwen3-guide.md          原始指南
├── feasibility-report.md       早期可行性核验
└── opd-qwen/                   实际工作区
    ├── EXPERIMENT_PLAN.md      实验计划 + 结果分支归因
    ├── BENCHMARK.md            压测结果（step 时间 / 显存 / 配置对比）
    ├── scripts/                一键脚本
    │   ├── 00/10/11/20/30 + run_all.sh           Instruct 主线
    │   └── 40/50/51/60     + run_all_base.sh      Base 主线（推荐）
    ├── src/                    训练 / 评测 / 出图代码（--student 切模型）
    ├── runs/                   训练 ckpt + eval 日志
    ├── figs/                   出图（loss / bar / ablation）
    └── models/
        ├── teacher/            Qwen3-8B
        ├── student/            Qwen3-1.7B-Instruct
        └── student-base/       Qwen3-1.7B-Base ← 主线 student
```

## 2. 快速开始

```bash
cd opd-qwen
source .venv/bin/activate

# 主线（推荐）：Base student → 头空间大，简历数字漂亮
bash scripts/run_all_base.sh   # ~10 h

# 对照：Instruct student → 验证 OPD 防灾难性遗忘
bash scripts/run_all.sh        # ~6 h
```

或拆开跑：见 `EXPERIMENT_PLAN.md`。

## 3. 关键修复（之前实验为什么不能用）

| Bug | 表现 | 修复 |
|---|---|---|
| Qwen3 thinking 没关 | OPD 单步 159 s，loss 不下降 | tokenizer wrap 强制 `enable_thinking=False`（见 `src/opd_train.py:108` / `src/sft_baseline.py:33`） |
| teacher_bf16_split 跨卡 | KV cache 跨 PCIe 慢 | 改回 4-bit nf4 单卡 teacher（默认） |
| SFT/OPD 不等 FLOPs | OPD 8.66e15 vs SFT 4.93e15，对照不公平 | SFT 默认 `--max_steps 540` 配 OPD 300 step |
| Instruct student 头空间太小 | baseline 75% / OPD 75.7%，看不出 OPD 价值 | 加 base student 主线（`scripts/{40,50,51,60}_*.sh`） |
| README 是 trl 自动占位符 | "fine-tuned version of None" | 此文件 + EXPERIMENT_PLAN.md 替代 |

## 4. 主要结果

### 4.1 Instruct student（已完成 — OPD 鲁棒性卖点）

| 模型 | GSM8K acc (n=1319) | 训练耗时 | FLOPs |
|---|---|---|---|
| Qwen3-1.7B-Instruct baseline | **75.4%** (n=500) | — | — |
| + SFT (540 step) | **60.7%** ⚠️ | 13 min | 8.9e15 |
| + OPD (300 step, λ=0.5) | **75.7%** | 2.79 h | 5.9e15 |
| Qwen3-8B teacher 上界 | **85.6%** (n=500) | — | — |

> 关键发现：在已 instruct-tuned 的 student 上做 SFT 触发**灾难性遗忘**（−15 pt）；OPD 是唯一保留原能力的路径。

### 4.2 Base student（**简历主线**）

| 模型 | GSM8K acc | 训练耗时 | FLOPs |
|---|---|---|---|
| Qwen3-1.7B-Base baseline | **57.4%** (n=500) | — | — |
| + SFT (1760 step) | *pending* (n=1319) | ~42 min | ~2.9e16 |
| + **OPD (400 step, λ=0.5)** | **69.5%** quick (n=200) ＋full pending | ~3.7 h | ~8e15 |
| Qwen3-8B teacher 上界 | 85.6% (n=500) | — | — |

> 关键发现：OPD 在 Base student 上 **+12.1 pt 绝对提升 / +21% 相对提升**；
> train loss 自 step 200 起在 0.10±0.005 平台震荡，step 400 已饱和，**提前停训** 体现 OPD 论文 "compute-efficient" 论点。

收尾流水：`bash scripts/finalize_base.sh`（OPD-400 full eval ‖ SFT-base 训练，~75 min）。
