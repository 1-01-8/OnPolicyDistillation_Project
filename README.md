# OPD复现项目

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

### 4.2 Base student（**简历主线 — 已完成**）

| 模型 | GSM8K acc (n=1319) | 训练耗时 | 步数 |
|---|---|---|---|
| Qwen3-1.7B-Base baseline | **57.4%** (n=500) | — | — |
| + SFT | **67.6%** | ~1.7 h | 1760 |
| + **OPD (λ=0.5)** | **70.9%** ✅ | ~3.5 h | **400** (提前停) |
| Qwen3-8B teacher 上界 | 85.6% (n=500) | — | — |

> **关键发现：** OPD 用 **400 step（仅 22.7% of SFT 步数）** 反而比 SFT 高 **+3.3 pt**，且比 baseline +13.5 pt（+23.5%）。Train loss 自 step 200 起进入 0.10±0.005 平台，提前停训符合 OPD 论文 "compute-efficient" 论点。

图：`figs/summary_bar_base.png`、`figs/summary_combined.png`、`figs/compute_efficiency.png`

### 4.3 CoT 行为分析（延伸 — 已完成）

在 acc 之上进一步问 **"OPD 和 SFT 的提升机制是否相同？"**。在 GSM8K test n=1319 上 dump 5 个模型的 CoT，ROUGE-L 参考为 **真 Qwen3-8B teacher**。

| 模型 | acc | tokens | steps | ROUGE-L vs Qwen3-8B Teacher | self-BLEU | `<<eq>>` 率 |
|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base + SFT  | 67.4% | 508 | **47.2** | **0.109** | 0.069 | **98.6%** |
| Qwen3-1.7B-Base + OPD  | **70.7%** | **271** | 18.2 | **0.534** | **0.441** | 0% |
| Qwen3-1.7B (instruct, baseline 学生) | 74.7% | 495 | 17.2 | 0.514 | – | 0% |
| **Qwen3-8B (Teacher)** | 83.3% | 453 | 17.1 | – | – | 0% |

**4 个核心发现**：
1. **风格对齐**：OPD ROUGE-L (0.534) **反超同体量 Qwen3-1.7B Instruct 学生 (0.514)**，比 SFT (0.109) 高 4.9×
2. **推理压缩**：OPD CoT 长度仅 271 (teacher 60%)，**步数 18.2 ≈ teacher 17.1**；SFT 反而 508 tokens / 47.2 步（爆 teacher 2.8×）
3. **SFT 表面学习实锤**：SFT 输出 **98.6%** 含 GSM8K 训练集特有的 `<<a×b=c>>` 格式标记，OPD/teacher/instruct 均为 0%
4. **策略稳定**：OPD 多次采样 self-BLEU 0.44 vs SFT 0.07 — OPD 收敛到稳定推理策略

图：`opd-qwen/figs/cot_3way_bars.png` / `cot_3way_dist.png` / `cot_3way_rouge.png`

→ 详见 `opd-qwen/COT_PLAN.md` 与 `opd-qwen/RESULTS.md` §6
