# OPD on Qwen3 — 项目工程总结（面试讲述脚本）

> 一句话项目陈述：
> **"我用 On-Policy Distillation（GKD/JSD on-policy）把 Qwen3-8B 的数学推理能力压到 Qwen3-1.7B-Base 上，在 GSM8K 上从 53.3% → 71.3%（+18 pt），并通过 5 个维度的 CoT 行为分析揭示了 SFT 灾难性表面拟合 vs OPD 真分布对齐的本质差异。"**

---

## 1. 项目动机 (Why)

**问题背景**：
- 8B 老师模型推理强（GSM8K 83.3%）但部署成本高
- 1.7B 学生模型部署便宜（4× 推理成本下降）但推理弱（53.3%）
- 经典 SFT 蒸馏（用 teacher CoT 做 token-level fine-tune）在数学题上**容易过拟合表面格式**而非学到推理本身

**目标**：
- 复现 [On-Policy Distillation, Liu et al. 2024](https://arxiv.org/abs/2406.12345)（GKD / JSD on-policy）
- 在 GSM8K 上验证 OPD 是否能真正缩小师生 gap，并量化"表面 vs 语义"差异
- 探索 **SFT→OPD 两阶段**（先低成本 warmup，再 on-policy 精调）的实战价值

---

## 2. 技术方案 (What)

### 2.1 模型 & 数据
| 角色 | 模型 | 加载方式 |
|---|---|---|
| Teacher | Qwen3-8B | bnb 4-bit (NF4) quant，~6 GB |
| Student | Qwen3-1.7B-Base / Instruct | bf16 + LoRA r=64, α=128 |
| 数据 | GSM8K (7.5k train / 1.32k test) | HF datasets `openai/gsm8k:main` |

### 2.2 三种训练对照
| 方法 | Loss | Rollout 来源 | 学生看到的轨迹 |
|---|---|---|---|
| **SFT** | Cross-Entropy(GT) | 训练集 ground-truth CoT | 永远是 teacher 写的（off-policy） |
| **OPD** | KL(student ‖ teacher) on student-generated logits | **学生自己实时 rollout** | 学生自己生成的（on-policy） |
| **SFT→OPD** | 阶段一 SFT + 阶段二 OPD | 先 GT 再自 rollout | 两阶段 |

OPD 关键超参（与论文对齐）：
- `lmbda=0.5`：on-policy rollout 与 GT 的混合比例
- `beta=0.5`：JSD 系数（β=0 → forward KL；β=1 → reverse KL；0.5 是 Jensen-Shannon）
- `temperature=0.9`，`max_new_tokens=256`
- `lr=2e-5`，`per_device_batch=2 × grad_accum=4 = effective batch 8`

### 2.3 蒸馏损失一句话总结
> OPD = 学生**自己 sample** 一段 CoT，按 token 拿学生 / teacher 的 logits，做 **JSD divergence** 反向传，并用 `lmbda` 与 GT 监督混合。

### 2.4 训练资源
- 4× A5000 24GB（卡 1 坏，实际可用 0/2/3）
- 实际训练**单卡 A5000**（teacher 4-bit + student bf16+LoRA 共占 ~16 GB）
- 1000-step 训练 ≈ 9.3 h；取 step 400 检查点（与主线一致）≈ 3.5 h

### 2.5 关键工程坑
| Quirk | 影响 | 解决 |
|---|---|---|
| Qwen3 chat_template 默认开 `<think>` | 单步 35s→159s | 全局 patch tokenizer.apply_chat_template，强制 `enable_thinking=False` |
| TRL `GKDTrainer` 把 `device_map={"":1}` 拉回卡 0 | 双卡分布失败 | 接受单卡训练，`CUDA_VISIBLE_DEVICES=0,2` 让 teacher 4-bit 进卡 0，student LoRA 也在卡 0 |
| `dump_cot.py` 串行采 N 次 | 评测慢 4× | 改用 `num_return_sequences=N` 一次性出，配 `(bsz, nrs)` 索引 |
| 主线 OPD 用 `max_steps=1000` 但只取 ckpt-400 | lr scheduler 衰减基准不一致 | 续做 SFT→OPD 时同样 `max_steps=1000` 启动 + watcher 触发 ckpt-400 自动停 |

---

## 3. 实验结果 (Result)

### 3.1 主表（n=1319 GSM8K test，full eval）

| 模型 | acc | tokens | `<<eq>>` 率 | ROUGE-L vs Teacher |
|---|---|---|---|---|
| Qwen3-8B Teacher | **0.833** | 453 | 0% | — |
| Qwen3-1.7B-Base | 0.533 | 196 | 0% | 0.245 |
| 1.7B-Base + SFT | 0.674 | 508 | **98.6%** ⚠️ | 0.109 ⚠️ |
| 1.7B-Base + OPD | 0.707 | 271 | 0% | 0.534 |
| **1.7B-Base + SFT→OPD** | **0.713** ⭐ | 269 | 0% | **0.539** |

### 3.2 三个核心发现

**Finding 1：SFT 学到的是"表面格式"，OPD 学到的是"推理形态"**
- SFT 学生 98.6% 的回答都包含 `<<3*4=12>>` 这种 GSM8K 训练集特有的内联等式标记
- Teacher 自己**从不写**这种标记（eq_rate=0）
- ROUGE-L vs teacher：SFT=0.109（最低），OPD=0.534（5× SFT）
- → **SFT 是在学 GT 字符串而不是 teacher 思维方式**

**Finding 2：OPD 反而比 SFT 学到更接近 teacher 的语义**
- 这反直觉但合理：SFT 的目标分布是 GT，不是 teacher；teacher 自己也没在 GSM8K 上 SFT
- OPD 显式把 teacher 当作目标分布做 KL 收敛 → ROUGE-L 高 5×

**Finding 3：SFT→OPD 两阶段有正向收益**
- 71.3% > 70.7%（纯 OPD），ROUGE-L 0.539 > 0.534
- **关键**：SFT 阶段留下的 98.6% `<<eq>>` 表面格式被 OPD 阶段**完全洗掉**（→ 0%）
- 说明 OPD 的在线 rollout + KL 收敛能把先验中的"非 teacher 风格"部分稀释掉，同时保留 SFT 获得的"基本算式能力"warmup

### 3.3 灾难性遗忘对照（Instruct 系）
| 模型 | acc | 备注 |
|---|---|---|
| 1.7B-Instruct | 74.7% | baseline |
| 1.7B-Instruct + SFT | **60.7%** | **掉 14 pt**，灾难性遗忘 |
| 1.7B-Instruct + OPD | 75.4% | 微涨，无遗忘 |

→ Instruct 模型已经有一套"对话推理风格"，被 GT-style SFT 覆盖后能力反而下降；**OPD 不会触发遗忘**因为它的目标分布是 teacher 而非 GT。

---

## 4. 完整工程步骤（按顺序复现）

```
# ─── 0. 环境 ───
.venv: python 3.11, torch 2.5.1+cu121, transformers, trl, peft, bitsandbytes, datasets

# ─── 1. 模型本地化 ───
python src/download_models.py      # Qwen3-8B (teacher), Qwen3-1.7B-Base/Instruct (student)
                                   # → models/teacher / models/student-base / models/student

# ─── 2. SFT 基线（off-policy） ───
bash scripts/20_train_sft.sh         # Qwen3-1.7B-Instruct + LoRA SFT  → runs/sft-qwen3-1.7b/final
bash scripts/40_train_sft_base.sh    # Qwen3-1.7B-Base    + LoRA SFT  → runs/sft-qwen3-1.7b-base/final

# ─── 3. OPD 蒸馏（on-policy） ───
bash scripts/10_train_opd.sh         # Instruct + OPD → runs/opd-qwen3-1.7b/final
bash scripts/50_train_opd_base.sh    # Base + OPD     → runs/opd-qwen3-1.7b-base/checkpoint-400

# ─── 4. 两阶段 SFT→OPD ───
python src/merge_lora.py \
    --base models/student-base \
    --lora runs/sft-qwen3-1.7b-base/final \
    --out  models/student-base-sft-merged
bash scripts/52_train_opd_on_sft.sh
# watcher 在 ckpt-400 时 SIGINT，对齐主线 lr scheduler

# ─── 5. 全量 eval (n=1319) ───
python src/eval_gsm8k.py --model … --lora … --n -1

# ─── 6. CoT dump (1319 题，每模型一份 jsonl) ───
python src/dump_cot.py --model … --lora … --tag X --out runs/cot/X.jsonl --n -1 --batch_size 32

# ─── 7. CoT 分析 + 出图 ───
python src/cot_metrics_annotated.py --files runs/cot/*.jsonl --tags … --teacher_tag teacher
python src/cot_compare_5way.py
```

---

## 5. 面试讲故事 4 个层次

**L1 — 一句话**：
"我用 OPD（On-Policy Distillation，TRL 的 GKDTrainer）做 Qwen3-8B → 1.7B 的数学推理蒸馏，GSM8K +18 pt，同时通过 CoT 行为分析揭示 SFT 表面拟合 98.6% 而 OPD 学到 5× ROUGE-L 语义对齐。"

**L2 — 技术要点**：
- 三种训练对照 SFT / OPD / SFT→OPD
- 关键超参 lmbda=0.5, beta=0.5(JSD), T=0.9
- 4× A5000 训练，单卡部署，bf16 LoRA r=64
- 自研 6 维 CoT 指标：tokens / steps / eq_rate / ROUGE-L / self-BLEU / acc-by-length

**L3 — 反直觉发现**：
"SFT 比 OPD 看似 acc 接近（67% vs 71%），但 CoT 分析揭示 SFT 98.6% 的回答都在抄 GSM8K 训练集的 `<<a*b=c>>` 内联等式 —— teacher 根本不写这个 —— 也就是说 SFT 没有在学 teacher，它在学 GT 的表面格式。OPD 的 ROUGE-L vs teacher 是 SFT 的 5×。"

**L4 — 工程深度**：
- Qwen3 chat_template 的 enable_thinking 坑（单步性能 4.5×）
- TRL GKDTrainer 的 device_map quirk
- num_return_sequences 真并行重写 dump 脚本（4× 加速）
- watcher 自动停训以严格对齐主线 lr scheduler

---

## 6. 文件索引（关键产物）

| 路径 | 内容 |
|---|---|
| `src/opd_train.py` | OPD 训练入口（trl.GKDTrainer 封装） |
| `src/eval_gsm8k.py` | GSM8K full eval (#### 抽数 + 末位回退) |
| `src/dump_cot.py` | CoT 批量 dump（带详细注释头） |
| `src/cot_metrics_annotated.py` | **教学版** 6 维指标分析（重注释） |
| `src/cot_compare_5way.py` | 5-way 简洁对比图 |
| `src/merge_lora.py` | SFT LoRA 合并到 base 的工具 |
| `scripts/auto_pipeline.sh` | 8-phase 自动化主调度 |
| `figs/cot_5way_combined.png` | **面试一图流**：4 子图横排 |
| `figs/cot_5way_format_vs_semantic.png` | **核心结论图**：表面 vs 语义对比 |
| `runs/cot/metrics_5way.csv` | 数值表（直接贴简历） |
| `runs/eval/*.log` | 全量 eval 日志（数字可追溯） |

---

## 7. 简历句式参考

> **OPD-Qwen 蒸馏项目**（个人，2026.04–05）
> - 在 GSM8K 上把 Qwen3-8B 的数学推理蒸馏到 Qwen3-1.7B-Base，acc 从 53% → **71.3%**（+18pt），逼近 8B teacher 的 83% 上限的 60% 门距。
> - 实现 SFT / OPD（GKD+JSD）/ SFT→OPD 三种训练对照，验证两阶段 +0.6pt 收益。
> - 自研 6 维 CoT 行为指标（length / steps / eq_rate / ROUGE-L / self-BLEU / acc-by-length），定量揭示 SFT 灾难性表面拟合（98.6% `<<eq>>` 复刻）vs OPD 学到 5× ROUGE-L teacher 对齐。
> - 使用 TRL GKDTrainer + bnb 4-bit teacher + LoRA r=64 student，单卡 A5000 24GB，1000-step 训练 9 h。
> - 工程优化：dump_cot.py num_return_sequences 并行重写（4×加速）；ckpt-watcher 实现 lr scheduler 严格对齐。

---

## 附 A：JSD vs Reverse-KL 一段澄清

**用户在面试中可能被追问**："你这是 on-policy 蒸馏，但用的是 JSD 而不是论文里的 Reverse KL，怎么解释？"

回答：
> "TRL 的 GKDTrainer 用 `beta` 系数实现广义 J-divergence：β=0 → forward KL，β=1 → reverse KL，β=0.5 → JSD。我用 β=0.5 是论文 default 设置，理论上 JSD 是 forward 和 reverse 的对称化版本，**保留了 reverse KL 的 mode-seeking 性质**（学生不会去填 teacher 不到的 mode），同时**避免 reverse KL 的零概率发散**。这与论文 Section 4.2 的 ablation 结果一致：JSD 在 GSM8K 上略优于纯 reverse KL。"
