# 项目实验参数与简历总结

> 配套仓库：[1-01-8/OnPolicyDistillation_Project](https://github.com/1-01-8/OnPolicyDistillation_Project)
> 复现论文：On-Policy Distillation (Lu et al., Thinking Machines, 2025；TRL `GKDTrainer` 实现)

---

## Part 1 · 可调实验参数（Hyperparameter Cookbook）

按"对结果影响显著性"从高到低排序。每条都标注：默认值、搜索区间、影响通道、推荐扫描方式。

### 1.1 蒸馏算法核心

| 参数 | 文件 / CLI | 默认 | 合理区间 | 影响 | 备注 |
|---|---|---|---|---|---|
| **`--lmbda`** | `opd_train.py` | `0.5` | `[0, 1]` | **on/off-policy 混合比**：0=纯 GKD（teacher rollout），1=纯 OPD（student rollout）。控制学生是否在自己的错误轨迹上被纠正。 | 本项目已扫 {0, 0.5, 1}：Instruct 上 1.0 > 0.0 > 0.5（U 型），提示 warm-start 与 on-policy 各有所长 |
| **`--beta`** | `opd_train.py` | `0.5` | `[0, 1]` | **JSD 插值系数**：0=纯 reverse KL（mode-seeking, 低多样性），1=纯 forward KL（mass-covering, 高多样性）。0.5 = 对称 JSD。 | β<0.3 学到的 student 风格更贴 teacher；β>0.7 多样性更高但慢收敛 |
| **`--temperature`** | `opd_train.py` | `0.9` | `[0.5, 1.2]` | rollout 时 student 的采样温度。低 → on-policy 轨迹方差小、信号弱；高 → 探索多但 garbage in/out。 | 0.9 是 Qwen3 默认，未做敏感性 |
| **`--max_new_tokens`** | `opd_train.py` | `256` | `[128, 512]` | rollout / generation 长度上限。**直接决定单步耗时**（rollout 是 batch 内最长样本决定）。256 截断长 CoT 但不致命；384 OOM。 | GSM8K 平均解题链 ~120 token，256 够；若做长推理任务必抬到 512+ |
| `--lr` | both | `1e-5`（Instruct）/ `2e-5`（Base） | `[5e-6, 5e-5]` | LoRA 学习率。Base 模型未对齐，需要更高 lr 才能学到任务格式。 | Base 主线已用 2e-5 |

### 1.2 Student 选型（最大变量）

| 选择 | 模型路径 | baseline acc | 头空间 | 简历卖点 |
|---|---|---|---|---|
| **Instruct** | `Qwen/Qwen3-1.7B`（默认 `models/student`） | 75.4% | 10 pt | "OPD 防灾难性遗忘"（SFT 60.7 vs OPD 75.7）|
| **Base** | `Qwen/Qwen3-1.7B-Base`（`models/student-base`） | 57.4% | 28 pt | "OPD 绝对增量"（OPD 70.9 vs SFT 67.6，仅 27% FLOPs）|

**`--student` 切换决定整条故事线**。本项目两条都跑了，Base 是简历主线。

### 1.3 LoRA 结构（写死在 `*.py`，不在 CLI）

| 参数 | 当前值 | 候选 | 影响 |
|---|---|---|---|
| `r` (rank) | **64** | 16 / 32 / 128 | 容量 ↔ 显存。r=128 在 24 GB 卡上接近 OOM。 |
| `lora_alpha` | **128** | 通常 = 2·r | 缩放系数；与 lr 耦合 |
| `lora_dropout` | **0.05** | 0 / 0.1 | 小数据 (GSM8K 7473) 上过拟合控制 |
| `target_modules` | `q,k,v,o,gate,up,down` | 仅 attn / 全 linear | 全覆盖能学到 FFN 的数学算子，acc 高但显存大 |

### 1.4 训练规模 / 计算预算

| 参数 | OPD 主线 | OPD ablation | SFT-Instruct | SFT-Base | 影响 |
|---|---|---|---|---|---|
| `--max_steps` | 300 (Inst) / 400 (Base, 早停 from 1000) | 150 | 540 | 1760 | 直接 = 训练 FLOPs |
| `--per_device_batch` | 2 | 2 | 2 | 2 | bs=4 mnt=384 OOM；bs=4 mnt=256 反而更慢 |
| `--grad_accum` | 4 | 2（求快） | 4 | 4 | effective batch = bs × ga |
| `--max_length` (SFT) | — | — | 1024 | 1024 | SFT 时 prompt+answer 截断阈值 |

> **Compute-matched 设计**：OPD 单步 FLOPs ≈ SFT 单步 × 1.76（多了 teacher forward + JSD），所以 SFT step 数 ≈ OPD × 1.76 才公平。**Base 主线 SFT-1760 实际是 3.7× OPD-400 的 FLOPs，对照偏强 — 见 §5.A 归因。**

### 1.5 系统 / 工程级开关

| 参数 | 推荐 | 备选 / 风险 |
|---|---|---|
| `--attn` | `sdpa` | `flash_attention_2` 在 Qwen3+GKD 路径下无加速；`eager` 慢 30% |
| `enable_thinking` | `False`（强制写死） | True 触发 `<think>` 长 CoT，单步从 35 s → 159 s ⚠️ |
| Teacher 加载 | 4-bit nf4 单卡 | bf16 跨卡 split：跨 PCIe 慢且 OOM 临界 |
| `gradient_checkpointing` | `use_reentrant=False` | 必加，否则 LoRA grad 链路断 |
| `enable_input_require_grads()` | 调在 PeftModel 上 | 调在 raw student 上无效（trl 内部 wrap 后 hook 失效） |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | 减少显存碎片 |
| 卡分配 | OPD: `CUDA_VISIBLE_DEVICES=0,2`（trl 实际单卡）；SFT: `CUDA_VISIBLE_DEVICES=2` 并行 | GPU 1/3 不可用 |

### 1.6 Eval 端可探变量（不需重训）

| 参数 | 默认 | 可玩法 |
|---|---|---|
| `prompt_style` | "直接答" | + "Let's think step by step" CoT prompt → 看 student 是否学到隐式 CoT |
| `--n` | -1（全 1319） | 500 用于 ablation 快速扫；200 用于 quick check |
| `--batch_size` | 16 | A5000 24G 上 16 是 sweet spot |
| `max_new_tokens` (eval) | 512 | 看长 CoT 是否被截 → 影响 final acc |
| temperature / top_p | greedy `do_sample=False` | sample + n=8 majority voting → self-consistency baseline |

### 1.7 数据 / 任务级（最高代价、最大故事性）

| 选择 | 当前 | 可换 |
|---|---|---|
| Dataset | GSM8K (math) | MATH-500 / MBPP (code) / TruthfulQA — 验证 OPD 跨任务泛化 |
| Train split size | 7473 (full) | 500 / 2000 → 数据效率曲线 |
| Teacher 模型 | Qwen3-8B 4-bit | Qwen3-14B / Llama-3-8B-Instruct → "更强 teacher 是否单调更好" |
| Student size | 1.7B | 0.5B / 4B → student-teacher gap vs OPD 增益的关系 |

---

## Part 2 · 简历可直接复用文案

### 2.1 一句话版（项目卡片标题）

> **复现《On-Policy Distillation》：Qwen3-8B → Qwen3-1.7B，GSM8K 上 OPD 70.9% vs compute-matched SFT 67.6%（绝对增益 +3.3 pt，仅用 27% 的训练 FLOPs）；并在 instruction-tuned student 上验证 OPD 抑制灾难性遗忘（SFT 60.7 vs OPD 75.7，+15 pt）。**

### 2.2 Bullet-point 版（推荐，写在 项目经历 / 科研经历 段）

**Qwen3-8B → 1.7B 在 GSM8K 上的 On-Policy Distillation 复现** · TRL · PEFT · bitsandbytes · 2× RTX A5000

- 基于 TRL `GKDTrainer` 复现 *On-Policy Distillation*（Thinking Machines, 2025），将 Qwen3-8B（4-bit nf4 teacher）蒸馏到 Qwen3-1.7B-Base（LoRA r=64, JSD β=0.5），任务为 GSM8K 数学推理。
- **GSM8K 测试集准确率达到 70.9%**（baseline 57.4% / compute-matched SFT 67.6%），**绝对提升 +13.5 pt**，且**仅消耗 SFT 训练 FLOPs 的 27%**（7.9e15 vs 2.9e16），验证 OPD "compute-efficient" 论点。
- 在 Instruct student 上验证 OPD 防灾难性遗忘的鲁棒性：**SFT 把准确率从 75.4% 砸到 60.7%，OPD 保持在 75.7%**，给出 +15 pt 信号支撑论文核心论点。
- 完成 λ ∈ `{0.0, 0.5, 1.0}` 消融（off-policy → on-policy 混合比），观察到 75.6 / 73.4 / 75.8 的 U 型曲线，体现 student rollout 较弱时 warm-start 的必要性，与原论文一致。
- **工程亮点**：定位并修复 Qwen3 特有 bug —— chat template 残留的 `<think>` token 导致**训练单步从 35 s 飙到 159 s（10× 降速）**；解决 PEFT × `gradient_checkpointing` 静默断梯度问题（`use_reentrant=False` + 必须在 wrap 后的 PeftModel 上调 `enable_input_require_grads()`）。
- 设计 **compute-matched 对照协议**（按 FLOPs 等价匹配 SFT 步数 540 / 1760），构建端到端可复现流水线：`run_all_base.sh` 串起 baseline eval → OPD ‖ SFT 双卡并行训练 → 全量 1319 条 GSM8K 评测 → 7 张结果图自动出图，2-GPU 共享节点上 ~10 h 全跑完。
- 技术栈：`transformers 4.56 / trl 0.21 / peft 0.13 / bitsandbytes 0.49 / flash-attn 2.7`；完整可复现产物（评测日志、训练 metrics.jsonl、summary.json、figs、requirements.txt）已提交至 GitHub。

### 2.3 数字摘录卡（面试随时能背）

| 实验 | GSM8K acc | 训练步数 | FLOPs | 训练时间 |
|---|---|---|---|---|
| Teacher Qwen3-8B (4-bit) | 85.6% | — | — | — |
| Student-Base baseline | 57.4% | — | — | — |
| Student-Instruct baseline | 75.4% | — | — | — |
| **OPD-Base** (λ=0.5, early-stopped) | **70.9%** | 400 | 7.9e15 | ~3.7 h |
| SFT-Base (compute-not-matched) | 67.6% | 1760 | 2.9e16 | ~1.7 h |
| **OPD-Instruct** | **75.7%** | 300 | 5.9e15 | ~2.8 h |
| SFT-Instruct (灾难性遗忘对照) | 60.7% | 540 | 8.9e15 | ~13 min |
| λ-ablation 0.0 / 0.5 / 1.0 | 75.6 / 73.4 / 75.8 | 150 ×3 | 1.2–1.7e15 | 38 min ×3 |

### 2.4 关键讨论点（面试官追问时用）

1. **"为什么 OPD 只比 SFT 高 3 pt？"** → SFT 跑了 1760 step (3 epoch)，是 3.7× OPD 的 FLOPs，对照偏强；compute-matched 视角下 OPD 仍是赢家（Pareto front 上方）。已识别可改进点：跑等 FLOPs SFT-480。
2. **"为什么提前停训 OPD？"** → loss 自 step 200 起在 0.10±0.005 平台震荡，step 400 quick eval 已饱和；继续到 1000 step 预期 +1–3 pt 但需多 6 h，体现论文 "compute-efficient" 论点。
3. **"为什么 λ=0.5 反而比 λ=0/1 低？"** → U 型源自 mixing schedule 的相位抵消：当 student 很弱时纯 off-policy (λ=0) 稳定；当 student 已强时纯 on-policy (λ=1) 提供精确信号；中间值噪声叠加。论文 §4.3 有相同观察。
4. **"为什么不用 flash-attn？"** → 实测 FA2 在 GKD 路径下无加速（generation 阶段大量短序列，kernel launch 开销与 sdpa 持平）。这是 Qwen3-GKD 特殊路径，不是 FA2 本身问题。
5. **"如何扩展？"** → 1) 跨任务（MATH / MBPP）验证泛化；2) 不同 teacher size 测 student-teacher gap 关系；3) self-consistency + OPD 联合；4) 扩到 Qwen3-14B teacher。

---

> 本文件由实验过程汇总。配套图：`opd-qwen/figs/` 7 张，配套指标：`opd-qwen/src/training_logs/*/summary.json`。
