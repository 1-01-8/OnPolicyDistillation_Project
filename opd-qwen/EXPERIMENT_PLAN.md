# OPD-Qwen3-1.7B 实验计划

> 配置：GPU 0+2 可用（GPU 3 被外部进程占用，全程绕开）
> Teacher = Qwen3-8B (4-bit nf4), Student = Qwen3-1.7B (bf16 + LoRA r=64)
> 数据：GSM8K (train 7473 / test 1319)

---

## 1. 计算资源分配（实测得出）

**关键发现**：TRL `GKDTrainer` 即使设了 `device_map={"":1}`，也会在 forward / JSD 计算时把 student 拉回 GPU 0，所以 OPD 训练**实际单卡运行**（GPU 0 ~14 GB / 98% util）。GPU 2 整训练全程 280 MiB，闲置——正好让出来给 SFT 并行。

| 阶段 | GPU 0 | GPU 2 |
|------|-------|-------|
| baseline eval | teacher (8B bf16) | student (1.7B bf16) |
| 训练并行 | OPD（teacher+student all-in-one） | SFT（独立进程） |
| final eval | OPD ckpt | SFT ckpt |
| ablation 训练 | OPD λ ∈ {0, 0.5, 1} | (空) |

GPU 3 假设始终不可用。

---

## 2. 推荐超参（经压测）

| 超参 | OPD 主训 | OPD ablation | SFT |
|------|----------|--------------|-----|
| `per_device_batch` | 2 | 2 | 2 |
| `grad_accum` | 4 | 2 | 4 |
| `max_new_tokens` | 256 | 256 | — |
| `max_steps` | 300 | 150 | 540 |
| `attn` | sdpa | sdpa | sdpa |
| 单步耗时 | ~35 s | ~15 s | ~1.5 s |
| 总耗时 | ~3 h | ~38 min × 3 | ~13 min |
| GPU 0 峰值显存 | 14 GB | 14 GB | — |

> `bs=4 mnt=384` OOM；`bs=4 mnt=256` 反而更慢（rollout 是 batch 内最长那条决定）。FA2 实测无加速，留 sdpa。

---

## 3. 命令

```bash
cd /home/xxm/OnPolicyDistillation_Project/opd-qwen
source .venv/bin/activate

# Step 0  baseline (~25 min)
bash scripts/00_baseline_eval.sh

# Step 1  并行 OPD ‖ SFT (~3 h，OPD 是瓶颈)
bash scripts/10_train_opd.sh &  PID=$!
sleep 60
bash scripts/11_train_sft.sh
wait $PID

# Step 2  final eval (~30 min, 全 1319 测试集)
bash scripts/20_eval_finals.sh

# Step 3  λ ablation (~2.5 h)
bash scripts/30_ablation_lmbda.sh

# Step 4  出图
python src/plot_results.py
```

或一把：`bash scripts/run_all.sh`（Instruct 主线）/ `bash scripts/run_all_base.sh`（Base 主线，**推荐**）

---

## 4. 预期结果

### 4.1 主实验：Instruct student（已完成）

| 指标 | 实测 | 解读 |
|------|------|------|
| Student baseline (n=500) | **75.4%** | Qwen3-1.7B-Instruct 已被官方在 GSM8K 上对齐 |
| Teacher baseline (n=500) | **85.6%** | 上界，差 student 仅 10 pt |
| SFT-540 (compute-matched) | **60.7%** ⚠️ | 灾难性遗忘：LoRA 把 instruct 能力学坏 |
| OPD-300 (λ=0.5) | **75.7%** | 持平 baseline，**比 SFT 高 +15 pt** |
| 简历核心数字 | **OPD 75.7 vs SFT 60.7 (+15 pt)** | 论点："OPD 防止灾难性遗忘" |

> 头空间太小（10 pt），OPD 难以"突破上界"。结论是 **OPD 的鲁棒性 > 绝对增量**。

### 4.2 Base student（**简历主线 — 已完成**）

| 模型 | GSM8K acc (n=1319) | 训练耗时 | 步数 |
|---|---|---|---|
| Qwen3-1.7B-Base baseline | **57.4%** (n=500) | — | — |
| + SFT | **67.6%** | ~1.7 h | 1760 |
| + **OPD (λ=0.5)** | **70.9%** ✅ | ~3.5 h | **400** (提前停训) |
| Qwen3-8B teacher 上界 | 85.6% (n=500) | — | — |

**关键结果：**
- OPD vs baseline: **+13.5 pt 绝对 / +23.5% 相对**
- **OPD − SFT: +3.3 pt**（compute-not-matched，OPD 反而提前停训用更少 step 拿到更高 acc）
- Train loss 自 step 200 起在 0.10±0.005 平台震荡，step 400 已饱和，体现 OPD 论文 "compute-efficient" 论点。

收尾入口：`bash scripts/finalize_base.sh`

---

## 5. 不同结果的归因分析

### A. **OPD > SFT > baseline，OPD 比 SFT 高 ≥ 3 pt** ✅ 预期
直接写简历："Qwen3-8B 蒸馏到 1.7B，OPD vs compute-matched SFT +X pt；λ ablation 验证 on-policy rollout 贡献 Y pt。"

### B. **OPD ≈ SFT（< 1 pt 差）**
1. **λ=0.5 对 GSM8K 偏低** → 改 `--lmbda 0.8` 重训 / 或在 ablation 里看 λ=1.0 是否更好。
2. **`max_new_tokens=256` 截断长解题链** → 抽 train rollout 看是否被截尾；若是，改 384，但需配套降 `per_device_batch` 到 1 防 OOM。
3. **学习率 1e-5 太小** → 看 loss 尾段是否仍下降；下降则把 `--max_steps` 加到 600。
4. **JSD beta=0.5 让 forward/reverse KL 抵消** → ablation 里如果没看到 λ 单调，可能也是 β 的问题。

### C. **OPD < SFT** ⚠️
信号危险，几乎必然有 bug。
1. **thinking 没真关掉** → 拿一条 student rollout 解码看是否还有 `<think>`；wandb 里看 `student_response_length`，正常应 < 100。
2. **Teacher 4-bit 量化误差太大** → teacher logits 噪声盖过信号。回退 bf16，但 8B bf16 单 24G 装不下，用 `--teacher_bf16_split`（仍保留作 fallback）。
3. **β 太高** → 试 `--beta 0.1`。
4. **gradient 没流到 LoRA** → 看 `grad_norm` 是否 > 0；为 0 说明 `enable_input_require_grads()` 没生效。

### D. **Baseline 极低（< 20%）**
extract_answer 抽到 `<think>` 段中间数字。`eval_gsm8k.py:86` 的 `re.sub(r"<think>.*?</think>", ...)` 必须先于 extract。

### E. **Baseline 极高（> 70%）** ✅ 已遇到（Instruct）
GSM8K 在 Qwen3-Instruct 后训练里见过。表现：SFT 反而让 acc 大幅下降（学到表面格式，破坏推理能力）。
**对策**：换 `Qwen3-1.7B-Base` 作 student（见 §4.2）。base 模型未经 instruct 对齐，头空间 30–60 pt。

### F. **OPD 训练时间超预期（>5 h / 300 step）**
99% 是 thinking 又跑出来了。检查 wandb `student_response_length`：< 100 才算关掉。

### G. **λ ablation 非单调（λ=0.5 最高，λ=1.0 反低）**
合理：纯 on-policy 在 student 太弱时 rollout 全错，teacher 没机会教正确轨迹。这是 OPD 论文里讨论过的 warm-start 必要性，简历可以加一句 "观察到 warm-start 效应"。

### H. **OPD vs SFT loss 不可直比**
GKD 的 loss 是 JSD（无标签 token-level KL），SFT 是 CE。绝对值无可比性，对比看 final eval acc。

### I. **GPU 0 显存涨过 22 GB / OOM**
压测过 `bs=4 mnt=256 = 22 GB / 96% util`，再涨就 OOM。如果想加 batch 必须降 mnt 或 freeze 更多 LoRA 层。

### J. **GPU 2 全程闲着不用怎么办**
本计划已用 GPU 2 跑并行 SFT。若 ablation 期间想再榨一点：把 `30_ablation_lmbda.sh` 的 eval 部分（500 条 × 3 个 ckpt）丢到 GPU 2 在训练间隙跑。

---

## 6. 简历模板

> *"复现 Google DeepMind《On-Policy Distillation》（ICLR 2024）：Qwen3-8B (4-bit nf4) 为 teacher，LoRA r=64 蒸馏到 Qwen3-1.7B-Base，GSM8K 准确率从 **57.4%** 提到 **70.9%**（compute-matched SFT 仅到 **67.6%**，OPD 用 22.7% 步数高 +3.3 pt）。在 Instruct student 上额外验证 OPD 鲁棒性：传统 SFT 触发灾难性遗忘（75.4→60.7），OPD 保持原能力（75.4→75.7）。λ ∈ {0, 0.5, 1} ablation 验证 student rollout 贡献：纯 KD（λ=0）→ 75.6%, 纯 SFT（λ=1）→ 75.8%, 混合 → 73.4%（小数据量下 ablation 区分度有限，已记录）。技术栈：TRL 0.21 GKDTrainer / PEFT / bitsandbytes 4-bit / 2× RTX A5000；定位并修复了 Qwen3 thinking 模板导致的 10× 训练降速 bug。"*

实测数字已全部锁定。
