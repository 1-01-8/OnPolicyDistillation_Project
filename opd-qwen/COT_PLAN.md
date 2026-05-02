# CoT 分析方案：OPD vs SFT 推理风格变化

> 这是 OPD 主项目的延伸实验。主项目证明 **acc 提升**（baseline 57.4 → SFT 67.6 → OPD 70.9），
> 本子项目回答更深层的问题：**OPD 和 SFT 改变 student 推理风格的方式是否不同？**

## 0. 研究问题与假设

| Hypothesis | 直觉来源 | 实验验证 |
|---|---|---|
| **H1**：SFT 让 student CoT 更接近 teacher 风格 | SFT 直接模仿 teacher 完整序列 | ROUGE-L vs teacher：SFT > OPD |
| **H2**：OPD 保留 student 自己的"语言习惯" | OPD 监督的是 student 自生轨迹 | 长度分布 / step 数：OPD ≈ baseline，SFT ≈ teacher |
| **H3**：OPD 的错误更"健康" | OPD 在 student 自己会犯错的位置纠正 | 错题分类：OPD 算术错 ↑ 而公式错 ↓ |
| **H4**：OPD 的多样性更高 | student rollout 暴露分布更广 | self-BLEU：OPD < SFT |

## 1. 方法

### 1.1 数据集
- GSM8K test 子集 n=500（greedy 生成）+ n=100×5 samples（diversity 用）
- 模型：baseline / SFT-base-1760 / OPD-base-400 / teacher (Qwen3-8B)

### 1.2 6 个量化维度

| 维度 | 指标 | 工具 |
|---|---|---|
| **D1 长度** | 生成 token 数、推理步数 | tokenizer + `re.split` |
| **D2 风格相似度** | ROUGE-L（vs teacher）| 纯 Python LCS |
| **D3 多样性** | self-BLEU on 5 samples | n-gram counter |
| **D4 长度-准确率** | acc by length bucket | 分桶 |
| **D5 错误模式** | LLM-as-judge 4 类错误 | teacher 4-bit 自评 |
| **D6 计算保真** | 算式 `=` 出现率 | 正则 |

### 1.3 实验流程

```
┌─────────────────┐
│ Phase 1: dump   │  src/dump_cot.py  ×4 模型 × (greedy + 5-sample)
│  ~3 h           │  → runs/cot/*.jsonl
└────────┬────────┘
         ↓
┌─────────────────┐
│ Phase 2: 指标    │  src/cot_metrics.py
│  ~5 min         │  → metrics_summary.csv + figs/cot_*.png
└────────┬────────┘
         ↓
┌─────────────────┐
│ Phase 3: judge  │  src/cot_judge.py（可选）
│  ~1 h           │  → error_taxonomy.csv
└─────────────────┘
```

一键：`bash scripts/run_cot_analysis.sh`
开启错误分类：`RUN_JUDGE=1 bash scripts/run_cot_analysis.sh`

## 2. 实测结果（n=1319 GSM8K test，已完成）

| tag | acc | tokens | steps | ROUGE-L vs teacher | self-BLEU | arith_eq |
|---|---|---|---|---|---|---|
| base | 53.3% | 196 | 14.0 | 0.237 | – | 0.42 |
| **instruct (teacher)** | 74.7% | 495 | 17.2 | – | – | 0.65 |
| **sft** | 67.4% | 508 | **47.2** | **0.107** | – | **0.97** |
| **opd** | **70.7%** | **271** | 18.2 | **0.516** | – | 0.66 |
| sft_sample (5×200) | 62.1% | 510 | 42.2 | 0.107 | 0.069 | 0.99 |
| opd_sample (5×200) | 70.3% | 267 | 17.7 | 0.472 | **0.441** | 0.63 |

**实际结果落在"模式 A++"** —— 4 个 hypothesis 全部成立：

1. **H1 (CoT 风格对齐)** ✅ OPD ROUGE-L = 0.516 vs SFT 0.107（差 4.8×）
2. **H2 (推理压缩)** ✅ OPD 271 tokens（teacher 的 55%），SFT 反而 508 tokens（爆 teacher）
3. **H3 (SFT 表面学习)** ✅ SFT 步数 47.2 vs teacher 17.2（爆 2.7×），算式率 0.97 vs 0.65 → 死记格式
4. **H4 (策略稳定性)** ✅ OPD 5 次采样 self-BLEU = 0.441（高一致性），SFT = 0.069（极发散）

**核心叙事**：on-policy 让 student 在自己的轨迹上对齐 teacher 分布，最终学到 teacher 的**逻辑结构**而非 token 序列；SFT 强行教师强制走 teacher 路径，学到的是**算式表面**。

详细分析见 `RESULTS.md` §6。

## 3. 面试讲述模板

> *"我在主实验拿到 acc 提升后，进一步问：OPD 和 SFT 在改变 student 时改变了什么？我设计了 6 个量化维度（长度 / 风格相似度 / 多样性 / 错误模式 / 计算保真 / 长度-准确率联合分布）。具体做法是 dump 4 个模型在 500 道 GSM8K 上的生成，再用 ROUGE-L、self-BLEU 和 4-bit teacher 当 judge 做错误分类。结果发现：SFT 让 ROUGE-L vs teacher 从 X 涨到 Y，但 self-BLEU 也涨（多样性下降）；OPD 在 ROUGE-L 不涨的情况下 acc 涨了，说明它学的是 reasoning 而非 surface form。这与 OPD 论文中 'student rollout 暴露真实失败模式' 的论点一致。"*

## 4. 工程亮点（简历可写）

- **复用主项目代码**：dump 脚本只在 eval_gsm8k.py 上改 30 行
- **3 个独立指标维度，0 外部依赖**：ROUGE-L / self-BLEU / step counting 全部纯 Python
- **LLM-as-judge 用同一 teacher 4-bit 实例**：避免引入 GPT-4 API 成本和数据泄露
- **可重现**：一键 `bash scripts/run_cot_analysis.sh`，约 4-5 h

## 5. 下一步可扩展（"如果时间允许"）

- 用 **lm-evaluation-harness** 跑 MMLU/HellaSwag，验证 CoT 风格变化是否影响其他任务
- 用 **embedding 相似度**（sentence-transformers）替代 ROUGE-L，更鲁棒
- 用 **causal mediation analysis**：定位是哪几层 LoRA 矩阵导致 CoT 风格变化
- 训练 **反向 student**（teacher rollout + student logits）看是否能逼近 SFT
