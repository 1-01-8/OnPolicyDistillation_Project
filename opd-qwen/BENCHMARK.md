# 性能压测记录

> 日期：2026-04-27
> 硬件：4× RTX A5000（24 GB），实测仅 GPU 0+2 可用（GPU 3 始终被外部进程占用）

## A. 配置 → 单步耗时矩阵

| 配置 | thinking | attn | bs | ga | mnt | 单步 | GPU 0 mem | GPU 2 用否 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 旧 | on | sdpa | 2 | 4 | 512 | **159 s** | 跨 0+2 | 用 | 之前的失败实验 |
| 新 | off | sdpa | 2 | 2 | 256 | **15 s** | 14 GB | 没用 | ⭐ ablation 用 |
| 新 | off | sdpa | 2 | 4 | 256 | **35 s** | 14 GB | 没用 | ⭐ 主训用（eff_batch=8） |
| 新 | off | sdpa | 4 | 2 | 256 | 31 s | 22 GB | 没用 | rollout 长尾拖慢 |
| 新 | off | sdpa | 4 | 2 | 384 | OOM | — | — | teacher+JSD stack 撑爆 |
| 新 | off | FA2 | 2 | 4 | 256 | 32 s | 14 GB | 没用 | FA2 没明显加速 |

## B. SFT 吞吐
- bs=2, ga=4, max_length=1024：**1.45 s/step**
- 540 步 ≈ 13 min

## C. 关键观察

1. **TRL `GKDTrainer` 单卡化**：尽管脚本设 `student device_map={"":1}`，trainer 在 JSD 计算时把所有张量搬到 teacher 所在 GPU 0，导致 GPU 2 全程 280 MiB / 0 % util。这是**不可绕过**的 trl 内部行为。
2. **利用闲置 GPU 2**：把 SFT 用独立进程 (`CUDA_VISIBLE_DEVICES=2`) 与 OPD 并行跑，互不干扰。
3. **rollout 长尾决定 batch 内步时**：`bs=4` 不能加速，因为最长样本拖慢全 batch；只增内存。
4. **FA2 在 Qwen3 + GKD 路径下没收益**：可能是 generation 阶段大量短序列、kernel launch 开销 vs sdpa 持平。
5. **OOM 阈值**：bs=2/mnt=256 → 14 GB；bs=4/mnt=256 → 22 GB；bs=4/mnt=384 → OOM。安全阈值在 ~20 GB。

## D. 推荐配置（最终采用）

```
OPD 主训:    bs=2, ga=4, mnt=256, sdpa, 300 step  →  ~3 h
OPD ablation: bs=2, ga=2, mnt=256, sdpa, 150 step  →  ~38 min × 3
SFT:         bs=2, ga=4, max_length=1024, 540 step →  ~13 min
```

## E. "压满" 策略
- GPU 0：OPD 主训独占，~14 GB / 98 % util — 没法再加 batch（会 OOM 或被长尾拖慢）
- GPU 2：独立 SFT 进程，~8 GB / 60-100 % util
- GPU 3：被外部用户占用，本项目不用
- 训练并行 + eval 并行 = 已榨干**项目可用算力**（2 卡）
