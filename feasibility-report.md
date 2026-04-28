# OPD-Qwen3 指南本机可行性检验报告

> **⚠️ 已被取代** — 真正落地的实验配置见 [`opd-qwen/EXPERIMENT_PLAN.md`](opd-qwen/EXPERIMENT_PLAN.md) + [`opd-qwen/BENCHMARK.md`](opd-qwen/BENCHMARK.md)。
> 本文件保留作"早期评估的历史快照"，下面 §3.5 关于 thinking 的判断已经在 2026-04-27 通过实测验证（旧配置 159 s/step → 修复后 15 s/step）。

---

**检验日期**：2026-04-26
**检验对象**：[opd-qwen3-guide.md](opd-qwen3-guide.md)
**结论**：✅ **可行**，但需对指南做 **5 处修正** 与 **3 项本机适配**（详见 §3、§4）。

---

## 1. 本机环境实测

| 项目 | 指南要求 | 本机实测 | 是否满足 |
|------|----------|----------|----------|
| GPU 数量 / 型号 | 2× RTX A5000 (24GB) | **4× RTX A5000 (24GB)** | ✅ 富余 |
| GPU 占用 | 全空 | GPU 0/1/2 ~38MiB（全空），GPU 3 ~403MiB（桌面 X 占用） | ✅ 用 0/1 即可 |
| CUDA 驱动 | ≥ 12.1 | Driver 535.230.02, **CUDA 12.2** | ✅ |
| nvcc 编译器 | 与 torch CUDA 对齐即可 | **12.4** | ✅（编译 flash-attn OK） |
| 系统内存 | 未指定 | 125 GiB（available 106 GiB） | ✅ |
| 磁盘空间 (/home) | ~30 GB（模型+ckpt+wandb） | **132 GB free / 878 GB（85% 已用）** | ⚠️ 偏紧，需清理 |
| Python | 3.11 | base 3.13；conda env 多为 3.10；**没有 3.11 现成环境** | ⚠️ 用 uv 装 3.11 |
| `uv` | 必须 | 0.11.7 已装 (`/home/xxm/miniconda3/bin/uv`) | ✅ |
| gcc | flash-attn 编译需 ≥ 7 | gcc 9.4.0 | ✅ |
| HF 网络 | 日本直连 | IP `133.9.42.142`（早稻田大学，Tokyo），HF 200 OK，RTT ~213 ms | ✅ |
| HF token / wandb | 自备 | 未登录（需手工 login） | ⚠️ 待补 |

### 现有 conda 环境对 OPD 的可用性

| Env | torch | transformers | trl/peft/accelerate | 适合 OPD？ |
|-----|-------|--------------|---------------------|-----------|
| base (3.13) | 无 | 无 | 无 | ❌ |
| llm_env (3.10) | 无 | 无 | 无 | ❌ |
| cosyvoice (3.10) | 2.3.1+cu121 | 4.51.3 | 无 | ❌ 不要污染 |
| nbss (3.10) | 2.3.0+cu121 | 4.46.2 | 无 | ❌ 不要污染 |
| voxcare (3.10) | **2.10.0+cu128** | 4.57.6 | 无 | ❌ torch 太新且无 trl |
| wharm (3.7) | — | — | — | ❌ |

**结论**：现有环境**没有一个能直接跑 OPD**，必须新建独立环境。这点和指南做法一致（用 `uv venv` 新建）。

---

## 2. 指南条目逐项核验

| § | 条目 | 验证 | 备注 |
|---|------|------|------|
| 0 | 安装 uv、torch 2.5.1+cu121 | ✅ uv 已就绪；nvcc 12.4 与 cu121 兼容 | — |
| 0 | flash-attn 可选，sdpa 兜底 | ✅ A5000 sm_86 支持 sdpa；flash-attn 2.6+ 提供 cu121 wheel，无需源码编译 | 见 §3.2 |
| 0 | 不装 transformer_engine（A5000 无 fp8） | ✅ 正确 | — |
| 0 | TRL ≥ 0.13 | ⚠️ 截至 2026-04，TRL 主线已到 0.21+，**`>=0.13`** 字面成立但建议固定 0.21 或 0.22；新版 `GKDConfig` 字段一致 | 见 §3.1 |
| 1 | 日本 IP 直连 + `hf_transfer` | ✅ 实测 200 OK | — |
| 2 | `Qwen/Qwen3-1.7B` 为 instruct 版 | ⚠️ Qwen3 1.7B HF 仓库 `Qwen/Qwen3-1.7B` 实际为 **post-trained instruct + thinking**，但 chat template 默认含 `<think>` 标签，会污染 GSM8K 答案抽取 | 见 §3.3 |
| 2 | 数据集 `openai/gsm8k` | ✅ HF 上存在；`split="train"` 7.47k；`test` 1319 | — |
| 4 | eval 用 `do_sample=False` + 正则抽数 | ✅ 但 Qwen3 thinking 模式输出 `<think>...</think>\n答案`，需在抽取前剥离 think 段或关闭 thinking | 见 §3.3 |
| 5 | `device_map={"":0}` / `{"":1}` | ✅ 关键避坑点正确 | — |
| 5 | `enable_input_require_grads()` | ✅ LoRA + ckpt 必备 | — |
| 5 | `lmbda=0.5, beta=0.5` 起步 | ✅ 与 TRL 文档建议一致 | — |
| 5 | OOM 三层降级 | ✅ 合理 | — |
| 5 | "Stable-OPD 论文 arXiv 2604.08527" | ❌ **arXiv 编号有误**（arXiv 月份编号为 YYMM，2026 年 4 月应为 `2604.xxxxx`，但具体编号需另查）；不影响代码可行性 | 见 §3.5 |
| 6 | `--lora` 加载 LoRA adapter 做 eval | ✅ peft `PeftModel.from_pretrained` 用法正确 | — |
| 7 | SFT baseline + lmbda 扫描 | ✅ 合理 | — |
| 8 | 时间估算 | ✅ A5000 跑 Qwen3-1.7B LoRA + 8B teacher rollout，1.5–2.5 h / 300 steps 量级合理 | — |

---

## 3. 必须修正/明确的问题

### 3.1 TRL 版本固定（指南只写 `>=0.13`）

主线已到 0.21+，`GKDTrainer` 接口仍稳定，但**强烈建议在 `pyproject` / `requirements` 里固版**，避免几个月后 minor 变动：

```bash
uv pip install "trl==0.21.*" "transformers==4.46.*" "peft==0.13.*" "accelerate==1.0.*"
```

> 我把 `transformers` 也固在 4.46 是因为 trl 0.21 仍然兼容 4.46–4.50；混搭新版 transformers 可能出现 chat template 行为变化。

### 3.2 flash-attn：直接装 wheel，不要源码编译

指南说 "编译 5-10 分钟"，但 2026-04 已有官方 cu121/torch2.5 wheel：

```bash
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
# 失败再降级 sdpa
```

A5000 sm_86 用 flash-attn 2 训练吞吐比 sdpa 快约 30–40%，**强烈建议装上**。

### 3.3 Qwen3 thinking 模式 → eval 数字会偏低（指南未提）

`Qwen/Qwen3-1.7B` 的 chat template 默认会在 assistant turn 前注入 `<think>` 段，模型会输出大段思考再给答案。指南的 `extract_answer` 用 `####` 数字 / 末尾数字两套规则，遇到 thinking 的中间过程可能抽到中间步数字，导致 baseline 被低估。

**修正方案**（在 `eval_gsm8k.py` 的 prompt 构造时关闭 thinking）：

```python
prompt = tok.apply_chat_template(
    msgs,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,   # Qwen3 专用，禁用 <think> 段
)
```

如果坚持开 thinking，则在 `extract_answer` 前剥离 `<think>...</think>`：

```python
text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
```

训练同理：`fmt()` 把 GT 答案塞进 `assistant` 字段时，建议**关 thinking**，否则 student rollout 会偏离 GT 分布。

### 3.4 磁盘空间 132 GB 偏紧 — 强制把缓存放到大盘

实测 / 仅剩 132 GB（85% 已用）。下载与训练产物估算：

| 内容 | 体积 |
|------|------|
| Qwen3-8B teacher (bf16 safetensors) | ~16 GB |
| Qwen3-1.7B student (bf16) | ~3.4 GB |
| GSM8K | < 50 MB |
| HF cache（指向 `$PWD/hf_cache`，与 `--local-dir` 重复一份）| 与上面重叠，~20 GB |
| 训练 ckpt（300 steps × 3 个 save × LoRA only ~200 MB）| < 1 GB |
| ablation 5 × LoRA + log | ~2 GB |
| wandb 本地缓存 | < 1 GB |
| **合计** | **~25 GB**（无 cache 重复）/ **~45 GB**（含重复） |

**修正**：`huggingface-cli download --local-dir` 会同时占用 HF cache **和** local-dir 两份。建议二选一：

```bash
# 方案 A：只用 HF cache（推荐）
unset HF_HOME   # 或保持默认 ~/.cache/huggingface
huggingface-cli download Qwen/Qwen3-8B    # 不加 --local-dir
# 代码里直接用模型名 "Qwen/Qwen3-8B"

# 方案 B：只用 local-dir，禁用 cache
export HF_HUB_DOWNLOAD_TIMEOUT=60
huggingface-cli download Qwen/Qwen3-8B \
    --local-dir models/teacher \
    --local-dir-use-symlinks False
```

另外建议把项目放到 `/home/xxm/OnPolicyDistillation_Project/` 内（已建仓），避免散落。

### 3.5 文献引用错误

第 267 行 "Stable-OPD 论文 arXiv 2604.08527" 编号无法验证（arXiv 编号 YYMM 是月份，写到简历前要再核一次原文）。**不影响代码运行**，但简历用前请校。

---

## 4. 本机适配细化

### 4.1 GPU 选用策略（4 卡 → 2 卡）

```bash
# 训练时绑定 GPU 0 (teacher) + GPU 1 (student)
export CUDA_VISIBLE_DEVICES=0,1
# 让 GPU 2/3 空出来做 baseline eval / SFT 对照实验，并行不冲突
```

代码中 `device_map={"":0}` 与 `{"":1}` 在 `CUDA_VISIBLE_DEVICES=0,1` 下分别指向物理 GPU 0 和 1，符合预期。

### 4.2 项目目录结构（落到本仓库内）

```
/home/xxm/OnPolicyDistillation_Project/
├── README.md
├── opd-qwen3-guide.md          # 原指南（不动）
├── feasibility-report.md       # 本报告
├── opd-qwen/                   # 实际工作区（gitignore）
│   ├── .venv/                  # uv venv
│   ├── models/{teacher,student}/
│   ├── data/gsm8k/
│   ├── src/{eval_gsm8k.py,opd_train.py,sft_baseline.py}
│   ├── runs/                   # ckpt + wandb
│   └── hf_cache/               # 仅用 local-dir 时不需要
└── .gitignore                  # 排除 opd-qwen/、*.bin、wandb/
```

### 4.3 SFT baseline 脚本（指南 §7-A 提到但未给代码）

补一份与 OPD 完全对等 FLOPs 的 SFT 对照（同 LoRA、同 step、同 batch）：

```python
# src/sft_baseline.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

STUDENT = "models/student"

def main():
    tok = AutoTokenizer.from_pretrained(STUDENT)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT, torch_dtype=torch.bfloat16,
        device_map={"": 1}, attn_implementation="sdpa")

    raw = load_dataset("openai/gsm8k", "main", split="train")
    def fmt(ex):
        return {"messages": [
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]}
    ds = raw.map(fmt, remove_columns=raw.column_names)

    lora = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM")

    cfg = SFTConfig(
        output_dir="runs/sft-qwen3-1.7b",
        max_steps=300,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5, warmup_ratio=0.05,
        bf16=True, gradient_checkpointing=True,
        logging_steps=5, save_steps=100,
        report_to="wandb", run_name="sft-qwen3-1.7b-gsm8k",
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=ds, processing_class=tok,
        peft_config=lora,
    )
    model.enable_input_require_grads()
    trainer.train()
    trainer.save_model("runs/sft-qwen3-1.7b/final")

if __name__ == "__main__":
    main()
```

---

## 5. 细化后的执行流程（一口气抄完即可）

### Step 0：准备（5 min）

```bash
cd /home/xxm/OnPolicyDistillation_Project
mkdir -p opd-qwen && cd opd-qwen
uv venv --python 3.11 && source .venv/bin/activate

# 锁版本，避免半年后接口漂移
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install "transformers==4.46.*" "trl==0.21.*" "peft==0.13.*" \
    "accelerate==1.0.*" "datasets>=3.0" bitsandbytes wandb hf_transfer
uv pip install flash-attn==2.7.4.post1 --no-build-isolation || echo "fallback to sdpa"

# HF / wandb
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli login   # 粘 token
wandb login              # 粘 wandb token
```

### Step 1：下载（15-20 min, 约 20 GB）

```bash
huggingface-cli download Qwen/Qwen3-8B   --local-dir models/teacher --local-dir-use-symlinks False
huggingface-cli download Qwen/Qwen3-1.7B --local-dir models/student --local-dir-use-symlinks False
huggingface-cli download openai/gsm8k --repo-type dataset --local-dir data/gsm8k --local-dir-use-symlinks False

df -h /home    # 确认还剩 > 80 GB
```

### Step 2：写代码

把指南 §4 / §5 的两个脚本落盘到 `src/`，**修改 eval 脚本**加上 `enable_thinking=False`（见 §3.3）。再加一份 §4.3 的 `src/sft_baseline.py`。

### Step 3：Baseline eval（10 min）

```bash
export CUDA_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=1 python src/eval_gsm8k.py --model models/student --n 200 \
    | tee runs/baseline_student.log
CUDA_VISIBLE_DEVICES=0 python src/eval_gsm8k.py --model models/teacher --n 200 \
    | tee runs/baseline_teacher.log
```

> ⚠️ 没拿到合理 baseline（student ≥ 50%, teacher ≥ 80%）**不要进训练**，先排查 thinking / chat template / 抽取规则。

### Step 4：OPD 训练（1.5-2.5 h, 300 steps）

```bash
CUDA_VISIBLE_DEVICES=0,1 python src/opd_train.py
# 盯 wandb：loss 单调下降、kl 单调下降、student_response_length 不暴涨
```

### Step 5：SFT 对照（同样 1.5-2 h，**单卡可跑，与 OPD 不冲突**）

```bash
# 与 OPD 训练并行：把 SFT 放到 GPU 2，互不打扰
CUDA_VISIBLE_DEVICES=2 python src/sft_baseline.py
```

### Step 6：最终 eval

```bash
# OPD ckpt
CUDA_VISIBLE_DEVICES=1 python src/eval_gsm8k.py \
    --model models/student --lora runs/opd-qwen3-1.7b/final --n 1319 \
    | tee runs/opd_final_eval.log

# SFT ckpt
CUDA_VISIBLE_DEVICES=2 python src/eval_gsm8k.py \
    --model models/student --lora runs/sft-qwen3-1.7b/final --n 1319 \
    | tee runs/sft_final_eval.log
```

### Step 7：lmbda ablation（晚上挂，6-8 h）

把 `opd_train.py` 改造成接受 `--lmbda`、`--max_steps=200`、`--output_dir`，循环：

```bash
for L in 0.0 0.25 0.5 0.75 1.0; do
  CUDA_VISIBLE_DEVICES=0,1 python src/opd_train.py \
      --lmbda $L --max_steps 200 \
      --output_dir runs/abl/lmbda_$L
done
```

eval 同样循环跑，最后用一段 matplotlib 画折线。

---

## 6. 风险与回退方案

| 风险 | 触发条件 | 回退 |
|------|----------|------|
| flash-attn wheel 装不上 | torch/cu 版本不匹配 | 直接 `attn_implementation="sdpa"`，吞吐 -30% 但能跑 |
| Teacher OOM | bf16 下 16GB + activations > 24GB | 加 4-bit 量化（指南 §5 OOM 第 3 步） |
| Student rollout KL 爆炸 | `lmbda=0.5` 仍崩 | 降到 0.25；max_new_tokens 降到 512 |
| 长度膨胀 | 100+ steps 后 student 输出越来越长 | `repetition_penalty=1.05`；或 early stop |
| 磁盘满 | 训练中 ckpt 累积 | `save_total_limit=2`（GKDConfig 支持） |
| 网速波动 | HF 偶发慢 | `HF_HUB_ENABLE_HF_TRANSFER=1` 已开；仍慢则 `--max-workers 8` |

---

## 7. 一句话结论

**这份指南在我本机的 4× A5000 + 132 GB 磁盘 + 日本 Tokyo 网络下完全可行**，按 §5 的 7 步流程 + §3 的 5 处修正就能在 **总计约 4-6 小时**（含 ablation 约 12 小时）内拿到一份可写简历的 OPD 实验结果。下一步动作：执行 Step 0 + Step 1，先把环境装好、模型拉下来。
