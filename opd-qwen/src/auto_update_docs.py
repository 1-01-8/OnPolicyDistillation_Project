"""自动 patch RESULTS.md / README.md / COT_PLAN.md 加 sft_then_opd 行。

读取 runs/cot/metrics_auto.csv（cot_compare 写出），把 sft_then_opd 行插入到
RESULTS.md §6.X 的 markdown 表格中。失败时 silent (主流程 || true)。
"""
import csv, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COT = os.path.join(ROOT, "runs/cot")

csv_path = os.path.join(COT, "metrics_auto.csv")
if not os.path.exists(csv_path):
    print(f"[doc] no {csv_path}, skip"); sys.exit(0)

rows = list(csv.reader(open(csv_path)))
header, body = rows[0], rows[1:]
print("=== final 7-way metrics ===")
print(",".join(header))
for r in body:
    print(",".join(r))

# 写入 RESULTS.md 末尾增补段
res = os.path.join(ROOT, "RESULTS.md")
marker = "<!-- AUTO_SFT_THEN_OPD -->"
with open(res, "r") as f:
    text = f.read()
if marker not in text:
    table = "| model | acc | tokens | steps | eq_rate | ROUGE-L |\n|---|---|---|---|---|---|\n"
    for r in body:
        table += "| " + " | ".join(r) + " |\n"
    addition = (
        f"\n{marker}\n"
        "## §6.5  SFT→OPD 两阶段实验（auto-generated）\n\n"
        "在 1.7B-Base+SFT (LoRA, 已合并) 之上**再训一阶段 OPD**，与主线 1.7B-Base+OPD "
        "完全同条件 (λ=0.5, β=0.5, T=0.9, max_steps=400, lr=2e-5)。\n\n"
        "### 7-way 全量结果\n\n" + table + "\n"
        "**关键观察**：见上表 `Qwen3-1.7B-Base + SFT→OPD` 一行 vs `+ OPD` / `+ SFT`：\n"
        "- 若 acc > 70.7% → SFT-warmup 对 OPD 有正面增益；\n"
        "- 若 ROUGE-L > 0.534 且 eq_rate 显著回升 → SFT 学到的表面格式被 OPD 部分保留；\n"
        "- 若 acc ≤ 70.7% 且 eq_rate=0 → OPD 阶段把 SFT 的 surface fitting 完全洗掉，先 SFT 再 OPD 等价于直接 OPD。\n"
    )
    with open(res, "a") as f:
        f.write(addition)
    print("[doc] RESULTS.md updated")

# README §4.3 后追加一段（不动既有表）
rm = os.path.join(ROOT, "..", "README.md")
if os.path.exists(rm):
    with open(rm, "r") as f:
        t = f.read()
    if marker not in t:
        addition = (
            f"\n{marker}\n"
            "### 4.3.1 SFT→OPD 两阶段实验（自动追加）\n\n"
            "完整 7-way CoT 分析见 `opd-qwen/figs/cot_auto_*.png` 和 `RESULTS.md §6.5`。\n"
        )
        with open(rm, "a") as f:
            f.write(addition)
        print("[doc] README.md updated")
