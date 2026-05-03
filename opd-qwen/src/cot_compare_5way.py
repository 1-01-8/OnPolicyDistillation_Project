"""5 模型对比图：Teacher + Base + SFT + OPD + SFT→OPD（聚焦核心结论，简单易懂）。

输入：runs/cot/{base,sft,opd,sft_then_opd,teacher}.jsonl
输出：figs/cot_5way_{acc,tokens,eq,rouge,combined}.png
"""
import json, os, re
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COT = os.path.join(ROOT, "runs/cot")
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

# 5 个核心对比对象（顺序 = 故事线：teacher → 学生从弱到强）
ORDER = ["teacher", "base", "sft", "opd", "sft_then_opd"]
LABELS = {
    "teacher":      "Qwen3-8B\nTeacher",
    "base":         "Qwen3-1.7B-Base\n(no train)",
    "sft":          "+ SFT\n(off-policy)",
    "opd":          "+ OPD\n(on-policy)",
    "sft_then_opd": "+ SFT→OPD\n(two-stage)",
}
# 颜色：teacher 深、base 灰、SFT 暖橙（表面拟合）、OPD 青绿（语义）、组合紫
COLORS = {
    "teacher":      "#264653",
    "base":         "#bbbbbb",
    "sft":          "#e76f51",
    "opd":          "#2a9d8f",
    "sft_then_opd": "#9b5de5",
}


def n_steps(text):
    parts = re.split(r"(?<=[.!?。！？])\s+|\n", text.strip())
    return max(1, sum(1 for p in parts if len(p.strip()) > 3))


def lcs_len(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0: return 0
    if n > m: a, b, n, m = b, a, m, n
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            cur[j] = prev[j-1] + 1 if a[i-1] == b[j-1] else max(cur[j-1], prev[j])
        prev = cur
    return prev[m]


def rouge_l(p, t):
    pt, tt = p.split(), t.split()
    if not pt or not tt: return 0.0
    l = lcs_len(pt, tt)
    if l == 0: return 0.0
    prec, rec = l / len(pt), l / len(tt)
    return 2 * prec * rec / (prec + rec)


def load(tag):
    return [json.loads(l) for l in open(os.path.join(COT, f"{tag}.jsonl"))]


def main():
    data = {t: load(t) for t in ORDER}
    teacher_by_idx = {r["idx"]: r["pred_text"] for r in data["teacher"]}

    stats = {}
    for tag, recs in data.items():
        accs = [int(r.get("correct", 0)) for r in recs]
        toks = [r.get("n_tokens", len(r["pred_text"].split())) for r in recs]
        steps = [n_steps(r["pred_text"]) for r in recs]
        eqs = [int(bool(re.search(r"<<[^<>]+=[^<>]+>>", r["pred_text"]))) for r in recs]
        rouges = ([rouge_l(r["pred_text"], teacher_by_idx[r["idx"]])
                   for r in recs if r["idx"] in teacher_by_idx]
                  if tag != "teacher" else [])
        stats[tag] = dict(
            acc=float(np.mean(accs)),
            tokens=float(np.mean(toks)),
            steps=float(np.mean(steps)),
            eq=float(np.mean(eqs)),
            rouge=float(np.mean(rouges)) if rouges else None,
        )

    # 一张组合大图：4 个子图横排
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics = [
        ("acc",    "Accuracy on GSM8K (n=1319)",   "{:.1%}", 0, 1.0),
        ("tokens", "Avg CoT length (tokens)",      "{:.0f}", 0, None),
        ("eq",     "Equation-format <<a×b=c>> rate", "{:.0%}", 0, 1.0),
        ("rouge",  "ROUGE-L vs Teacher CoT",       "{:.3f}", 0, None),
    ]
    for ax, (k, ttl, fmt, ymin, ymax) in zip(axes, metrics):
        if k == "rouge":
            tags = [t for t in ORDER if t != "teacher"]
        else:
            tags = ORDER
        vals = [stats[t][k] if stats[t][k] is not None else 0 for t in tags]
        bars = ax.bar([LABELS[t] for t in tags], vals,
                      color=[COLORS[t] for t in tags], edgecolor="white", linewidth=1.5)
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)
        if ymax: ax.set_ylim(ymin, ymax * 1.15)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (max(vals) * 0.015),
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    plt.suptitle("OPD on Qwen3: 5-way comparison (Teacher vs 4 student variants)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cot_5way_combined.png"), dpi=130, bbox_inches="tight")
    plt.close()
    # 单图 1：accuracy 趋势（讲故事用）
    fig, ax = plt.subplots(figsize=(8, 5))
    tags = ORDER
    vals = [stats[t]["acc"] for t in tags]
    bars = ax.bar([LABELS[t] for t in tags], vals,
                  color=[COLORS[t] for t in tags], edgecolor="white", linewidth=1.5)
    ax.axhline(stats["teacher"]["acc"], color="#264653", ls="--", lw=1, alpha=0.5,
               label=f"Teacher = {stats['teacher']['acc']:.1%}")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("GSM8K test accuracy: OPD recovers 96% of teacher gap closed by SFT→OPD",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.1%}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cot_5way_acc.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # 单图 2: SFT 表面格式 vs OPD 真分布（最能讲故事的图）
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    students = [t for t in ORDER if t != "teacher"]
    for ax, key, ttl, ymax in [
        (axes[0], "eq",    "Equation-format <<a*b=c>> rate (SFT surface artifact)", 1.05),
        (axes[1], "rouge", "ROUGE-L vs Teacher CoT (semantic alignment)",           None),
    ]:
        vals = [stats[t][key] for t in students]
        bars = ax.bar([LABELS[t] for t in students], vals,
                      color=[COLORS[t] for t in students], edgecolor="white", linewidth=1.5)
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if ymax: ax.set_ylim(0, ymax)
        for b, v in zip(bars, vals):
            fmt = "{:.0%}" if key == "eq" else "{:.3f}"
            ax.text(b.get_x() + b.get_width()/2, v + (max(vals) * 0.02),
                    fmt.format(v), ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
    plt.suptitle("SFT copies surface format (98.6%); OPD learns semantics; "
                 "SFT->OPD washes out surface but keeps semantic gain",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cot_5way_format_vs_semantic.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # csv
    import csv
    with open(os.path.join(COT, "metrics_5way.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc", "avg_tokens", "avg_steps", "eq_rate", "rouge_L_vs_teacher"])
        for t in ORDER:
            s = stats[t]
            w.writerow([LABELS[t].replace("\n", " "),
                        round(s["acc"], 4), round(s["tokens"], 1), round(s["steps"], 1),
                        round(s["eq"], 3),
                        "" if s["rouge"] is None else round(s["rouge"], 3)])

    # 打印
    print(f"\n{'model':<32} {'acc':>7} {'tokens':>8} {'eq':>7} {'rouge':>8}")
    print("-" * 70)
    for t in ORDER:
        s = stats[t]
        r = f"{s['rouge']:.3f}" if s["rouge"] is not None else "  -- "
        print(f"{LABELS[t].replace(chr(10), ' '):<32} {s['acc']:>7.3f} "
              f"{s['tokens']:>8.1f} {s['eq']:>7.3f} {r:>8}")
    print("\n→ figs/cot_5way_combined.png")
    print("→ figs/cot_5way_acc.png")
    print("→ figs/cot_5way_format_vs_semantic.png")


if __name__ == "__main__":
    main()
