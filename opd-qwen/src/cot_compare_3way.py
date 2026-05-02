"""SFT vs OPD vs Teacher 三方对比图（用正确的模型命名）。

输入：runs/cot/{sft,opd,instruct}.jsonl  (instruct = Qwen3-8B teacher)
输出：figs/cot_3way_*.png
"""
import json, os, re
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COT = os.path.join(ROOT, "runs/cot")
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)

LABELS = {
    "base": "Qwen3-1.7B-Base",
    "instruct1p7b": "Qwen3-1.7B (Instruct)",
    "instruct_sft": "Qwen3-1.7B-Instruct + SFT",
    "instruct_opd": "Qwen3-1.7B-Instruct + OPD",
    "sft": "Qwen3-1.7B-Base + SFT",
    "opd": "Qwen3-1.7B-Base + OPD",
    "sft_then_opd": "Qwen3-1.7B-Base + SFT→OPD",
    "teacher": "Qwen3-8B (Teacher)",
}
COLORS = {
    "base": "#bbbbbb", "instruct1p7b": "#888888",
    "instruct_sft": "#f4a261", "instruct_opd": "#a8dadc",
    "sft": "#e76f51", "opd": "#2a9d8f",
    "sft_then_opd": "#9b5de5",
    "teacher": "#264653",
}
DEFAULT_ORDER_FULL = [
    "base", "instruct1p7b",
    "instruct_sft", "instruct_opd",
    "sft", "opd", "sft_then_opd",
    "teacher",
]
DEFAULT_ORDER_BASE = ["base", "sft", "opd", "sft_then_opd", "teacher"]
DEFAULT_ORDER_3 = ["sft", "opd", "teacher"]


def _resolve_order():
    import os as _os
    mode = _os.environ.get("COT_ORDER", "auto")
    if mode == "3way":
        return DEFAULT_ORDER_3, "3way"
    if mode == "base":
        return DEFAULT_ORDER_BASE, "base"
    if mode == "full":
        return DEFAULT_ORDER_FULL, "full"
    # auto: include any tag whose jsonl exists
    present = [t for t in DEFAULT_ORDER_FULL
               if _os.path.exists(_os.path.join(COT, f"{t}.jsonl"))]
    return present, "auto"


ORDER, ORDER_MODE = _resolve_order()


def n_steps(text):
    parts = re.split(r"(?<=[.!?。！？])\s+|\n", text.strip())
    return max(1, sum(1 for p in parts if len(p.strip()) > 3))


def lcs_len(a, b):
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    if n > m:
        a, b = b, a; n, m = m, n
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(cur[j - 1], prev[j])
        prev = cur
    return prev[m]


def rouge_l(p, t):
    pt, tt = p.split(), t.split()
    if not pt or not tt:
        return 0.0
    l = lcs_len(pt, tt)
    if l == 0:
        return 0.0
    prec, rec = l / len(pt), l / len(tt)
    return 2 * prec * rec / (prec + rec)


def load(tag):
    return [json.loads(l) for l in open(os.path.join(COT, f"{tag}.jsonl"))]


def main():
    data = {t: load(t) for t in ORDER}
    teacher_by_idx = {r["idx"]: r["pred_text"] for r in data["teacher"]}

    stats = {}
    for tag, recs in data.items():
        toks = [r["gen_tokens"] for r in recs]
        steps = [n_steps(r["pred_text"]) for r in recs]
        acc = np.mean([r["correct"] for r in recs])
        eq_rate = np.mean([1 if "<<" in r["pred_text"] else 0 for r in recs])
        if tag == "teacher":
            rouge = None
        else:
            scores = [rouge_l(r["pred_text"], teacher_by_idx[r["idx"]])
                      for r in recs if r["idx"] in teacher_by_idx]
            rouge = float(np.mean(scores)) if scores else 0.0
        stats[tag] = dict(acc=acc, tokens=np.array(toks), steps=np.array(steps),
                          eq=eq_rate, rouge=rouge,
                          mean_tokens=float(np.mean(toks)),
                          mean_steps=float(np.mean(steps)))

    suffix = ORDER_MODE  # auto / 3way / base / full
    n_models = len(ORDER)
    fig_w = max(8, 2.0 * n_models)

    # ---- Figure 1: bar group (acc / mean_tokens / mean_steps / equation_rate) ----
    fig, axes = plt.subplots(1, 4, figsize=(fig_w, 4.5))
    metric_keys = [("acc", "Accuracy", 1), ("mean_tokens", "Avg CoT tokens", 1),
                   ("mean_steps", "Avg # reasoning steps", 1),
                   ("eq", "Equation-format rate (<<a×b=c>>)", 1)]
    for ax, (k, ttl, _) in zip(axes, metric_keys):
        vals = [stats[t][k] for t in ORDER]
        bars = ax.bar([LABELS[t] for t in ORDER], vals,
                      color=[COLORS[t] for t in ORDER])
        ax.set_title(ttl)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}" if v < 10 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"CoT comparison ({suffix}, GSM8K test n=1319)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, f"cot_{suffix}_bars.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ---- Figure 2: distribution (tokens + steps) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins_t = np.linspace(0, 800, 40)
    bins_s = np.arange(0, 80, 2)
    for tag in ORDER:
        axes[0].hist(stats[tag]["tokens"], bins=bins_t, alpha=0.5,
                     label=LABELS[tag], color=COLORS[tag])
        axes[1].hist(stats[tag]["steps"], bins=bins_s, alpha=0.5,
                     label=LABELS[tag], color=COLORS[tag])
    axes[0].set_xlabel("CoT length (tokens)"); axes[0].set_ylabel("count")
    axes[0].set_title("CoT length distribution"); axes[0].legend()
    axes[1].set_xlabel("# reasoning steps"); axes[1].set_ylabel("count")
    axes[1].set_title("Reasoning step count distribution"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, f"cot_{suffix}_dist.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ---- Figure 3: ROUGE-L vs teacher (students only) ----
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * (n_models - 1)), 4.5))
    students = [t for t in ORDER if t != "teacher"]
    vals = [stats[t]["rouge"] for t in students]
    bars = ax.bar([LABELS[t] for t in students], vals,
                  color=[COLORS[t] for t in students])
    ax.set_ylabel("ROUGE-L vs Qwen3-8B Teacher CoT")
    ax.set_title("CoT style similarity to Qwen3-8B Teacher")
    ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, f"cot_{suffix}_rouge.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ---- summary table ----
    print(f"\n{'model':<35} {'acc':>7} {'tokens':>8} {'steps':>7} "
          f"{'eq_rate':>8} {'ROUGE-L':>9}")
    print("-" * 80)
    for t in ORDER:
        s = stats[t]
        r = f"{s['rouge']:.3f}" if s["rouge"] is not None else "  -- "
        print(f"{LABELS[t]:<35} {s['acc']:>7.3f} {s['mean_tokens']:>8.1f} "
              f"{s['mean_steps']:>7.1f} {s['eq']:>8.3f} {r:>9}")
    print(f"\n→ figs/cot_{suffix}_bars.png  cot_{suffix}_dist.png  cot_{suffix}_rouge.png")

    # save csv
    import csv
    with open(os.path.join(COT, f"metrics_{suffix}.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc", "avg_tokens", "avg_steps", "eq_rate", "rouge_L_vs_teacher"])
        for t in ORDER:
            s = stats[t]
            w.writerow([LABELS[t], round(s["acc"], 4), round(s["mean_tokens"], 1),
                        round(s["mean_steps"], 1), round(s["eq"], 3),
                        "" if s["rouge"] is None else round(s["rouge"], 3)])


if __name__ == "__main__":
    main()
