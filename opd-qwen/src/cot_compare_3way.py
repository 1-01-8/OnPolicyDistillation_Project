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
    "base": "Qwen3-1.7B-Base (student init)",
    "instruct1p7b": "Qwen3-1.7B (student-instruct baseline)",
    "sft": "Qwen3-1.7B-Base + SFT",
    "opd": "Qwen3-1.7B-Base + OPD",
    "teacher": "Qwen3-8B (Teacher)",
}
COLORS = {"base": "#bbbbbb", "instruct1p7b": "#888888",
          "sft": "#e76f51", "opd": "#2a9d8f", "teacher": "#264653"}
ORDER = ["sft", "opd", "teacher"]


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

    # ---- Figure 1: bar group (acc / mean_tokens / mean_steps / equation_rate) ----
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
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
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("CoT comparison: SFT vs OPD vs Teacher (GSM8K test, n=1319)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cot_3way_bars.png"), dpi=120, bbox_inches="tight")
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
    plt.savefig(os.path.join(FIGS, "cot_3way_dist.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ---- Figure 3: ROUGE-L vs teacher (sft vs opd only) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    students = [t for t in ORDER if t != "teacher"]
    vals = [stats[t]["rouge"] for t in students]
    bars = ax.bar([LABELS[t] for t in students], vals,
                  color=[COLORS[t] for t in students])
    ax.set_ylabel("ROUGE-L vs Qwen3-8B Teacher CoT")
    ax.set_title("CoT style similarity to Qwen3-8B Teacher")
    ax.set_ylim(0, max(vals) * 1.25)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cot_3way_rouge.png"), dpi=120, bbox_inches="tight")
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
    print(f"\n→ figs/cot_3way_bars.png  cot_3way_dist.png  cot_3way_rouge.png")

    # save csv
    import csv
    with open(os.path.join(COT, "metrics_3way.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["model", "acc", "avg_tokens", "avg_steps", "eq_rate", "rouge_L_vs_teacher"])
        for t in ORDER:
            s = stats[t]
            w.writerow([LABELS[t], round(s["acc"], 4), round(s["mean_tokens"], 1),
                        round(s["mean_steps"], 1), round(s["eq"], 3),
                        "" if s["rouge"] is None else round(s["rouge"], 3)])


if __name__ == "__main__":
    main()
