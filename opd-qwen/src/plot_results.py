"""聚合 runs/eval/*.log + training_logs/*/metrics.jsonl，画图：
  figs/loss_curve.png         — Instruct: OPD JSD vs SFT CE loss
  figs/loss_curve_base.png    — Base:     OPD JSD vs SFT CE loss
  figs/lmbda_ablation.png     — lmbda 扫描精度曲线
  figs/summary_bar.png        — Instruct: baseline / SFT / OPD / teacher
  figs/summary_bar_base.png   — Base:     baseline / SFT / OPD / teacher  (简历主线)
  figs/summary_combined.png   — Instruct vs Base 双子图（两条故事线）
  figs/compute_efficiency.png — FLOPs vs acc，OPD vs SFT compute-matched
"""
import json, os, re, glob
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "src", "training_logs")
EVAL = os.path.join(ROOT, "runs", "eval")
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)


def read_acc(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in reversed(f.read().splitlines()):
            m = re.search(r"Final Acc:\s*([\d.]+)", line)
            if m:
                return float(m.group(1))
    return None


def read_loss(jsonl):
    xs, ys = [], []
    if not os.path.exists(jsonl):
        return xs, ys
    for line in open(jsonl):
        d = json.loads(line)
        if "loss" in d and "train_runtime" not in d:
            xs.append(d["step"]); ys.append(d["loss"])
    return xs, ys


def plot_loss():
    for outname, runs in [
        ("loss_curve.png", [("opd-qwen3-1.7b-gsm8k", "OPD (JSD)"),
                            ("sft-qwen3-1.7b-gsm8k", "SFT (CE)")]),
        ("loss_curve_base.png", [("opd-qwen3-1.7b-base-gsm8k", "OPD (JSD) — Base"),
                                 ("sft-qwen3-1.7b-base-gsm8k", "SFT (CE) — Base")]),
    ]:
        plt.figure(figsize=(6, 4))
        plotted = False
        for tag, name in runs:
            xs, ys = read_loss(os.path.join(LOGS, tag, "metrics.jsonl"))
            if xs:
                plt.plot(xs, ys, label=name); plotted = True
        if not plotted:
            plt.close(); print(f"no loss logs for {outname}, skip"); continue
        plt.xlabel("step"); plt.ylabel("train loss"); plt.legend(); plt.grid(alpha=0.3)
        plt.title(outname.replace(".png", ""))
        plt.tight_layout(); plt.savefig(os.path.join(FIGS, outname), dpi=140); plt.close()
        print(f"→ figs/{outname}")


def _bar(items, outname, title=None, teacher_line=None):
    items = [(k, v) for k, v in items if v is not None]
    if not items: print(f"no eval logs for {outname}, skip"); return
    plt.figure(figsize=(6.5, 4.2))
    ks, vs = zip(*items)
    colors = ["#888", "#e08040", "#48c", "#c84"][:len(items)]
    bars = plt.bar(ks, vs, color=colors)
    for b, v in zip(bars, vs):
        plt.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=10)
    if teacher_line is not None:
        plt.axhline(teacher_line, color="#c84", linestyle="--", alpha=0.5,
                    label=f"teacher = {teacher_line:.3f}")
        plt.legend(loc="lower right")
    plt.ylabel("GSM8K acc"); plt.ylim(0, max(vs)+0.12); plt.grid(alpha=0.3, axis="y")
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, outname), dpi=140); plt.close()
    print(f"→ figs/{outname}")


def plot_bar():
    teacher = read_acc(os.path.join(EVAL, "baseline_teacher.log"))
    # Instruct
    _bar([
        ("Student\n(Instruct)",  read_acc(os.path.join(EVAL, "baseline_student.log"))),
        ("+ SFT-540",            read_acc(os.path.join(EVAL, "sft_final.log"))),
        ("+ OPD-300\n(λ=0.5)",   read_acc(os.path.join(EVAL, "opd_final.log"))),
        ("Teacher\n(8B)",        teacher),
    ], "summary_bar.png",
       title="Instruct student — OPD prevents catastrophic forgetting", teacher_line=teacher)
    # Base (resume main line)
    _bar([
        ("Student\n(Base)",      read_acc(os.path.join(EVAL, "baseline_student_base.log"))),
        ("+ SFT-1760",           read_acc(os.path.join(EVAL, "sft_base_final.log"))),
        ("+ OPD-400\n(λ=0.5)",   read_acc(os.path.join(EVAL, "opd_base_final.log"))),
        ("Teacher\n(8B)",        teacher),
    ], "summary_bar_base.png",
       title="Base student — OPD absolute gain (resume main line)", teacher_line=teacher)


def plot_combined():
    teacher = read_acc(os.path.join(EVAL, "baseline_teacher.log"))
    instruct = [
        ("Student",   read_acc(os.path.join(EVAL, "baseline_student.log"))),
        ("+ SFT",     read_acc(os.path.join(EVAL, "sft_final.log"))),
        ("+ OPD",     read_acc(os.path.join(EVAL, "opd_final.log"))),
    ]
    base = [
        ("Student",   read_acc(os.path.join(EVAL, "baseline_student_base.log"))),
        ("+ SFT",     read_acc(os.path.join(EVAL, "sft_base_final.log"))),
        ("+ OPD",     read_acc(os.path.join(EVAL, "opd_base_final.log"))),
    ]
    if any(v is None for _, v in instruct + base):
        print("combined skipped (missing eval)"); return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, items, title in [
        (axes[0], instruct, "Instruct student\n(no catastrophic forgetting)"),
        (axes[1], base,     "Base student\n(resume main line — absolute gain)"),
    ]:
        ks, vs = zip(*items)
        colors = ["#888", "#e08040", "#48c"]
        bars = ax.bar(ks, vs, color=colors)
        for b, v in zip(bars, vs):
            ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=10)
        if teacher is not None:
            ax.axhline(teacher, color="#c84", linestyle="--", alpha=0.6,
                       label=f"teacher {teacher:.3f}")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("GSM8K acc"); axes[0].set_ylim(0, (teacher or 0.9)+0.08)
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "summary_combined.png"), dpi=140); plt.close()
    print("→ figs/summary_combined.png")


def _flops(tag):
    """Read total_flos from summary.json. If missing (e.g. early-stopped run),
    estimate from a reference run with same per-step config."""
    p = os.path.join(LOGS, tag, "summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        for d in s.get("log_history_tail", []):
            if "total_flos" in d: return d["total_flos"]
    # Fallback: estimate from metrics.jsonl using Instruct OPD as per-step reference
    REF = {
        "opd-qwen3-1.7b-base-gsm8k": ("opd-qwen3-1.7b-gsm8k", 400),  # early-stopped at 400
    }
    if tag in REF:
        ref_tag, est_step = REF[tag]
        ref_p = os.path.join(LOGS, ref_tag, "summary.json")
        if os.path.exists(ref_p):
            s = json.load(open(ref_p))
            for d in s.get("log_history_tail", []):
                if "total_flos" in d:
                    per_step = d["total_flos"] / d["step"]
                    return per_step * est_step
    return None


def plot_compute_efficiency():
    """FLOPs vs GSM8K acc, OPD vs SFT, both Instruct and Base lines."""
    points = []  # (flops, acc, label, marker, color)
    for tag, eval_log, label, color, marker in [
        ("opd-qwen3-1.7b-gsm8k",      "opd_final.log",      "OPD-Instruct",  "#48c", "o"),
        ("sft-qwen3-1.7b-gsm8k",      "sft_final.log",      "SFT-Instruct",  "#e08040", "s"),
        ("opd-qwen3-1.7b-base-gsm8k", "opd_base_final.log", "OPD-Base",      "#1a5a99", "o"),
        ("sft-qwen3-1.7b-base-gsm8k", "sft_base_final.log", "SFT-Base",      "#a04020", "s"),
    ]:
        f = _flops(tag); a = read_acc(os.path.join(EVAL, eval_log))
        if f and a: points.append((f, a, label, color, marker))
    if not points: print("no flops/eval, skip compute_efficiency"); return
    plt.figure(figsize=(6.8, 4.4))
    for f, a, label, color, marker in points:
        suffix = " *" if label == "OPD-Base" else ""
        plt.scatter(f/1e15, a, s=120, c=color, marker=marker,
                    label=f"{label}{suffix}  ({f/1e15:.2f}e15, {a:.3f})",
                    edgecolors="black", linewidths=0.6)
    teacher = read_acc(os.path.join(EVAL, "baseline_teacher.log"))
    if teacher:
        plt.axhline(teacher, color="#c84", linestyle="--", alpha=0.5, label=f"teacher {teacher:.3f}")
    plt.xlabel("Training FLOPs  (×10¹⁵)")
    plt.ylabel("GSM8K acc")
    plt.title("Compute-matched efficiency — OPD vs SFT\n(* OPD-Base early-stopped at step 400, FLOPs extrapolated)")
    plt.grid(alpha=0.3); plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "compute_efficiency.png"), dpi=140); plt.close()
    print("→ figs/compute_efficiency.png")


def plot_ablation():
    pts = []
    for f in sorted(glob.glob(os.path.join(EVAL, "abl_lmbda_*.log"))):
        m = re.search(r"abl_lmbda_([\d.]+)\.log", f)
        if m:
            acc = read_acc(f)
            if acc is not None:
                pts.append((float(m.group(1)), acc))
    if not pts: print("no ablation logs yet, skip"); return
    pts.sort()
    xs, ys = zip(*pts)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, "o-", linewidth=2, markersize=8)
    plt.xlabel("lmbda  (0=off-policy / 1=on-policy)")
    plt.ylabel("GSM8K acc")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "lmbda_ablation.png"), dpi=140)
    print("→ figs/lmbda_ablation.png")


if __name__ == "__main__":
    plot_loss(); plot_bar(); plot_combined(); plot_ablation(); plot_compute_efficiency()
