"""6 个维度的 CoT 指标分析。

输入：runs/cot/*.jsonl（每个模型一份；同 idx 可对齐）
输出：
  runs/cot/metrics_summary.csv      汇总表
  figs/cot_length_dist.png          长度分布对比
  figs/cot_step_count.png           推理步数分布
  figs/cot_similarity.png           ROUGE-L vs teacher
  figs/cot_diversity.png            self-BLEU
  figs/cot_acc_by_length.png        长度-准确率联合分布

用法:
  python src/cot_metrics.py \
      --files runs/cot/baseline_base.jsonl runs/cot/sft_base.jsonl runs/cot/opd_base.jsonl runs/cot/teacher.jsonl \
      --tags  baseline sft opd teacher \
      --teacher_tag teacher
"""
import argparse, json, os, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)


def load(jsonl):
    out = []
    with open(jsonl) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def n_steps(text):
    """估计推理步数：按句子计数 + 排除空"""
    sents = re.split(r"(?<=[.!?。！？])\s+|\n", text.strip())
    return sum(1 for s in sents if len(s.strip()) > 4)


def has_arith(text):
    return bool(re.search(r"\d+\s*[+\-*/×÷]\s*\d+\s*=", text))


def ngrams(toks, n):
    return [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]


def distinct_n(texts, n=2):
    """所有 text 合并后 distinct n-gram 比例"""
    all_ng = []
    for t in texts:
        all_ng += ngrams(t.split(), n)
    if not all_ng: return 0.0
    return len(set(all_ng)) / len(all_ng)


def self_bleu(texts, n=2, max_pairs=200):
    """同组内两两相似度（粗略估计：n-gram 重叠率均值）"""
    if len(texts) < 2: return 0.0
    import random; random.seed(0)
    pairs = [(i, j) for i in range(len(texts)) for j in range(i+1, len(texts))]
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    scores = []
    for i, j in pairs:
        a, b = ngrams(texts[i].split(), n), ngrams(texts[j].split(), n)
        if not a or not b: continue
        ca, cb = Counter(a), Counter(b)
        overlap = sum((ca & cb).values())
        scores.append(overlap / max(len(a), 1))
    return np.mean(scores) if scores else 0.0


def rouge_l(ref, hyp):
    """LCS-based ROUGE-L F1，纯 Python 避开依赖"""
    a, b = ref.split(), hyp.split()
    if not a or not b: return 0.0
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    if lcs == 0: return 0.0
    p, r = lcs/n, lcs/m
    return 2*p*r/(p+r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--teacher_tag", default=None,
                    help="哪个 tag 是 teacher，用作 ROUGE 参考")
    args = ap.parse_args()
    assert len(args.files) == len(args.tags)

    data = {tag: load(f) for tag, f in zip(args.tags, args.files)}

    # 索引：按 (idx, sample) 对齐
    aligned = defaultdict(dict)
    for tag, recs in data.items():
        for r in recs:
            aligned[(r["idx"], r["sample"])][tag] = r

    rows = []
    teacher_recs = {}
    if args.teacher_tag and args.teacher_tag in data:
        for r in data[args.teacher_tag]:
            teacher_recs[(r["idx"], r["sample"])] = r["pred_text"]

    for tag in args.tags:
        recs = data[tag]
        L_tok = [r["gen_tokens"] for r in recs]
        L_step = [n_steps(r["pred_text"]) for r in recs]
        acc = np.mean([r["correct"] for r in recs])
        avg_tok = np.mean(L_tok); avg_step = np.mean(L_step)
        # ROUGE-L vs teacher (per question)
        rouges = []
        if teacher_recs and tag != args.teacher_tag:
            for r in recs:
                tk = teacher_recs.get((r["idx"], r["sample"]))
                if tk:
                    rouges.append(rouge_l(tk, r["pred_text"]))
        rouge = np.mean(rouges) if rouges else float("nan")
        # diversity (within tag, per-question across samples)
        per_q = defaultdict(list)
        for r in recs:
            per_q[r["idx"]].append(r["pred_text"])
        diversities = [self_bleu(v) for v in per_q.values() if len(v) > 1]
        div = np.mean(diversities) if diversities else float("nan")
        # arith / step coverage
        arith_rate = np.mean([has_arith(r["pred_text"]) for r in recs])
        rows.append({
            "tag": tag, "n": len(recs), "acc": round(acc, 3),
            "avg_tokens": round(avg_tok, 1),
            "avg_steps": round(avg_step, 2),
            "rouge_L_vs_teacher": round(rouge, 3) if not np.isnan(rouge) else None,
            "self_bleu_2": round(div, 3) if not np.isnan(div) else None,
            "arith_eq_rate": round(arith_rate, 3),
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, "runs", "cot", "metrics_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\n→ {out_csv}")

    # ─── plots ───
    # 1. 长度分布
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for tag in args.tags:
        recs = data[tag]
        ax[0].hist([r["gen_tokens"] for r in recs], bins=30, alpha=0.5, label=tag)
        ax[1].hist([n_steps(r["pred_text"]) for r in recs], bins=20, alpha=0.5, label=tag)
    ax[0].set_xlabel("gen tokens"); ax[0].set_ylabel("count"); ax[0].legend(); ax[0].set_title("CoT length distribution")
    ax[1].set_xlabel("# reasoning steps"); ax[1].legend(); ax[1].set_title("Reasoning steps distribution")
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "cot_length_dist.png"), dpi=120); plt.close()

    # 2. 准确率-长度联合
    plt.figure(figsize=(7, 4))
    for tag in args.tags:
        recs = data[tag]
        # 分桶
        toks = np.array([r["gen_tokens"] for r in recs])
        cor = np.array([r["correct"] for r in recs])
        bins = [0, 50, 100, 150, 200, 300, 512]
        accs = []
        for i in range(len(bins)-1):
            mask = (toks >= bins[i]) & (toks < bins[i+1])
            accs.append(cor[mask].mean() if mask.sum() > 0 else np.nan)
        plt.plot(bins[:-1], accs, marker="o", label=tag)
    plt.xlabel("CoT length (tokens)"); plt.ylabel("accuracy")
    plt.legend(); plt.title("Accuracy vs CoT length")
    plt.tight_layout(); plt.savefig(os.path.join(FIGS, "cot_acc_by_length.png"), dpi=120); plt.close()

    # 3. ROUGE / diversity bar
    if args.teacher_tag:
        plt.figure(figsize=(6, 4))
        labels = [r for r in args.tags if r != args.teacher_tag]
        vals = [next((row["rouge_L_vs_teacher"] for row in rows if row["tag"] == l), 0) for l in labels]
        plt.bar(labels, vals, color=["#888","#e76f51","#2a9d8f"][:len(labels)])
        plt.ylabel(f"ROUGE-L vs {args.teacher_tag}"); plt.title("CoT style similarity (vs teacher)")
        plt.tight_layout(); plt.savefig(os.path.join(FIGS, "cot_similarity.png"), dpi=120); plt.close()

    print("→ figs/cot_*.png")


if __name__ == "__main__":
    main()
