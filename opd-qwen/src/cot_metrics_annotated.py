"""CoT (Chain-of-Thought) 行为分析的教学版脚本。
=================================================

【背景】
本脚本分析 GSM8K 上不同蒸馏方法（SFT / OPD / SFT→OPD / Teacher）的"思维链行为"，
回答一个核心问题：**学生模型究竟是模仿了 teacher 的"表面格式"还是"推理语义"？**

【6 个 CoT 维度】
1) acc            —— 答案是否正确（GSM8K 末位数字 == ground truth）
2) avg_tokens     —— CoT 平均长度（token 数），反映"啰嗦程度"
3) avg_steps      —— 推理步数（按句号/换行切句），反映"推理粒度"
4) eq_rate        —— GSM8K 训练集特有的 "<<a*b=c>>" 等式标记出现率，是 SFT 表面拟合的"指纹"
5) ROUGE-L vs teacher —— 与 teacher CoT 的最长公共子序列相似度（语义对齐的代理指标）
6) self-BLEU      —— 同一问题不同采样间的相互重叠（多样性，越低越多样）

【输入数据格式】 runs/cot/{tag}.jsonl，每行：
    {"idx": 0, "sample": 0, "question": "...", "pred_text": "...",
     "gold": "72", "pred": "72", "correct": 1, "gen_tokens": 134}

【为什么这些指标重要？】
- 一个 student 即使 acc 高，如果它的 CoT 是"乱猜数字 + 巧合命中"（短、无步骤、与 teacher 风格相反），
  说明它学的是 surface shortcut；OPD/SFT-then-OPD 应该展示更接近 teacher 的"推理形态"。
- eq_rate 是 GSM8K 数据集自带的怪癖标记。如果 student 大量生成 <<...>>，几乎必然是
  off-policy SFT 在做 token-level 复制（teacher 不写这个），这是"灾难性表面拟合"的强证据。
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 项目根目录：…/opd-qwen
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)


# ────────────────────────────────────────────────────────────
#  1. I/O 工具
# ────────────────────────────────────────────────────────────
def load(jsonl):
    """读取一份 dump_cot.py 的输出（jsonl）。每行一个 dict。"""
    out = []
    with open(jsonl) as f:
        for line in f:
            out.append(json.loads(line))
    return out


# ────────────────────────────────────────────────────────────
#  2. CoT 形态指标
# ────────────────────────────────────────────────────────────
def n_steps(text):
    """估计推理"步骤数" = 切分后的非空句数。

    切分规则：英文句号/感叹号/问号 + 空格，或换行符。
    > 4 字符的过滤是为了排除 "OK." "Yes." 这类非推理片段。

    GSM8K 经典 CoT 一般 5–25 步；过短(=Base 模型乱写) 或过长(>50, =SFT 灾难性重复) 都是异常信号。
    """
    sents = re.split(r"(?<=[.!?。！？])\s+|\n", text.strip())
    return sum(1 for s in sents if len(s.strip()) > 4)


def has_arith(text):
    """是否出现 '数字 OP 数字 ='，即学生有"算式形式"。注意这个比 eq_rate 宽松：
    eq_rate 要求严格 '<<a*b=c>>'（GSM8K 训练数据特有），has_arith 只要 'a*b=' 即可。"""
    return bool(re.search(r"\d+\s*[+\-*/×÷]\s*\d+\s*=", text))


def eq_rate(text):
    """GSM8K 训练集特有的 '<<3*4=12>>' 内联等式占比 —— SFT 表面拟合的指纹。

    Teacher (Qwen3-8B) 不会写这个标记（它没在 GSM8K 上做过专门 SFT）；
    SFT student (Base) 因为对 GT 做 token-level 拟合，几乎 100% 复刻；
    OPD student 因为是 on-policy + 蒸馏 teacher 分布，eq_rate 会归零。
    """
    return bool(re.search(r"<<[^<>]+=[^<>]+>>", text))


# ────────────────────────────────────────────────────────────
#  3. 文本相似度：n-gram 多样性 + ROUGE-L
# ────────────────────────────────────────────────────────────
def ngrams(toks, n):
    """切 n-gram。例如 ['a','b','c','d'] + n=2 → [('a','b'),('b','c'),('c','d')]"""
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(texts, n=2):
    """distinct-n：所有文本合并后唯一 n-gram 数 / 总 n-gram 数。
    越高 → 越多样；越低 → 越重复（SFT 经常 0.05 以下）。"""
    all_ng = []
    for t in texts:
        all_ng += ngrams(t.split(), n)
    if not all_ng:
        return 0.0
    return len(set(all_ng)) / len(all_ng)


def self_bleu(texts, n=2, max_pairs=200):
    """Self-BLEU：同一组里两两文本的 n-gram 重叠率均值。
    1.0 = 完全相同；0 = 完全不同。
    用于看"温度采样下不同 sample 的多样性"。
    采样上限 max_pairs 防止 O(N²) 爆炸。
    """
    if len(texts) < 2:
        return 0.0
    import random
    random.seed(0)
    pairs = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    scores = []
    for i, j in pairs:
        a = ngrams(texts[i].split(), n)
        b = ngrams(texts[j].split(), n)
        if not a or not b:
            continue
        ca, cb = Counter(a), Counter(b)
        # 交集计数 = ∑ min(ca[g], cb[g])
        overlap = sum((ca & cb).values())
        scores.append(overlap / max(len(a), 1))
    return np.mean(scores) if scores else 0.0


def rouge_l(ref, hyp):
    """ROUGE-L F1：基于 LCS（最长公共子序列）的 F-measure。
    ROUGE-L 的优点是允许"非连续匹配"（teacher 与 student 之间常有同义改写、插入步骤），
    所以比 BLEU 更适合衡量"推理路径相似度"。

    实现：经典 O(m·n) 动态规划。这里用纯 Python，避免引入 rouge_score 依赖。
    F1 = 2*P*R/(P+R)，P=LCS/|hyp|, R=LCS/|ref|。
    """
    a, b = ref.split(), hyp.split()
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / n
    r = lcs / m
    return 2 * p * r / (p + r)


# ────────────────────────────────────────────────────────────
#  4. 主流程
# ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="多份 jsonl 路径，对应每个模型的 dump_cot 输出")
    ap.add_argument("--tags", nargs="+", required=True,
                    help="与 --files 一一对应的标签（例 base sft opd teacher）")
    ap.add_argument("--teacher_tag", default=None,
                    help="哪个 tag 作为 ROUGE-L 的参考对象（一般填 teacher）")
    args = ap.parse_args()
    assert len(args.files) == len(args.tags), "files 和 tags 数量必须一致"

    # 4.1 读所有 jsonl
    data = {tag: load(f) for tag, f in zip(args.tags, args.files)}

    # 4.2 按 (idx, sample) 对齐 —— 同一题不同模型 / 同一模型不同采样
    aligned = defaultdict(dict)
    for tag, recs in data.items():
        for r in recs:
            aligned[(r["idx"], r["sample"])][tag] = r

    # 4.3 准备 teacher 参照表（用 (idx, sample) → text）
    teacher_recs = {}
    if args.teacher_tag and args.teacher_tag in data:
        for r in data[args.teacher_tag]:
            teacher_recs[(r["idx"], r["sample"])] = r["pred_text"]

    # 4.4 逐 tag 计算指标
    rows = []
    for tag in args.tags:
        recs = data[tag]
        L_tok = [r["gen_tokens"] for r in recs]
        L_step = [n_steps(r["pred_text"]) for r in recs]
        acc = np.mean([r["correct"] for r in recs])

        # ROUGE-L vs teacher（每条样本对齐 (idx, sample) 后求平均）
        rouges = []
        if teacher_recs and tag != args.teacher_tag:
            for r in recs:
                tk = teacher_recs.get((r["idx"], r["sample"]))
                if tk:
                    rouges.append(rouge_l(tk, r["pred_text"]))
        rouge = np.mean(rouges) if rouges else float("nan")

        # 多样性：同一题多个 sample 的 self-BLEU 均值
        per_q = defaultdict(list)
        for r in recs:
            per_q[r["idx"]].append(r["pred_text"])
        diversities = [self_bleu(v) for v in per_q.values() if len(v) > 1]
        div = np.mean(diversities) if diversities else float("nan")

        # SFT 表面格式信号
        eq = np.mean([eq_rate(r["pred_text"]) for r in recs])
        arith = np.mean([has_arith(r["pred_text"]) for r in recs])

        rows.append({
            "tag": tag,
            "n": len(recs),
            "acc": round(acc, 3),
            "avg_tokens": round(np.mean(L_tok), 1),
            "avg_steps": round(np.mean(L_step), 2),
            "rouge_L_vs_teacher": round(rouge, 3) if not np.isnan(rouge) else None,
            "self_bleu_2": round(div, 3) if not np.isnan(div) else None,
            "arith_eq_rate": round(arith, 3),
            "gsm8k_eq_rate": round(eq, 3),
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, "runs", "cot", "metrics_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\n→ {out_csv}")

    # 4.5 三张图
    # ─ 图1：长度 / 步数分布（直方叠加）→ 一眼看出 SFT 是否过长
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for tag in args.tags:
        recs = data[tag]
        ax[0].hist([r["gen_tokens"] for r in recs], bins=30, alpha=0.5, label=tag)
        ax[1].hist([n_steps(r["pred_text"]) for r in recs], bins=20, alpha=0.5, label=tag)
    ax[0].set_xlabel("gen tokens"); ax[0].set_ylabel("count")
    ax[0].legend(); ax[0].set_title("CoT length distribution")
    ax[1].set_xlabel("# reasoning steps"); ax[1].legend()
    ax[1].set_title("Reasoning steps distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cot_length_dist.png"), dpi=120)
    plt.close()

    # ─ 图2：长度桶 vs accuracy → 看是否"越长越对"或反之
    plt.figure(figsize=(7, 4))
    bins = [0, 50, 100, 150, 200, 300, 512]
    for tag in args.tags:
        recs = data[tag]
        toks = np.array([r["gen_tokens"] for r in recs])
        cor = np.array([r["correct"] for r in recs])
        accs = []
        for i in range(len(bins) - 1):
            mask = (toks >= bins[i]) & (toks < bins[i + 1])
            accs.append(cor[mask].mean() if mask.sum() > 0 else np.nan)
        plt.plot(bins[:-1], accs, marker="o", label=tag)
    plt.xlabel("CoT length (tokens)")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy vs CoT length")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "cot_acc_by_length.png"), dpi=120)
    plt.close()

    # ─ 图3：ROUGE-L vs teacher（学生间对比）→ 谁的语义最像 teacher
    if args.teacher_tag:
        plt.figure(figsize=(6, 4))
        labels = [r for r in args.tags if r != args.teacher_tag]
        vals = [next((row["rouge_L_vs_teacher"] for row in rows if row["tag"] == l), 0)
                for l in labels]
        plt.bar(labels, vals, color=["#888", "#e76f51", "#2a9d8f", "#9b5de5"][:len(labels)])
        plt.ylabel(f"ROUGE-L vs {args.teacher_tag}")
        plt.title("CoT style similarity (vs teacher)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS, "cot_similarity.png"), dpi=120)
        plt.close()

    print("→ figs/cot_*.png")


if __name__ == "__main__":
    main()
