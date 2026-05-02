"""用 teacher (Qwen3-8B) 当 judge，对每个模型在错题上的 CoT 做 4 类分类：
  - arithmetic   : 算术错误（公式对，但算错）
  - missing_step : 跳步 / 推理不完整
  - wrong_formula: 公式选错 / 列错方程
  - other        : 其他

输入：runs/cot/*.jsonl (只挑 correct=False 且 gt_answer != None)
输出：runs/cot/error_taxonomy.csv

用法:
  CUDA_VISIBLE_DEVICES=0 python src/cot_judge.py \
      --files runs/cot/sft_base.jsonl runs/cot/opd_base.jsonl \
      --tags  sft opd \
      --max_per_tag 100
"""
import argparse, json, os, re, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import pandas as pd

JUDGE_PROMPT = """你是数学题判官。学生回答错了，请基于学生的推理找出主要错误类型，仅输出一个标签：
- arithmetic   : 计算步骤算错（公式对但数字错）
- missing_step : 推理跳步、缺中间步
- wrong_formula: 列错公式 / 用错关系
- other        : 其他

题目：
{q}

正确答案：{ga}

学生回答：
{cot}
学生最终答案：{pa}

直接输出标签（一个英文词）："""


def load_judge():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained("models/teacher")
    m = AutoModelForCausalLM.from_pretrained(
        "models/teacher", quantization_config=bnb, device_map="auto",
        attn_implementation="sdpa",
    )
    m.eval()
    return tok, m


def classify(tok, m, q, ga, cot, pa):
    msgs = [{"role": "user",
             "content": JUDGE_PROMPT.format(q=q, ga=ga, cot=cot[:1200], pa=pa)}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    enc = tok(p, return_tensors="pt").to(next(m.parameters()).device)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=10, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).lower()
    for k in ("arithmetic", "missing_step", "wrong_formula", "other"):
        if k in txt: return k
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--max_per_tag", type=int, default=100)
    args = ap.parse_args()

    tok, m = load_judge()
    summary = {}
    for tag, f in zip(args.tags, args.files):
        recs = [json.loads(l) for l in open(f)]
        bad = [r for r in recs if r.get("sample", 0) == 0
               and not r["correct"] and r["gt_answer"]]
        bad = bad[:args.max_per_tag]
        cnt = Counter()
        for r in tqdm(bad, desc=tag):
            cnt[classify(tok, m, r["question"], r["gt_answer"],
                          r["pred_text"], r["pred_answer"])] += 1
        summary[tag] = dict(cnt)
        print(tag, dict(cnt))

    df = pd.DataFrame(summary).fillna(0).astype(int)
    out = "runs/cot/error_taxonomy.csv"
    df.to_csv(out)
    print(df); print(f"→ {out}")


if __name__ == "__main__":
    main()
