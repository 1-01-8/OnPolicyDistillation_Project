"""Dump 每个模型在 GSM8K test 子集上的 CoT 输出（含元数据）到 jsonl。

输出 schema:
  {idx, question, gt_answer, gt_cot, pred_text, pred_answer, correct,
   gen_tokens, prompt_tokens, gen_time}

用法:
  CUDA_VISIBLE_DEVICES=0 python src/dump_cot.py \
      --model models/student-base \
      --lora runs/opd-qwen3-1.7b-base/checkpoint-400 \
      --n 500 --batch_size 16 \
      --tag opd_base \
      --out runs/cot/opd_base.jsonl
  # 多次采样（diversity 用）：
  python src/dump_cot.py ... --n_samples 5 --temperature 0.7 --top_p 0.95
"""
import argparse, json, os, re, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def extract_answer(text):
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def make_prompt(tok, question):
    msgs = [{"role": "user", "content": question}]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lora", default=None)
    p.add_argument("--tag", required=True, help="模型标签，写入每行 jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--n_samples", type=int, default=1,
                   help="同一题 sample 多少次（多样性用，>1 时自动开 do_sample）")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    do_sample = args.n_samples > 1

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={"": args.device}, attn_implementation="sdpa",
    )
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    test = load_dataset("openai/gsm8k", "main", split="test")
    end = args.offset + args.n if args.n != -1 else len(test)
    ds = test.select(range(args.offset, min(end, len(test))))

    fout = open(args.out, "w")
    pbar = tqdm(total=len(ds) * args.n_samples, desc=f"dump[{args.tag}]")

    def run_batch(qs, gts, idxs):
        prompts = [make_prompt(tok, q) for q in qs]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(args.device)
        plen = enc["input_ids"].shape[1]
        bsz = len(qs)
        nrs = args.n_samples if do_sample else 1
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else 1.0,
                top_p=args.top_p if do_sample else 1.0,
                num_return_sequences=nrs,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
        dt = (time.time() - t0) / (bsz * nrs)
        # out shape: (bsz*nrs, seq_len), order = [b0_s0, b0_s1, ..., b1_s0, ...]
        for i in range(bsz):
            q, gt, idx = qs[i], gts[i], idxs[i]
            prompt_tok = int(enc["attention_mask"][i].sum().item())
            for s in range(nrs):
                gen_ids = out[i * nrs + s]
                gen_tokens = (gen_ids[plen:] != tok.pad_token_id).sum().item()
                txt = tok.decode(gen_ids[plen:], skip_special_tokens=True)
                txt_clean = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
                pa = extract_answer(txt_clean)
                ga = extract_answer(gt)
                rec = {
                    "idx": idx, "sample": s, "tag": args.tag,
                    "question": q, "gt_cot": gt, "gt_answer": ga,
                    "pred_text": txt_clean, "pred_answer": pa,
                    "correct": pa == ga,
                    "gen_tokens": gen_tokens,
                    "prompt_tokens": prompt_tok,
                    "gen_time_s": dt,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pbar.update(1)

    qs, gts, idxs = [], [], []
    for i, ex in enumerate(ds):
        qs.append(ex["question"]); gts.append(ex["answer"])
        idxs.append(args.offset + i)
        if len(qs) == args.batch_size:
            run_batch(qs, gts, idxs); qs, gts, idxs = [], [], []
    if qs:
        run_batch(qs, gts, idxs)

    pbar.close(); fout.close()
    print(f"[dump_cot] wrote {args.out}")


if __name__ == "__main__":
    main()
