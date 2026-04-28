import re, torch, argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def extract_answer(text):
    m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", text)
    if m: return m.group(1).replace(",", "").rstrip(".")
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
    p.add_argument("--n", type=int, default=200,
                   help="评测条数；-1 表示全量")
    p.add_argument("--offset", type=int, default=0,
                   help="从第几条开始，配合 --n 实现数据集分片（双卡并行）")
    p.add_argument("--batch_size", type=int, default=8,
                   help="推理 batch size，越大越快；显存不够就调小")
    p.add_argument("--device", type=str, default=None,
                   help="指定单卡，如 cuda:0；不填则 device_map=auto")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    # 生成时必须左填充，否则 attention mask 对齐出错
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device_map = {"": args.device} if args.device else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    total_test = load_dataset("openai/gsm8k", "main", split="test")
    end = args.offset + args.n if args.n != -1 else len(total_test)
    ds = total_test.select(range(args.offset, min(end, len(total_test))))

    correct = 0
    total = 0
    pbar = tqdm(total=len(ds), desc="Evaluating GSM8K")

    # 按 batch_size 分组处理
    batch_questions, batch_answers = [], []

    def process_batch(questions, answers):
        nonlocal correct, total
        prompts = [make_prompt(tok, q) for q in questions]
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(next(model.parameters()).device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for gen_ids, ans in zip(out, answers):
            text = tok.decode(gen_ids[input_len:], skip_special_tokens=True)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            if extract_answer(text) == extract_answer(ans):
                correct += 1
            total += 1
            pbar.update(1)
            pbar.set_postfix({"acc": f"{correct/total:.3f}", "correct": f"{correct}/{total}"})

    for ex in ds:
        batch_questions.append(ex["question"])
        batch_answers.append(ex["answer"])
        if len(batch_questions) == args.batch_size:
            process_batch(batch_questions, batch_answers)
            batch_questions, batch_answers = [], []

    # 处理尾部不足一个 batch 的样本
    if batch_questions:
        process_batch(batch_questions, batch_answers)

    pbar.close()
    end_idx = args.offset + total
    print(f"Final Acc: {correct/total:.3f} ({correct}/{total})  "
          f"[samples {args.offset}–{end_idx-1}]")


if __name__ == "__main__":
    main()
