"""Merge SFT LoRA into base, save as a new HF dir for the OPD second-stage student.

Output:
  models/student-base-sft-merged/  (safetensors + tokenizer)
"""
import argparse
import os
import shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="models/student-base")
    p.add_argument("--lora", default="runs/sft-qwen3-1.7b-base/final")
    p.add_argument("--out", default="models/student-base-sft-merged")
    args = p.parse_args()

    if os.path.exists(args.out) and os.listdir(args.out):
        print(f"[merge] {args.out} already exists, skipping merge")
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"[merge] loading base {args.base}")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    print(f"[merge] applying lora {args.lora}")
    m = PeftModel.from_pretrained(base, args.lora)
    print("[merge] merge_and_unload ...")
    m = m.merge_and_unload()
    print(f"[merge] saving to {args.out}")
    m.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out)
    print("[merge] done")


if __name__ == "__main__":
    main()
