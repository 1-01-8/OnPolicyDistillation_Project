---
library_name: transformers
model_name: lmbda_0.0
tags:
- generated_from_trainer
- gkd
- trl
licence: license
---

# Model Card for lmbda_0.0

This model is a fine-tuned version of [None](https://huggingface.co/None).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/xiexuemaixxm/huggingface/runs/1pe9e9v8) 


This model was trained with GKD, a method introduced in [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://huggingface.co/papers/2306.13649).

### Framework versions

- TRL: 0.21.0
- Transformers: 4.56.2
- Pytorch: 2.5.1+cu121
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citations

Cite GKD as:

```bibtex
@inproceedings{agarwal2024on-policy,
    title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
    author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
    year         = 2024,
    booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
    publisher    = {OpenReview.net},
    url          = {https://openreview.net/forum?id=3zKtaqxLhW},
}
```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```