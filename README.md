---
license: apache-2.0
language:
- zh
- en
pipeline_tag: translation
tags:
- wmt26
- video-subtitle-translation
- hy-mt2
base_model:
- tencent/Hy-MT2-1.8B
---

# Hy-MT2-1.8B WMT26 — Video Subtitle Translation (zh→en)

QLoRA fine-tune of [tencent/Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) on [TVsub](https://github.com/longyuewangdcu/tvsub) for the WMT26 Video Subtitle Translation shared task.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| **Base model** | `tencent/Hy-MT2-1.8B` (1.8B params) |
| **Fine-tuning** | 4-bit QLoRA (NF4, double quant) |
| **LoRA rank / alpha / dropout** | 16 / 32 / 0.05 |
| **Target modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Trainable params** | 19,398,656 / 1,810,479,104 (1.07%) |
| **Data** | TVsub processed split — 60K train / 500 dev / 500 test pairs |
| **Max sequence length** | 640 tokens total (prompt + target) |
| **Epochs** | 3 (early stopping on dev sacreBLEU, patience=2, min delta=0.05) |
| **Batch size** | 4 (train), 4 (eval), gradient accumulation 8 |
| **Optimizer** | PagedAdamW8bit (falls back to torch AdamW) |
| **Learning rate** | 2e-4, cosine schedule with 3% warmup |
| **Weight decay** | 0.0 |
| **Max gradient norm** | 1.0 |
| **Precision** | BF16 (FP16 fallback), GradScaler for FP16 |
| **Hardware** | 1× NVIDIA L4 (23.7 GB) — Lightning AI Studio |

---

## Data Processing

- **Source:** TVsub dataset cloned from `github.com/longyuewangdcu/tvsub`
- **Cleaning:** HTML unescaping, detokenization (BPE `@@ `, `@-@`), NFKC normalization, whitespace collapse
- **Filtering:** Pairs where src/tgt length exceeds 2× max_len, or length ratio < 0.3 / > 4.0, are dropped
- **Prompt template:**
  ```
  Translate the following text into {tgt_lang}. Note that you must ONLY
  output the translated result without any additional explanation:

  {src}
  ```
- **Labels:** Only target tokens contribute to loss (prompt positions masked with `-100`)

---

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("aashish6969/wmt26-hymt2-finetuned")
tokenizer = AutoTokenizer.from_pretrained("aashish6969/wmt26-hymt2-finetuned")

prompt = (
    "Translate the following text into English. Note that you must ONLY "
    "output the translated result without any additional explanation:\n\n"
    "你好，世界"
)
msgs = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
output = model.generate(inputs, max_new_tokens=160, num_beams=4)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

---

## Generation Parameters

| Parameter | Value |
|-----------|-------|
| Max new tokens | 160 |
| Beam size | 4 |
| Do sample | False |
| Temperature | 1.0 |

---

## Performance

Final evaluation on the official TVsub test split (500 pairs):

| Metric | Value |
|--------|-------|
| BLEU | 25.49 |
| sacreBLEU | 24.26 |
| chrF | 39.80 |
| TER | 84.15 |

---

## License

Apache 2.0
