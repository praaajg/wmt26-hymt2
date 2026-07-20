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

## Repository Structure

| File | Description |
|------|-------------|
| `Hy-MT2-1.8B-train.ipynb` | QLoRA training notebook |
| `evaluate_hymt2_tvsub.ipynb` | Inference & evaluation on TVsub test set |
| `adapter_config.json` / `adapter_model.safetensors` | LoRA adapter weights |
| `tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` | Tokenizer files |
| `chat_template.jinja` | Chat template used for prompt formatting |
| `test_hypotheses.txt` | Model translations on TVsub test set (200 pairs) |
| `test_hypotheses.metrics.json` | Evaluation metrics

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

## Notes

- **SGM multi-reference parsing fix:** `parse_sgm_multiref` originally grouped references purely by numeric `<seg id>`, ignoring `<DOC>` boundaries. Since dev/test SGM files contain two documents (`SUB01`, `SUB02`) that both restart segment numbering at 1, the old function merged references from two different TV shows — corrupting validation sacreBLEU (which drives early stopping & checkpoint selection) and the final test report. The fix parses each document independently and offsets global IDs so references align correctly with concatenated source files.

---

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "tencent/Hy-MT2-1.8B",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "path/to/this/repo")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("tencent/Hy-MT2-1.8B", trust_remote_code=True)

prompt = (
    "Translate the following text into English. Note that you must ONLY "
    "output the translated result without any additional explanation:\n\n"
    "你好，世界"
)
msgs = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
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

Evaluation on TVsub test set (200 pairs, multi-reference with up to 3 refs per segment):

| Metric | Value |
|--------|-------|
| BLEU | 38.56 |
| sacreBLEU | 37.63 |
| chrF | 52.76 |
| TER | 53.93 |

Metrics computed by `evaluate_hymt2_tvsub.ipynb` using sacrebleu (tokenize="13a", smooth="exp") for sacreBLEU/chrF/TER and HuggingFace evaluate for BLEU.

---

## License

Apache 2.0
