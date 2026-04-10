# prog-tutor-gemma2-2b-qlora

> A QLoRA fine-tuned Gemma-2-2B instruction model specialized as a **Programming Tutor** for Python, Data Structures & Algorithms, and SQL.

---

## Overview

This project fine-tunes [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) using **Supervised Fine-Tuning (SFT) with QLoRA** to produce a structured, pedagogically consistent programming tutor. The model is trained to always respond in a fixed tutoring format:

```
Goal → Concept → Step-by-step → Worked Example → Checkpoint
```

Every response guides the student through what they'll learn, builds intuition, walks through the approach step by step, shows working code, and ends with a comprehension question.

---

## Table of Contents

- [Model Details](#model-details)
- [Dataset](#dataset)
- [Training Setup](#training-setup)
- [Evaluation Results](#evaluation-results)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Limitations & Next Steps](#limitations--next-steps)

---

## Model Details

| Property | Value |
|---|---|
| **Base model** | `google/gemma-2-2b-it` |
| **Method** | Supervised Fine-Tuning (SFT) + QLoRA |
| **Quantization** | 4-bit NF4 + double quantization |
| **LoRA rank / alpha** | 16 / 32 |
| **LoRA dropout** | 0.05 |
| **Target modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Trainable parameters** | ~0.5–1% of total |
| **Hardware** | Google Colab T4 (16 GB VRAM) |

**Why Gemma-2-2B-IT?**  
The `-it` (instruction-tuned) checkpoint already understands conversational turn structure, reducing the data needed to steer format behaviour. At 2B parameters it fits on a Colab T4 under 4-bit NF4 quantization — larger alternatives like Llama 3 8B and Mistral 7B were ruled out due to OOM risk.

---

## Dataset

| Source | License | Size | Purpose |
|---|---|---|---|
| Hand-authored synthetic (`synth_data.json`) | Original / Unrestricted | 50 × 10 upsample = 500 | Format anchor |
| [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) | Apache 2.0 | ~280 filtered → ~180 used | Topic coverage |

**Total split:** ~610–630 train / ~68–70 validation (10% held-out)

**Tutoring format (system prompt enforced):**
```
Goal      — one sentence: what the student will learn
Concept   — intuitive explanation, use analogies
Step-by-step — numbered steps breaking down the approach
Worked Example — clean working code in a ```python block
Checkpoint — one question to verify understanding
```

**Data cleaning & filtering:**
- **Length filter** — retained only examples between 200–3072 characters
- **Structure filter** (CodeAlpaca) — kept only examples with code blocks or numbered steps and output > 100 words
- **Safety regex** — discarded entries containing `hack`, `exploit`, or `malware`
- No copyrighted dumps or private data used

---

## Training Setup

| Hyperparameter | Value |
|---|---|
| Epochs | 5 |
| Learning rate | 5e-5 |
| LR scheduler | Cosine |
| Warmup ratio | 0.1 |
| Effective batch size | 1 × 8 grad accum = 8 |
| Max sequence length | 1024 |
| Max generated tokens | 512 |
| Compute dtype | FP16 (BF16 unsupported on T4) |
| Optimizer | `paged_adamw_32bit` |
| Weight decay / grad clip | 0.01 / 1.0 |
| Est. training time | ~25 min |

**Key fix — DataCollator label masking:**  
`DataCollatorForCompletionOnlyLM` response-template token IDs were extracted **in-context** (not via a standalone tokenizer call) to prevent the BOS-prefix bug that masks all labels to `-100` and zeros gradient signal. A runtime assertion verifies that unmasked tokens exist before training starts.

---

## Evaluation Results

Base vs. fine-tuned compared on **32 held-out prompts** (27 in-scope, 5 out-of-scope) across three evaluation lenses:

```
==================================================================
EVALUATION RESULTS SUMMARY
==================================================================
Metric                                  Base   Fine-tuned    Delta
------------------------------------------------------------------
Ped. structure score (0–6)              5.69         5.69    +0.00
OOS refusal rate                        0.0%        80.0%
Code unit test pass rate              100.0%        90.0%
==================================================================
```

> Note: 1 code unit test failed in the fine-tuned model due to a syntax error.

**Evaluation lenses:**

1. **Lens 1 — Pedagogical Structure Score (0–6):** Rule-based check for the presence of `Goal`, `Concept`, `Step-by-step`, code block, `Checkpoint`, and minimum length. Applied to in-scope prompts only.

2. **Lens 2 — Out-of-scope Refusal Quality:** 5 off-topic prompts. Pass if the response contains a domain-redirect keyword (e.g. `'outside'`, `'programming tutor'`). The fine-tuned model went from 0% → 80% refusal rate.

3. **Lens 3 — Code Unit Tests (10 cases):** Prompts paired with executable Python test harnesses. First fenced code block extracted and run in a subprocess with a 10-second timeout.

---

## Installation & Usage

### Requirements

```bash
pip install transformers==4.44.0 peft==0.12.0 trl bitsandbytes accelerate==0.31.0 datasets
```

### Load the adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    quantization_config=bnb,
    device_map="auto",
    attn_implementation="eager",
)

model = PeftModel.from_pretrained(base, "./prog_tutor_qlora/final_adapter")
tokenizer = AutoTokenizer.from_pretrained("./prog_tutor_qlora/final_adapter")
```

### Run inference

```python
SYSTEM_PROMPT = (
    "You are a programming tutor specializing in Python, "
    "Data Structures & Algorithms, and SQL.\n\n"
    "REQUIRED FORMAT — you MUST use this exact structure for every answer:\n"
    "**Goal** — one sentence: what the student will learn\n"
    "**Concept** — intuitive explanation, use analogies\n"
    "**Step-by-step** — numbered steps breaking down the approach\n"
    "**Worked Example** — clean working code in a ```python block\n"
    "**Checkpoint** — end with one question to verify understanding\n"
)

@torch.inference_mode()
def ask_tutor(question, model, tokenizer, max_new_tokens=512):
    messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print(ask_tutor("What is a heap and when should I use min-heap vs max-heap?", model, tokenizer))
```

### Reproduce training

Open `final.ipynb` in Google Colab with a **T4 GPU runtime** and run all cells in order.

> **Note:** You will need a Hugging Face token with access to `google/gemma-2-2b-it`. Set it in Cell 15 or via `huggingface-cli login`.

---

## Project Structure

```
.
├── final.ipynb              # Main training & evaluation notebook
├── synth_data.json          # 50 hand-authored synthetic tutoring examples
├── unit_tests.json          # 10 Python unit test harnesses for code eval
├── prog_tutor_qlora/
│   └── final_adapter/       # Saved LoRA adapter weights + tokenizer
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files
└── README.md
```

---

## Limitations & Next Steps

**Current limitations:**
- **Small / upsampled dataset:** ×10 replication risks format-overfitting; a larger diverse corpus is needed for broad generalization.
- **No preference tuning:** Hallucination and code-correctness errors that SFT missed require DPO/RLHF with ranked response pairs.
- **Rule-based scorer:** Measures header presence, not actual pedagogical effectiveness; an LLM-as-judge rubric would give a richer signal.
- **Single syntax failure:** One code unit test failed in the fine-tuned model due to a syntax error in the generated output.

**Planned next steps:**
- [ ] Scale synthetic data to 500–1000 non-upsampled examples
- [ ] Add LLM-as-judge evaluation for response quality
- [ ] Apply DPO for preference tuning
- [ ] Merge adapter and export to GGUF format
- [ ] Publish to Hugging Face Hub with a Gradio demo

---

## References

- [Gemma 2 (Google DeepMind)](https://huggingface.co/google/gemma-2-2b-it)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [CodeAlpaca-20k Dataset](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
- [TRL – Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [PEFT – Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

---
