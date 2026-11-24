# Phylogeny LM — Structural Phylogeny QA with a 135M Fine-Tuned Model

A small, fully reproducible research workbench for **rooted binary phylogenetic tree question answering**. A 135M-parameter `SmolLM2-135M-Instruct` model is fine-tuned with LoRA to answer four structural queries directly from Newick topology:

| Task | Question shape | Answer |
| :--- | :--- | :--- |
| `mrca` | “What is the most recent common ancestor of A and B?” | Internal clade label |
| `sister` | “What is the direct sister taxon of A?” | Leaf name (cherry siblings only) |
| `monophyletic` | “Do A, B[, C] form a monophyletic clade?” | `Yes` / `No` |
| `distance` | “What is the topological edge distance between A and B?” | Integer edge count |

The project includes a dataset generator, LoRA training, three evaluation suites, a curated 48-case stress test, and a zero-CDN single-page web workbench with an exact algorithmic verifier.

---

## خلاصه فارسی

این پروژه یک مدل زبانی کوچک ۱۳۵ میلیون پارامتری (`SmolLM2-135M-Instruct`) را با LoRA روی پرسش‌های ساختاری درخت‌های فیلوژنتیک نیویک ریشه‌دار دودویی آموزش می‌دهد. چهار وظیفه پشتیبانی می‌شود: نزدیک‌ترین نیای مشترک (MRCA)، خواهر مستقیم، تک‌تباری (monophyletic)، و فاصلهٔ یالی توپولوژیک.

نتایج تأییدشده:

- مجموعهٔ آزمون مصنوعی کنارگذاشته‌شده `data/test.jsonl` (۸۰ پرسش، تاکسون‌ها و درخت‌های کاملاً جدا از آموزش):
  - مدل پایه: **۰٪ (۰/۸۰)**
  - LoRA v1: **۶۲.۵۰٪ (۵۰/۸۰)**
  - LoRA v2: **۹۸.۷۵٪ (۷۹/۸۰)**
  - LoRA v3 بسته‌بندی‌شده: **۹۷.۵۰٪ (۷۸/۸۰)**
- درخت‌های زیستی واقعی و دیده‌نشده (`demo_unseen_primates.py`، ۱۲ پرسش):
  - پایه: **۰/۱۲**، LoRA v2: **۱۱/۱۲ (۹۱.۶۷٪)**، LoRA v3: **۱۲/۱۲ (۱۰۰٪)**
- آزمون فشار علمی ۴۸ موردی (`stress_test_phylogeny.py`):
  - پایه: **۰/۴۸**، LoRA v2: **۴۴/۴۸ (۹۱.۶۷٪)**، LoRA v3: **۴۷/۴۸ (۹۷.۹۲٪)**

نسخهٔ نهایی در پوشهٔ `model/phylogeny-lora/` قرار دارد (LoRA با `r=32`، `alpha=64`، حدود ۹.۷۷ میلیون پارامتر آموزش‌پذیر، ۶.۷۷٪ پارامترها، حجم آداپتر ۳۸ مگابایت). در رابط وب، پاسخ مدل با «تأییدکنندهٔ الگوریتمی دقیق» مقایسه می‌شود و حقیقت زمینی از همان توپولوژی نرمال‌شده دوباره محاسبه می‌گردد.

**دامنهٔ کار:** فقط درخت‌های نیویک ریشه‌دار دودویی با ساختار توپولوژیک؛ طول شاخه‌ها حذف می‌شود. این پروژه ابزار هم‌ترازی FASTA/MSA، استنباط درست‌نمایی (likelihood)، یا بازسازی فیلوژنتیک احتمالاتی نیست. برای درخت‌های بزرگ‌تر از ۸ برگ ارزیابی گسترده انجام نشده است.

---

## Table of Contents

- [Why this project exists](#why-this-project-exists)
- [Architecture](#architecture)
- [Antigravity ⇄ OpenCode multi-agent workflow](#antigravity--opencode-multi-agent-workflow)
- [Packaged model](#packaged-model)
- [Verified benchmarks](#verified-benchmarks)
- [Web workbench](#web-workbench)
- [Quickstart](#quickstart)
- [HTTP API](#http-api)
- [Repository layout](#repository-layout)
- [Training details](#training-details)
- [Scope & limitations](#scope--limitations)
- [Reproducibility notes](#reproducibility-notes)

---

## Why this project exists

Phylogenetic reasoning is a useful stress test for small language models because the answer must be **derived from an explicit symbolic structure**, not recalled from pretraining text. The tree is given in the prompt; the model must follow parent/child relationships, distinguish cherries from non-cherries, detect paraphyly, and count edges. A base instruct model tends to answer with fluent but incorrect biology prose. A small LoRA adapter can instead learn the *algorithmic shape* of the task.

The central design rule is:

> **The language model is never the source of truth.** Ground truth is always recomputed by an independent exact solver in `phylogeny_utils.py`, and the web UI exposes both answers side by side.

This makes the project useful as a compact case study in verifier-backed fine-tuning: the adapter can be scored honestly, and any hallucination is immediately visible.

---

## Architecture

```text
                         ┌─────────────────────────────────────────────┐
                         │              phylogeny_utils.py             │
                         │  Newick parser · normalizer · exact solvers │
                         │  MRCA · sister · monophyletic · distance    │
                         └───────────────┬─────────────────────────────┘
                                         │ single source of truth
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
   ┌──────────▼──────────┐    ┌──────────▼──────────┐    ┌──────────▼──────────┐
   │ generate_dataset.py │    │     train_lora.py    │    │      evaluate.py     │
   │ disjoint taxa pools │    │  SmolLM2 + LoRA      │    │  greedy exact-match  │
   │ 4 tasks · verify    │    │  label masking       │    │  per-task metrics    │
   └─────────────────────┘    └──────────┬───────────┘    └─────────────────────┘
                                         │
                         ┌───────────────▼────────────────┐
                         │      model/phylogeny-lora/     │
                         │  packaged LoRA v3 adapter      │
                         └───────────────┬────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
┌───────▼──────────────┐      ┌──────────▼──────────────┐      ┌──────────▼──────────────┐
│ demo_unseen_primates │      │ stress_test_phylogeny.py │      │          app.py          │
│ real biological trees│      │ 48 cases · 4 categories  │      │ stdlib ThreadingHTTPServer│
└──────────────────────┘      └──────────────────────────┘      │ + static/index.html      │
                                                                 │ SVG cladogram + verifier │
                                                                 └──────────────────────────┘
```

### Inference prompt

Every question uses one canonical, compact prompt:

```text
Tree: (((Homo,Pan)Hominini,Gorilla)Homininae,(Pongo,Hylobates)Pongidae)Hominoidea;
Task (mrca): What is the most recent common ancestor of Homo and Gorilla?
Answer with only the exact value.
```

The chat template is applied by the tokenizer, generation is greedy (`do_sample=False`), and the first non-empty output line is compared against the verifier answer case-insensitively.

---

## Antigravity ⇄ OpenCode multi-agent workflow

This repository was built through a multi-round orchestrator/coder handoff:

- **Antigravity** acted as orchestrator, reviewer, GPU runner, error analyst, and git auditor.
- **OpenCode** acted as the coding sub-agent, driven by the free OpenCode Zen model `opencode/deepseek-v4.1-flash-free` at `$0` cost.
- A persistent OpenCode session was resumed across rounds so the agent retained project context.
- GPU execution and benchmark verification were performed by the orchestrator, not assumed by the coding agent.

| Round | Handoff | Outcome |
| :--- | :--- | :--- |
| Init | Repository and `.gitignore` initialized by Antigravity | Clean git workspace |
| Round 1 | OpenCode authored `generate_dataset.py` with an independent Newick parser, disjoint `train`/`val`/`test` taxon pools, and all four tasks | 600 train / 80 val / 80 test, 100% ground-truth verified, zero split overlap |
| Round 2 | OpenCode authored `evaluate.py` and `train_lora.py`; LoRA on attention + MLP projections with prompt-token `-100` label masking | Base **0.00%** → LoRA v1 **62.50%** |
| Round 3 | Per-task error analysis fed back: harder data, mixed clade labels, best-val checkpointing, and a real-biological-tree demo | LoRA v2 **98.75%** on held-out; **11/12** on unseen real trees |
| Round 4 | OpenCode built a 48-case four-category scientific stress test and upgraded training data to 4–8 leaves with hard paraphyly and synthetic Latinate clades | LoRA v3 **97.92%** stress test; **12/12** unseen real trees |
| Round 5 | stdlib web server + zero-CDN scientific UI with exact verifier | This workbench |

The workflow demonstrates a practical division of labor: the orchestrator owns measurement and judgement, while the coding agent owns reproducible implementation.

---

## Packaged model

| Property | Value |
| :--- | :--- |
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Adapter directory | `model/phylogeny-lora/` |
| Method | PEFT LoRA, causal language modeling |
| Rank `r` | `32` |
| `lora_alpha` | `64` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | `9,768,960` |
| Share of parameters | `6.77%` |
| Best epoch | `5` |
| Best validation loss | `0.02107` |
| `adapter_model.safetensors` | ~38 MB |
| PEFT version | `0.22.0` |

Training data for v3: **3,600** records, trees with **4–8 leaves**, hard paraphyly traps, and synthetic Latinate clade names. Six epochs were run; the best validation checkpoint was selected.

---

## Verified benchmarks

All numbers below are exact-match accuracy from the packaged adapter unless noted otherwise. No numbers are estimated.

### 1. Held-out synthetic test set — `data/test.jsonl`

80 questions with **disjoint taxa and trees** from training and validation.

| Model | Overall | `mrca` (20) | `sister` (20) | `monophyletic` (20) | `distance` (20) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Base `SmolLM2-135M-Instruct` | **0.00%** (0/80) | 0.00% (0/20) | 0.00% (0/20) | 0.00% (0/20) | 0.00% (0/20) |
| LoRA v1 — 600 train, 4 epochs, `r=16` | **62.50%** (50/80) | 55.00% (11/20) | 90.00% (18/20) | 70.00% (14/20) | 35.00% (7/20) |
| LoRA v2 — 2,400 train, 6 epochs, `r=32` | **98.75%** (79/80) | **100.00%** (20/20) | **100.00%** (20/20) | **100.00%** (20/20) | 95.00% (19/20) |
| **LoRA v3 / packaged** — 3,600 train, 4–8 leaves, 6 epochs, `r=32` | **97.50%** (78/80) | **100.00%** (20/20) | **100.00%** (20/20) | **100.00%** (20/20) | 90.00% (18/20) |

The packaged v3 is slightly lower than v2 on this synthetic split but stronger on real biological and adversarial trees, which is the intended trade-off after increasing structural difficulty.

### 2. Unseen real biological trees — `demo_unseen_primates.py`

12 runtime-verified questions across three real phylogenies whose taxon and clade names never appeared in training:

1. **Primates** — `(((Homo,Pan)Hominini,Gorilla)Homininae,(Pongo,Hylobates)Pongidae)Hominoidea;`
2. **Carnivores** — `(((Canis,Vulpes)Canidae,(Ursus,Ailuropoda)Ursidae)Caniformia,(Felis,Panthera)Felidae)Carnivora;`
3. **Vertebrates** — `(((Gallus,Alligator)Archosauria,Anolis)Sauropsida,(Mus,Ornithorhynchus)Mammalia)Amniota;`

| Model | Correct | Accuracy |
| :--- | :---: | :---: |
| Base `SmolLM2-135M-Instruct` | 0/12 | **0.00%** |
| LoRA v2 | 11/12 | **91.67%** |
| **LoRA v3 / packaged** | 12/12 | **100.00%** |

### 3. Scientific stress test — `stress_test_phylogeny.py`

48 cases across four categories, including real biology with paraphyly traps, adversarial topology swaps, out-of-distribution tree sizes, and branch-length/formatting noise.

#### Overall

| Model | Correct | Accuracy |
| :--- | :---: | :---: |
| Base `SmolLM2-135M-Instruct` | 0/48 | **0.00%** |
| LoRA v2 | 44/48 | **91.67%** |
| **LoRA v3 / packaged** | 47/48 | **97.92%** |

#### By category

| Category | Cases | Base | LoRA v2 | **LoRA v3 / packaged** |
| :--- | :---: | :---: | :---: | :---: |
| `real_biology_and_paraphyly` | 16 | 0/16 (0.00%) | 13/16 (81.25%) | **16/16 (100.00%)** |
| `adversarial_topology_swap` | 12 | 0/12 (0.00%) | 12/12 (100.00%) | 11/12 (91.67%) |
| `ood_tree_sizes` | 12 | 0/12 (0.00%) | 12/12 (100.00%) | **12/12 (100.00%)** |
| `branch_lengths_and_formatting` | 8 | 0/8 (0.00%) | 7/8 (87.50%) | **8/8 (100.00%)** |

#### Packaged v3 by task

| Task | Correct | Accuracy |
| :--- | :---: | :---: |
| `sister` | 12/12 | **100.00%** |
| `monophyletic` | 13/13 | **100.00%** |
| `distance` | 11/11 | **100.00%** |
| `mrca` | 11/12 | **91.67%** |

The single v3 stress-test miss is an adversarial MRCA case. The UI’s verifier is specifically designed to surface this kind of residual failure rather than hide it.

---

## Web workbench

`app.py` serves a single-page scientific UI from `static/index.html` using only Python’s standard-library `ThreadingHTTPServer`. There is **no Flask, Gradio, FastAPI, or CDN dependency**.

Features:

- Six curated preset trees:
  - Primates
  - Carnivores
  - Vertebrates Paraphyly Trap
  - Land Plants
  - Adversarial Counterfactual Primates
  - Branch-Length Primates
- Custom Newick textarea with branch-length stripping and automatic labeling of unlabeled internal nodes.
- Normalization status pill showing whether branch lengths were removed and how many internal labels were auto-assigned.
- Task pills for `MRCA`, `Sister Taxon`, `Monophyletic Clade`, and `Edge Distance`.
- Clickable taxon chips with selection-count enforcement; in `sister` mode, only cherry leaves are selectable and eligible leaves are marked.
- A pure-JavaScript SVG cladogram renderer drawing rectangular elbow branches, internal clade badges, leaf labels, queried-taxon highlights, predicted-clade highlights, predicted-sister highlights, and predicted distance paths.
- Side-by-side result cards for **Fine-Tuned LoRA v3** and **Base SmolLM2-135M** with exact predictions, correctness badges, and measured latencies in milliseconds.
- An **Exact Algorithmic Verifier** badge (`Verified ✓` / `Mismatch ✗`) with the recomputed ground truth.
- A collapsible inspector showing the exact question, canonical prompt, and normalized Newick.

Because the base model shares the same weights and the adapter is toggled for base inference, concurrent inference is serialized with a lock. Base predictions are produced inside `model.disable_adapter()` (with a `disable_adapter_layers()` / `enable_adapter_layers()` `try/finally` fallback), always restoring adapter state.

---

## Quickstart

### Requirements

- Python 3.10+
- `torch`, `transformers`, `peft`, `accelerate`, `safetensors`
- CPU works with `float32`; CUDA uses `bfloat16` when supported, otherwise `float16`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the web workbench

```bash
python app.py
# open http://127.0.0.1:7860
```

Useful flags:

```bash
python app.py --host 127.0.0.1 --port 7860 \
  --model_name HuggingFaceTB/SmolLM2-135M-Instruct \
  --adapter_dir model/phylogeny-lora \
  --max_new_tokens 16
```

Add `--no_base` to skip base-model predictions and halve inference work.

### Evaluate the packaged adapter

```bash
python evaluate.py \
  --adapter_dir model/phylogeny-lora \
  --test_file data/test.jsonl \
  --output_json results/packaged_model_eval.json
```

### Run the unseen real-tree demo

```bash
python demo_unseen_primates.py \
  --adapter_dir model/phylogeny-lora \
  --output_json results/unseen_bio_demo_v3.json
```

### Run the 48-case stress test

```bash
python stress_test_phylogeny.py \
  --adapter_dir model/phylogeny-lora \
  --output_json results/stress_test_v3.json
```

Verify the stress-test ground truth without loading a model:

```bash
python stress_test_phylogeny.py --verify_only
```

### Regenerate and verify the dataset

```bash
python generate_dataset.py --verify
```

### Train from scratch

```bash
python train_lora.py \
  --train_file data/train_v3.jsonl \
  --val_file data/val.jsonl \
  --output_dir outputs/phylogeny-lora-v3 \
  --metrics_json results/train_metrics_v3.json \
  --epochs 6 \
  --batch_size 8 \
  --lr 3e-4 \
  --lora_r 32 \
  --lora_alpha 64
```

---

## HTTP API

`app.py` exposes four JSON endpoints plus the UI.

### `GET /`

Serves `static/index.html`.

### `GET /api/health`

```json
{
  "status": "ok",
  "model_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "adapter_dir": "model/phylogeny-lora",
  "device": "cuda",
  "base_enabled": true
}
```

### `GET /api/presets`

Returns a JSON array of six preset objects. Each object contains `id`, `name`, `description`, `raw_newick`, `normalized_newick`, `metadata`, `leaves`, `internal_clades`, `cherry_leaves`, `tree_dict`, `suggested_task`, and `suggested_taxa`.

### `POST /api/parse`

Request:

```json
{ "newick": "(((Homo,Pan)Hominini,Gorilla)Homininae,(Pongo,Hylobates)Pongidae)Hominoidea;" }
```

Returns `normalized_newick`, `metadata`, `leaves`, `internal_clades`, `cherry_leaves`, and `tree_dict`. Malformed input returns HTTP `400` with `{"error": "..."}`.

### `POST /api/predict`

Request:

```json
{
  "newick": "(((Homo,Pan)Hominini,Gorilla)Homininae,(Pongo,Hylobates)Pongidae)Hominoidea;",
  "task": "mrca",
  "taxa": ["Homo", "Gorilla"]
}
```

Returns:

```json
{
  "normalized_newick": "...",
  "metadata": { "branch_lengths_stripped": false, "auto_labeled_nodes": [], "leaf_count": 5, "internal_count": 4 },
  "tree_dict": { "...": "..." },
  "task": "mrca",
  "taxa": ["Homo", "Gorilla"],
  "question": "What is the most recent common ancestor of Homo and Gorilla?",
  "prompt": "Tree: ...\nTask (mrca): ...\nAnswer with only the exact value.",
  "ground_truth": "Homininae",
  "lora_prediction": "Homininae",
  "lora_correct": true,
  "lora_ms": 41.2,
  "base_prediction": "The most recent...",
  "base_correct": false,
  "base_ms": 37.8
}
```

Validation rules:

- `mrca`: exactly two distinct leaves.
- `distance`: exactly two distinct leaves.
- `sister`: exactly one leaf, and it must be inside a cherry.
- `monophyletic`: two or three distinct leaves, canonicalized in alphabetical order.

When `--no_base` is used, `base_prediction` and `base_ms` are `null` and `base_correct` is `null`.

---

## Repository layout

```text
phylogeny-lm-test/
├── app.py                      # stdlib ThreadingHTTPServer + LoRA inference + API
├── static/
│   └── index.html              # zero-CDN single-page scientific workbench
├── phylogeny_utils.py          # parser, normalizer, exact solvers, summaries
├── generate_dataset.py         # disjoint-split synthetic dataset generator
├── train_lora.py               # LoRA training with label masking and checkpointing
├── evaluate.py                 # held-out greedy exact-match evaluation
├── demo_unseen_primates.py     # unseen real biological tree demo
├── stress_test_phylogeny.py    # 48-case, four-category stress test
├── requirements.txt
├── README.md
├── orchestration_log.md        # Antigravity ⇄ OpenCode handoff record
├── data/
│   ├── train.jsonl
│   ├── train_v2.jsonl
│   ├── train_v3.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── model/
│   └── phylogeny-lora/         # packaged LoRA v3 adapter + tokenizer
├── outputs/                    # training run artifacts (git-ignored)
└── results/                    # evaluation JSON artifacts
```

---

## Training details

- **Objective:** causal language modeling on the assistant answer only; prompt tokens are masked with `-100`.
- **Prompt format:** canonical tree/task/question text rendered through the tokenizer chat template.
- **Data mixtures:** v1 (600 records) → v2 (2,400 records, mixed clade labels) → v3 (3,600 records, 4–8 leaves, hard paraphyly, synthetic Latinate clades).
- **Splits:** `train`, `val`, and `test` use disjoint taxon pools and disjoint trees; test is never used for checkpoint selection.
- **Checkpointing:** best validation loss is retained, which is why v3 reports epoch 5 even when more epochs are configured.
- **Decoding:** greedy (`do_sample=False`), capped at 12–16 new tokens depending on the script.
- **Scoring:** first non-empty line, case-insensitive exact match against the exact solver.
- **Verification:** every generated example is recomputed by `phylogeny_utils.recompute_answer`; the stress test can be run with `--verify_only`.

---

## Scope & limitations

This is a focused structural-QA benchmark and demo, not a general phylogenetics package.

- **Rooted binary Newick only.** Polytomies, unrooted trees, reticulation, and quoted/complex Newick labels are out of scope.
- **Topology only.** Branch lengths are parsed and then stripped; distance means **topological edge count**, not patristic or substitution distance.
- **No alignment or likelihood inference.** There is no FASTA/MSA parsing, no substitution model, no maximum-likelihood or Bayesian reconstruction, and no tree inference from sequences.
- **Taxon names are treated as tokens.** The question parser expects simple names matching `[A-Za-z][A-Za-z0-9_]*`; whitespace is removed and punctuation-heavy labels should be avoided.
- **Leaf-count coverage.** Training and evaluation focus on 4–8 leaf trees. Behavior on larger or highly unbalanced trees is not broadly validated.
- **The model can still hallucinate.** Residual errors remain, especially on adversarial MRCA cases. The exact verifier, not the model, is authoritative.
- **v3 is a trade-off, not a strict upgrade.** It is slightly below v2 on the synthetic held-out split (97.50% vs. 98.75%) but stronger on unseen real biology and the stress test (97.92% vs. 91.67%).
- **No production hardening.** `app.py` is a small stdlib server intended for local demos and orchestrator testing; it is not a hardened multi-tenant deployment.

---

## Reproducibility notes

- Training uses seed `42` by default.
- The CPU path uses `float32`; CUDA uses `bfloat16` when supported, otherwise `float16`.
- The packaged adapter includes its tokenizer files and `chat_template.jinja`, so `python app.py` works without a separate tokenizer download beyond the base model.
- Evaluation artifacts are written as JSON to `results/` for auditability.
- The base-model comparison in the UI is produced by temporarily disabling the LoRA adapter on the same loaded model, under a thread lock, so the comparison is fair and state-safe.
