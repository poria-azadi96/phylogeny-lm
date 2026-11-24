# Phylogeny LM - Structural Phylogeny QA with a 135M Fine-Tuned Model

A small, fully reproducible research workbench for **rooted binary phylogenetic tree question answering** and **real 16S rRNA gene-tree reconstruction**. A 135M-parameter `SmolLM2-135M-Instruct` model is fine-tuned with LoRA to answer four structural queries directly from Newick topology:

| Task | Question shape | Answer |
| :--- | :--- | :--- |
| `mrca` | "What is the most recent common ancestor of A and B?" | Internal clade label |
| `sister` | "What is the direct sister taxon of A?" | Leaf name (cherry siblings only) |
| `monophyletic` | "Do A, B[, C] form a monophyletic clade?" | `Yes` / `No` |
| `distance` | "What is the topological edge distance between A and B?" | Integer edge count |

The project includes a dataset generator, LoRA training, four evaluation suites (including real NCBI RefSeq 16S rRNA sequences), a pure-Python K2P + Neighbor-Joining reconstruction pipeline, and a zero-CDN single-page web workbench with an exact algorithmic verifier.

---

## Table of Contents

- [Why this project exists](#why-this-project-exists)
- [Architecture](#architecture)
- [Antigravity - OpenCode multi-agent workflow](#antigravity---opencode-multi-agent-workflow)
- [Packaged model](#packaged-model)
- [Verified benchmarks](#verified-benchmarks)
- [Real Genetic Phylogenetics: NCBI RefSeq 16S rRNA](#real-genetic-phylogenetics-ncbi-refseq-16s-rrna-archaea--bacteria)
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

Round 6 extended this rule from synthetic trees to real biology: instead of asking a 135M language model to parse raw DNA strings, `bio_phylo.py` performs pairwise alignment, computes Kimura 2-parameter distances, runs exact Neighbor-Joining, midpoint-roots the unrooted tree, and only then hands a normalized Newick topology to the fine-tuned model. The raw-FASTA control is reported honestly in the benchmark section.

---

## Architecture

```text
                         +----------------------------------------+
                         |           phylogeny_utils.py           |
                         | Newick parser / normalizer / solvers   |
                         | MRCA / sister / monophyletic / distance |
                         +-------------------+--------------------+
                                             | single source of truth
              +--------------------------------------------------+
              |                                                  |
   +----------+----------+                          +-------------+-------------+
   | generate_dataset.py |                          |        bio_phylo.py       |
   | disjoint taxa pools |                          | FASTA / K2P / NJ / rooting |
   | 4 tasks / verify    |                          +-------------+-------------+
   +----------+----------+                                        |
              |                                                   |
   +----------+----------+                          +-------------+-------------+
   |    train_lora.py    |                          | test_16s_rrna_phylogeny.py |
   | SmolLM2 + LoRA v4   |                          | 8 NCBI RefSeq sequences    |
   +----------+----------+                          +-------------+-------------+
              |                                                   |
              +---------------------------+-----------------------+
                                          |
                          +---------------+---------------+
                          |  model/phylogeny-lora/        |
                          |  packaged LoRA v4 adapter     |
                          +---------------+---------------+
                                          |
                  +-----------------------+------------------------+
                  |                                                |
        +---------+----------+                          +-----------+------------+
        | evaluate.py / demo  |                          |          app.py        |
        | stress_test.py      |                          | stdlib HTTP server     |
        +---------------------+                          | + static/index.html    |
                                                         | Newick + FASTA UI      |
                                                         | SVG cladogram + verify |
                                                         +------------------------+
```

### Inference prompt

Every question uses one canonical, compact prompt:

```text
Tree: ((Halobacterium_salinarum,((Methanocaldococcus_jannaschii,Pyrococcus_furiosus)Clade1,Sulfolobus_acidocaldarius)Clade2)Clade3,((Bacillus_subtilis,Staphylococcus_aureus)Clade4,(Escherichia_coli,Pseudomonas_aeruginosa)Clade5)Clade6)Root;
Task (mrca): What is the most recent common ancestor of Escherichia_coli and Pseudomonas_aeruginosa?
Answer with only the exact value.
```

The chat template is applied by the tokenizer, generation is greedy (`do_sample=False`), and the first non-empty output line is compared against the verifier answer case-insensitively. The packaged server uses `--max_new_tokens 24` so long binomial taxon names such as `Methanocaldococcus_jannaschii` are never truncated.

---

## Antigravity - OpenCode multi-agent workflow

This repository was built through a multi-round orchestrator/coder handoff:

- **Antigravity** acted as orchestrator, reviewer, GPU runner, error analyst, and git auditor.
- **OpenCode** acted as the coding sub-agent.
- A persistent OpenCode session was resumed across rounds so the agent retained project context.
- GPU execution and benchmark verification were performed by the orchestrator, not assumed by the coding agent.

| Round | Handoff | Outcome |
| :--- | :--- | :--- |
| Init | Repository and `.gitignore` initialized by Antigravity | Clean git workspace |
| Round 1 | OpenCode authored `generate_dataset.py` with an independent Newick parser, disjoint `train`/`val`/`test` taxon pools, and all four tasks | 600 train / 80 val / 80 test, 100% ground-truth verified, zero split overlap |
| Round 2 | OpenCode authored `evaluate.py` and `train_lora.py`; LoRA on attention + MLP projections with prompt-token `-100` label masking | Base **0.00%** to LoRA v1 **62.50%** |
| Round 3 | Per-task error analysis fed back: harder data, mixed clade labels, best-val checkpointing, and a real-biological-tree demo | LoRA v2 **98.75%** held-out; **11/12** unseen real trees |
| Round 4 | OpenCode built a 48-case four-category scientific stress test and upgraded training data to 4-8 leaves with hard paraphyly and synthetic Latinate clades | LoRA v3 **97.92%** stress test; **12/12** unseen real trees |
| Round 5 | stdlib web server + zero-CDN scientific UI with exact verifier | First single-page workbench |
| Round 6 | OpenCode authored `bio_phylo.py` and `test_16s_rrna_phylogeny.py`; generated `train_v4.jsonl` with binomial taxon names and `Root` clade labels; packaged LoRA v4 | Real 16S rRNA pipeline: **4/4 PASS** biological validation, **14/16 (87.50%)** gene-tree QA |
| Round 7 | OpenCode added FASTA mode to the web UI, two new JSON routes, `biopython`/`numpy` requirements, and English-only documentation | Newick or FASTA input, K2P distance matrix readout, automatic gene-tree loading into the visualizer and taxon picker |

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
| Training data | `data/train_v4.jsonl` (4,400 records, binomial taxon names, `Root` clade labels) |
| Best epoch | `6` |
| Best validation loss | `0.02518` |
| Inference token cap | `24` new tokens |
| `adapter_model.safetensors` | ~38 MB |
| PEFT version | `0.22.0` |

The v4 data mixture keeps the v3 structural difficulty (4-8 leaves, paraphyly traps, mixed clade labels) and adds full binomial taxon identifiers plus `Root` as an explicit top-level clade label. This is what makes real RefSeq taxa such as `Escherichia_coli` and `Sulfolobus_acidocaldarius` valid prompt tokens.

---

## Verified benchmarks

All numbers below are exact-match accuracy from runtime-verified evaluation runs. No numbers are estimated.

### 1. Held-out synthetic test set - `data/test.jsonl`

80 questions with **disjoint taxa and trees** from training and validation.

| Model | Overall | `mrca` (20) | `sister` (20) | `monophyletic` (20) | `distance` (20) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Base `SmolLM2-135M-Instruct` | **0.00%** (0/80) | 0.00% (0/20) | 0.00% (0/20) | 0.00% (0/20) | 0.00% (0/20) |
| LoRA v1 - 600 train, 4 epochs, `r=16` | **62.50%** (50/80) | 55.00% (11/20) | 90.00% (18/20) | 70.00% (14/20) | 35.00% (7/20) |
| LoRA v2 - 2,400 train, 6 epochs, `r=32` | **98.75%** (79/80) | **100.00%** (20/20) | **100.00%** (20/20) | **100.00%** (20/20) | 95.00% (19/20) |
| LoRA v3 - 3,600 train, 4-8 leaves | **97.50%** (78/80) | **100.00%** (20/20) | **100.00%** (20/20) | **100.00%** (20/20) | 90.00% (18/20) |
| **LoRA v4 (packaged)** - 4,400 train, binomial taxa | **95.00%** (76/80) | 95.00% (19/20) | **100.00%** (20/20) | **100.00%** (20/20) | 85.00% (17/20) |

v4 deliberately trades a little synthetic accuracy for the ability to answer questions about real binomial taxon names and explicit `Root` labels. Its strength is on real biological and adversarial trees, not on the synthetic split alone.

### 2. Unseen real biological trees - `demo_unseen_primates.py`

12 runtime-verified questions across three real phylogenies whose taxon and clade names never appeared in training:

1. **Primates** - `(((Homo,Pan)Hominini,Gorilla)Homininae,(Pongo,Hylobates)Pongidae)Hominoidea;`
2. **Carnivores** - `(((Canis,Vulpes)Canidae,(Ursus,Ailuropoda)Ursidae)Caniformia,(Felis,Panthera)Felidae)Carnivora;`
3. **Vertebrates** - `(((Gallus,Alligator)Archosauria,Anolis)Sauropsida,(Mus,Ornithorhynchus)Mammalia)Amniota;`

| Model | Correct | Accuracy |
| :--- | :---: | :---: |
| Base `SmolLM2-135M-Instruct` | 0/12 | **0.00%** |
| LoRA v2 | 11/12 | **91.67%** |
| LoRA v3 | **12/12** | **100.00%** |
| **LoRA v4 (packaged)** | 11/12 | **91.67%** |

### 3. Scientific stress test - `stress_test_phylogeny.py`

48 cases across four categories: real biology with paraphyly traps, adversarial topology swaps, out-of-distribution tree sizes, and branch-length/formatting noise.

#### Overall

| Model | Correct | Accuracy |
| :--- | :---: | :---: |
| Base `SmolLM2-135M-Instruct` | 0/48 | **0.00%** |
| LoRA v2 | 44/48 | **91.67%** |
| LoRA v3 | **47/48** | **97.92%** |
| **LoRA v4 (packaged)** | 46/48 | **95.83%** |

#### By category

| Category | Cases | Base | LoRA v2 | LoRA v3 | **LoRA v4 (packaged)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `real_biology_and_paraphyly` | 16 | 0/16 (0.00%) | 13/16 (81.25%) | **16/16 (100.00%)** | 15/16 (93.75%) |
| `adversarial_topology_swap` | 12 | 0/12 (0.00%) | **12/12 (100.00%)** | 11/12 (91.67%) | **12/12 (100.00%)** |
| `ood_tree_sizes` | 12 | 0/12 (0.00%) | **12/12 (100.00%)** | **12/12 (100.00%)** | **12/12 (100.00%)** |
| `branch_lengths_and_formatting` | 8 | 0/8 (0.00%) | 7/8 (87.50%) | **8/8 (100.00%)** | 7/8 (87.50%) |

#### Packaged v4 by task

| Task | Correct | Accuracy |
| :--- | :---: | :---: |
| `sister` | **12/12** | **100.00%** |
| `monophyletic` | **13/13** | **100.00%** |
| `distance` | **11/11** | **100.00%** |
| `mrca` | 10/12 | 83.33% |

The residual v4 stress-test misses are MRCA cases. The UI verifier exists specifically to surface this kind of failure rather than hide it.

---

## Real Genetic Phylogenetics: NCBI RefSeq 16S rRNA (Archaea & Bacteria)

`test_16s_rrna_phylogeny.py` and `bio_phylo.py` add a real-sequence case study on top of the synthetic benchmark. The hybrid pipeline is:

```text
FASTA -> Pairwise Alignment -> Kimura 2-Parameter distances -> Neighbor-Joining
      -> Midpoint Rooting -> Fine-Tuned LoRA v4 + Exact Verifier
```

### Input sequences

The bundled preset `data/16s_rrna_archaea_bacteria.fasta` contains 8 real NCBI RefSeq 16S rRNA sequences (4 Bacteria, 4 Archaea) with metadata in `data/16s_rrna_metadata.json`:

| Taxon | RefSeq accession | Length | Domain / phylum |
| :--- | :--- | ---: | :--- |
| `Escherichia_coli` | `NR_114042.1` | 1467 bp | Bacteria / Pseudomonadota (Gammaproteobacteria) |
| `Pseudomonas_aeruginosa` | `NR_117678.1` | 1527 bp | Bacteria / Pseudomonadota (Gammaproteobacteria) |
| `Bacillus_subtilis` | `NR_102783.2` | 1550 bp | Bacteria / Bacillota |
| `Staphylococcus_aureus` | `NR_037007.2` | 1552 bp | Bacteria / Bacillota |
| `Halobacterium_salinarum` | `NR_025555.1` | 1434 bp | Archaea / Euryarchaeota (Halobacteria) |
| `Methanocaldococcus_jannaschii` | `NR_113292.1` | 1458 bp | Archaea / Euryarchaeota (Methanococci) |
| `Pyrococcus_furiosus` | `NR_113294.1` | 1321 bp | Archaea / Euryarchaeota (Thermococci) |
| `Sulfolobus_acidocaldarius` | `NR_115499.1` | 1463 bp | Archaea / Thermoproteota (Crenarchaeota) |

### Step A - Biological validation of the reconstructed tree

The K2P + NJ + midpoint-rooting reconstruction recovers the expected domain and phylum structure. The exact solver verifies **4/4 PASS**:

| Biological claim | Clade | Verified result |
| :--- | :--- | :---: |
| Domain Bacteria is monophyletic | `Clade6` | PASS |
| Domain Archaea is monophyletic | `Clade3` | PASS |
| Gammaproteobacteria (`Escherichia_coli`, `Pseudomonas_aeruginosa`) is monophyletic | `Clade5` | PASS |
| Bacillota (`Bacillus_subtilis`, `Staphylococcus_aureus`) is monophyletic | `Clade4` | PASS |

The distance structure is also biologically sensible: **intra-phylum K2P distances are 0.0798-0.1631**, while **cross-domain distances are 0.3669-0.4561**. The reconstructed weighted Newick is:

```text
((Halobacterium_salinarum:0.165811,((Methanocaldococcus_jannaschii:0.063517,Pyrococcus_furiosus:0.062177)Clade1:0.029647,Sulfolobus_acidocaldarius:0.14479)Clade2:0.024758)Clade3:0.058535,((Bacillus_subtilis:0.027163,Staphylococcus_aureus:0.052618)Clade4:0.075384,(Escherichia_coli:0.095078,Pseudomonas_aeruginosa:0.068048)Clade5:0.040556)Clade6:0.092449)Root;
```

### Step B - 16 gene-tree questions

The 16 questions are split four per task (`mrca`, `sister`, `monophyletic`, `distance`), with `mrca` and `distance` evenly divided between intra-domain and cross-domain pairs. All ground truth is recomputed by `recompute_answer` from the reconstructed normalized Newick.

| Model | Overall | `mrca` (4) | `sister` (4) | `monophyletic` (4) | `distance` (4) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Base `SmolLM2-135M-Instruct` | **0/16 (0.00%)** | 0/4 (0.00%) | 0/4 (0.00%) | 0/4 (0.00%) | 0/4 (0.00%) |
| LoRA v3 | 12/16 (75.00%) | - | - | - | - |
| **LoRA v4 (packaged)** | **14/16 (87.50%)** | **4/4 (100%)** | **4/4 (100%)** | **4/4 (100%)** | 2/4 (50%) |

v4 is perfect on the three symbolic topology tasks for real 16S rRNA; both remaining misses are edge-distance questions, where the model must count edges between long binomial leaves across a deeper topology.

### Step C - Honest Raw-FASTA control

`test_16s_rrna_phylogeny.py` also runs a control that gives the model only 240 bp sequence snippets and asks which two species are most closely related, with **no tree provided**. The packaged v4 model scores **2/4 = 50.00%** on this control.

This is an honest negative result: a 135M language model is not a substitution-model phylogenetics engine. Feeding raw DNA strings directly to the LM discards the explicit symbolic structure that the adapter was trained to read. The hybrid pipeline is therefore essential:

1. Pairwise global alignment makes positions homologous.
2. Kimura 2-parameter distances convert differences into evolutionary distances.
3. Neighbor-Joining infers an unrooted topology from those distances.
4. Midpoint rooting produces a rooted binary tree with verifiable clade labels.
5. Fine-Tuned LoRA v4 answers structural questions on the normalized Newick.
6. The exact verifier recomputes ground truth from the same topology.

---

## Web workbench

`app.py` serves a single-page scientific UI from `static/index.html` using only Python's standard-library `ThreadingHTTPServer`. There is **no Flask, Gradio, FastAPI, or CDN dependency**.

### Newick Tree mode

- Six curated preset trees: Primates, Carnivores, Vertebrates Paraphyly Trap, Land Plants, Adversarial Counterfactual Primates, and Branch-Length Primates.
- Custom Newick textarea with branch-length stripping and automatic labeling of unlabeled internal nodes.
- Normalization status pill showing whether branch lengths were removed and how many internal labels were auto-assigned.

### FASTA / 16S rRNA Sequences mode

- **Load Real NCBI 16S rRNA (Archaea & Bacteria)** button calls `GET /api/fasta_preset` and fills the FASTA textarea.
- Organism / domain / accession chips for all eight RefSeq records.
- Multi-FASTA textarea and **Reconstruct Gene Tree (K2P + NJ)** button calling `POST /api/reconstruct`.
- Compact scrollable K2P distance matrix table and reconstructed weighted Newick readout.
- The reconstructed `normalized_newick` and `tree_summary` are loaded automatically into the SVG cladogram and taxon picker, so all four tasks are immediately available.

### Shared inference UI

- Task pills for `MRCA`, `Sister Taxon`, `Monophyletic Clade`, and `Edge Distance`.
- Clickable taxon chips with selection-count enforcement; in `sister` mode, only cherry leaves are selectable.
- Pure-JavaScript SVG cladogram with rectangular elbow branches, internal clade badges, queried-taxon highlights, predicted-clade highlights, and distance paths.
- Side-by-side result cards for **Fine-Tuned LoRA v4** and **Base SmolLM2-135M** with exact predictions, correctness badges, and measured latencies.
- An **Exact Algorithmic Verifier** badge (`Verified` / `Mismatch`) with the recomputed ground truth.
- A collapsible inspector showing the exact question, canonical prompt, and normalized Newick.
- Hero stats: **87.5%** real 16S rRNA, **95.8%** on the 48-case stress test, **95.0%** held-out test, and **100%** on 16S rRNA `mrca` / `sister` / `monophyletic`.

Because the base model shares the same weights and the adapter is toggled for base inference, concurrent inference is serialized with a lock. Base predictions are produced inside `model.disable_adapter()` (with a `disable_adapter_layers()` / `enable_adapter_layers()` fallback), always restoring adapter state.

---

## Quickstart

### Requirements

- Python 3.10+
- `torch`, `transformers`, `peft`, `accelerate`, `safetensors`, `biopython`, `numpy`
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
  --max_new_tokens 24
```

In the UI, switch to **FASTA / 16S rRNA Sequences**, load the real RefSeq preset, click **Reconstruct Gene Tree (K2P + NJ)**, then select taxa and click **Run Inference** to compare LoRA v4 against the base model with exact verification.

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
  --output_json results/unseen_bio_demo_v4.json
```

### Run the 48-case stress test

```bash
python stress_test_phylogeny.py \
  --adapter_dir model/phylogeny-lora \
  --output_json results/stress_test_v4.json
```

Verify the stress-test ground truth without loading a model:

```bash
python stress_test_phylogeny.py --verify_only
```

### Run the real 16S rRNA case study

```bash
python test_16s_rrna_phylogeny.py \
  --adapter_dir model/phylogeny-lora \
  --output_json results/16s_rrna_eval_v4.json
```

Verify the K2P + NJ reconstruction and biological clades without loading a model:

```bash
python test_16s_rrna_phylogeny.py --verify_only
```

### Regenerate and verify the dataset

```bash
python generate_dataset.py --verify
```

### Train from scratch (v4)

```bash
python train_lora.py \
  --train_file data/train_v4.jsonl \
  --val_file data/val.jsonl \
  --output_dir outputs/phylogeny-lora-v4 \
  --metrics_json results/train_metrics_v4.json \
  --epochs 6 \
  --batch_size 8 \
  --lr 3e-4 \
  --lora_r 32 \
  --lora_alpha 64
```

---

## HTTP API

`app.py` exposes the UI plus seven JSON endpoints.

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

Returns a JSON array of six preset Newick trees. Each object contains `id`, `name`, `description`, `raw_newick`, `normalized_newick`, `metadata`, `leaves`, `internal_clades`, `cherry_leaves`, `tree_dict`, `suggested_task`, and `suggested_taxa`.

### `GET /api/fasta_preset`

Returns the bundled real-sequence preset:

```json
{
  "name": "NCBI RefSeq 16S rRNA (Archaea & Bacteria)",
  "fasta": ">Escherichia_coli\n...",
  "metadata": [
    {
      "taxon_id": "Escherichia_coli",
      "organism": "Escherichia coli",
      "domain": "Bacteria",
      "accession": "NR_114042.1",
      "length_bp": 1467
    }
  ]
}
```

### `POST /api/parse`

Request:

```json
{ "newick": "(((Homo,Pan)Hominini,Gorilla)Homininae,(Pongo,Hylobates)Pongidae)Hominoidea;" }
```

Returns `normalized_newick`, `metadata`, `leaves`, `internal_clades`, `cherry_leaves`, and `tree_dict`. Malformed input returns HTTP `400` with `{"error": "..."}`.

### `POST /api/reconstruct`

Request:

```json
{ "fasta": ">Escherichia_coli\nACGT...\n>Pseudomonas_aeruginosa\nACGT...\n..." }
```

Returns HTTP `200` with:

```json
{
  "weighted_newick": "((Taxon:0.1,...)Clade1:0.2)Root;",
  "normalized_newick": "((Taxon,...)Clade1)Root;",
  "taxa": ["Escherichia_coli", "Pseudomonas_aeruginosa"],
  "lengths": { "Escherichia_coli": 1467 },
  "distance_matrix": [[0.0, 0.1631], [0.1631, 0.0]],
  "identity_matrix": [[100.0, 85.38], [85.38, 100.0]],
  "aligned_sites_matrix": [[1467, 1412], [1412, 1527]],
  "tree_summary": {
    "normalized_newick": "((Taxon,...)Clade1)Root;",
    "metadata": { "branch_lengths_stripped": false, "leaf_count": 8 },
    "leaves": ["Escherichia_coli", "Pseudomonas_aeruginosa"],
    "internal_clades": ["Clade1", "Root"],
    "cherry_leaves": ["Escherichia_coli", "Pseudomonas_aeruginosa"],
    "tree_dict": { "...": "..." }
  }
}
```

Invalid FASTA, `NewickError`, and `ValueError` return HTTP `400` with `{"error": "..."}`.

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
├── app.py                        # stdlib HTTP server, inference API, FASTA routes
├── static/
│   └── index.html                # Newick + FASTA scientific workbench
├── phylogeny_utils.py            # Newick parser, normalizer, exact solvers, summaries
├── bio_phylo.py                  # FASTA parsing, K2P distances, NJ, midpoint rooting
├── generate_dataset.py           # disjoint-split synthetic dataset generator
├── train_lora.py                 # LoRA training with label masking and checkpointing
├── evaluate.py                   # held-out greedy exact-match evaluation
├── demo_unseen_primates.py       # unseen real biological tree demo
├── stress_test_phylogeny.py      # 48-case, four-category stress test
├── test_16s_rrna_phylogeny.py    # real NCBI RefSeq 16S rRNA case study
├── requirements.txt
├── README.md
├── orchestration_log.md          # Antigravity - OpenCode handoff record
├── data/
│   ├── 16s_rrna_archaea_bacteria.fasta
│   ├── 16s_rrna_metadata.json
│   ├── train.jsonl
│   ├── train_v2.jsonl
│   ├── train_v3.jsonl
│   ├── train_v4.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── model/
│   └── phylogeny-lora/           # packaged LoRA v4 adapter + tokenizer
├── outputs/                      # training run artifacts (git-ignored)
└── results/                      # evaluation JSON artifacts
```

---

## Training details

- **Objective:** causal language modeling on the assistant answer only; prompt tokens are masked with `-100`.
- **Prompt format:** canonical tree/task/question text rendered through the tokenizer chat template.
- **Data mixtures:** v1 (600 records) to v2 (2,400 records, mixed clade labels) to v3 (3,600 records, 4-8 leaves, hard paraphyly, synthetic Latinate clades) to v4 (4,400 records, binomial taxon names and explicit `Root` labels).
- **Splits:** `train`, `val`, and `test` use disjoint taxon pools and disjoint trees; test is never used for checkpoint selection.
- **Checkpointing:** best validation loss is retained.
- **Decoding:** greedy (`do_sample=False`), capped at 24 new tokens in the packaged server.
- **Scoring:** first non-empty line, case-insensitive exact match against the exact solver.
- **Verification:** every generated example is recomputed by `phylogeny_utils.recompute_answer`; the stress test and 16S case study can be run with `--verify_only`.

---

## Scope & limitations

This is a focused structural-QA benchmark and demo, not a general phylogenetics package.

- **Rooted binary Newick only.** Polytomies, unrooted trees, reticulation, and quoted/complex Newick labels are out of scope.
- **Topology-only QA.** For `distance`, the answer is a **topological edge count**, not a K2P or patristic distance. K2P values are shown in the FASTA-mode matrix readout but are not what the model is asked to predict.
- **Sequence reconstruction is intentionally narrow.** FASTA mode uses pairwise global alignment, Kimura 2-parameter distances, exact Neighbor-Joining, and midpoint rooting; it is not a maximum-likelihood or Bayesian phylogenetics suite.
- **Raw DNA is not fed to the language model.** The Step C control shows why: 2/4 (50.00%) without an explicit tree, versus 14/16 (87.50%) with the hybrid pipeline.
- **Taxon names are treated as tokens.** The question parser expects simple names matching `[A-Za-z][A-Za-z0-9_]*`; whitespace is removed and punctuation-heavy labels should be avoided.
- **Leaf-count coverage.** Training and evaluation focus on 4-8 leaf trees. The 16S case study has 8 real taxa; larger trees are not broadly validated.
- **The model can still hallucinate.** Residual errors remain, especially on MRCA and edge-distance cases. The exact verifier, not the model, is authoritative.
- **v4 is a trade-off, not a strict upgrade.** It is below v2/v3 on some synthetic splits but adds real binomial-taxon coverage and perfect 16S rRNA `mrca` / `sister` / `monophyletic` accuracy.
- **No production hardening.** `app.py` is a small stdlib server intended for local demos and orchestrator testing.

---

## Reproducibility notes

- Training uses seed `42` by default.
- The CPU path uses `float32`; CUDA uses `bfloat16` when supported, otherwise `float16`.
- The packaged adapter includes its tokenizer files and `chat_template.jinja`, so `python app.py` works without a separate tokenizer download beyond the base model.
- Evaluation artifacts are written as JSON to `results/` for auditability.
- The base-model comparison in the UI is produced by temporarily disabling the LoRA adapter on the same loaded model, under a thread lock, so the comparison is fair and state-safe.
- FASTA preset paths in `app.py` are resolved relative to the repository root, so `GET /api/fasta_preset` works regardless of the current working directory.
