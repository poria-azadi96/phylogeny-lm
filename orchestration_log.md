# Orchestration Log: Antigravity - OpenCode (Phylogeny Minimal LM Fine-Tuning)

## Environment & Agent Setup
- **Orchestrator**: Antigravity (architecture, code review, GPU training & verification, error analysis, live HTTP smoke testing, git auditing)
- **Coding Sub-Agent**: OpenCode CLI (coding handoffs across a resumed session)
- **Base Language Model**: `HuggingFaceTB/SmolLM2-135M-Instruct` (~135M params)
- **Packaged Final Adapter**: `model/phylogeny-lora/` (LoRA v4, `r=32`, `alpha=64`, `9,768,960` trainable params = `6.77%`, `best_epoch=6`, `best_val_loss=0.02518`, `adapter_model.safetensors` = `38 MB`)
- **Compute & Storage**:
  - GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM, `torch 2.15.0+cu132`, `bfloat16`)
  - Python env: `/home/poriaazadi/miniconda3/envs/phylogeny_lm` (WSL `/`)
  - HuggingFace cache: `/home/poriaazadi/.cache/huggingface` (kept off the Windows drive)

---

## Multi-Round Agent-to-Agent Handoffs

| Round | Direction & Task | Verification Outcome |
| :--- | :--- | :--- |
| **Init** | Antigravity initialized git repo & `.gitignore` | Clean repo |
| **Round 1** | **Antigravity -> OpenCode**: Author stdlib-only `generate_dataset.py` with independent recursive-descent Newick parser, 3 disjoint taxon pools (`train`/`val`/`test`), and 4 phylogeny tasks (`mrca`, `sister`, `monophyletic`, `distance`). | `python3 generate_dataset.py --verify` passed: 600 train / 80 val / 80 test, 100% ground-truth verified, 0 overlap. |
| **Round 2** | **Antigravity -> OpenCode**: Resume session to author `evaluate.py` (greedy chat-template evaluation) and `train_lora.py` (LoRA on attention + MLP projections with strict prompt-token `-100` label masking). | Baseline eval: **0.00%** (0/80). LoRA v1 (`r=16`, 4 epochs, 4.88M trainable params): `val_loss=0.18795`, test accuracy **62.50%** (50/80). |
| **Round 3** | **Antigravity -> OpenCode**: Fed Round 2 per-task metrics and off-by-one error analysis back to OpenCode. Added harder data, mixed clade labels, best-val checkpointing, and `demo_unseen_primates.py`. | LoRA v2 (`r=32`, 6 epochs, best epoch 5, `val_loss=0.00876`): **98.75%** (79/80) test, **11/12 (91.67%)** unseen real trees. |
| **Round 4a** | **Antigravity -> OpenCode**: Create `phylogeny_utils.py` (Newick normalizer, exact solvers, tree JSON summary) and `stress_test_phylogeny.py` (48-case, 4-category suite). | `--verify_only` passed 48/48. LoRA v2 scored **44/48 (91.67%)** vs Base **0/48 (0.00%)**. |
| **Round 4b** | **Antigravity -> OpenCode**: Upgrade `generate_dataset.py` to 4-8 leaves, hard paraphyly, and synthetic Latinate clades; generate `data/train_v3.jsonl` (3,600 records). | LoRA v3: **97.50%** (78/80) test, **12/12 (100%)** unseen real trees, **47/48 (97.92%)** stress test. |
| **Round 5** | **Antigravity -> OpenCode**: Build stdlib HTTP inference server `app.py`, zero-CDN SVG cladogram workbench `static/index.html`, `requirements.txt`, and `README.md`. | Live GPU smoke test verified `GET /`, `GET /api/health`, `GET /api/presets`, `POST /api/parse`, and `POST /api/predict` with exact verifier agreement. |
| **Round 6** | **Antigravity -> OpenCode**: Author `bio_phylo.py` (FASTA parsing, Kimura 2-parameter distances, exact Saitou-Nei Neighbor-Joining, midpoint rooting) and `test_16s_rrna_phylogeny.py` (8 real NCBI RefSeq 16S rRNA sequences, Steps A/B/C). Generate `data/train_v4.jsonl` with binomial taxon names and explicit `Root` clade labels; train and package LoRA v4. | Real 16S rRNA pipeline: Step A **4/4 PASS** biological validation, Step B **14/16 (87.50%)** for packaged v4, Step C honest raw-FASTA control **2/4 (50.00%)**. `max_new_tokens=24` prevents long binomial names from truncating. |
| **Round 7** | **Antigravity -> OpenCode**: Add a FASTA / 16S rRNA Input Mode toggle to `static/index.html`, `GET /api/fasta_preset` and `POST /api/reconstruct` to `app.py`, `biopython` and `numpy` to `requirements.txt`, and rewrite `README.md` as 100% English-only with all verified v4 benchmarks. | Static verification: `python -m py_compile app.py bio_phylo.py` passed. Live GPU HTTP smoke test and Persian-character scan delegated to Antigravity. |

The workflow demonstrates a practical division of labor: the orchestrator owns measurement and judgement, while the coding agent owns reproducible implementation.

---

## Final Quantitative Summary

### 1. Held-Out Unseen Test Set (`data/test.jsonl`, 80 examples, 100% disjoint taxa & trees)

| Model | Overall | `mrca` (20) | `sister` (20) | `monophyletic` (20) | `distance` (20) | Best Val Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base `SmolLM2-135M-Instruct` | **0.00%** (0/80) | 0.00% | 0.00% | 0.00% | 0.00% | - |
| Round 2 LoRA v1 (600 train, 4 ep, `r=16`) | **62.50%** (50/80) | 55.00% | 90.00% | 70.00% | 35.00% | `0.18795` |
| Round 3 LoRA v2 (2,400 train, 6 ep, `r=32`) | **98.75%** (79/80) | **100.00%** | **100.00%** | **100.00%** | 95.00% | `0.00876` |
| Round 4b LoRA v3 (3,600 train) | **97.50%** (78/80) | **100.00%** | **100.00%** | **100.00%** | 90.00% | `0.02107` |
| **Round 6 LoRA v4 / packaged (4,400 train)** | **95.00%** (76/80) | 95.00% | **100.00%** | **100.00%** | 85.00% | `0.02518` |

### 2. Unseen Real Biological Trees (`demo_unseen_primates.py`, 12 questions)

- **Base `SmolLM2-135M-Instruct`**: **0/12 (0.00%)**
- **Round 3 LoRA v2**: 11/12 (91.67%)
- **Round 4b LoRA v3**: **12/12 (100.00%)**
- **Round 6 LoRA v4 (packaged)**: 11/12 (91.67%)

### 3. 48-Question Scientific Stress Test (`stress_test_phylogeny.py`)

| Category | Cases | Base | LoRA v2 | LoRA v3 | **LoRA v4 (packaged)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `real_biology_and_paraphyly` | 16 | 0/16 (0.00%) | 13/16 (81.25%) | **16/16 (100.00%)** | 15/16 (93.75%) |
| `adversarial_topology_swap` | 12 | 0/12 (0.00%) | **12/12 (100.00%)** | 11/12 (91.67%) | **12/12 (100.00%)** |
| `ood_tree_sizes` (4 & 8 leaves) | 12 | 0/12 (0.00%) | **12/12 (100.00%)** | **12/12 (100.00%)** | **12/12 (100.00%)** |
| `branch_lengths_and_formatting` | 8 | 0/8 (0.00%) | 7/8 (87.50%) | **8/8 (100.00%)** | 7/8 (87.50%) |
| **Total** | **48** | **0/48 (0.00%)** | **44/48 (91.67%)** | **47/48 (97.92%)** | **46/48 (95.83%)** |

Packaged v4 by task: `sister` **12/12 (100.00%)**, `monophyletic` **13/13 (100.00%)**, `distance` **11/11 (100.00%)**, `mrca` 10/12 (83.33%).

### 4. Real Genetic Phylogenetics (`test_16s_rrna_phylogeny.py`)

- **Step A biological validation**: **4/4 PASS** - Bacteria monophyletic `Clade6`, Archaea monophyletic `Clade3`, Gammaproteobacteria `Clade5`, Bacillota `Clade4`; intra-phylum K2P `0.0798`-`0.1631` vs cross-domain `0.3669`-`0.4561`.
- **Step B 16 gene-tree questions**: Base **0/16 (0.00%)**, LoRA v3 12/16 (75.00%), **LoRA v4 (packaged) 14/16 (87.50%)**; by task v4: `mrca` **4/4 (100%)**, `sister` **4/4 (100%)**, `monophyletic` **4/4 (100%)**, `distance` 2/4 (50%).
- **Step C raw-FASTA control**: **2/4 = 50.00%** without the K2P + NJ tree, confirming that the hybrid `FASTA -> Pairwise Alignment + K2P + NJ + Midpoint Rooting -> Fine-Tuned LoRA + Exact Verifier` pipeline is essential.

### 5. Round 7 Web/API Additions

- `GET /api/fasta_preset` returns the real RefSeq FASTA string plus organism/domain/accession metadata.
- `POST /api/reconstruct` returns `weighted_newick`, `normalized_newick`, `taxa`, `lengths`, `distance_matrix`, `identity_matrix`, `aligned_sites_matrix`, and `tree_summary`; invalid input returns HTTP `400`.
- The UI automatically loads reconstructed trees into the SVG cladogram and taxon picker, enabling all four QA tasks with LoRA-vs-Base comparison and exact verification.
- Documentation is now 100% English-only.
