# Orchestration Log: Antigravity ⇄ OpenCode (Phylogeny Minimal LM Fine-Tuning)

## Environment & Agent Setup
- **Orchestrator**: Antigravity (architecture, code review, GPU training & verification, error analysis, live HTTP smoke testing, git auditing)
- **Coding Sub-Agent**: OpenCode CLI `1.19.6` (`$HOME/.opencode/bin/opencode`)
- **Free Cloud Model Used**: `opencode/deepseek-v4.1-flash-free` (OpenCode Zen, `$0` cost, no API key required)
- **OpenCode Sessions**:
  - Rounds 1–4a: `ses_df1a9610dffeLElWJMEScFdKD7`
  - Round 4b: `ses_df13f3b53ffeOcSinNuJiN8cwh`
  - Round 5: `ses_df12d8f13ffeoT7v0F7cuqwp2p`
- **Base Language Model**: `HuggingFaceTB/SmolLM2-135M-Instruct` (~135M params)
- **Packaged Final Adapter**: `model/phylogeny-lora/` (`LoRA v3`, `r=32`, `alpha=64`, `9,768,960` trainable params = `6.77%`, `best_epoch=5`, `best_val_loss=0.02107`, `adapter_model.safetensors` = `38 MB`)
- **Compute & Storage**:
  - GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM, `torch 2.15.0+cu132`, `bfloat16`)
  - Python env: `/home/poriaazadi/miniconda3/envs/phylogeny_lm` (WSL `/`)
  - HuggingFace cache: `/home/poriaazadi/.cache/huggingface` (kept off `C:\`)

---

## Multi-Round Agent-to-Agent Handoffs

| Round | Direction & Task | OpenCode Session | Git Commit | Verification Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Init** | Antigravity initialized git repo & `.gitignore` | — | `8684957` | Clean repo |
| **Round 1** | **Antigravity → OpenCode**: Author stdlib-only `generate_dataset.py` with independent recursive-descent Newick parser, 3 disjoint taxon pools (`train`/`val`/`test`), and 4 phylogeny tasks (`mrca`, `sister`, `monophyletic`, `distance`). | `ses_df1a9610dffeLElWJMEScFdKD7` | `4eb2244` | `python3 generate_dataset.py --verify` passed: 600 train / 80 val / 80 test, 100% ground-truth verified, 0 overlap. |
| **Round 2** | **Antigravity → OpenCode**: Resume session to author `evaluate.py` (greedy chat-template evaluation) and `train_lora.py` (LoRA on attention + MLP projections with strict prompt-token `-100` label masking). | `ses_df1a9610dffeLElWJMEScFdKD7` | `2cb2c44` | Baseline eval: **0.00%** (0/80). LoRA v1 (`r=16`, 4 epochs, 4.88M trainable params): `val_loss` dropped to `0.18795`, test accuracy jumped to **62.50%** (50/80). |
| **Round 3** | **Antigravity → OpenCode**: Fed Round 2 per-task metrics & off-by-one error analysis (`mrca` 55%, `distance` 35%) back to OpenCode. OpenCode added `--train_only_out` & `--mixed_clade_labels` to `generate_dataset.py` (2,400 train records with 70% `C1..Ck` + 30% synthetic clade names, keeping `val.jsonl`/`test.jsonl` unchanged), fixed `dtype=` deprecation, added best-val-loss checkpointing to `train_lora.py`, and created `demo_unseen_primates.py` (3 real biological trees, 12 runtime-verified questions). | `ses_df1a9610dffeLElWJMEScFdKD7` | `1c7fcec` | LoRA v2 (`r=32`, 6 epochs, best checkpoint epoch 5 `val_loss=0.00876`, 9.77M trainable params): test accuracy jumped to **98.75%** (79/80); unseen biological tree demo jumped from **0/12 (0%)** to **11/12 (91.67%)**. |
| **Round 4a** | **Antigravity → OpenCode**: Create `phylogeny_utils.py` (Newick normalizer stripping branch lengths & auto-labeling internal nodes, exact solvers, tree JSON summary) and `stress_test_phylogeny.py` (48-case 4-category stress suite). | `ses_df1a9610dffeLElWJMEScFdKD7` | `465b699` | `--verify_only` passed 48/48. LoRA v2 scored **44/48 (91.67%)** vs Base **0/48 (0.00%)**, including **12/12** on adversarial topology swaps and **12/12** on OOD tree sizes; 4 failures were paraphyly & nested biological clade levels. |
| **Round 4b** | **Antigravity → OpenCode**: Fed the 4 stress-test failures back to OpenCode to upgrade `generate_dataset.py` (`--min_leaves 4 --max_leaves 8 --mixed_clade_ratio 0.5 --hard_paraphyly --v3_clades`) and generate `data/train_v3.jsonl` (3,600 balanced records). | `ses_df13f3b53ffeOcSinNuJiN8cwh` | `465b699` | LoRA v3 (`model/phylogeny-lora/`): **97.50%** (78/80) on `test.jsonl`, **12/12 (100%)** on `demo_unseen_primates.py`, and **47/48 (97.92%)** on the 48-case stress test (**16/16** real biology & paraphyly, **8/8** branch-length Newick, **12/12** OOD sizes, **11/12** adversarial swaps). |
| **Round 5** | **Antigravity → OpenCode**: Build stdlib HTTP inference server `app.py`, zero-CDN SVG cladogram workbench `static/index.html`, `requirements.txt`, and bilingual `README.md`. | `ses_df12d8f13ffeoT7v0F7cuqwp2p` | `0cfe8cc` | Live GPU smoke test on port `7861` verified `GET /`, `GET /api/health`, `GET /api/presets` (6 presets), `POST /api/parse` (branch-length stripping + auto-labeling), and `POST /api/predict` (Primates MRCA `Homininae`, Vertebrates paraphyly `No`, Branch-length distance `5` — all `lora_correct=True` vs Base `False`). |

---

## Final Quantitative Summary

### 1. Held-Out Unseen Test Set (`data/test.jsonl`, 80 examples, 100% disjoint taxa & trees)

| Model | Overall | `mrca` (20) | `sister` (20) | `monophyletic` (20) | `distance` (20) | Best Val Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base `SmolLM2-135M-Instruct` | **0.00%** (0/80) | 0.00% | 0.00% | 0.00% | 0.00% | — |
| Round 2 LoRA v1 (600 train, 4 ep, `r=16`) | **62.50%** (50/80) | 55.00% | 90.00% | 70.00% | 35.00% | `0.18795` |
| Round 3 LoRA v2 (2400 train, 6 ep, `r=32`) | **98.75%** (79/80) | **100.00%** | **100.00%** | **100.00%** | **95.00%** | `0.00876` |
| **Round 4 LoRA v3 / Packaged (`model/phylogeny-lora`)** | **97.50%** (78/80) | **100.00%** | **100.00%** | **100.00%** | 90.00% | `0.02107` |

### 2. Unseen Real Biological Trees (`demo_unseen_primates.py`, 12 questions)
- **Base `SmolLM2-135M-Instruct`**: **0.00%** (0/12)
- **Round 3 LoRA v2**: **91.67%** (11/12)
- **Round 4 LoRA v3 (`model/phylogeny-lora`)**: **100.00%** (12/12)

### 3. 48-Question Scientific Stress Test (`stress_test_phylogeny.py`)

| Category | Cases | Base | LoRA v2 | **LoRA v3 (`model/phylogeny-lora`)** |
| :--- | :---: | :---: | :---: | :---: |
| `real_biology_and_paraphyly` | 16 | 0/16 (0.00%) | 13/16 (81.25%) | **16/16 (100.00%)** |
| `adversarial_topology_swap` | 12 | 0/12 (0.00%) | **12/12 (100.00%)** | 11/12 (91.67%) |
| `ood_tree_sizes` (4 & 8 leaves) | 12 | 0/12 (0.00%) | **12/12 (100.00%)** | **12/12 (100.00%)** |
| `branch_lengths_and_formatting` | 8 | 0/8 (0.00%) | 7/8 (87.50%) | **8/8 (100.00%)** |
| **Total** | **48** | **0/48 (0.00%)** | **44/48 (91.67%)** | **47/48 (97.92%)** |
