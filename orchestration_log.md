# Orchestration Log: Antigravity ⇄ OpenCode (Phylogeny Minimal LM Fine-Tuning)

## Environment & Agent Setup
- **Orchestrator**: Antigravity (planning, code review, GPU execution, error analysis, git auditing)
- **Coding Sub-Agent**: OpenCode CLI `1.19.6` (`$HOME/.opencode/bin/opencode`)
- **Free Cloud Model Used**: `opencode/deepseek-v4.1-flash-free` (OpenCode Zen, `$0` cost, no API key required)
- **Persistent OpenCode Session ID**: `ses_df1a9610dffeLElWJMEScFdKD7` (resumed across all 3 rounds via `opencode run --auto --format json -s <sessionID>`)
- **Base Language Model**: `HuggingFaceTB/SmolLM2-135M-Instruct` (~135M params)
- **Compute & Storage**:
  - GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM, `torch 2.15.0+cu132`, `bfloat16`)
  - Python env: `/home/poriaazadi/miniconda3/envs/phylogeny_lm` (WSL `/`, 922 GB free)
  - HuggingFace cache: `/home/poriaazadi/.cache/huggingface` (261 MB, kept off `C:\`)
  - Project folder size: 65 MB (`phylogeny-lm-test`)

---

## Multi-Round Agent-to-Agent Handoffs

| Round | Direction & Task | OpenCode Session | Git Commit | Verification Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Init** | Antigravity initialized git repo & `.gitignore` | — | `8684957` | Clean repo |
| **Round 1** | **Antigravity → OpenCode**: Author stdlib-only `generate_dataset.py` with independent recursive-descent Newick parser, 3 disjoint taxon pools (`train`/`val`/`test`), and 4 phylogeny tasks (`mrca`, `sister`, `monophyletic`, `distance`). | `ses_df1a9610dffeLElWJMEScFdKD7` | `4eb2244` | `python3 generate_dataset.py --verify` passed: 600 train / 80 val / 80 test, 100% ground-truth verified, 0 overlap. |
| **Round 2** | **Antigravity → OpenCode**: Resume session to author `evaluate.py` (greedy chat-template evaluation) and `train_lora.py` (LoRA on attention + MLP projections with strict prompt-token `-100` label masking). | `ses_df1a9610dffeLElWJMEScFdKD7` | `2cb2c44` | Baseline eval: **0.00%** (0/80). LoRA v1 (`r=16`, 4 epochs, 4.88M trainable params): `val_loss` dropped to `0.18795`, test accuracy jumped to **62.50%** (50/80). |
| **Round 3** | **Antigravity → OpenCode**: Fed Round 2 per-task metrics & off-by-one error analysis (`mrca` 55%, `distance` 35%) back to OpenCode. OpenCode added `--train_only_out` & `--mixed_clade_labels` to `generate_dataset.py` (2,400 train records with 70% `C1..Ck` + 30% synthetic clade names, keeping `val.jsonl`/`test.jsonl` unchanged), fixed `dtype=` deprecation, added best-val-loss checkpointing to `train_lora.py`, and created `demo_unseen_primates.py` (3 real biological trees, 12 runtime-verified questions). | `ses_df1a9610dffeLElWJMEScFdKD7` | `1c7fcec` | LoRA v2 (`r=32`, 6 epochs, best checkpoint epoch 5 `val_loss=0.00876`, 9.77M trainable params): test accuracy jumped to **98.75%** (79/80); unseen biological tree demo jumped from **0/12 (0%)** to **11/12 (91.67%)**. |

---

## Quantitative Comparison

### 1. Held-Out Unseen Test Set (`data/test.jsonl`, 80 examples, 100% disjoint taxa & trees)

| Model | Overall | `mrca` (20) | `sister` (20) | `monophyletic` (20) | `distance` (20) | Best Val Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base `SmolLM2-135M-Instruct` | **0.00%** (0/80) | 0.00% | 0.00% | 0.00% | 0.00% | — |
| Round 2 LoRA v1 (600 train, 4 ep, `r=16`) | **62.50%** (50/80) | 55.00% | 90.00% | 70.00% | 35.00% | `0.18795` |
| Round 3 LoRA v2 (2400 train, 6 ep, `r=32`) | **98.75%** (79/80) | **100.00%** | **100.00%** | **100.00%** | **95.00%** | `0.00876` |

### 2. Unseen Real Biological Trees (`demo_unseen_primates.py`, 12 questions)

Trees tested (never seen during training, real biological taxon & clade names):
1. **Primates**: `(((Homo,Pan)Hominini,Gorilla)Homininae,(Pongo,Hylobates)Pongidae)Hominoidea;`
2. **Carnivores**: `(((Canis,Vulpes)Canidae,(Ursus,Ailuropoda)Ursidae)Caniformia,(Felis,Panthera)Felidae)Carnivora;`
3. **Vertebrates**: `(((Gallus,Alligator)Archosauria,Anolis)Sauropsida,(Mus,Ornithorhynchus)Mammalia)Amniota;`

| Tree | Task | Ground Truth | Base Prediction | LoRA v2 Prediction | Match |
| :--- | :--- | :--- | :--- | :--- | :---: |
| Primates | `mrca` (Homo, Gorilla) | `Homininae` | `The most recent common ancestor of Homo and Gorilla is the genus Homo, which` | `Hominini` | No |
| Primates | `sister` (Homo) | `Pan` | `The direct sister taxon of Homo is Pan.` | `Pan` | Yes |
| Primates | `monophyletic` (Homo, Pan) | `Yes` | `No, the tree is not monophyletic. The tree is a monophy` | `Yes` | Yes |
| Primates | `distance` (Homo, Pongo) | `5` | `The topological edge distance between Homo and Pongo is 1.` | `5` | Yes |
| Carnivores | `mrca` (Canis, Ursus) | `Caniformia` | `The most recent common ancestor of Canis and Ursus is the domestic dog (` | `Caniformia` | Yes |
| Carnivores | `sister` (Canis) | `Vulpes` | `The direct sister taxon of Canis is Felis.` | `Vulpes` | Yes |
| Carnivores | `monophyletic` (Felis, Panthera) | `Yes` | `Yes, the tree is a monophyletic clade. The term "monophy` | `Yes` | Yes |
| Carnivores | `distance` (Canis, Felis) | `5` | `The topological edge distance between Canis and Felis is 1.` | `5` | Yes |
| Vertebrates | `mrca` (Gallus, Anolis) | `Sauropsida` | `The most recent common ancestor of Gallus and Anolis is the Anolis species` | `Sauropsida` | Yes |
| Vertebrates | `sister` (Mus) | `Ornithorhynchus` | `The direct sister taxon of Mus is Anolis.` | `Ornithorhynchus` | Yes |
| Vertebrates | `monophyletic` (Mus, Ornithorhynchus) | `Yes` | `(Gallus,Alligator)Archosauria,Anolis)Sau` | `Yes` | Yes |
| Vertebrates | `distance` (Gallus, Mus) | `5` | `Gallus is a member of the family Gallidae, which includes the Anolis` | `5` | Yes |

- **Base Model Total**: `0 / 12 (0.00%)`
- **Fine-Tuned LoRA v2 Total**: `11 / 12 (91.67%)`
