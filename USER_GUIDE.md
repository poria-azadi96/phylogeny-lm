# PhyloScope User Guide

Welcome to **PhyloScope**, an interactive phylogenetic tree workbench and 16S rRNA gene-tree analyzer powered by **PhyloClade-135M** and an exact algorithmic verifier.

---

## 1. What PhyloScope Does

PhyloScope supports two complementary workflows:

1. **Newick Tree Mode**: Parse, normalize, visualize, and query any rooted binary Newick tree.
2. **FASTA / 16S rRNA Sequence Mode**: Reconstruct a real evolutionary gene tree from raw DNA sequences (pairwise alignment, Kimura 2-Parameter distances, Saitou-Nei Neighbor-Joining, and midpoint rooting), inspect the distance matrix, and query the resulting tree.

For every query, PhyloScope runs:

- **PhyloClade-135M** (a specialized 135M-parameter phylogenetic reasoning model)
- **Untuned Baseline** (optional reference run showing untuned behavior)
- **Exact Algorithmic Verifier** (deterministic ground truth computed directly from the normalized topology)

### Supported Phylogenetic Tasks

| Task | Description | Required Taxa | Example Question |
| :--- | :--- | :--- | :--- |
| `mrca` | Most Recent Common Ancestor | 2 distinct leaves | What is the MRCA of `Escherichia_coli` and `Pseudomonas_aeruginosa`? |
| `sister` | Direct Sister Taxon (cherry siblings only) | 1 leaf inside a cherry | What is the direct sister taxon of `Homo`? |
| `monophyletic` | Monophyletic Clade Check | 2 or 3 distinct leaves | Do `Alligator` and `Anolis` form a monophyletic clade? |
| `distance` | Topological Edge Distance | 2 distinct leaves | What is the edge distance between `Bacillus_subtilis` and `Halobacterium_salinarum`? |

---

## 2. Installation & Requirements

### Prerequisites
- Python 3.10+
- PyTorch (CUDA GPU recommended for fast inference; CPU works too)
- Dependencies listed in `requirements.txt`: `torch`, `transformers`, `peft`, `accelerate`, `datasets`, `biopython`, `numpy`

### Setup

```bash
git clone https://github.com/poria-azadi96/phylogeny-lm.git
cd phylogeny-lm
pip install -r requirements.txt
```

> **Note:** On the first run, the base tokenizer/weights referenced by `model/phyloclade-135m/adapter_config.json` will be cached automatically if not already present locally.

---

## 3. Launching the Web Workbench

Start the local HTTP server from the repository root:

```bash
python app.py --port 7860
```

Then open your browser to:

```text
http://127.0.0.1:7860
```

### Useful Server Flags

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--host` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on LAN) |
| `--port` | `7860` | Listening HTTP port |
| `--adapter_dir` | `model/phyloclade-135m` | Path to the packaged PhyloClade-135M model directory |
| `--max_new_tokens` | `24` | Maximum generated tokens (kept at 24 to support long `Genus_species` names) |
| `--no_base` | `false` | Disable the Untuned Baseline comparison for ~2x faster responses |

---

## 4. Mode 1: Newick Tree Workflow

1. **Choose a Preset or Paste Newick**:
   - Click any built-in preset (e.g., *Primates*, *Carnivores*, *Vertebrates Paraphyly Trap*, *Adversarial Counterfactual Primates*), or paste your own rooted binary Newick string into the input box.
   - Branch lengths (e.g. `:0.012`) and unlabeled internal nodes are automatically normalized and labeled (`Clade1`, `Clade2`, ..., `Root`).
2. **Inspect the Interactive SVG Cladogram**:
   - Click **Parse & Visualize**. The right-hand panel renders a live SVG cladogram.
   - Leaves marked with a cyan indicator dot belong to a cherry and are valid targets for `sister` queries.
3. **Select Task & Taxa**:
   - Pick `MRCA`, `Sister Taxon`, `Monophyletic Clade`, or `Edge Distance`.
   - Click taxa directly on the SVG cladogram or via the taxon pills.
4. **Run Inference**:
   - Click **Run Inference** to see:
     - **Exact Algorithmic Verifier** ground truth and match status
     - **PhyloClade-135M** prediction and latency
     - **Untuned Baseline** comparison
     - **Canonical Prompt & Normalized Newick** (expandable inspection drawer)

---

## 5. Mode 2: FASTA / 16S rRNA Sequence Workflow

1. Switch the top-left toggle to **FASTA / 16S rRNA Sequences**.
2. **Load Sequences**:
   - Click **Load Real NCBI 16S rRNA (Archaea & Bacteria)** to load 8 verified RefSeq 16S rRNA genes (~1,321-1,552 bp each across Bacteria and Archaea), **or** paste at least 4 aligned/unaligned DNA sequences in FASTA format (`>Species_Name` followed by nucleotide bases).
3. **Reconstruct Gene Tree**:
   - Click **Reconstruct Gene Tree (K2P + NJ)**.
   - PhyloScope performs pairwise global alignment, computes the Kimura 2-Parameter (K2P) substitution distance matrix, builds an unrooted tree via Saitou-Nei Neighbor-Joining, and midpoint-roots it.
   - The K2P distance matrix appears in the scrollable table, and the reconstructed tree automatically loads into the SVG cladogram.
4. **Query the Reconstructed Tree**:
   - Select any task (`mrca`, `sister`, `monophyletic`, `distance`) and taxa on the reconstructed tree, then click **Run Inference**.

---

## 6. Command-Line Evaluation & API Usage

### Run Benchmark Suites

```bash
# Real NCBI 16S rRNA biological validation + PhyloClade-135M evaluation
python test_16s_rrna_phylogeny.py --adapter_dir model/phyloclade-135m --max_new_tokens 24

# 48-case adversarial & OOD stress test
python stress_test_phylogeny.py --adapter_dir model/phyloclade-135m --max_new_tokens 24

# Held-out test split (80 unseen trees)
python evaluate.py --adapter_dir model/phyloclade-135m --split data/test.jsonl --max_new_tokens 24
```

### HTTP API Example (`curl`)

**1) Reconstruct a tree from FASTA:**

```bash
curl -s -X POST http://127.0.0.1:7860/api/reconstruct \
  -H "Content-Type: application/json" \
  -d '{"fasta": ">Escherichia_coli\nAGAGTTTG...\n>Pseudomonas_aeruginosa\nAGAGTTTG...\n>Bacillus_subtilis\nAGAGTTTG...\n>Halobacterium_salinarum\nATTCCGGT..."}'
```

**2) Run inference on a Newick tree:**

```bash
curl -s -X POST http://127.0.0.1:7860/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "newick": "(((Homo,Pan)Hominini,Gorilla)Homininae,(Pongo,Hylobates)Pongidae)Hominoidea;",
    "task": "mrca",
    "taxa": ["Homo", "Gorilla"]
  }'
```

---

## 7. Limitations & Troubleshooting

- **Rooted Binary Trees Only**: Newick mode requires a strictly binary rooted tree. Multifurcations (polytomies with 3+ children at one node) will be rejected by the parser with a clear error message.
- **Minimum Sequences for FASTA**: Neighbor-Joining reconstruction requires at least 4 distinct FASTA records with valid nucleotide sequences.
- **Why FASTA Reconstruction Runs Before LM Inference**: A 135M language model cannot accurately align 1,500-bp raw DNA sequences in context (`50%` accuracy on raw-FASTA control). PhyloScope therefore delegates alignment and evolutionary distance modeling to `bio_phylo.py` and uses PhyloClade-135M strictly for structural tree reasoning, always verified by `phylogeny_utils.py`.
- **Port Already in Use**: Start the server on another port with `python app.py --port 7861`.
- **Slow CPU Inference**: Use `--no_base` to skip the Untuned Baseline pass when running without a CUDA GPU.

---

## 8. Standalone Offline Apps (Windows `.exe` & Android `.apk`)

If you prefer a **100% offline, zero-setup** experience without installing Python or running a local server manually, download the prebuilt packages from **[GitHub Releases v1.0](https://github.com/poria-azadi96/phylogeny-lm/releases/tag/v1.0)**:

### Windows (`PhyloScope-Windows-x64.zip`)
1. Download and extract `PhyloScope-Windows-x64.zip`.
2. Double-click `PhyloScope.exe`.
3. It automatically starts the local `PhyloClade-135M (Q8_0)` GGUF engine in the background and opens the PhyloScope workbench in your browser.
4. Close the `PhyloScope Offline` control window when finished to cleanly shut down the local engine.

### Android (`PhyloScope-v1.0-arm64.apk`)
1. Download `PhyloScope-v1.0-arm64.apk` on an `arm64-v8a` Android device and install it (allow installation from your browser/file manager if prompted).
2. On first launch, wait a few seconds while the embedded `PhyloClade-135M (Q4_K_M)` native C++ engine initializes.

### Offline App Capabilities (Windows & Android)
- **Database Link / Accession Downloader**: Paste a direct FASTA URL, an NCBI page link (e.g., `https://www.ncbi.nlm.nih.gov/nuccore/NR_114042.1`), or comma-separated NCBI Accession IDs (e.g., `NR_114042.1, NR_117678.1, NR_102783.2, NR_037007.2`) and click **Download from Database**.
- **Local File & Offline Preset**: Open `.fasta`/`.fna` files from disk/phone or click **Load Built-in 16S rRNA (Offline)**.
- **Tree Reconstruction & Export**: Reconstruct K2P + Neighbor-Joining + Midpoint-rooted trees offline, inspect the K2P matrix and SVG cladogram, and export `.nwk` or `.svg` files.
- **Offline AI + Exact Verifier**: Query `MRCA`, `Sister Taxon`, `Monophyletic Clade`, and `Edge Distance` completely offline with side-by-side deterministic verification.

