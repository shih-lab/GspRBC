# GspRBC

Analysis code accompanying **"Rational rubisco engineering through mutational scanning and phylogenetic inference."**

This repository contains the scripts and data used to design, compare, and statistically evaluate engineered Rubisco variants, including sequence-alignment-based mutant design, alanine-scanning kinetics analysis, cross-species fitness mapping, and phylogenetic Shannon entropy analysis.

**Key naming convention:** `4_9` = MmRBC, `4_10` = GspRBC

Code was generated and refined with the assistance of ChatGPT and Claude.

## Repository structure

```
GspRBC/
├── 4_9x Design/
│   ├── Code/                  Parses Clustal alignments and identifies sequence
│   │                          differences to guide MmRBC (4_9) variant design
│   └── Data Files/            Alignment outputs, difference tables, mutated
│                              sequence sets (4_9x Design Round 3, etc.)
│
├── RrRBC and GspRBC Comparison/
│   ├── GspRBC and RrRBC comparison.py   Maps alanine-scan fitness values from
│   │                                    GspRBC and RrRBC onto a shared sequence
│   │                                    alignment for direct positional comparison
│   └── Data Files/            Alignment (FASTA), per-residue Vmax/kcat tables,
│                              disagreement analysis between the two enzymes
│
├── Shannon Entropy/
│   ├── Shannon Entropy and kcat Layering.py   Computes per-position Shannon
│   │                                          entropy across a phylogenetic
│   │                                          alignment and layers it against
│   │                                          measured kcat to flag positions
│   │                                          associated with catalytic speed
│   └── Data Files/            Aligned FASTA of homologs, entropy/kcat tables
│
└── Statistical Analysis GspRBC Alanine Scan/
    ├── Code/                  Statistical comparisons (e.g. WT-on-the-day vs.
    │                          backbone control vs. variant) on alanine-scan
    │                          kinetics data
    └── Data Files/            Raw absorbance values, consolidated kinetics
                               data, normalized alanine-scan output
```

## Requirements

Scripts are written in Python 3 and rely on:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scipy`, `scikit-learn`
- `biopython` (`Bio.AlignIO`, `Bio.SeqIO`)

Install with:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn biopython
```

## Usage

Each script is a Python-file export of a Jupyter notebook (visible from the `# In[N]:` cell markers throughout). Only the **first cell** of each file is a self-contained, standalone-runnable demo using the data bundled in this repo — later cells in the same files contain additional analysis from the full project and reference files from the original working environment that aren't included here. The Demo section below documents only the verified, reproducible first-cell portion of each script.

Paths in the first cell of each script now resolve relative to the script's own location, so each demo below can be run directly from a clone of this repo without editing any paths.

## Demo

All four demos below were run and timed on a standard desktop-class environment (Python 3.12, no GPU).

### 1. `4_9x Design/Code/4_9x Design.py` (first cell only)

**Run:**
```bash
cd "4_9x Design/Code"
python "4_9x Design.py"
```

**Expected output:** Parses the bundled Clustal alignment and writes two files into `4_9x Design/Data Files/`:
- `clustalo-...-p1m_differences.csv` — one row per alignment position, one column per protein (RBC4.9, RBC2.3, RBC4.11, RBC4.10, 13), showing the residue at each differing position
- `clustalo-...-p1m_differences.xlsx` — the same table, formatted

The script also prints a summary count of differing positions and a preview of the first 5 rows to the terminal.

**Expected run time:** <1 second

---

### 2. `RrRBC and GspRBC Comparison/GspRBC and RrRBC comparison.py` (first cell only)

**Run:**
```bash
cd "RrRBC and GspRBC Comparison"
python "GspRBC and RrRBC comparison.py"
```

**Expected output:** Maps GspRBC and RrRBC alanine-scan fitness values onto their shared sequence alignment and writes `fitness_parallel_mapped_blocks_BY_ALIGNMENT.svg` into the same folder — a figure showing per-position fitness effects for both enzymes side by side, aligned by position.

**Expected run time:** ~6 seconds

---

### 3. `Shannon Entropy/Shannon Entropy and kcat Layering.py` (first cell only)

**Run:**
```bash
cd "Shannon Entropy"
python "Shannon Entropy and kcat Layering.py"
```

**Expected output:** Computes per-position Shannon entropy across the bundled homolog alignment, fits a Gaussian Mixture Model to the delta-entropy distribution, and writes three files into `Shannon Entropy/Data Files/`:
- `delta_entropy_GMM_variable_positions.svg` — GMM fit diagnostic plot
- `delta_entropy_top_quarter_candidates.csv` — top-quartile variable positions by delta entropy
- `delta_entropy_variable_positions_quantiled.csv` — full set of variable positions with quantile bins

Also prints the GMM component breakdown (means/variances/weights) and a preview of the top 20 candidate positions to the terminal.

**Expected run time:** ~6 seconds

---

### 4. `Statistical Analysis GspRBC Alanine Scan/Code/Statistical Analysis Alanine Scan Data.py` (first cell only)

**Run:**
```bash
cd "Statistical Analysis GspRBC Alanine Scan/Code"
python "Statistical Analysis Alanine Scan Data.py"
```

**Expected output:** Compares WT-on-the-day vs. backbone-control kcat distributions from the bundled kinetics data. Prints medians, standard deviations, a t-test, Shapiro-Wilk normality tests, and a Mann-Whitney U test (with plain-language significance calls), then repeats summary statistics for untagged variants. Displays histogram, Q-Q, and boxplot figures (no files are written by this cell — plots are shown interactively; run in a Jupyter environment or add `plt.savefig(...)` calls to save them headlessly).

**Expected run time:** ~3 seconds


## Citation

If you use this code, please cite the associated manuscript (citation to be added upon publication).

## Versions

Python 3.10.9
pandas: 1.5.3
numpy: 1.24.3
matplotlib: 3.6.3
seaborn: 0.13.2
scipy: 1.10.0
scikit-learn: 1.7.1
biopython: 1.82
MAC, 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:38:00 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T8103 arm64


