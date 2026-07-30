Analysis code accompanying "Rational rubisco engineering through mutational scanning and phylogenetic inference."

This repository contains the scripts and data used to design, compare, and statistically evaluate engineered Rubisco variants, including sequence-alignment-based mutant design, alanine-scanning kinetics analysis, cross-species fitness mapping, and phylogenetic Shannon entropy analysis.

Key naming convention: 4_9 = MmRBC, 4_10 = GspRBC

Code was generated and refined with the assistance of Claude.

Repository structure
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

Requirements

Scripts are written in Python 3 and rely on:
 pandas, numpy
matplotlib, seaborn
scipy, scikit-learn
biopython (Bio.AlignIO, Bio.SeqIO)

Install with:
bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn biopython

Usage

Each script is self-contained within its folder and reads from/writes to the adjacent Data Files/ directory. Several scripts (e.g. Shannon Entropy and kcat Layering.py) use placeholder or local file paths (pathname, absolute paths from the original analysis environment) — update these to point to your local copy of the Data Files/ folder before running.

Citation

If you use this code, please cite the associated manuscript (citation to be added upon publication).
