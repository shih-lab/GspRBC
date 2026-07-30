#!/usr/bin/env python
# coding: utf-8

# In[75]:


# === Imports ===
import os
from Bio import AlignIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# === File paths ===
# Resolves relative to this script's location, so it works regardless of
# the directory the script is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data Files")
alignment_path = os.path.join(DATA_DIR, 'RRand4_10aligned.fasta')
rr_csv = os.path.join(DATA_DIR, 'Rr Alanine Vmax and Residue.csv')
_410_csv = os.path.join(DATA_DIR, 'GspRBC alanine_scan_normalized_output.csv')
output_path = os.path.join(SCRIPT_DIR, "fitness_parallel_mapped_blocks_BY_ALIGNMENT.svg")

# === Column references ===
rr_pos_col = "position"
rr_val_col = "Vmax_median"
_410_pos_col = "#"
_410_val_col = "normalized_kcat_bc"

# === Load alignment ===
alignment = AlignIO.read(alignment_path, "fasta")
seq_dict = {record.id: str(record.seq) for record in alignment}
rr_seq = seq_dict.get("Rr")
_410_seq = seq_dict.get("4_10")
if rr_seq is None or _410_seq is None:
    raise ValueError("Both Rr and 4_10 must be present in the alignment.")

# === Map alignment positions to sequence positions ===
rr_map, _410_map = [], []
rr_counter, _410_counter = 1, 1
for aa_rr, aa_410 in zip(rr_seq, _410_seq):
    rr_map.append(rr_counter if aa_rr != '-' else None)
    _410_map.append(_410_counter if aa_410 != '-' else None)
    if aa_rr != '-': rr_counter += 1
    if aa_410 != '-': _410_counter += 1

alignment_df = pd.DataFrame({
    "alignment_pos": range(1, len(rr_seq) + 1),
    "Rr_seq_pos": rr_map,
    "4_10_seq_pos": _410_map,
    "Rr_aa": list(rr_seq),
    "4_10_aa": list(_410_seq)
})

# === Load fitness data ===
rr_df = pd.read_csv(rr_csv)
_410_df = pd.read_csv(_410_csv)

# === Merge onto alignment ===
merged = alignment_df.merge(rr_df[[rr_pos_col, rr_val_col]], left_on="Rr_seq_pos", right_on=rr_pos_col, how="left")
merged = merged.merge(_410_df[[_410_pos_col, _410_val_col]], left_on="4_10_seq_pos", right_on=_410_pos_col, how="left")
merged.drop(columns=[rr_pos_col, _410_pos_col], inplace=True)

# === Create blocks ===
block_size = 53
position_index = list(zip(
    merged["alignment_pos"],
    merged["Rr_aa"],
    merged["4_10_aa"],
    merged["Vmax_median"],
    merged["normalized_kcat_bc"]
))
blocks = [position_index[i:i + block_size] for i in range(0, len(position_index), block_size)]

# === Define emphasize_white colormap truncator ===
def emphasize_white(cmap, minval=0.0, maxval=0.7, n=100):
    return LinearSegmentedColormap.from_list(
        f'{cmap.name}_truncated',
        cmap(np.linspace(minval, maxval, n))
    )

# === Normalization and colormaps ===
rr_norm = mcolors.Normalize(vmin=merged["Vmax_median"].min(), vmax=merged["Vmax_median"].max())
_410_norm = mcolors.Normalize(vmin=merged["normalized_kcat_bc"].min(), vmax=merged["normalized_kcat_bc"].max())
rr_cmap = emphasize_white(cm.Blues_r, 0.0, 1.5)
_410_cmap = emphasize_white(cm.Reds_r, 0.0, 1.5)


# === Plotting ===
fig, axs = plt.subplots(len(blocks) * 3, 1,
                        figsize=(block_size * 0.3, len(blocks) * 2.2),
                        gridspec_kw={'height_ratios': [0.05, 0.05, 0.075] * len(blocks)})
fig.subplots_adjust(hspace=0.2)


for block_index, block in enumerate(blocks):
    rr_colors = [rr_cmap(rr_norm(val)) if pd.notna(val) else (0, 0, 0, 1) for _, _, _, val, _ in block]
    _410_colors = [_410_cmap(_410_norm(val)) if pd.notna(val) else (0, 0, 0, 1) for _, _, _, _, val in block]

    # Rr fitness row
    axs[block_index * 3].imshow([rr_colors], aspect="equal")
    axs[block_index * 3].set_yticks([0])
    axs[block_index * 3].set_yticklabels(["Rr Fitness"])
    axs[block_index * 3].set_xticks([])
    axs[block_index * 3].set_xlim(-0.5, len(block) - 0.5)

    # 4_10 fitness row
    axs[block_index * 3 + 1].imshow([_410_colors], aspect="equal")
    axs[block_index * 3 + 1].set_yticks([0])
    axs[block_index * 3 + 1].set_yticklabels(["4_10 Fitness"])
    axs[block_index * 3 + 1].set_xticks([])
    axs[block_index * 3 + 1].set_xlim(-0.5, len(block) - 0.5)

    # Sequence labels
    axs[block_index * 3 + 2].set_xlim(-0.5, len(block) - 0.5)
    axs[block_index * 3 + 2].set_ylim(-1.3, 1.3)
    axs[block_index * 3 + 2].axis("off")
    for i, (pos, aa_rr, aa_410, _, _) in enumerate(block):
        axs[block_index * 3 + 2].text(i, 0.9, aa_rr, ha='center', va='center',
                                  fontsize=10, fontweight='bold', family='monospace', color='blue')
        axs[block_index * 3 + 2].text(i, -0.015, aa_410, ha='center', va='center',
                                  fontsize=10, fontweight='bold', family='monospace', color='darkred')
        axs[block_index * 3 + 2].text(i, -0.8, str(pos), ha='center', va='center',
                                  fontsize=8, color='black')


# === Colorbars ===
sm_rr = cm.ScalarMappable(cmap=rr_cmap, norm=rr_norm)
sm_410 = cm.ScalarMappable(cmap=_410_cmap, norm=_410_norm)
cbar_ax1 = fig.add_axes([0.92, 0.5, 0.010, 0.25])
cbar_ax2 = fig.add_axes([0.92, 0.2, 0.010, 0.25])
fig.colorbar(sm_rr, cax=cbar_ax1, orientation='vertical', label='Rr Fitness')
fig.colorbar(sm_410, cax=cbar_ax2, orientation='vertical', label='4_10 Fitness')

# === Save and show (same layout for both) ===
plt.savefig(output_path, format="svg", dpi=300)
plt.show()
print(f"✅ Exported figure saved to: {output_path}")


# In[43]:


#!/usr/bin/env python3
"""
Rubisco Alanine Scan Disagreement Analysis
==========================================

This script compares fitness effects (Rr) vs catalytic activity (4_10) 
using sequence alignment to identify positions where organismal fitness 
disagrees with biochemical measurements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import AlignIO
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (18, 12)
plt.rcParams['font.size'] = 10

# File paths
ALIGNMENT_PATH = 'RRand4_10aligned.fasta'
RR_CSV = 'Rr Alanine Vmax and Residue.csv'
FOURTENTEN_CSV = 'GspRBC alanine_scan_normalized_output.csv'

# Column mappings
RR_POSITION_COL = "position"
RR_VALUE_COL = "Vmax_median"
FOURTENTEN_POSITION_COL = "#"
FOURTENTEN_VALUE_COL = "normalized_kcat_bc"

# Reference value for Rr normalized kcat
RR_NORMALIZED_KCAT = 0.371744105

# Analysis parameters
TOP_N_POSITIONS = 50
LOW_KCAT_THRESHOLD = -0.1
FITNESS_ADVANTAGE_THRESHOLD = 0.2

def load_and_align_data():
    """Load alignment and data files, create merged dataset."""
    print("Loading alignment and data files...")
    
    # Load alignment
    alignment = AlignIO.read(ALIGNMENT_PATH, "fasta")
    seq_dict = {record.id: str(record.seq) for record in alignment}
    
    rr_seq = seq_dict.get("Rr")
    fourtenten_seq = seq_dict.get("4_10")
    
    if rr_seq is None or fourtenten_seq is None:
        print(f"Available sequence IDs: {list(seq_dict.keys())}")
        raise ValueError("Both Rr and 4_10 must be present in the alignment.")
    
    print(f"Loaded alignment: Rr ({len(rr_seq)} chars) vs 4_10 ({len(fourtenten_seq)} chars)")
    
    # Map alignment positions to sequence positions
    rr_map, fourtenten_map = [], []
    rr_counter, fourtenten_counter = 1, 1
    
    for aa_rr, aa_410 in zip(rr_seq, fourtenten_seq):
        rr_map.append(rr_counter if aa_rr != '-' else np.nan)
        fourtenten_map.append(fourtenten_counter if aa_410 != '-' else np.nan)
        if aa_rr != '-': 
            rr_counter += 1
        if aa_410 != '-': 
            fourtenten_counter += 1
    
    alignment_df = pd.DataFrame({
        "alignment_pos": range(1, len(rr_seq) + 1),
        "Rr_seq_pos": rr_map,
        "4_10_seq_pos": fourtenten_map,
        "Rr_aa": list(rr_seq),
        "4_10_aa": list(fourtenten_seq)
    })
    
    print(f"Created alignment mapping: {len(alignment_df)} alignment positions")
    
    # Load data files
    rr_df = pd.read_csv(RR_CSV)
    fourtenten_df = pd.read_csv(FOURTENTEN_CSV)
    
    print(f"Loaded Rr data: {len(rr_df)} rows")
    print(f"Loaded 4_10 data: {len(fourtenten_df)} rows")
    
    # Merge data onto alignment
    merged = alignment_df.merge(
        rr_df[[RR_POSITION_COL, RR_VALUE_COL]], 
        left_on="Rr_seq_pos", right_on=RR_POSITION_COL, how="left"
    )
    merged = merged.merge(
        fourtenten_df[[FOURTENTEN_POSITION_COL, FOURTENTEN_VALUE_COL]], 
        left_on="4_10_seq_pos", right_on=FOURTENTEN_POSITION_COL, how="left"
    )
    merged.drop(columns=[RR_POSITION_COL, FOURTENTEN_POSITION_COL], inplace=True)
    
    # Keep only rows with both values
    valid_data = merged.dropna(subset=[RR_VALUE_COL, FOURTENTEN_VALUE_COL])
    
    print(f"Successfully aligned {len(valid_data)} positions with both datasets")
    
    return valid_data

def identify_disagreements(data, top_n=TOP_N_POSITIONS):
    """Identify positions with largest disagreements between fitness and kcat."""
    data = data.copy()
    data['difference'] = data[RR_VALUE_COL] - data[FOURTENTEN_VALUE_COL]
    data['abs_difference'] = data['difference'].abs()
    
    # Sort ALL positions by disagreement for complete export
    all_positions_ranked = data.sort_values('abs_difference', ascending=False)
    
    # Identify biologically interesting positions
    interesting = data[
        (data[FOURTENTEN_VALUE_COL] < LOW_KCAT_THRESHOLD) &
        (data['difference'] > FITNESS_ADVANTAGE_THRESHOLD)
    ].copy()
    
    # Get top disagreements for plotting
    top_disagreements = data.nlargest(top_n, 'abs_difference')
    
    # Combine and prioritize interesting positions for plotting
    if len(interesting) > 0:
        combined = pd.concat([interesting, top_disagreements]).drop_duplicates(subset=['alignment_pos'])
        result = combined.nlargest(top_n, 'abs_difference')
    else:
        result = top_disagreements
    
    print(f"\nDisagreement Analysis Summary:")
    print(f"  Total valid positions: {len(data)}")
    print(f"  Biologically interesting positions: {len(interesting)}")
    print(f"  Top disagreements selected for plot: {len(result)}")
    
    return result, interesting, all_positions_ranked

def create_disagreement_plot(disagreement_data, title_suffix=""):
    """Create the main disagreement visualization."""
    
    n_positions = len(disagreement_data)
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, max(8, n_positions * 0.3)), 
        gridspec_kw={'width_ratios': [3, 1, 3], 'wspace': 0.1}
    )
    
    # Sort data for display
    plot_data = disagreement_data.sort_values('abs_difference', ascending=True)
    y_positions = np.arange(len(plot_data))
    
    # Amino acid conversion dictionary
    aa_dict = {
        'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
        'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
        'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
        'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'
    }
    
    # Left plot: Rr fitness (bars point right)
    ax1.barh(y_positions, plot_data[RR_VALUE_COL], 
             color='blue', alpha=0.3)
    ax1.set_xlabel('Rr Fitness (Vmax_median)', fontweight='bold')
    ax1.set_title('Rr Organismal Fitness', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax1.set_ylim(-0.5, len(plot_data) - 0.5)
    ax1.invert_xaxis()  # Flip to point toward center
    
    # Add value labels on bars
    for i in range(len(plot_data)):
        row = plot_data.iloc[i]
        value = row[RR_VALUE_COL]
        ax1.text(value - 0.02, y_positions[i], f'{value:.2f}', 
                va='center', ha='right', fontweight='bold', fontsize=9)
    
    ax1.set_yticks([])
    
    # Middle: Position and amino acid information
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-0.5, len(plot_data) - 0.5)
    
    for i in range(len(plot_data)):
        row = plot_data.iloc[i]
        y_pos = y_positions[i]
        
        # Get information
        rr_aa = aa_dict.get(row['Rr_aa'], '---') if pd.notna(row['Rr_aa']) else '---'
        fourtenten_aa = aa_dict.get(row['4_10_aa'], '---') if pd.notna(row['4_10_aa']) else '---'
        rr_pos = f"{int(row['Rr_seq_pos'])}" if pd.notna(row['Rr_seq_pos']) else "gap"
        fourtenten_pos = f"{int(row['4_10_seq_pos'])}" if pd.notna(row['4_10_seq_pos']) else "gap"
        
        # Layout: Rr AA, positions, 4_10 AA
        ax2.text(-0.6, y_pos, rr_aa, ha='center', va='center', fontweight='bold', fontsize=10)
        ax2.text(0, y_pos, f"{rr_pos}/{fourtenten_pos}", ha='center', va='center', fontweight='bold', fontsize=9)
        ax2.text(0.6, y_pos, fourtenten_aa, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Add tick marks
        ax2.plot([-0.8, -0.8], [y_pos-0.1, y_pos+0.1], 'k-', linewidth=1)
        ax2.plot([0.8, 0.8], [y_pos-0.1, y_pos+0.1], 'k-', linewidth=1)
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Rr AA | Pos | 4_10 AA', fontweight='bold')
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    # Right plot: 4_10 kcat (bars point left)
    ax3.barh(y_positions, plot_data[FOURTENTEN_VALUE_COL], 
             color='red', alpha=0.3)
    ax3.set_xlabel('4_10 Normalized Kcat', fontweight='bold')
    ax3.set_title('4_10 Catalytic Activity', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax3.set_ylim(-0.5, len(plot_data) - 0.5)
    
    # Add value labels on bars
    for i in range(len(plot_data)):
        row = plot_data.iloc[i]
        value = row[FOURTENTEN_VALUE_COL]
        ax3.text(value, y_positions[i], f'{value:.2f}', 
                va='center', ha='center', fontweight='bold', fontsize=9, color='black')
    
    # Add reference line for Rr kcat
    ax3.axvline(RR_NORMALIZED_KCAT, color='black', linestyle='--', linewidth=1, alpha=0.8, 
                label=f'R. rubrum ({RR_NORMALIZED_KCAT:.1f})')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_yticks([])
    
    # Highlight biologically interesting positions
    for i, (_, row) in enumerate(plot_data.iterrows()):
        if row[FOURTENTEN_VALUE_COL] < LOW_KCAT_THRESHOLD and row['difference'] > FITNESS_ADVANTAGE_THRESHOLD:
            y_pos = y_positions[i]
            ax1.axhspan(y_pos-0.4, y_pos+0.4, alpha=0.2, color='yellow', zorder=0)
            ax2.axhspan(y_pos-0.4, y_pos+0.4, alpha=0.2, color='yellow', zorder=0)
            ax3.axhspan(y_pos-0.4, y_pos+0.4, alpha=0.2, color='yellow', zorder=0)
    
    plt.suptitle(f'Top {len(plot_data)} Positions with Largest Disagreements{title_suffix}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add interpretation guide
    fig.text(0.02, 0.02, 
             'Yellow highlighting: 4_10 kcat is impaired but Rr fitness is less affected\n' +
             'These positions may affect catalysis without limiting organismal growth',
             fontsize=10, style='italic', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Main analysis workflow."""
    print("Rubisco Alanine Scan Disagreement Analysis")
    print("=" * 50)
    
    # Load and align data
    valid_data = load_and_align_data()
    
    # Identify disagreements
    top_disagreements, interesting_positions, all_positions_ranked = identify_disagreements(valid_data, TOP_N_POSITIONS)
    
    # Create visualization
    fig = create_disagreement_plot(top_disagreements, " (Prioritizing Biological Interest)")
    
    # Export SVG to desktop
    svg_filename = "/Users/leah-shihlab/Desktop/rubisco_disagreement_analysis.svg"
    fig.savefig(svg_filename, format='svg', dpi=300, bbox_inches='tight')
    print(f"SVG exported to: {svg_filename}")
    
    plt.show()
    
    # Report interesting positions
    if len(interesting_positions) > 0:
        print(f"\nBiologically Interesting Positions (Low 4_10 kcat, Better Rr fitness):")
        for i, (_, row) in enumerate(interesting_positions.head(10).iterrows()):
            print(f"  {i+1:2d}. Alignment pos {int(row['alignment_pos'])}: " +
                  f"Rr={row[RR_VALUE_COL]:.3f}, 4_10={row[FOURTENTEN_VALUE_COL]:.3f}, " +
                  f"Difference={row['difference']:.3f}")
    
    # Save complete results - ALL positions ranked by disagreement
    complete_results = all_positions_ranked[[
        'alignment_pos', 'Rr_seq_pos', '4_10_seq_pos', 
        'Rr_aa', '4_10_aa', RR_VALUE_COL, FOURTENTEN_VALUE_COL, 
        'difference', 'abs_difference'
    ]].copy()
    
    # Add amino acid names in full
    aa_dict = {
        'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
        'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
        'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
        'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'
    }
    
    complete_results['Rr_amino_acid'] = complete_results['Rr_aa'].map(aa_dict)
    complete_results['4_10_amino_acid'] = complete_results['4_10_aa'].map(aa_dict)
    
    # Reorder columns for better readability
    complete_results = complete_results[[
        'alignment_pos', 'Rr_seq_pos', '4_10_seq_pos',
        'Rr_aa', 'Rr_amino_acid', '4_10_aa', '4_10_amino_acid',
        RR_VALUE_COL, FOURTENTEN_VALUE_COL, 'difference', 'abs_difference'
    ]]
    
    # Save complete ranked results to desktop
    complete_csv_filename = "complete_disagreement_analysis_ranked.csv"
    complete_results.to_csv(complete_csv_filename, index=False)
    print(f"Complete results (all {len(complete_results)} positions) saved to: {complete_csv_filename}")
    
    # Also save the top 50 used for plotting to desktop
    top_50_filename = "/Users/leah-shihlab/Desktop/disagreement_analysis_results.csv"
    top_disagreements.to_csv(top_50_filename, index=False)
    print(f"Top {TOP_N_POSITIONS} results saved to: {top_50_filename}")
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()


# In[37]:


#!/usr/bin/env python3
"""
Rubisco Alanine Scan Disagreement Analysis - Single CSV Mode
============================================================

Updated to read the enriched CSV (with Vmax_qbcov column) and grey out
any Vmax bar where qbcov > 1 (unreliable fit per Prywes et al. 2025).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 6  # Base font size in points

# --- UPDATE THIS PATH to point at the enriched CSV with qbcov column ---
INPUT_CSV = 'ENG_ACTIVE_LOOP6_with_qbcov.csv'

# Figure dimensions for publication (180mm width)
FIG_WIDTH_MM = 180
FIG_WIDTH_INCHES = FIG_WIDTH_MM / 25.4

# Reference value for Rr normalized kcat
RR_NORMALIZED_KCAT = 0.371744105

# Vmax reliability threshold (from Prywes et al. 2025)
QBCOV_THRESHOLD = 1.0


def load_data():
    """Load data from enriched CSV file."""
    print("Loading data from CSV...")
    data = pd.read_csv(INPUT_CSV)

    print(f"Loaded {len(data)} positions")
    print(f"Columns: {list(data.columns)}")

    required = ['Rr_seq_pos', '4_10_seq_pos', 'Rr_aa', '4_10_aa',
                'Vmax_median', 'normalized_kcat_bc', 'Vmax_qbcov']
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\n"
                         f"Available: {list(data.columns)}\n"
                         f"Make sure you are using the enriched CSV "
                         f"(ENG_ACTIVE_LOOP6_with_qbcov.csv).")

    return data


def create_disagreement_plot(data, title_suffix=""):
    """Create the disagreement visualization with grouping by tags."""

    if 'difference' not in data.columns:
        data['difference'] = data['Vmax_median'] - data['normalized_kcat_bc']
    if 'abs_difference' not in data.columns:
        data['abs_difference'] = data['difference'].abs()

    # Identify tag column
    tag_col = None
    for col in ['tag', 'Tag', 'tags', 'Tags', 'category', 'Category', 'group', 'Group']:
        if col in data.columns:
            tag_col = col
            break

    if tag_col is None:
        print("Warning: No tag column found. Plotting without grouping.")
        plot_data = data.sort_values('abs_difference', ascending=True)
        groups = [('All Positions', plot_data)]
    else:
        print(f"Grouping by column: '{tag_col}'")

        group_order = ['Active Site', 'Loop 6/Active Site', 'Loop 6', 'Engineering']
        groups = []
        for group_name in group_order:
            group_data = data[data[tag_col] == group_name].copy()
            if len(group_data) > 0:
                group_data = group_data.sort_values('abs_difference', ascending=True)
                groups.append((group_name, group_data))
                print(f"  {group_name}: {len(group_data)} positions")

        # Reverse so Active Site is at top
        groups = groups[::-1]
        plot_data = pd.concat([g[1] for g in groups])

    n_positions = len(plot_data)

    height_per_position = 0.12
    fig_height_inches = max(3, n_positions * height_per_position)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(FIG_WIDTH_INCHES, fig_height_inches),
        gridspec_kw={'width_ratios': [3, 1, 3], 'wspace': 0.1}
    )

    y_positions = np.arange(len(plot_data))

    aa_dict = {
        'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
        'Q': 'Gln', 'E': 'Glu', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
        'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
        'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'
    }

    # ---------------------------------------------------------------
    # Left plot: Rr Vmax bars
    # Colour logic:
    #   qbcov > 1              -> gray (unreliable fit, original 'gray')
    #   qbcov <= 1, Vmax < 0.23 -> gray (reliable but low)
    #   qbcov <= 1, Vmax >= 0.23 -> blue (reliable, near-WT)
    # ---------------------------------------------------------------
    def vmax_bar_color(row):
        qbcov = row.get('Vmax_qbcov', np.nan)
        try:
            qbcov = float(qbcov)
        except (ValueError, TypeError):
            return 'gray'
        if qbcov > QBCOV_THRESHOLD:
            return 'gray'
        return 'gray' if row['Vmax_median'] < 0.23 else 'blue'

    bar_colors = [vmax_bar_color(plot_data.iloc[i]) for i in range(n_positions)]

    for i in range(n_positions):
        ax1.barh(y_positions[i], plot_data.iloc[i]['Vmax_median'],
                 color=bar_colors[i], alpha=0.3)

    ax1.set_xlabel('Rr Fitness (Vmax_median)', fontsize=7, fontweight='bold')
    ax1.set_title('Rr Organismal Fitness', fontsize=7, fontweight='bold')
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    ax1.set_ylim(-0.5, n_positions - 0.5)
    ax1.invert_xaxis()
    ax1.tick_params(labelsize=5)

    # Value labels — bold black throughout, same as original
    for i in range(n_positions):
        row = plot_data.iloc[i]
        value = row['Vmax_median']
        ax1.text(value - 0.02, y_positions[i], f'{value:.2f}',
                 va='center', ha='right', fontweight='bold', fontsize=5)

    ax1.set_yticks([])

    # ---------------------------------------------------------------
    # Middle: position labels
    # ---------------------------------------------------------------
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-0.5, n_positions - 0.5)

    for i in range(n_positions):
        row = plot_data.iloc[i]
        y_pos = y_positions[i]

        rr_aa        = aa_dict.get(row['Rr_aa'], '---') if pd.notna(row['Rr_aa']) else '---'
        fourtenten_aa = aa_dict.get(row['4_10_aa'], '---') if pd.notna(row['4_10_aa']) else '---'
        rr_pos       = f"{int(row['Rr_seq_pos'])}" if pd.notna(row['Rr_seq_pos']) else "gap"
        fourtenten_pos = f"{int(row['4_10_seq_pos'])}" if pd.notna(row['4_10_seq_pos']) else "gap"

        ax2.text(-0.6, y_pos, rr_aa,              ha='center', va='center', fontsize=6)
        ax2.text( 0,   y_pos, f"{rr_pos}/{fourtenten_pos}", ha='center', va='center', fontsize=5)
        ax2.text( 0.6, y_pos, fourtenten_aa,       ha='center', va='center', fontsize=6)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Rr AA | Pos | 4_10 AA', fontsize=7, fontweight='bold')
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # ---------------------------------------------------------------
    # Right plot: 4_10 kcat bars
    # ---------------------------------------------------------------
    ax3.barh(y_positions, plot_data['normalized_kcat_bc'],
             color='#f12492ff', alpha=0.3)
    ax3.set_xlabel('4_10 Normalized Kcat', fontsize=7, fontweight='bold')
    ax3.set_title('4_10 Catalytic Activity', fontsize=7, fontweight='bold')
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    ax3.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)
    ax3.set_ylim(-0.5, n_positions - 0.5)
    ax3.tick_params(labelsize=5)

    for i in range(n_positions):
        row = plot_data.iloc[i]
        value = row['normalized_kcat_bc']
        ax3.text(value, y_positions[i], f'{value:.2f}',
                 va='center', ha='center', fontweight='bold', fontsize=5, color='black')

    ax3.axvline(RR_NORMALIZED_KCAT, color='blue', linestyle='--',
                linewidth=1.5, alpha=0.3,
                label=f'R. rubrum ({RR_NORMALIZED_KCAT:.2f})')
    ax3.legend(loc='lower right', fontsize=5)
    ax3.set_yticks([])

    # ---------------------------------------------------------------
    # Group separators and labels
    # ---------------------------------------------------------------
    if tag_col is not None and len(groups) > 1:
        current_position = 0
        for group_name, group_data in groups:
            group_size = len(group_data)
            if current_position > 0:
                sep_y = current_position - 0.5
                for ax in (ax1, ax2, ax3):
                    ax.axhline(sep_y, color='gray', linestyle='-',
                               linewidth=1, alpha=0.5)
            group_center = current_position + (group_size - 1) / 2
            ax3.text(1.15, group_center, group_name,
                     transform=ax3.get_yaxis_transform(),
                     rotation=90, va='center', ha='center',
                     fontsize=6, fontweight='bold', color='gray')
            current_position += group_size

    # ---------------------------------------------------------------
    # Legend for Vmax bar colours
    # ---------------------------------------------------------------
    legend_patches = [
        mpatches.Patch(color='blue', alpha=0.3, label='Reliable fit, near-WT (qbcov ≤ 1)'),
        mpatches.Patch(color='gray', alpha=0.3, label='Low Vmax or unreliable fit (qbcov > 1)'),
    ]
    ax1.legend(handles=legend_patches, loc='lower left', fontsize=4.5,
               framealpha=0.8, title='Vmax reliability', title_fontsize=5)

    plt.suptitle(
        f'Disagreement Analysis — {n_positions} Selected Positions{title_suffix}',
        fontsize=7, fontweight='bold', y=0.98
    )

    fig.text(0.02, 0.01,
             'Grey bars: Vmax qbcov > 1 — fit unreliable per Prywes et al. 2025 criteria',
             fontsize=5, style='italic',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    return fig


def main():
    print("Rubisco Alanine Scan Disagreement Analysis v2")
    print("=" * 50)

    data = load_data()
    fig  = create_disagreement_plot(data)

    svg_filename = "rubisco_disagreement_analysis_v2.svg"
    fig.savefig(svg_filename, format='svg', dpi=300, bbox_inches='tight')
    print(f"\nSVG exported to: {svg_filename}")

    plt.show()
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()


# In[11]:


"""
Plot 1: Journal formatting with adjustable font sizes
WITH ACTIVE SITE RESIDUES IN BLACK AND POSITION LABELS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# formatting specifications
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 7  # Base font size
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2

# Load data
file_path = 'GspRBC and RrRBC complete_disagreement_analysis.csv'
df = pd.read_excel(file_path, sheet_name=0)

# Clean data
mask = ~(df['Vmax_median'].isna() | df['normalized_kcat_bc'].isna() | df['Same amino acid?'].isna())
df_clean = df[mask].copy()

#############################################################################
# ACTIVE SITE POSITIONS (from table - using GspRBC positions)
#############################################################################
ACTIVE_SITE_POSITIONS = [53, 175, 200, 201, 202, 203, 298, 299, 332, 334, 379]

# Mark active site positions in dataframe
# Assuming you have a column like '4_10_seq_pos' or 'alignment_pos' for GspRBC positions
if '4_10_seq_pos' in df_clean.columns:
    df_clean['is_active_site'] = df_clean['4_10_seq_pos'].isin(ACTIVE_SITE_POSITIONS)
elif 'alignment_pos' in df_clean.columns:
    df_clean['is_active_site'] = df_clean['alignment_pos'].isin(ACTIVE_SITE_POSITIONS)
else:
    print("Warning: Could not find position column. Available columns:", df_clean.columns.tolist())
    df_clean['is_active_site'] = False

# Separate into categories
active_site_df = df_clean[df_clean['is_active_site'] == True].copy()
same_aa_df = df_clean[(df_clean['Same amino acid?'] == 1) & (df_clean['is_active_site'] == False)].copy()
diff_aa_df = df_clean[(df_clean['Same amino acid?'] == 0) & (df_clean['is_active_site'] == False)].copy()

# Calculate statistics for ALL data
slope_all, intercept_all, r_all, p_all, _ = stats.linregress(
    df_clean['Vmax_median'], df_clean['normalized_kcat_bc']
)

print(f"All data: n={len(df_clean)}, R²={r_all**2:.3f}, p={p_all:.2e}")
print(f"Active site: n={len(active_site_df)}")
print(f"Same AA (non-AS): n={len(same_aa_df)}")
print(f"Different AA (non-AS): n={len(diff_aa_df)}")

#############################################################################
# FONT SIZE CONTROLS - ADJUST THESE
#############################################################################
AXIS_LABEL_SIZE = 7      # "RrRBC alanine scan (Vmax)"
TICK_LABEL_SIZE = 7      # Numbers on axes (0.0, 0.5, 1.0, etc.)
LEGEND_SIZE = 5          # Legend text
POSITION_LABEL_SIZE = 6  # Position labels
POSITIONS_TO_LABEL = [104, 151, 155, 236, 241]


fig, ax = plt.subplots(figsize=(3.46, 3.46), dpi=300)

# Plot different AA (pink, background)
ax.scatter(diff_aa_df['Vmax_median'], diff_aa_df['normalized_kcat_bc'],
          alpha=0.3, s=40, c='#f12492ff', edgecolors='none',
          label=f'Different AA (n={len(diff_aa_df)})', zorder=1)

# Plot same AA (blue)
ax.scatter(same_aa_df['Vmax_median'], same_aa_df['normalized_kcat_bc'],
          alpha=0.3, s=40, c='#0000ff', edgecolors='none',
          label=f'Same AA (n={len(same_aa_df)})', zorder=2)

# Plot active site residues (BLACK, on top)
ax.scatter(active_site_df['Vmax_median'], active_site_df['normalized_kcat_bc'],
          alpha=0.8, s=50, c='black', edgecolors='white', linewidths=0.5,
          label=f'Active site (n={len(active_site_df)})', zorder=3)


# Add position labels with arrows
for pos in POSITIONS_TO_LABEL:
    # Find the position in the dataframe
    if '4_10_seq_pos' in df_clean.columns:
        pos_row = df_clean[df_clean['4_10_seq_pos'] == pos]
    elif 'alignment_pos' in df_clean.columns:
        pos_row = df_clean[df_clean['alignment_pos'] == pos]
    elif 'Rr_seq_pos' in df_clean.columns:
        pos_row = df_clean[df_clean['Rr_seq_pos'] == pos]
    else:
        print(f"Warning: Could not find position column. Available columns: {df_clean.columns.tolist()}")
        break
    
    if len(pos_row) > 0:
        x_val = pos_row['Vmax_median'].values[0]
        y_val = pos_row['normalized_kcat_bc'].values[0]
        
        # Add text label with arrow pointing to the dot
        ax.annotate(str(pos), 
                   xy=(x_val, y_val),  # Point at the dot
                   xytext=(20, 20),     # Text offset (points)
                   textcoords='offset points',
                   fontsize=POSITION_LABEL_SIZE,
                   fontweight='bold',
                   color='black',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor='black',
                           linewidth=0.5,
                           alpha=0.9),
                   arrowprops=dict(arrowstyle='->', 
                                 connectionstyle='arc3,rad=0',
                                 color='black',
                                 linewidth=0.75,
                                 shrinkA=0,  # Don't shrink at text end
                                 shrinkB=3))  # Shrink 3 points at dot end
        
        print(f"Position {pos}: x={x_val:.3f}, y={y_val:.3f}")
    else:
        print(f"Warning: Position {pos} not found in data")

# Labels with journal-appropriate font sizes
ax.set_xlabel('RrRBC alanine scan (Vmax)', fontsize=AXIS_LABEL_SIZE)
ax.set_ylabel('GspRBC alanine scan (kcat)', fontsize=AXIS_LABEL_SIZE)

# Set limits with small padding
ax.set_xlim(-0.05, 1.3)
ax.set_ylim(-0.05, 1.2)

# Add quadrant lines
ax.axvline(x=0.75, color='gray', linewidth=0.75, linestyle='--', alpha=0.5, zorder=0)
ax.axhline(y=0.8, color='gray', linewidth=0.75, linestyle='--', alpha=0.5, zorder=0)

# Legend with smaller font
ax.legend(fontsize=LEGEND_SIZE, frameon=False, loc='upper left', 
         handletextpad=0.5, borderaxespad=0.5)

# Tick parameters
ax.tick_params(labelsize=TICK_LABEL_SIZE, pad=2)

# Tighten layout
plt.tight_layout(pad=0.3)

# Save in multiple formats for submission
plt.savefig('/Users/leah-shihlab/Desktop/Fig_epistasis_ActiveSite.svg', 
           bbox_inches='tight', pad_inches=0.02)

plt.show()

print("\n" + "="*70)
print("JOURNAL-FORMATTED FIGURE SAVED (WITH ACTIVE SITE)")
print("="*70)
print(f"""
Font sizes used:
  - Axis labels: {AXIS_LABEL_SIZE}pt
  - Tick labels (numbers): {TICK_LABEL_SIZE}pt
  - Legend: {LEGEND_SIZE}pt
  - Position labels: {POSITION_LABEL_SIZE}pt

Categories:
  - Active site residues (black): {len(active_site_df)}
  - Same AA (blue): {len(same_aa_df)}
  - Different AA (pink): {len(diff_aa_df)}
  - Total: {len(df_clean)}

Active site positions (GspRBC): {ACTIVE_SITE_POSITIONS}
Labeled positions: {POSITIONS_TO_LABEL}
""")

# In[36]:


"""
Conservation-colored epistasis plots from SI Data Workbook
Plotting positions with both kcat and Vmax data
Colored by Shannon Entropy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

# Jupyter inline plotting
get_ipython().run_line_magic('matplotlib', 'inline')

# Journal formatting
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.major.size'] = 2
plt.rcParams['ytick.major.size'] = 2

#############################################################################
# QUADRANT SETTINGS - ADJUST THESE
#############################################################################
QUADRANT_X = 0.75  # Vertical line position (X coordinate)
QUADRANT_Y = 0.8  # Horizontal line position (Y coordinate)
QUADRANT_LINE_COLOR = 'gray'
QUADRANT_LINE_STYLE = '--'
QUADRANT_LINE_WIDTH = 1.0
QUADRANT_LINE_ALPHA = 0.5
QUADRANT_LABEL_SIZE = 8
QUADRANT_LABEL_COLOR = 'gray'

#############################################################################
# LOAD DATA FROM SI WORKBOOK
#############################################################################

# Load from uploaded file
si_file = 'Supplementary Data.csv'

print("="*80)
print("LOADING DATA FROM SI WORKBOOK")
print("="*80)

# Read S2 sheet
df = pd.read_excel(si_file, sheet_name='S2')

print(f"\nLoaded {len(df)} rows from S2")
print(f"Columns: {list(df.columns)}")

# Clean data - keep only rows with both kcat and Vmax
mask = ~(df['RrRBC Ala variant VmaxC normalized'].isna() | 
         df['GspRBC Ala variant kcatC normalized'].isna() |
         df['Shannon Entropy all FIIs (H_all)'].isna())
df_clean = df[mask].copy()

print(f"\nAfter filtering for complete data: {len(df_clean)} positions")

# Entropy statistics
print(f"\nEntropy range: {df_clean['Shannon Entropy all FIIs (H_all)'].min():.3f} to {df_clean['Shannon Entropy all FIIs (H_all)'].max():.3f}")
print(f"Mean entropy: {df_clean['Shannon Entropy all FIIs (H_all)'].mean():.3f}")
print(f"Median entropy: {df_clean['Shannon Entropy all FIIs (H_all)'].median():.3f}")

# Filter for highly conserved (entropy ≤ 0.5)
df_conserved = df_clean[df_clean['Shannon Entropy all FIIs (H_all)'] <= 0.5].copy()
print(f"\nPositions with entropy ≤ 0.5: {len(df_conserved)} ({100*len(df_conserved)/len(df_clean):.1f}%)")

#############################################################################
# CALCULATE CORRELATIONS (Pearson AND Spearman)
#############################################################################

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# All positions - Pearson
slope_all, intercept_all, r_all_pearson, p_all_pearson, _ = stats.linregress(
    df_clean['RrRBC Ala variant VmaxC normalized'], df_clean['GspRBC Ala variant kcatC normalized']
)

# All positions - Spearman
r_all_spearman, p_all_spearman = stats.spearmanr(
    df_clean['RrRBC Ala variant VmaxC normalized'], df_clean['GspRBC Ala variant kcatC normalized']
)

print(f"\nAll positions (n={len(df_clean)}):")
print(f"  Pearson:  r={r_all_pearson:.3f}, R²={r_all_pearson**2:.3f}, p={p_all_pearson:.2e}")
print(f"  Spearman: ρ={r_all_spearman:.3f}, ρ²={r_all_spearman**2:.3f}, p={p_all_spearman:.2e}")

# Conserved positions
if len(df_conserved) >= 3:
    # Pearson
    slope_cons, intercept_cons, r_cons_pearson, p_cons_pearson, _ = stats.linregress(
        df_conserved['RrRBC Ala variant VmaxC normalized'], df_conserved['GspRBC Ala variant kcatC normalized']
    )
    
    # Spearman
    r_cons_spearman, p_cons_spearman = stats.spearmanr(
        df_conserved['RrRBC Ala variant VmaxC normalized'], df_conserved['GspRBC Ala variant kcatC normalized']
    )
    
    print(f"\nConserved positions (entropy ≤ 0.5, n={len(df_conserved)}):")
    print(f"  Pearson:  r={r_cons_pearson:.3f}, R²={r_cons_pearson**2:.3f}, p={p_cons_pearson:.2e}")
    print(f"  Spearman: ρ={r_cons_spearman:.3f}, ρ²={r_cons_spearman**2:.3f}, p={p_cons_spearman:.2e}")

# Variable positions
df_variable = df_clean[df_clean['Shannon Entropy all FIIs (H_all)'] > 0.5].copy()
if len(df_variable) >= 3:
    r_var_pearson, p_var_pearson = stats.pearsonr(
        df_variable['RrRBC Ala variant VmaxC normalized'], df_variable['GspRBC Ala variant kcatC normalized']
    )
    r_var_spearman, p_var_spearman = stats.spearmanr(
        df_variable['RrRBC Ala variant VmaxC normalized'], df_variable['GspRBC Ala variant kcatC normalized']
    )
    
    print(f"\nVariable positions (entropy > 0.5, n={len(df_variable)}):")
    print(f"  Pearson:  r={r_var_pearson:.3f}, R²={r_var_pearson**2:.3f}, p={p_var_pearson:.2e}")
    print(f"  Spearman: ρ={r_var_spearman:.3f}, ρ²={r_var_spearman**2:.3f}, p={p_var_spearman:.2e}")

#############################################################################
# COLORMAP SETUP
#############################################################################

# Create colormap: Blue (conserved/low entropy) → Yellow (variable/high entropy)
colors = ['#0D0887', '#7E03A8', '#CC4778', '#F89540', '#F0F921']
cmap = LinearSegmentedColormap.from_list('blue_yellow', colors, N=256)

# Calculate max entropy BEFORE any plotting
max_entropy = df_clean['Shannon Entropy all FIIs (H_all)'].max()

#############################################################################
# HELPER FUNCTION TO ADD QUADRANTS
#############################################################################

def add_quadrants(ax, x_pos, y_pos):
    """Add quadrant lines and labels to axis."""
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Add vertical line
    ax.axvline(x_pos, color=QUADRANT_LINE_COLOR, linestyle=QUADRANT_LINE_STYLE, 
              linewidth=QUADRANT_LINE_WIDTH, alpha=QUADRANT_LINE_ALPHA, zorder=0)
    
    # Add horizontal line
    ax.axhline(y_pos, color=QUADRANT_LINE_COLOR, linestyle=QUADRANT_LINE_STYLE, 
              linewidth=QUADRANT_LINE_WIDTH, alpha=QUADRANT_LINE_ALPHA, zorder=0)
    
    # Add quadrant labels
    # Quadrant 1: Top Right (high X, high Y)
    ax.text(xlim[1] - 0.05, ylim[1] - 0.05, '1', 
           fontsize=QUADRANT_LABEL_SIZE, color=QUADRANT_LABEL_COLOR, 
           weight='bold', ha='right', va='top')
    
    # Quadrant 2: Top Left (low X, high Y)
    ax.text(xlim[0] + 0.05, ylim[1] - 0.05, '2', 
           fontsize=QUADRANT_LABEL_SIZE, color=QUADRANT_LABEL_COLOR, 
           weight='bold', ha='left', va='top')
    
    # Quadrant 3: Bottom Left (low X, low Y)
    ax.text(xlim[0] + 0.05, ylim[0] + 0.05, '3', 
           fontsize=QUADRANT_LABEL_SIZE, color=QUADRANT_LABEL_COLOR, 
           weight='bold', ha='left', va='bottom')
    
    # Quadrant 4: Bottom Right (high X, low Y)
    ax.text(xlim[1] - 0.05, ylim[0] + 0.05, '4', 
           fontsize=QUADRANT_LABEL_SIZE, color=QUADRANT_LABEL_COLOR, 
           weight='bold', ha='right', va='bottom')

#############################################################################
# MAIN FIGURE: Side-by-side comparison with BOTH correlations AND QUADRANTS
#############################################################################

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=300)

# Left: All data
scatter1 = ax1.scatter(
    df_clean['RrRBC Ala variant VmaxC normalized'], 
    df_clean['GspRBC Ala variant kcatC normalized'],
    c=df_clean['Shannon Entropy all FIIs (H_all)'],
    cmap=cmap,
    s=25,
    alpha=0.7,
    edgecolors='none',
    vmin=0,
    vmax=max_entropy,
    zorder=2
)

# Add colorbar to ax1
cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.02)
cbar1.set_label('Shannon Entropy', fontsize=7, rotation=270, labelpad=12)
cbar1.ax.tick_params(labelsize=5)

ax1.set_xlabel('RrRBC Vmax', fontsize=7)
ax1.set_ylabel('GspRBC kcat', fontsize=7)
ax1.set_xlim(-0.05, 1.3)
ax1.set_ylim(-0.05, 1.2)

# Add quadrants to left plot
add_quadrants(ax1, QUADRANT_X, QUADRANT_Y)

# Title with r and ρ values
ax1.set_title(f'All positions (n={len(df_clean)})\n' + 
             f'r={r_all_pearson:.3f} (R²={r_all_pearson**2:.3f}), ρ={r_all_spearman:.3f} (ρ²={r_all_spearman**2:.3f})', 
             fontsize=6.5, weight='bold')
ax1.tick_params(labelsize=5)

# Right: Conserved only
scatter2 = ax2.scatter(
    df_conserved['RrRBC Ala variant VmaxC normalized'], 
    df_conserved['GspRBC Ala variant kcatC normalized'],
    c=df_conserved['Shannon Entropy all FIIs (H_all)'],
    cmap=cmap,
    s=25,
    alpha=0.7,
    edgecolors='none',
    vmin=0,
    vmax=max_entropy,
    zorder=2
)

# Add colorbar to ax2
cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.02)
cbar2.set_label('Shannon Entropy', fontsize=7, rotation=270, labelpad=12)
cbar2.ax.tick_params(labelsize=5)

ax2.set_xlabel('RrRBC Vmax', fontsize=7)
ax2.set_ylabel('GspRBC kcat', fontsize=7)
ax2.set_xlim(-0.05, 1.3)
ax2.set_ylim(-0.05, 1.2)

# Add quadrants to right plot
add_quadrants(ax2, QUADRANT_X, QUADRANT_Y)

# Title with r and ρ values
ax2.set_title(f'Conserved (entropy ≤ 0.5, n={len(df_conserved)})\n' + 
             f'r={r_cons_pearson:.3f} (R²={r_cons_pearson**2:.3f}), ρ={r_cons_spearman:.3f} (ρ²={r_cons_spearman**2:.3f})', 
             fontsize=6.5, weight='bold')
ax2.tick_params(labelsize=5)

plt.tight_layout()
plt.savefig('pathname/Figure.svg', 
           bbox_inches='tight', pad_inches=0.02)
plt.show()

print("\n✓ Side-by-side comparison saved with r, ρ, R², ρ², and quadrants")
print(f"✓ Quadrant lines at X={QUADRANT_X}, Y={QUADRANT_Y}")

#############################################################################
# STATISTICAL COMPARISON
#############################################################################

print("\n" + "="*80)
print("STATISTICAL COMPARISON")
print("="*80)

print(f"\nAll positions (n={len(df_clean)}):")
print(f"  Pearson:  r={r_all_pearson:.3f}, R²={r_all_pearson**2:.3f}, p={p_all_pearson:.2e}")
print(f"  Spearman: ρ={r_all_spearman:.3f}, ρ²={r_all_spearman**2:.3f}, p={p_all_spearman:.2e}")

if len(df_conserved) >= 3:
    print(f"\nHighly conserved only (entropy ≤ 0.5, n={len(df_conserved)}):")
    print(f"  Pearson:  r={r_cons_pearson:.3f}, R²={r_cons_pearson**2:.3f}, p={p_cons_pearson:.2e}")
    print(f"  Spearman: ρ={r_cons_spearman:.3f}, ρ²={r_cons_spearman**2:.3f}, p={p_cons_spearman:.2e}")
    
    # Calculate change
    r2_change = r_cons_pearson**2 - r_all_pearson**2
    pct_change = 100 * r2_change / r_all_pearson**2
    
    print(f"\nChange in Pearson R²:")
    print(f"  ΔR² = {r2_change:+.3f}")
    print(f"  % change = {pct_change:+.1f}%")
    
    if r_cons_pearson**2 > r_all_pearson**2:
        print("\n✓ Conserved positions show STRONGER correlation!")
        print("  → Conservation predicts better transferability")
    elif r_cons_pearson**2 < r_all_pearson**2:
        print("\n✓ Conserved positions show WEAKER correlation!")
        print("  → Conservation does NOT predict better transferability")
    else:
        print("\n✓ Similar correlation in both")
        print("  → Conservation doesn't affect transferability")

print("\n" + "="*80)
print("SUMMARY: PEARSON vs SPEARMAN")
print("="*80)
print(f"""
Pearson (r): Linear relationship (affected by outliers)
Spearman (ρ): Monotonic relationship (rank-based, robust)

All positions:
  r = {r_all_pearson:.3f}, R²  = {r_all_pearson**2:.3f}
  ρ = {r_all_spearman:.3f}, ρ² = {r_all_spearman**2:.3f}
  Difference  = {abs(r_all_spearman**2 - r_all_pearson**2):.3f}

Conserved positions:
  r = {r_cons_pearson:.3f}, R²  = {r_cons_pearson**2:.3f}
  ρ = {r_cons_spearman:.3f}, ρ² = {r_cons_spearman**2:.3f}
  Difference  = {abs(r_cons_spearman**2 - r_cons_pearson**2):.3f}

KEY FINDING: Conservation does NOT improve transferability.
Even conserved positions show poor epistatic correlation.
""")


# In[28]:


"""
Distance-binned epistasis analysis: journal formatting
4 panels: All data, 0-10Å, 10-20Å, 20+Å
Journlal figure specifications:
  - Single column: 88 mm (~3.46 in)
  - Double column: 180 mm (~7.09 in)
  - Max height: 225 mm (~8.86 in)
  - Font: Arial, 6-8 pt body, 6 pt min
  - Line weights: axes 0.5 pt, data lines 0.75 pt
  - Resolution: 300 dpi (print), 72 dpi (screen)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy import stats

# ============================================================
# FORMATTING — global rcParams
# ============================================================
plt.rcParams.update({
    # Font
    'font.family':          'sans-serif',
    'font.sans-serif':      ['Arial'],
    'font.size':            6,
    'axes.titlesize':       7,
    'axes.labelsize':       7,
    'xtick.labelsize':      6,
    'ytick.labelsize':      6,
    'legend.fontsize':      6,

    # Axes
    'axes.linewidth':       0.5,
    'axes.spines.top':      True,
    'axes.spines.right':    True,

    # Ticks
    'xtick.major.width':    0.5,
    'ytick.major.width':    0.5,
    'xtick.major.size':     2.5,
    'ytick.major.size':     2.5,
    'xtick.minor.width':    0.35,
    'ytick.minor.width':    0.35,
    'xtick.minor.size':     1.5,
    'ytick.minor.size':     1.5,
    'xtick.direction':      'out',
    'ytick.direction':      'out',
    'xtick.top':            False,
    'ytick.right':          False,

    # Lines / markers
    'lines.linewidth':      0.75,

    # Layout
    'figure.dpi':           300,
    'savefig.dpi':          300,
    'pdf.fonttype':         42,   # embed fonts as TrueType
    'ps.fonttype':          42,
    'svg.fonttype':         'none',  # editable text in Illustrator
})

#color palette (colorblind-accessible)
BLUE   = '#3A86C8'   # same-AA data points / main series
GREY   = '#888888'   # reference lines
BLACK  = '#000000'

# ============================================================
# LOAD DATA — SI_Data_Draft_5.xlsx
#   S3: epistasis data (RrRBC Vmax vs GspRBC kcat, per position)
#   S1: GspRBC alanine-scan kcat + distance from active site (Lys200)
# ============================================================
si_file = 'Pathname/Supplementary Data'

df_s3 = pd.read_excel(si_file, sheet_name='S3')
df_s1 = pd.read_excel(si_file, sheet_name='S1')

# S1 contains wildtype control rows where Residue # is "GspRBC" — drop these
df_s1 = df_s1[pd.to_numeric(df_s1['Residue #'], errors='coerce').notna()].copy()
df_s1['Residue #'] = df_s1['Residue #'].astype(int)

print("=" * 80)
print("MERGING DATASETS")
print("=" * 80)
print(f"S3 (epistasis) : {len(df_s3)} positions")
print(f"S1 (distance)  : {len(df_s1)} positions")

# ============================================================
# MERGE & CLEAN
# ============================================================
# Derive same-AA flag directly from S3 AA columns
df_s3 = df_s3.copy()
df_s3['Same amino acid?'] = (df_s3['RrRBC AA'] == df_s3['GspRBC AA']).astype(int)

df_merged = pd.merge(
    df_s3,
    df_s1[['Residue #', 'Distance from GspRBClLys200']].rename(
        columns={'Residue #':                    'Residue_Number',
                 'Distance from GspRBClLys200':  'Distance_Angstrom'}
    ),
    left_on='GspRBC Residue #',
    right_on='Residue_Number',
    how='inner'
)

print(f"After merge    : {len(df_merged)} positions")

# Rename to consistent internal names used throughout the rest of the script
df_merged = df_merged.rename(columns={
    'RrRBC Vmax':              'Vmax_median',
    'GspRBC Normalized kcatC': 'normalized_kcat_bc',
})

mask = ~(
    df_merged['Vmax_median'].isna()              |
    df_merged['normalized_kcat_bc'].isna()       |
    df_merged['Distance_Angstrom'].isna()
)
df_clean = df_merged[mask].copy()

print(f"After NaN removal: {len(df_clean)} positions")
print(f"Distance range: {df_clean['Distance_Angstrom'].min():.1f}"
      f"–{df_clean['Distance_Angstrom'].max():.1f} Å")

same_aa_df = df_clean[df_clean['Same amino acid?'] == 1].copy()
diff_aa_df = df_clean[df_clean['Same amino acid?'] == 0].copy()
print(f"\nSame AA: {len(same_aa_df)}    Different AA: {len(diff_aa_df)}")

# ============================================================
# DISTANCE BINS
# ============================================================
bins = [
    ('All',    0,  np.inf),
    ('0–10 Å', 0,  10),
    ('10–20 Å', 10, 20),
    ('≥20 Å',  20, np.inf),
]

print("\n" + "=" * 80)
print("DISTANCE BINNING ANALYSIS")
print("=" * 80)

bin_stats = []
for bin_name, min_dist, max_dist in bins:
    if bin_name == 'All':
        subset = same_aa_df
    else:
        subset = same_aa_df[
            (same_aa_df['Distance_Angstrom'] >= min_dist) &
            (same_aa_df['Distance_Angstrom'] <  max_dist)
        ]

    if len(subset) >= 3:
        r, p = stats.pearsonr(subset['Vmax_median'], subset['normalized_kcat_bc'])
        rho, p_spearman = stats.spearmanr(subset['Vmax_median'], subset['normalized_kcat_bc'])
        slope, intercept, _, _, _ = stats.linregress(
            subset['Vmax_median'], subset['normalized_kcat_bc']
        )
        bin_stats.append({
            'bin':        bin_name,
            'n':          len(subset),
            'r':          r,
            'r2':         r ** 2,
            'p':          p,
            'rho':        rho,
            'p_spearman': p_spearman,
            'subset':     subset,
            'slope':      slope,
            'intercept':  intercept,
        })
        print(f"\n{bin_name}: n={len(subset)}, R²={r**2:.3f}, p={p:.2e} | "
              f"ρ={rho:.3f}, p_spearman={p_spearman:.2e}")
    else:
        print(f"\n{bin_name}: n={len(subset)} (insufficient for regression)")

# ============================================================
# FIGURE 1 — 4-panel distance-binned scatter
# Double column width, 2×2 layout
# ============================================================
panel_labels = ['a', 'b', 'c', 'd']

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.09, 6.50),   # 180 mm wide, ~165 mm tall — within max height
    dpi=300
)
axes_flat = axes.flatten()

for i, stat in enumerate(bin_stats):
    ax     = axes_flat[i]
    subset = stat['subset']

    # --- scatter ---
    ax.scatter(
        subset['Vmax_median'],
        subset['normalized_kcat_bc'],
        s=20,
        color=BLUE,
        alpha=0.55,
        linewidths=0,
        rasterized=True,   # keeps SVG/PDF file size manageable for many points
        zorder=2,
    )

    # --- regression line ---
    x_line = np.array([0, 1.3])
    y_line = stat['slope'] * x_line + stat['intercept']
    ax.plot(x_line, y_line,
            color=BLACK, linewidth=0.75, alpha=0.85, zorder=3)

    # --- axes cosmetics ---
    ax.set_xlim(-0.05, 1.35)
    ax.set_ylim(-0.05, 1.25)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    ax.set_xlabel('RrRBC V$_{max}$ (normalised)', fontsize=7)
    ax.set_ylabel('GspRBC $k_{cat}$ (normalised)', fontsize=7)
    ax.tick_params(which='both', pad=2)

    # --- significance asterisks ---
    if   stat['p'] < 0.001: sig = '***'
    elif stat['p'] < 0.01:  sig = '**'
    elif stat['p'] < 0.05:  sig = '*'
    else:                   sig = 'n.s.'

    title_str = (
        f"{stat['bin']}    "
        f"$n$ = {stat['n']},  "
        f"$R^2$ = {stat['r2']:.3f},  "
        f"\u03c1 = {stat['rho']:.3f},  "
        f"{sig}"
    )
    ax.set_title(title_str, fontsize=6, pad=4)

    # --- panel label (a, b, c, d) — top-left, bold ---
    ax.text(
        -0.12, 1.06,
        panel_labels[i],
        transform=ax.transAxes,
        fontsize=8,
        fontweight='bold',
        va='top',
        ha='left',
    )

plt.tight_layout(pad=0.8, w_pad=1.2, h_pad=1.5)

out_base = '/pathname'
plt.savefig(f'{out_base}.svg', bbox_inches='tight', pad_inches=0.02)
plt.savefig(f'{out_base}.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig(f'{out_base}.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.show()

print("\n4-panel figure saved (SVG / PDF / PNG)")

# ============================================================
# FIGURE 2 — Sliding-window R² vs distance
# Single column width
# ============================================================
WINDOW   = 30   # positions per window
STEP     =  5   # step size

distances_sorted = np.sort(same_aa_df['Distance_Angstrom'].values)
sorted_indices   = np.argsort(same_aa_df['Distance_Angstrom'].values)

window_centers, window_r2 = [], []

for i in range(0, len(sorted_indices) - WINDOW, STEP):
    idx    = sorted_indices[i : i + WINDOW]
    subset = same_aa_df.iloc[idx]
    if len(subset) >= 10:
        r, _ = stats.pearsonr(
            subset['Vmax_median'], subset['normalized_kcat_bc']
        )
        window_centers.append(np.median(subset['Distance_Angstrom']))
        window_r2.append(r ** 2)

fig2, ax2 = plt.subplots(figsize=(3.46, 2.60), dpi=300)   # 88 mm single column

ax2.plot(
    window_centers, window_r2,
    'o-',
    color=BLUE,
    markersize=3.5,
    linewidth=0.75,
    alpha=0.8,
    markeredgewidth=0,
    zorder=3,
)

overall_r2 = bin_stats[0]['r2']
ax2.axhline(
    y=overall_r2,
    color=GREY,
    linestyle='--',
    linewidth=0.5,
    zorder=2,
)

# Legend handle for dashed line
legend_elements = [
    Line2D([0], [0], color=GREY, linewidth=0.5, linestyle='--',
           label=f'Overall $R^2$ = {overall_r2:.3f}')
]
ax2.legend(
    handles=legend_elements,
    frameon=False,
    fontsize=5,
    loc='upper right',
)

ax2.set_xlabel('Distance from active site (Å)', fontsize=7)
ax2.set_ylabel('$R^2$ (sliding window, $n$ = 30)', fontsize=7)
ax2.set_title('Epistatic correlation vs. distance', fontsize=7, weight='bold', pad=4)
ax2.set_ylim(0, max(window_r2) * 1.15 if window_r2 else 0.5)
ax2.tick_params(which='both', pad=2)

# Panel label
ax2.text(
    -0.15, 1.06, 'e',
    transform=ax2.transAxes,
    fontsize=8, fontweight='bold',
    va='top', ha='left',
)

plt.tight_layout(pad=0.4)

out_base2 = '/pathname'
plt.savefig(f'{out_base2}.svg', bbox_inches='tight', pad_inches=0.02)
plt.savefig(f'{out_base2}.pdf', bbox_inches='tight', pad_inches=0.02)
plt.savefig(f'{out_base2}.png', bbox_inches='tight', pad_inches=0.02, dpi=300)
plt.show()

print("Sliding-window figure saved (SVG / PDF / PNG)")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY: DOES DISTANCE PREDICT EPISTATIC TRANSFERABILITY?")
print("=" * 80)

for stat in bin_stats:
    if   stat['p'] < 0.001: sig = '***'
    elif stat['p'] < 0.01:  sig = '**'
    elif stat['p'] < 0.05:  sig = '*'
    else:                   sig = 'n.s.'
    print(f"  {stat['bin']:12s}  R² = {stat['r2']:.3f}   ρ = {stat['rho']:.3f}   "
          f"n = {stat['n']:4d}   p_pearson = {stat['p']:.2e}   p_spearman = {stat['p_spearman']:.2e}   {sig}")

r2_values        = [s['r2'] for s in bin_stats[1:]]   # exclude 'All'
distances_binned = [5, 15, 25]

if len(r2_values) >= 3:
    corr, p_trend = stats.pearsonr(distances_binned, r2_values)
    print(f"\nTrend — R² vs. distance bin centre: r = {corr:.3f}, p = {p_trend:.3f}")
    if p_trend > 0.05:
        print("\n  No significant trend (p > 0.05)")
        print("  → Distance does NOT predict epistatic transferability")
        print("  → Amino acid chemistry > spatial position")
    else:
        direction = "Closer" if corr < 0 else "Further"
        print(f"\n  Significant trend (p < 0.05)")
        print(f"  → {direction} from active site = stronger correlation")


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

# ── Tunable parameters ────────────────────────────────────────────────────────
FILE_PATH   = '/Pathname/Supplementary Data.csv'
SHEET       = 'S2'
VALUE_COL   = 'ΔH (H_all-H_fast)'
ENTROPY_COL = 'Shannon Entropy all FIIs (H_all)'
KCAT_COL    = 'GspRBC Ala variant kcatC normalized'
GROUPS      = ['Complete data?', 'Q1', 'Q2', 'Q3', 'Q4']
LABELS      = ['Complete data', 'Q1', 'Q2', 'Q3', 'Q4']
COLORS      = ['#555555', '#4477AA', '#EE6677', '#228833', '#CCBB44']
BIN_WIDTH   = 0.10
FONT_FAMILY = 'Arial'
FONT_SIZE   = 7
TICK_SIZE   = 6
AXIS_LW     = 0.5
FIG_WIDTH   = 88 / 25.4
FIG_HEIGHT  = 155 / 25.4
DPI_DISPLAY = 150
DPI_SAVE    = 300
ALPHA       = 0.75

OUT_DIR = '/pathname/Histogram Q1-4'
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_SVG          = os.path.join(OUT_DIR, 'Histogram Q1-4.svg')
OUTPUT_PNG          = os.path.join(OUT_DIR, 'Histogram Q1-4.png')
OUTPUT_SVG_COMPLETE = os.path.join(OUT_DIR, 'Histogram Q1-4 (complete only).svg')
OUTPUT_PNG_COMPLETE = os.path.join(OUT_DIR, 'Histogram Q1-4 (complete only).png')
# ─────────────────────────────────────────────────────────────────────────────

# Load data
df = pd.read_excel(FILE_PATH, sheet_name=SHEET)

# ── Collect subsets for BOTH versions ─────────────────────────────────────────
ALL_LABEL = 'All positions'
ALL_COLOR = '#222222'

subsets_all = {}
subsets_all[ALL_LABEL] = df[VALUE_COL].dropna().values
for grp, label in zip(GROUPS, LABELS):
    subsets_all[label] = df[df[grp] == 'Y'][VALUE_COL].dropna().values

subsets_complete = {}
subsets_complete[ALL_LABEL] = df[df['Complete data?'] == 'Y'][VALUE_COL].dropna().values
for grp, label in zip(GROUPS, LABELS):
    if grp == 'Complete data?':
        mask = df[grp] == 'Y'
    else:
        mask = (df[grp] == 'Y') & (df['Complete data?'] == 'Y')
    subsets_complete[label] = df.loc[mask, VALUE_COL].dropna().values

# Full label/color lists (6 panels)
ALL_LABELS = [ALL_LABEL] + LABELS
ALL_COLORS = [ALL_COLOR] + COLORS

# Shared bin edges
all_vals = np.concatenate(
    list(subsets_all.values()) + list(subsets_complete.values())
)
x_min = np.floor(all_vals.min() / BIN_WIDTH) * BIN_WIDTH
x_max = np.ceil(all_vals.max() / BIN_WIDTH) * BIN_WIDTH
bins  = np.arange(x_min, x_max + BIN_WIDTH, BIN_WIDTH)

# ── Plotting helper ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':       FONT_FAMILY,
    'font.size':         FONT_SIZE,
    'axes.linewidth':    AXIS_LW,
    'xtick.major.width': AXIS_LW,
    'ytick.major.width': AXIS_LW,
    'xtick.labelsize':   TICK_SIZE,
    'ytick.labelsize':   TICK_SIZE,
})


def make_figure(subsets, title_suffix, out_svg, out_png):
    n_panels = len(ALL_LABELS)
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI_DISPLAY)
    gs  = gridspec.GridSpec(n_panels, 1, figure=fig, hspace=0.55)

    axes = []
    for i, (label, color) in enumerate(zip(ALL_LABELS, ALL_COLORS)):
        ax   = fig.add_subplot(gs[i])
        vals = subsets[label]
        ax.hist(vals, bins=bins, color=color, alpha=ALPHA,
                edgecolor='none', linewidth=0)

        med = np.median(vals)
        ax.axvline(med, color=color, lw=0.8, ls='--')
        ax.axvline(0, color='#999999', lw=0.4, ls='-', zorder=0)

        # Shapiro-Wilk test
        if len(vals) >= 3:
            sw_stat, sw_p = stats.shapiro(vals)
            if sw_p < 0.001:
                p_str = f'p < 0.001'
            else:
                p_str = f'p = {sw_p:.3f}'
            sw_text = f'S-W: W = {sw_stat:.3f}, {p_str}'
        else:
            sw_text = 'S-W: n too small'

        ax.set_ylabel('Count', fontsize=FONT_SIZE)
        n_str = f'n = {len(vals)}'
        ax.set_title(f'{label}{title_suffix}   ({n_str})',
                     fontsize=FONT_SIZE, fontweight='bold',
                     loc='left', pad=2)

        # Add Shapiro-Wilk result to upper-right corner of panel
        ax.text(0.98, 0.95, sw_text,
                transform=ax.transAxes,
                fontsize=FONT_SIZE - 1,
                ha='right', va='top',
                color='#333333',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', edgecolor='#cccccc',
                          alpha=0.8, linewidth=0.4))

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', which='both', length=2)

        axes.append(ax)

    axes[-1].set_xlabel('ΔH (H_all − H_fast)', fontsize=FONT_SIZE)
    for ax in axes[:-1]:
        ax.set_xticklabels([])
    for ax in axes:
        ax.set_xlim(x_min, x_max)

    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    fig.savefig(out_png, format='png', dpi=DPI_SAVE, bbox_inches='tight')
    print(f'Saved:\n  {out_svg}\n  {out_png}')
    plt.show()


# ── Generate both figures ─────────────────────────────────────────────────────
make_figure(subsets_all, '', OUTPUT_SVG, OUTPUT_PNG)
make_figure(subsets_complete, ' (complete only)', OUTPUT_SVG_COMPLETE, OUTPUT_PNG_COMPLETE)

# ── Reporting helper ──────────────────────────────────────────────────────────

def run_report(pool, pool_name):
    print(f'\n{"─" * 70}')
    print(f'  POOL: {pool_name}')
    print(f'{"─" * 70}')

    # --- 1. ΔH == 0 per quadrant ---
    print(f'\n  --- ΔH (H_all-H_fast) == 0 per group ---')

    dh_all = pool[VALUE_COL].dropna()
    n_zero_all = (dh_all == 0).sum()
    pct = 100 * n_zero_all / len(dh_all) if len(dh_all) > 0 else 0
    print(f'    {"All positions":20s}:  {n_zero_all} / {len(dh_all)}  ({pct:.1f}%)')

    for grp, label in zip(GROUPS, LABELS):
        grp_df = pool[pool[grp] == 'Y']
        dh_vals = grp_df[VALUE_COL].dropna()
        n_zero = (dh_vals == 0).sum()
        pct = 100 * n_zero / len(dh_vals) if len(dh_vals) > 0 else 0
        print(f'    {label:20s}:  {n_zero} / {len(dh_vals)}  ({pct:.1f}%)')

    # --- 2. H_all >= 1.0 → kcatC > 0.8 ---
    print(f'\n  --- Shannon Entropy all FIIs (H_all) >= 1.0 ---')
    any_q_mask = (pool['Q1'] == 'Y') | (pool['Q2'] == 'Y') | \
                 (pool['Q3'] == 'Y') | (pool['Q4'] == 'Y')
    q_pool = pool[any_q_mask]

    h_all_valid = q_pool[ENTROPY_COL].dropna()
    high_entropy_mask = q_pool[ENTROPY_COL] >= 1.0
    n_high_entropy = high_entropy_mask.sum()
    print(f'    Positions with H_all >= 1.0:  {n_high_entropy} / {len(h_all_valid)}')

    high_entropy_df = q_pool[high_entropy_mask]
    kcat_valid = high_entropy_df[KCAT_COL].dropna()
    n_kcat_high = (kcat_valid > 0.8).sum()
    pct = 100 * n_kcat_high / len(kcat_valid) if len(kcat_valid) > 0 else 0
    print(f'    Of those, kcatC normalized > 0.8:  {n_kcat_high} / {len(kcat_valid)}  '
          f'({pct:.1f}%)')

    # --- 3. ΔH >= 0.25 → kcatC < 0.8 ---
    print(f'\n  --- ΔH (H_all-H_fast) >= 0.265 ---')
    dh_valid = q_pool[VALUE_COL].dropna()
    high_dh_mask = q_pool[VALUE_COL] >= 0.265
    n_high_dh = high_dh_mask.sum()
    print(f'    Positions with ΔH >= 0.265:  {n_high_dh} / {len(dh_valid)}')

    high_dh_df = q_pool[high_dh_mask]
    kcat_valid_dh = high_dh_df[KCAT_COL].dropna()
    n_kcat_high_dh = (kcat_valid_dh < 0.8).sum()
    pct = 100 * n_kcat_high_dh / len(kcat_valid_dh) if len(kcat_valid_dh) > 0 else 0
    print(f'    Of those, kcatC normalized < 0.8:  {n_kcat_high_dh} / {len(kcat_valid_dh)}  '
          f'({pct:.1f}%)')

    # --- 4. ΔH < 0.25 → how many have measured kcat values ---
    print(f'\n  --- ΔH (H_all-H_fast) < 0.265 ---')
    low_dh_mask = q_pool[VALUE_COL] < 0.265
    n_low_dh = low_dh_mask.sum()
    print(f'    Positions with ΔH < 0.265:  {n_low_dh} / {len(dh_valid)}')

    low_dh_df = q_pool[low_dh_mask]
    kcat_has_value = low_dh_df[KCAT_COL].dropna()
    n_has_kcat = len(kcat_has_value)
    pct = 100 * n_has_kcat / n_low_dh if n_low_dh > 0 else 0
    print(f'    Of those, have measured kcatC value:  {n_has_kcat} / {n_low_dh}  '
          f'({pct:.1f}%)')

    n_kcat_high_low = (kcat_has_value < 0.8).sum()
    pct2 = 100 * n_kcat_high_low / n_has_kcat if n_has_kcat > 0 else 0
    print(f'    Of those with kcat, kcatC normalized < 0.8:  {n_kcat_high_low} / {n_has_kcat}  '
          f'({pct2:.1f}%)')


# ── Run reports for both pools ────────────────────────────────────────────────
print('\n' + '=' * 70)
print('REPORT')
print('=' * 70)

run_report(df, 'ALL')

df_complete = df[df['Complete data?'] == 'Y'].copy()
run_report(df_complete, 'COMPLETE DATA ONLY')

print('\n' + '=' * 70)


# In[ ]:
