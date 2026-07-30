#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# === Paths ===
# Resolves relative to this script's location, so it works regardless of
# the directory the script is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(SCRIPT_DIR, "Data Files") + os.sep

full_fasta = base_dir + 'FIIsMiloOnly90cutoffprioritysorted_aligned_88seqs.fasta'
output_csv = base_dir + 'entropycalculation.csv'

# === Load alignment ===
records = list(SeqIO.parse(full_fasta, "fasta"))

# === Group sequences ===
ultrafast = [str(r.seq) for r in records if "ultrafast" in r.description.lower()]
slow      = [str(r.seq) for r in records if "slow"      in r.description.lower()]
all_seqs  = [str(r.seq) for r in records]

# === Get aligned reference sequence ===
def get_ref_seq(records, keyword="OGS68397.1|_UltraFast"):
    for rec in records:
        if keyword.lower() in rec.description.lower():
            return str(rec.seq)
    raise ValueError(f"Reference '{keyword}' not found.")

ref_aligned = get_ref_seq(records)

# === Build residue number map ===
ref_res_nums = []
res_idx = 0
for aa in ref_aligned:
    if aa != "-":
        res_idx += 1
        ref_res_nums.append(res_idx)
    else:
        ref_res_nums.append(np.nan)

# === Compute entropy ===
def compute_entropy(sequences):
    entropy = []
    for i in range(len(sequences[0])):
        col = [s[i] for s in sequences if s[i] not in ["-", "."]]
        if len(col) <= 1:
            entropy.append(np.nan)
            continue
        if len(set(col)) == 1:
            entropy.append(0.0)
            continue
        freqs = Counter(col)
        probs = np.array(list(freqs.values())) / len(col)
        entropy.append(-np.sum(probs * np.log2(probs)))
    return entropy

ultrafast_entropy = compute_entropy(ultrafast)
slow_entropy      = compute_entropy(slow)
all_entropy       = compute_entropy(all_seqs)

# === Build DataFrame ===
df = pd.DataFrame({
    "MSA_Column":             range(1, len(ref_aligned) + 1),
    "Ref_Residue":            list(ref_aligned),
    "Protein_Residue_Number": ref_res_nums,
    "UltraFast_Entropy":      ultrafast_entropy,
    "Slow_Entropy":           slow_entropy,
    "All_Entropy":            all_entropy,
    "Delta_Entropy":          np.array(all_entropy) - np.array(ultrafast_entropy),
})

# === Save ===
df.to_csv(output_csv, index=False)
print(f"✅ UltraFast sequences: {len(ultrafast)}")
print(f"✅ Slow sequences:      {len(slow)}")
print(f"✅ All sequences:       {len(all_seqs)}")
print(f"✅ Output saved to:     {output_csv}")

# ============================================================
# === GAUSSIAN MIXTURE MODEL ON DELTA ENTROPY ===
# ============================================================

def run_gmm(delta_series, all_entropy_series, ultrafast_entropy_series, 
            title_suffix, filename_suffix):
    """Fit GMM and plot results for a given delta entropy array."""
    
    delta_vals = delta_series.dropna().values.reshape(-1, 1)

    # Fit GMMs k=1 to 5, select by BIC
    max_components = 5
    bic_scores = []
    models = []
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(delta_vals)
        bic_scores.append(gmm.bic(delta_vals))
        models.append(gmm)

    best_k   = np.argmin(bic_scores) + 1
    best_gmm = models[best_k - 1]

    print(f"\n📊 GMM Results ({title_suffix}):")
    print(f"   Positions analysed: {len(delta_vals)}")
    print(f"   Best fit: {best_k} component(s) (lowest BIC = {bic_scores[best_k-1]:.2f})")
    for i, (mean, var, weight) in enumerate(zip(
            best_gmm.means_.flatten(),
            best_gmm.covariances_.flatten(),
            best_gmm.weights_.flatten())):
        print(f"   Component {i+1}: μ={mean:.3f}, σ={np.sqrt(var):.3f}, weight={weight:.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Delta entropy GMM — {title_suffix}', fontsize=12)

    # Left: BIC
    axes[0].plot(range(1, max_components + 1), bic_scores, 
                 marker='o', color='steelblue', linewidth=2)
    axes[0].axvline(best_k, color='tomato', linestyle='--', 
                    label=f'Best k={best_k}')
    axes[0].set_xlabel('Number of GMM components')
    axes[0].set_ylabel('BIC score (lower = better)')
    axes[0].set_title('Model selection by BIC')
    axes[0].legend()

    # Right: histogram + GMM
    x_range = np.linspace(delta_vals.min() - 0.2, 
                           delta_vals.max() + 0.2, 1000).reshape(-1, 1)
    counts, _, _ = axes[1].hist(delta_vals, bins=40, density=True,
                                 alpha=0.5, color='steelblue', 
                                 label='Delta entropy')
    axes[1].set_ylim(0, counts.max() * 1.2)

    colors = ['tomato', 'seagreen', 'darkorange', 'purple', 'saddlebrown']
    for i, (mean, var, weight) in enumerate(zip(
            best_gmm.means_.flatten(),
            best_gmm.covariances_.flatten(),
            best_gmm.weights_.flatten())):
        component_curve = weight * norm.pdf(x_range.flatten(), mean, np.sqrt(var))
        axes[1].plot(x_range, component_curve, color=colors[i], linewidth=2,
                     label=f'Component {i+1}: μ={mean:.2f}, σ={np.sqrt(var):.2f}')

    total_density = np.exp(best_gmm.score_samples(x_range))
    axes[1].plot(x_range, total_density, 'k--', linewidth=1.5, 
                 label='Total GMM fit')
    axes[1].set_xlabel('Delta entropy (All − UltraFast)')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'GMM fit (k={best_k})')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    save_path = base_dir + f'delta_entropy_GMM_{filename_suffix}.svg'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Plot saved to: {save_path}")


# --- Analysis 1: all positions including universally conserved ---
run_gmm(
    delta_series             = df["Delta_Entropy"],
    all_entropy_series       = df["All_Entropy"],
    ultrafast_entropy_series = df["UltraFast_Entropy"],
    title_suffix             = "all positions",
    filename_suffix          = "all_positions"
)

# --- Analysis 2: exclude positions fully conserved in both groups ---
mask = (
    df["Delta_Entropy"].notna() &
    ~(df["All_Entropy"] == 0)
)
run_gmm(
    delta_series             = df.loc[mask, "Delta_Entropy"],
    all_entropy_series       = df.loc[mask, "All_Entropy"],
    ultrafast_entropy_series = df.loc[mask, "UltraFast_Entropy"],
    title_suffix             = "variable positions only",
    filename_suffix          = "variable_positions"
)

print(f"\nPositions excluded as universally conserved: {(~mask).sum()}")
print(f"Positions retained for variable analysis:    {mask.sum()}")

# ============================================================
# === QUANTILE BINNING OF DELTA ENTROPY ===
# ============================================================

# Work on variable positions only (same mask as GMM analysis 2)
df_var = df[mask].copy()

# Assign quantile bins
df_var["Delta_Quantile"] = pd.qcut(df_var["Delta_Entropy"], q=4, 
                                    labels=["Q1_low", "Q2", "Q3", "Q4_high"])

# Top quarter — positions where UltraFast is most constrained vs full library
top_quarter = df_var[df_var["Delta_Quantile"] == "Q4_high"].copy()
top_quarter = top_quarter.sort_values("Delta_Entropy", ascending=False)

print(f"\n📊 Quantile binning results:")
print(f"   Total variable positions: {len(df_var)}")
print(f"   Top quarter (Q4) positions: {len(top_quarter)}")
print(f"   Delta entropy range in Q4: {top_quarter['Delta_Entropy'].min():.3f} – {top_quarter['Delta_Entropy'].max():.3f}")
print(f"\n   Top 20 candidate positions:")
print(top_quarter[["Protein_Residue_Number", "Ref_Residue", 
                    "UltraFast_Entropy", "All_Entropy", 
                    "Delta_Entropy"]].head(20).to_string(index=False))

# Save top quarter to its own CSV
top_quarter_path = base_dir + 'delta_entropy_top_quarter_candidates.csv'
top_quarter.to_csv(top_quarter_path, index=False)
print(f"\n✅ Top quarter candidates saved to: {top_quarter_path}")

# Also save the full variable-positions DataFrame with quantile labels
df_var_path = base_dir + 'delta_entropy_variable_positions_quantiled.csv'
df_var.to_csv(df_var_path, index=False)
print(f"✅ Full quantiled variable positions saved to: {df_var_path}")


# In[2]:


# this merges the entropy analysis with the normalized kcat data

import pandas as pd
from Bio import AlignIO
from collections import Counter

# === Step 1: Load alanine scan data ===
alanine_df = pd.read_csv('alanine_scan_GspRBC.csv')
alanine_df.columns = alanine_df.columns.str.strip()

# Standardize the column name
if 'position' in alanine_df.columns:
    alanine_df.rename(columns={'position': '#'}, inplace=True)

# === Step 2: Load MSA and extract UltraFast sequence ===
msa_file = '90%_ID_cutoff_aligned.fasta'
records = list(AlignIO.read(msa_file, "fasta"))
ultrafast_seq = str(next(r for r in records if "OGS68397.1|_UltraFast" in r.description).seq)

# === Step 3: Map # → MSA alignment column ===
def map_alanine_positions_to_msa(ultrafast_seq, alanine_positions):
    mapping = {}
    aa_counter = 0
    for msa_index, aa in enumerate(ultrafast_seq):
        if aa != "-":
            aa_counter += 1
        if aa_counter in alanine_positions:
            mapping[aa_counter] = msa_index + 1  # MSA is 1-based
            if len(mapping) == len(alanine_positions):
                break
    return mapping

# Treat "#" as the true protein residue number
alanine_df["Protein_Residue_Number"] = alanine_df["#"]

# === Step 4: Load entropy data and merge in correct direction ===
entropy_df = pd.read_csv('entropy_allFII90ID_sequences.csv')
entropy_df.columns = entropy_df.columns.str.strip()
entropy_df.rename(columns={'Alignment_Column': 'Protein_Residue_Number'}, inplace=True)

# Merge FROM entropy data to retain all conservation sites
merged_df = pd.merge(entropy_df, alanine_df, how='left', on='Protein_Residue_Number')

# === Step 5: Save and report ===
output_path = "/Users/leah-shihlab/Desktop/merged_conservation_allFII90ID.csv"
merged_df.to_csv(output_path, index=False)

print(f"✅ Merged dataset saved to: {output_path}")
print(f"🧬 Total positions with entropy: {len(entropy_df)}")
print(f"✅ Positions with both entropy and kcat data: {merged_df['normalized_kcat_bc'].notna().sum()}")


# In[41]:


from Bio import AlignIO
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# === CONFIGURATION ===
fasta_file = '90cutoffprioritysorted_aligned_88seqs.fasta'
csv_path = 'Supplementarydata.csv'
df = pd.read_excel(excel_path, sheet_name='S2')
df.columns = df.columns.str.strip()

# === Column names from S2 ===
entropy_col = 'Shannon Entropy all FIIs (H_all)'
kcat_col    = 'GspRBC Ala variant kcatC normalized'
ultrafast_id = "gi|1085068552|gb|OGS68397.1|_UltraFast"
block_size = 50
start_res = "MDQSN"
end_res = "GV---HK"

# === Load Data ===
# === Load Data ===
df = pd.read_excel(excel_path, sheet_name='S2')
df.columns = df.columns.str.strip()
df = df[df['GspRBC Residue #'].notna()].copy()
df['GspRBC Residue #'] = df['GspRBC Residue #'].astype(int)


# === Load MSA ===
msa = AlignIO.read(fasta_file, "fasta")
ultrafast_seq = [str(rec.seq) for rec in msa if ultrafast_id in rec.description][0]

# === Trim ultrafast sequence to region of interest ===
start_index = ultrafast_seq.find(start_res)
end_index = ultrafast_seq.find(end_res)
if start_index == -1 or end_index == -1:
    raise ValueError("Start or end sequence not found in ultrafast reference.")
end_index += len(end_res)
ultrafast_seq = ultrafast_seq[start_index:end_index]

# === Map alignment positions (include gaps) ===
position_index = []
res_counter = 1
for i, aa in enumerate(ultrafast_seq):
    pos = res_counter if aa != "-" else None
    position_index.append((i + 1, aa, pos))  # MSA column, residue, protein position
    if aa != "-":
        res_counter += 1

blocks = [position_index[i:i+block_size] for i in range(0, len(position_index), block_size)]

# === Truncate colormaps to emphasize white ===
def emphasize_white(cmap, minval=0.0, maxval=0.7, n=100):
    return LinearSegmentedColormap.from_list(
        f'{cmap.name}_truncated',
        cmap(np.linspace(minval, maxval, n))
    )

# === Color maps ===
kcat_norm    = mcolors.Normalize(vmin=df[kcat_col].min(),    vmax=df[kcat_col].max())
entropy_norm = mcolors.Normalize(vmin=df[entropy_col].min(), vmax=df[entropy_col].max())
kcat_cmap    = emphasize_white(cm.Reds_r,  0.0, 1.5)
entropy_cmap = emphasize_white(cm.Blues_r, 0.0, 2)

kcat_map = df.set_index('GspRBC Residue #')[kcat_col].apply(
    lambda x: kcat_cmap(kcat_norm(x)) if pd.notna(x) else (0, 0, 0, 1)
).to_dict()
entropy_map = df.set_index('GspRBC Residue #')[entropy_col].apply(
    lambda x: entropy_cmap(entropy_norm(x)) if pd.notna(x) else (0, 0, 0, 1)
).to_dict()

low_entropy_set = set(df[df[entropy_col] < -0.405]['GspRBC Residue #'])

# === Plot layout ===
fig, axs = plt.subplots(len(blocks)*3, 1, figsize=(block_size * 0.3, len(blocks)*1.5),
                        gridspec_kw={'height_ratios': [0.6, 0.6, 0.8]*len(blocks)})
fig.subplots_adjust(hspace=0.2)  # Space between kcat and entropy bars

for block_index, block in enumerate(blocks):
    kcat_colors = [kcat_map.get(pos, (0, 0, 0, 1)) for _, _, pos in block]
    entropy_colors = [entropy_map.get(pos, (0, 0, 0, 1)) for _, _, pos in block]

    # === kcat row ===
    axs[block_index * 3].imshow([kcat_colors], aspect="equal")
    axs[block_index * 3].set_yticks([0])
    axs[block_index * 3].set_yticklabels(["kcat"])
    axs[block_index * 3].set_xticks([])
    axs[block_index * 3].set_xlim(-0.5, len(block) - 0.5)

    # === entropy row ===
    axs[block_index * 3 + 1].imshow([entropy_colors], aspect="equal")
    axs[block_index * 3 + 1].set_yticks([0])
    axs[block_index * 3 + 1].set_yticklabels(["Entropy"])
    axs[block_index * 3 + 1].set_xticks([])
    axs[block_index * 3 + 1].set_xlim(-0.5, len(block) - 0.5)

    # === label row ===
    axs[block_index * 3 + 2].set_xlim(-0.5, len(block) - 0.5)
    axs[block_index * 3 + 2].set_ylim(-1, 1)
    axs[block_index * 3 + 2].axis("off")
    for i, (_, aa, pos) in enumerate(block):
        color = "red" if pos in low_entropy_set else "black"
        axs[block_index * 3 + 2].text(i, 0.4, aa, ha='center', va='center',
                                      fontsize=10, fontweight='bold', family='monospace',
                                      color=color)
        axs[block_index * 3 + 2].text(i, -0.4, str(pos) if pos is not None else "-", 
                                      ha='center', va='center',
                                      fontsize=9, color='black')

# === Colorbars ===
sm_kcat = cm.ScalarMappable(cmap=kcat_cmap, norm=kcat_norm)
sm_entropy = cm.ScalarMappable(cmap=entropy_cmap, norm=entropy_norm)
cbar_ax1 = fig.add_axes([0.92, 0.5, 0.010, 0.25])   # kcat colorbar
cbar_ax2 = fig.add_axes([0.92, 0.2, 0.010, 0.25])   # entropy colorbar
fig.colorbar(sm_kcat, cax=cbar_ax1, orientation='vertical', label='kcat')
fig.colorbar(sm_entropy, cax=cbar_ax2, orientation='vertical', label='Entropy')

# === Save and Show ===
output_path = "/Users/leah-shihlab/Desktop/conservation_alanine_mapped_blocks_BY_PROTEIN_RESIDUE_NUMBER.svg"
plt.savefig(output_path, format="svg", dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Exported figure saved to: {output_path}")


# In[ ]:





# In[25]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# ============================================
# DISPLAY CONTROL - ADJUST THIS!
# ============================================
DISPLAY_SCALE = 3  # Multiply figure size for Jupyter viewing (use 2-4)
# ============================================

# === CONFIGURATION ===
point_color = "#001F5B"      # Customize: blue scatter points
regression_color = "darkred" # Customize: regression line color
alpha_value = 0.6            # Point transparency
output_svg = 'entropy_allFII90IDvskcarC.svg'

# === Load data from Excel S2 tab ===
excel_path = 'SupplementaryData.csv'
df = pd.read_excel(excel_path, sheet_name='S2')
df.columns = df.columns.str.strip()

# === Column names from S2 ===
entropy_col = 'Shannon Entropy all FIIs (H_all)'
kcat_col    = 'GspRBC Ala variant kcatC normalized'

# === Filter usable data ===
scatter_df = df[df[kcat_col].notna() & df[entropy_col].notna()]

# === Correlation ===
r, p = stats.pearsonr(scatter_df[entropy_col], scatter_df[kcat_col])
r2 = r**2

# === single column width ===
width_mm = 88
width_inches = width_mm / 25.4
height_inches = width_inches * 0.75  # 4:3 aspect ratio

display_width = width_inches * DISPLAY_SCALE
display_height = height_inches * DISPLAY_SCALE

print(f"Figure dimensions:")
print(f"  Output: {width_mm}mm x {height_inches*25.4:.1f}mm")
print(f"  Display: {display_width:.1f}\" x {display_height:.1f}\"")
print(f"\nPearson r² = {r2:.3f}, p = {p:.2e}")

# ============================================
# DISPLAY VERSION (for Jupyter)
# ============================================

font_scale = DISPLAY_SCALE * 0.5

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 6 * font_scale
plt.rcParams['axes.linewidth'] = 0.5 * font_scale
plt.rcParams['xtick.major.width'] = 0.5 * font_scale
plt.rcParams['ytick.major.width'] = 0.5 * font_scale

fig, ax = plt.subplots(figsize=(display_width, display_height))

sns.regplot(
    data=scatter_df,
    x=entropy_col,
    y=kcat_col,
    ax=ax,
    scatter_kws={'alpha': alpha_value, 'color': point_color, 's': 20 * font_scale},
    line_kws={'color': regression_color, 'linewidth': 0.8 * font_scale}
)

ax.set_xlabel("Shannon Entropy (All Sequences)", fontsize=7 * font_scale)
ax.set_ylabel("Normalized k$_{cat}$C", fontsize=7 * font_scale)
ax.set_title("Conservation vs Catalytic Activity", fontsize=7 * font_scale)

ax.text(
    0.05, 0.95,
    f"Pearson r² = {r2:.2f}\np = {p:.2e}",
    transform=ax.transAxes,
    ha='left', va='top', fontsize=6 * font_scale,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', linewidth=0.5 * font_scale)
)

ax.tick_params(axis='both', which='major', labelsize=6 * font_scale,
               width=0.5 * font_scale, length=2 * font_scale)

plt.tight_layout()
plt.show()

# ============================================
# PUBLICATION VERSION (Journal format)
# ============================================

print("\nSaving publication version...")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

fig_pub, ax_pub = plt.subplots(figsize=(width_inches, height_inches))

sns.regplot(
    data=scatter_df,
    x=entropy_col,
    y=kcat_col,
    ax=ax_pub,
    scatter_kws={'alpha': alpha_value, 'color': point_color, 's': 20},
    line_kws={'color': regression_color, 'linewidth': 0.8}
)

ax_pub.set_xlabel("Shannon Entropy (All Sequences)", fontsize=7)
ax_pub.set_ylabel("Normalized k$_{cat}$C", fontsize=7)
ax_pub.set_title("Conservation vs Catalytic Activity", fontsize=7)

ax_pub.text(
    0.05, 0.95,
    f"Pearson r² = {r2:.2f}\np = {p:.2e}",
    transform=ax_pub.transAxes,
    ha='left', va='top', fontsize=6,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', linewidth=0.5)
)

ax_pub.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)

plt.tight_layout()
plt.savefig(output_svg, format="svg", bbox_inches='tight', dpi=300)
print(f"✅ Saved SVG to: {output_svg}")
plt.show()


# In[ ]:
