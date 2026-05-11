#!/usr/bin/env python
# coding: utf-8

# In[19]:


#This code looks at the distribution of wild-type on the day (WTOTD), BC and variant values. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# === Load file ===
df = pd.read_excel('pathname/Tidy_Consolidated_Kinetics_Data.xlsx')

# === Clean column names ===
df.columns = df.columns.str.strip().str.lower()
df['tags'] = df['tags'].fillna('').astype(str).str.strip().str.upper()

# === Confirm tag values ===
print("✅ Unique tag values:", sorted(df['tags'].unique()))

# === Extract WTOTD and BC groups ===
groupWT = df[df['tags'] == 'WTOTD']['kcat'].dropna()
groupBC = df[df['tags'] == 'BC']['kcat'].dropna()

print("\n--- WTOTD vs. BC Analysis ---")
print("WTOTD kcat:", groupWT.values)
print("BC kcat:", groupBC.values)

# === Summary stats
def summary_stats(label, data):
    median = data.median()
    stddevp = data.std(ddof=0)
    print(f"{label} — Median: {median:.4f}, StdDevP: {stddevp:.4f}")
    return median, stddevp

median_WT, stddevp_WT = summary_stats("WTOTD", groupWT)
median_BC, stddevp_BC = summary_stats("BC", groupBC)

combined = pd.concat([groupWT, groupBC])
median_combined, stddevp_combined = summary_stats("Combined", combined)

# === T-test
t_stat, p_value = stats.ttest_ind(groupWT, groupBC, nan_policy='omit')
print(f"\nT-test: t-stat={t_stat:.4f}, p={p_value:.4g}")
print("→ Statistically significant." if p_value < 0.05 else "→ Not statistically significant.")

# === Shapiro-Wilk Normality
for label, group in {"WTOTD": groupWT, "BC": groupBC}.items():
    if len(group) >= 3:
        stat, p = stats.shapiro(group)
        print(f"{label} Shapiro-Wilk: Stat={stat:.4f}, p={p:.4g}")
    else:
        print(f"{label} — Not enough data for Shapiro-Wilk test.")

# === Mann-Whitney U
u_stat, p_mwu = stats.mannwhitneyu(groupWT, groupBC, alternative='two-sided')
print(f"\nMann-Whitney U: U={u_stat}, p={p_mwu:.4g}")
print("→ Distributions differ." if p_mwu < 0.05 else "→ No strong difference.")

# === Distribution plots
for label, group in {"WTOTD": groupWT, "BC": groupBC}.items():
    print(f"\n--- {label} Distribution ---")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(group, kde=True, bins=10, color='teal' if label == "WTOTD" else 'steelblue')
    plt.title(f"{label} Histogram")
    plt.xlabel("kcat")

    plt.subplot(1, 2, 2)
    stats.probplot(group, dist="norm", plot=plt)
    plt.title(f"{label} Q-Q Plot")
    plt.tight_layout()
    plt.show()

    skew = stats.skew(group)
    kurt = stats.kurtosis(group)
    print(f"{label} Skewness: {skew:.4f}")
    print(f"{label} Kurtosis: {kurt:.4f}")

    if skew > 0.5:
        log_data = np.log(group)
        plt.figure()
        stats.probplot(log_data, dist="norm", plot=plt)
        plt.title(f"{label} Q-Q Plot (Log-Transformed)")
        plt.tight_layout()
        plt.show()

# === Boxplot for WTOTD vs. BC
plot_df = pd.DataFrame({
    'kcat': np.concatenate([groupWT.values, groupBC.values]),
    'Group': ['WTOTD'] * len(groupWT) + ['BC'] * len(groupBC)
})

plt.figure(figsize=(6, 6))
sns.boxplot(data=plot_df, x='Group', y='kcat', palette={'WTOTD': 'teal', 'BC': 'steelblue'})
sns.stripplot(data=plot_df, x='Group', y='kcat', color='black', size=4, alpha=0.6, jitter=True)
plt.title('kcat Comparison: WTOTD vs. BC')
plt.ylabel('kcat (s⁻¹)')
plt.xlabel('')
plt.tight_layout()
plt.show()

# === OTHER VARIANTS ANALYSIS ===
print("\n=== Analysis of Untagged Variants (Not BC or WTOTD) ===")
other_variants = df[
    (df['#'].between(1, 470)) &
    (df['tags'] != 'BC') &
    (df['tags'] != 'WTOTD') &
    (df['kcat'].notna())
]

if other_variants.empty:
    print("No untagged variant data found.")
else:
    groupV = other_variants['kcat']
    print(f"\nOther Variants: N = {len(groupV)}")
    median = groupV.median()
    stddev = groupV.std(ddof=0)
    skew = stats.skew(groupV)
    kurt = stats.kurtosis(groupV)
    print(f"Median: {median:.4f}, StdDevP: {stddev:.4f}")
    print(f"Skewness: {skew:.4f}, Kurtosis: {kurt:.4f}")

    # Histogram
    plt.figure(figsize=(6, 5))
    sns.histplot(groupV, kde=True, bins=12, color='slategray')
    plt.title('Untagged Variants Histogram')
    plt.xlabel('kcat')
    plt.tight_layout()
    plt.show()

    # Q-Q Plot
    plt.figure(figsize=(6, 5))
    stats.probplot(groupV, dist="norm", plot=plt)
    plt.title('Untagged Variants Q-Q Plot')
    plt.tight_layout()
    plt.show()

    # Boxplot with WTOTD and BC
    full_df = pd.DataFrame({
        'kcat': np.concatenate([groupWT.values, groupBC.values, groupV.values]),
        'Group': ['WTOTD'] * len(groupWT) + ['BC'] * len(groupBC) + ['Other'] * len(groupV)
    })

    plt.figure(figsize=(7, 6))
    sns.boxplot(data=full_df, x='Group', y='kcat', palette={'WTOTD': 'teal', 'BC': 'steelblue', 'Other': 'gray'})
    sns.stripplot(data=full_df, x='Group', y='kcat', color='black', size=4, alpha=0.6, jitter=True)
    plt.title('kcat Comparison: WTOTD vs. BC vs. Other Variants')
    plt.ylabel('kcat (s⁻¹)')
    plt.xlabel('')
    plt.tight_layout()
    plt.show()


# In[4]:


#this code normalizes the kcatC data to that of the blind control median
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


DISPLAY_SCALE = 3  # Multiply figure size for Jupyter viewing (use 2-4)


# === File path ===
excel_path = 'pathname/Tidy_Consolidated_Kinetics_Data.xlsx'

# === Load data ===
df = pd.read_excel(excel_path)
df.columns = df.columns.str.strip().str.lower()

# === Normalize and clean tags ===
df['tags'] = df['tags'].fillna('').astype(str).str.strip().str.upper()
df['#'] = pd.to_numeric(df['#'], errors='coerce')

# === Define group column using tags ===
def classify_from_tags(row):
    if row['tags'] == 'WTOTD':
        return 'WT'
    elif row['tags'] == 'BC':
        return 'BC'
    elif pd.notna(row['#']) and row['#'] > 0:
        return 'Variant'
    else:
        return 'Unknown'

df['group'] = df.apply(classify_from_tags, axis=1)

# === Activity labeling ===
def label_activity(value):
    if pd.isna(value):
        return "Missing"
    elif value == 0:
        return "Inactive"
    else:
        return "Active"

df['activity_label'] = df['kcat'].apply(label_activity)

# === Summary statistics from BC group ===
bc_vals = df[(df['group'] == 'BC') & df['kcat'].notna()]['kcat']
bc_median = np.nanmedian(bc_vals)
bc_std = np.nanstd(bc_vals)

# === Normalize by BC median ===
df['normalized_kcat_bc'] = df['kcat'] / bc_median

# === Output file (now includes relative abundance) ===
output_cols = ['#', 'kcat', 'normalized_kcat_bc', 'relative abundance', 'group', 'activity_label']
output_df = df[output_cols].copy()
output_df.to_csv("/Users/leah-shihlab/Desktop/alanine_scan_normalized_output.csv", index=False)

# === Display summary ===
print(f"BC Median: {bc_median:.4f}")
print(f"BC StdDev: {bc_std:.4f}")

# === Filter data for plotting (exclude WTOTD) ===
plot_df = df[df['tags'] != 'WTOTD'].copy()

print(f"\nTotal variants: {len(df)}")
print(f"Variants for plotting (excluding WTOTD): {len(plot_df)}")
print(f"Variants with normalized_kcat_bc = 0: {len(plot_df[plot_df['normalized_kcat_bc'] == 0])}")
print(f"Variants with normalized_kcat_bc > 0: {len(plot_df[plot_df['normalized_kcat_bc'] > 0])}")

# === Calculate correlation statistics (excluding WTOTD and NaN values) ===
# Remove rows with NaN in either column
corr_df = plot_df[['relative abundance', 'normalized_kcat_bc']].dropna()

# Pearson correlation
pearson_r, pearson_p = stats.pearsonr(corr_df['relative abundance'], corr_df['normalized_kcat_bc'])
pearson_r2 = pearson_r ** 2

# Spearman correlation
spearman_rho, spearman_p = stats.spearmanr(corr_df['relative abundance'], corr_df['normalized_kcat_bc'])
spearman_rho2 = spearman_rho ** 2

print(f"\n=== Correlation Statistics ===")
print(f"Pearson r: {pearson_r:.3f} (p = {pearson_p:.2e})")
print(f"Pearson R²: {pearson_r2:.3f}")
print(f"Spearman ρ: {spearman_rho:.3f} (p = {spearman_p:.2e})")
print(f"Spearman ρ²: {spearman_rho2:.3f}")

# ============================================
# DISPLAY VERSION (for Jupyter)
# ============================================

# Nature single column width: 88 mm = 3.465 inches
width_mm = 88
width_inches = width_mm / 25.4
height_inches = width_inches * 0.75  # 4:3 aspect ratio

display_width = width_inches * DISPLAY_SCALE
display_height = height_inches * DISPLAY_SCALE

print(f"\nFigure dimensions:")
print(f"  Output: {width_mm}mm x {height_inches*25.4:.1f}mm")
print(f"  Display: {display_width:.1f}\" x {display_height:.1f}\"")

font_scale = DISPLAY_SCALE * 0.5

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 6 * font_scale
plt.rcParams['axes.linewidth'] = 0.5 * font_scale
plt.rcParams['xtick.major.width'] = 0.5 * font_scale
plt.rcParams['ytick.major.width'] = 0.5 * font_scale

fig, ax = plt.subplots(figsize=(display_width, display_height))

# Scatter plot
ax.scatter(plot_df['relative abundance'], plot_df['normalized_kcat_bc'], 
           alpha=0.6, s=10*font_scale, color='#2E86AB', edgecolors='black', linewidth=0.3*font_scale)

# Labels and formatting
ax.set_xlabel('Relative Abundance', fontsize=7*font_scale)
ax.set_ylabel('Normalized k$_{cat}$ (BC)', fontsize=7*font_scale)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.3*font_scale)

# Tick parameters
ax.tick_params(axis='both', which='major', labelsize=6*font_scale, width=0.5*font_scale, length=2*font_scale)

# Tight layout
plt.tight_layout()

# Display plot
plt.show()

# ============================================
# PUBLICATION VERSION (Nature format)
# ============================================

print("\nSaving publication version...")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

fig_pub, ax_pub = plt.subplots(figsize=(width_inches, height_inches))

# Scatter plot
ax_pub.scatter(plot_df['relative abundance'], plot_df['normalized_kcat_bc'], 
               alpha=0.6, s=20, color='#2E86AB', edgecolors='black', linewidth=0.3)

# Labels and formatting
ax_pub.set_xlabel('Relative Abundance', fontsize=7)
ax_pub.set_ylabel('Normalized k$_{cat}$ (BC)', fontsize=7)

# Grid
ax_pub.grid(True, alpha=0.3, linestyle='--', linewidth=0.3)

# Tick parameters
ax_pub.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2)

# Tight layout
plt.tight_layout()

# Save to SVG and PNG
plt.savefig('/Users/leah-shihlab/Desktop/relative_abundance_vs_normalized_kcat.svg', 
            bbox_inches='tight', dpi=300)
plt.savefig('/Users/leah-shihlab/Desktop/relative_abundance_vs_normalized_kcat.png', 
            bbox_inches='tight', dpi=300)

print("\n✓ Plot saved to Desktop!")
print("  Filename: relative_abundance_vs_normalized_kcat.svg/png")
print(f"  Pearson R² = {pearson_r2:.3f}")
print(f"  Spearman ρ² = {spearman_rho2:.3f}")


# In[2]:


#this builds a histogram of all data and plots the bc mean std devs for reference
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap

# === Custom emphasize_white function ===
def emphasize_white(cmap, midpoint=0.0, stretch=1.5):
    """Stretches the lower range of a colormap to emphasize white."""
    colors = cmap(np.linspace(0, 1, 256))
    n = len(colors)
    indices = np.linspace(midpoint, 1, n)**stretch
    indices = (indices - indices.min()) / (indices.max() - indices.min())
    stretched_colors = cmap(indices)
    return LinearSegmentedColormap.from_list("stretched", stretched_colors)

# === Load Data ===
df = pd.read_csv('pathname/alanine_scan_normalized_output.csv')
df.columns = df.columns.str.strip().str.lower()
df['group'] = df['group'].astype(str).str.strip()

# === Define plotting function ===
def plot_variant_vs_bc_histogram_by_kcat(
    data,
    title='Histogram of Normalized kcat: WT/BC vs Other Variants',
    filename="filename.svg",
    bin_count=30,
    std_line_color='black',
    axis_label_size=7,
    tick_label_size=6,
    legend_size=6,
    title_size=7,
    annotation_size=5,
):
    # === Separate data into two groups ===
    wt_bc_data = data[data['group'].isin(['WT', 'BC'])]['normalized_kcat_bc'].dropna()
    variant_data = data[~data['group'].isin(['WT', 'BC'])]['normalized_kcat_bc'].dropna()
    
    # Get BC data specifically for reference lines
    bc_data = data[data['group'] == 'BC']['normalized_kcat_bc'].dropna()
    if variant_data.empty:
        print("⚠️ No variant data to plot.")
        return
    
    # === Histogram ===
    counts, bins = np.histogram(variant_data, bins=bin_count)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bar_width = 1.0 * (bins[1] - bins[0])
    
    # === Compute BC reference values ===
    bc_mean = np.mean(bc_data)
    bc_std = np.std(bc_data)
    
    
    # === Print BC standard deviation boundaries ===
    print(f"BC Mean: {bc_mean:.4f}")
    for i in [1, 2, 3]:
        lower = bc_mean - i * bc_std
        upper = bc_mean + i * bc_std
        print(f"  ±{i}σ → Lower: {lower:.4f}, Upper: {upper:.4f}")
    
    # === Normalize bin centers and apply custom-stretched colormap ===
    norm = Normalize(vmin=min(bin_centers), vmax=max(bin_centers))
    stretched_cmap = emphasize_white(cm.Reds_r, midpoint=0.0, stretch=1.5)
    colors = [stretched_cmap(norm(x)) for x in bin_centers]
    
    # === Compute WT/BC histogram for overlay ===
    wt_bc_counts, _ = np.histogram(wt_bc_data, bins=bins)
    
    # === DIMENSIONS: Height fixed, width scales proportionally ===
    figure_height = 2.5  # inches - fixed
    aspect_ratio = 1.5  # Adjust this to control width scaling
    figure_width = figure_height * aspect_ratio
    
    # === Plot with constrained_layout ===
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), 
                           constrained_layout=True, dpi=300)
    
    # Set font to Arial explicitly for this figure
    plt.rcParams['font.family'] = 'Arial'
    
    ax.bar(bin_centers, counts, width=bar_width, color=colors, label='Other Variants')
    ax.bar(bin_centers, wt_bc_counts, width=bar_width, color='grey', alpha=0.5, label='WT/BC')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, label='Inactive')
    
    # Annotate ±1/2/3 SD
    for i in [1, 2, 3]:
        for direction in [-1, 1]:
            x = bc_mean + direction * i * bc_std
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.5)
            ax.text(x, ax.get_ylim()[1]*0.9, f'±{i}σ', 
                   color='black', fontsize=annotation_size, fontname='Arial',
                   ha='right', va='top', rotation=90)
    
    # Set labels with explicit font sizes
    ax.set_xlabel('Normalized kcat (relative to BC mean)', 
                  fontsize=axis_label_size, fontname='Arial')
    ax.set_ylabel('Count', 
                  fontsize=axis_label_size, fontname='Arial')
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size, 
                   width=0.5, length=2)
    ax.set_title(title, fontsize=title_size, fontname='Arial')
    
    # Set tick label font explicitly
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(tick_label_size)
    
    # Set spine width
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    # Legend with explicit font
    legend = ax.legend(fontsize=legend_size, frameon=False, prop={'family': 'Arial'})
    
    plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('.svg', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Raw histogram saved to: {filename}")

# === Run the plot ===
plot_variant_vs_bc_histogram_by_kcat(df)


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Load WT absorbance data ===
wt_file = 'pathname/Absorbance Values for WT in Each Assay.csv'
wt_df = pd.read_csv(wt_file)
wt_df.columns = wt_df.columns.str.strip().str.lower()

wt_values = wt_df['volume wt (int)'].dropna().astype(float)
wt_mean = wt_values.mean()
wt_std = wt_values.std()
wt_mad = np.median(np.abs(wt_values - np.median(wt_values)))

# Normalized range thresholds
thresholds = {
    '1σ_lower': 1 - wt_std / wt_mean,
    '1σ_upper': 1 + wt_std / wt_mean,
    '2σ_lower': 1 - 2 * wt_std / wt_mean,
    '2σ_upper': 1 + 2 * wt_std / wt_mean,
    'MAD_lower': 1 - wt_mad / wt_mean,
    'MAD_upper': 1 + wt_mad / wt_mean
}

# === Print threshold summary ===
print("📊 WT absorbance variation statistics (normalized relative to WT = 1):\n")
print(f"WT mean absorbance: {wt_mean:.2f}")
print(f"WT standard deviation (σ): {wt_std:.2f}")
print(f"WT median absolute deviation (MAD): {wt_mad:.2f}\n")
print("Normalized threshold ranges:")
print(f" - 1σ range:  {thresholds['1σ_lower']:.2f} to {thresholds['1σ_upper']:.2f}")
print(f" - 2σ range:  {thresholds['2σ_lower']:.2f} to {thresholds['2σ_upper']:.2f}")
print(f" - MAD range: {thresholds['MAD_lower']:.2f} to {thresholds['MAD_upper']:.2f}")

# === Load variant solubility data ===
variant_file = '/Users/leah-shihlab/Desktop/Tidy_Consolidated_Kinetics_Data.xlsx'
df = pd.read_excel(variant_file)
df.columns = df.columns.str.strip().str.lower()

# Define helper column
def flag_variation(val):
    if val < thresholds['2σ_lower'] or val > thresholds['2σ_upper']:
        return 'Beyond 2σ'
    elif val < thresholds['1σ_lower'] or val > thresholds['1σ_upper']:
        return 'Beyond 1σ'
    else:
        return 'Within 1σ'

df['solubility_range_flag'] = df['relative abundance'].apply(flag_variation)

# === Plot (variants + BCs overlaid at x ≈ 0) ===
plt.figure(figsize=(10, 5))

# Prepare data
variants = df[df['tags'].isin(['BC', 'WTOTD']) == False].copy()
bcs = df[df['tags'] == 'BC'].copy()

# All dots at x ≈ 0, with small jitter
x_variants = np.random.normal(loc=0, scale=0.08, size=len(variants))
x_bcs = np.random.normal(loc=0, scale=0.08, size=len(bcs))

# Plot variants
plt.scatter(
    x=x_variants,
    y=variants['relative abundance'],
    color='lightgray',
    label='Variants',
    alpha=0.7,
    edgecolors='k',
    linewidths=0.3
)

# Overlay BCs on same axis
plt.scatter(
    x=x_bcs,
    y=bcs['relative abundance'],
    color='teal',
    label='Blind Controls (BC)',
    alpha=0.9,
    edgecolors='k',
    linewidths=0.3
)

# Threshold lines
plt.axhline(thresholds['1σ_upper'], linestyle='--', color='red', label='+1σ')
plt.axhline(thresholds['1σ_lower'], linestyle='--', color='red')
plt.axhline(thresholds['2σ_upper'], linestyle=':', color='orange', label='+2σ')
plt.axhline(thresholds['2σ_lower'], linestyle=':', color='orange')

# Formatting
plt.xticks([0], ['Variants + BCs'])
plt.xlim(-0.5, 0.5)  # Adds horizontal whitespace on both sides
plt.ylabel('Relative Abundance (Normalized to WT)')
plt.title('Overlay of Variant and BC Solubility with WT-Based Thresholds')
plt.legend()
plt.tight_layout()
plt.show()



# In[ ]:




