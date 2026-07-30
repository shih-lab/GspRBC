#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Analyzes aligned protein sequences and lists ALL differences by protein name
import os
import pandas as pd
from collections import defaultdict
import re

def parse_clustal_file(file_path):
    """Parse Clustal alignment file and extract sequences"""
    sequences = {}
    sequence_order = []
    
    print(f"🔍 Reading Clustal file: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find where sequences start (after CLUSTAL header)
    seq_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('CLUSTAL') or line.strip() == '':
            continue
        if any(char.isalpha() for char in line):
            seq_start = i
            break
    
    # Parse sequences
    current_block = []
    for i in range(seq_start, len(lines)):
        line = lines[i].strip()
        
        # Skip empty lines and conservation lines (* : .)
        if not line or line.startswith('*') or line.startswith(':') or line.startswith('.') or all(c in '*:. ' for c in line):
            if current_block:
                # Process current block
                for seq_line in current_block:
                    parts = seq_line.split()
                    if len(parts) >= 2:
                        seq_name = parts[0]
                        seq_fragment = parts[1]
                        
                        if seq_name not in sequences:
                            sequences[seq_name] = ""
                            sequence_order.append(seq_name)
                        sequences[seq_name] += seq_fragment
                current_block = []
            continue
        
        # Add line to current block
        current_block.append(line)
    
    # Process final block
    if current_block:
        for seq_line in current_block:
            parts = seq_line.split()
            if len(parts) >= 2:
                seq_name = parts[0]
                seq_fragment = parts[1]
                
                if seq_name not in sequences:
                    sequences[seq_name] = ""
                    sequence_order.append(seq_name)
                sequences[seq_name] += seq_fragment
    
    print(f"📊 Found {len(sequences)} sequences:")
    for name in sequence_order:
        print(f"   {name}: {len(sequences[name])} residues")
    
    return sequences, sequence_order

def find_all_differences(sequences, sequence_order):
    """Find ALL differences between sequences at every position"""
    if not sequences:
        return {}
    
    # Get alignment length
    alignment_length = len(list(sequences.values())[0])
    print(f"🔍 Analyzing alignment of length {alignment_length}")
    
    # Store differences for each sequence
    differences = defaultdict(list)
    
    # Check each position
    for pos in range(alignment_length):
        # Get residue at this position for each sequence
        residues_at_pos = {}
        for seq_name in sequence_order:
            residues_at_pos[seq_name] = sequences[seq_name][pos] if pos < len(sequences[seq_name]) else '-'
        
        # Check if there are any differences at this position
        unique_residues = set(residues_at_pos.values())
        
        if len(unique_residues) > 1:  # There are differences
            # Record difference for each sequence
            for seq_name in sequence_order:
                residue = residues_at_pos[seq_name]
                differences[seq_name].append({
                    'position': pos + 1,  # 1-based position
                    'residue': residue,
                    'other_residues': [residues_at_pos[other_seq] for other_seq in sequence_order if other_seq != seq_name]
                })
    
    return differences

def create_output_files(differences, sequence_order, output_base_path):
    """Create both CSV and Excel output files with separate columns for each protein"""
    
    # Get all positions with differences
    all_positions = set()
    for seq_name in sequence_order:
        if seq_name in differences:
            for diff in differences[seq_name]:
                all_positions.add(diff['position'])
    
    # Sort positions
    sorted_positions = sorted(all_positions)
    
    # Prepare data for output
    output_data = []
    
    for pos in sorted_positions:
        row = {'Position': pos}
        
        # Add amino acid for each protein at this position
        for seq_name in sequence_order:
            residue = '-'  # Default
            if seq_name in differences:
                for diff in differences[seq_name]:
                    if diff['position'] == pos:
                        residue = diff['residue']
                        break
            row[seq_name] = residue
        
        output_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(output_data)
    
    # Reorder columns to have Position first, then proteins in sequence order
    column_order = ['Position'] + sequence_order
    df = df[column_order]
    
    # Create output files
    csv_file = f"{output_base_path}_differences.csv"
    xlsx_file = f"{output_base_path}_differences.xlsx"
    
    # Save CSV
    df.to_csv(csv_file, index=False)
    print(f"✅ CSV file saved: {csv_file}")
    
    # Save Excel with better formatting
    with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Differences', index=False)
        
        # Get the worksheet to apply formatting
        worksheet = writer.sheets['All_Differences']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 20)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"✅ Excel file saved: {xlsx_file}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"Total positions with differences: {len(sorted_positions)}")
    print(f"Columns in output: Position + {len(sequence_order)} protein columns")
    print(f"Protein columns: {', '.join(sequence_order)}")
    
    # Show a preview of the first few rows
    print(f"\n📋 PREVIEW (first 5 positions):")
    print(df.head().to_string(index=False))

def main():
    print("=" * 60)
    print("🧬 CLUSTAL ALIGNMENT DIFFERENCE ANALYZER")
    print("=" * 60)
    
    # Resolves relative to this script's location (Code/), so it works
    # regardless of the directory the script is launched from.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clustal_file = os.path.join(
        script_dir, "..", "Data Files",
        "clustalo-I20250904-171241-0868-39628282-p1m.aln-clustal_num"
    )
    
    print(f"📁 Using file: {clustal_file}")
    
    # Check if file exists
    if not os.path.exists(clustal_file):
        print(f"❌ Error: File not found at '{clustal_file}'")
        print("Make sure the file path is correct and the file exists.")
        return
    
    try:
        # Parse the Clustal file
        sequences, sequence_order = parse_clustal_file(clustal_file)
        
        if not sequences:
            print("❌ No sequences found in the file.")
            return
        
        # Find all differences
        print(f"\n🔍 Finding ALL differences between {len(sequences)} sequences...")
        differences = find_all_differences(sequences, sequence_order)
        
        # Create output files
        output_base = os.path.splitext(clustal_file)[0]
        create_output_files(differences, sequence_order, output_base)
        
        print(f"\n🎉 Analysis complete!")
        print(f"📂 Output files created in same directory as input file")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


# In[4]:


def apply_criteria_3(merged_df, consensus_threshold=0.8):
    """
    Criteria 3: Strong consensus - RBC4.9 differs from conserved consensus
    ONLY identifies positions where:
    1. >=80% of OTHER proteins agree on one amino acid (strong conservation)
    2. RBC4.9 has a DIFFERENT amino acid from this consensus
    
    If RBC4.9 matches the consensus, position is NOT flagged.
    """
    print(f"\n🎯 Applying Criteria 3: RBC4.9 differs from strong consensus (≥{consensus_threshold*100:.0f}% of others agree)")
    
    protein_columns = [col for col in merged_df.columns if col not in ['Position', '#', 'kcat', 'normalized_kcat_bc', 'group', 'activity_label']]
    
    # Make sure RBC4.9 is in the protein columns
    if 'RBC4.9' not in protein_columns:
        print("⚠️ RBC4.9 not found in protein columns!")
        return pd.DataFrame()
    
    criteria3_positions = []
    
    for _, row in merged_df.iterrows():
        # Get RBC4.9 amino acid
        rbc49_aa = row['RBC4.9']
        if pd.isna(rbc49_aa) or rbc49_aa == '':
            continue
            
        # Get amino acids for all OTHER proteins (excluding RBC4.9)
        other_proteins = [col for col in protein_columns if col != 'RBC4.9']
        other_amino_acids = []
        
        for protein in other_proteins:
            aa#!/usr/bin/env python3
"""
Kinetic Data + Alignment Analyzer
Maps alanine scan kinetic data to protein alignment differences
and identifies positions based on functional and sequence criteria
"""

import pandas as pd
import os
import numpy as np
from collections import Counter

def load_alignment_differences(differences_file):
    """Load the alignment differences CSV file"""
    print(f"🔍 Loading alignment differences: {differences_file}")
    
    df = pd.read_csv(differences_file)
    print(f"📊 Found {len(df)} positions with differences")
    print(f"📋 Proteins in alignment: {', '.join([col for col in df.columns if col != 'Position'])}")
    
    return df

def load_kinetic_data(kinetic_file):
    """Load the alanine scan kinetic data"""
    print(f"🔍 Loading kinetic data: {kinetic_file}")
    
    df = pd.read_csv(kinetic_file)
    print(f"📊 Found {len(df)} kinetic measurements")
    
    # Show data preview
    print(f"📋 Kinetic data columns: {', '.join(df.columns.tolist())}")
    print(f"📈 normalized_kcat_bc range: {df['normalized_kcat_bc'].min():.3f} to {df['normalized_kcat_bc'].max():.3f}")
    
    return df

def merge_data(alignment_df, kinetic_df):
    """Merge alignment differences with kinetic data"""
    print(f"\n🔗 Merging alignment and kinetic data...")
    
    # Merge on position number
    merged_df = pd.merge(alignment_df, kinetic_df, left_on='Position', right_on='#', how='left')
    
    print(f"📊 Merged dataset: {len(merged_df)} positions")
    
    # Show positions with kinetic data
    has_kinetic = merged_df['normalized_kcat_bc'].notna().sum()
    print(f"📈 Positions with kinetic data: {has_kinetic}/{len(merged_df)}")
    
    return merged_df

def apply_criteria_1(merged_df):
    """
    Criteria 1: normalized_kcat_bc < 0.8 AND RBC4.10 != RBC4.9 at that position
    """
    print(f"\n🎯 Applying Criteria 1: Low activity (< 0.8) + RBC4.10 ≠ RBC4.9")
    
    # Filter conditions
    condition1 = merged_df['normalized_kcat_bc'] < 0.8
    condition2 = merged_df['RBC4.10'] != merged_df['RBC4.9']
    
    # Apply both conditions
    criteria1_df = merged_df[condition1 & condition2].copy()
    
    print(f"📊 Positions meeting Criteria 1: {len(criteria1_df)}")
    
    if len(criteria1_df) > 0:
        print(f"📋 Preview of positions:")
        for _, row in criteria1_df.head(10).iterrows():
            pos = int(row['Position'])
            kcat = row['normalized_kcat_bc']
            rbc410 = row['RBC4.10']
            rbc49 = row['RBC4.9']
            print(f"   Position {pos}: kcat={kcat:.3f}, RBC4.10={rbc410}, RBC4.9={rbc49}")
    
    return criteria1_df

def apply_criteria_2(merged_df):
    """
    Criteria 2: All proteins agree EXCEPT RBC4.9 (strong phenotype signal)
    """
    print(f"\n🎯 Applying Criteria 2: All proteins agree except RBC4.9 (strong phenotype signal)")
    
    protein_columns = [col for col in merged_df.columns if col not in ['Position', '#', 'kcat', 'normalized_kcat_bc', 'group', 'activity_label']]
    
    criteria2_positions = []
    
    for _, row in merged_df.iterrows():
        # Get amino acids for all proteins at this position
        amino_acids = {}
        for protein in protein_columns:
            amino_acids[protein] = row[protein]
        
        # Check if all proteins except RBC4.9 have the same amino acid
        non_rbc49_proteins = [p for p in protein_columns if p != 'RBC4.9']
        
        if len(non_rbc49_proteins) > 1:
            # Get amino acids for non-RBC4.9 proteins
            non_rbc49_aas = [amino_acids[p] for p in non_rbc49_proteins]
            
            # Check if all non-RBC4.9 proteins agree
            all_same = len(set(non_rbc49_aas)) == 1
            
            # Check if RBC4.9 is different from the consensus
            rbc49_different = amino_acids.get('RBC4.9') not in non_rbc49_aas
            
            if all_same and rbc49_different:
                criteria2_positions.append(row)
    
    criteria2_df = pd.DataFrame(criteria2_positions)
    
    print(f"📊 Positions meeting Criteria 2: {len(criteria2_df)}")
    
    if len(criteria2_df) > 0:
        print(f"📋 Preview of positions:")
        for _, row in criteria2_df.head(10).iterrows():
            pos = int(row['Position'])
            rbc49 = row['RBC4.9']
            others = [row[col] for col in protein_columns if col != 'RBC4.9']
            consensus = others[0] if others else 'N/A'
            kcat = row['normalized_kcat_bc'] if pd.notna(row['normalized_kcat_bc']) else 'N/A'
            print(f"   Position {pos}: RBC4.9={rbc49}, Others={consensus}, kcat={kcat}")
    
    return criteria2_df

def apply_criteria_3(merged_df, consensus_threshold=0.8):
    """
    Criteria 3: Strong consensus - high conservation except for specific differences
    Identifies positions where >=80% of proteins have the same amino acid (conservation in protein evolution sense)
    """
    print(f"\n🎯 Applying Criteria 3: Strong consensus (≥{consensus_threshold*100:.0f}% conservation)")
    
    protein_columns = [col for col in merged_df.columns if col not in ['Position', '#', 'kcat', 'normalized_kcat_bc', 'group', 'activity_label']]
    
    criteria3_positions = []
    
    for _, row in merged_df.iterrows():
        # Get amino acids for all proteins at this position
        amino_acids = [row[protein] for protein in protein_columns]
        amino_acids = [aa for aa in amino_acids if pd.notna(aa) and aa != '']  # Remove NaN and empty values
        
        if len(amino_acids) < 2:  # Skip if insufficient data
            continue
            
        # Count frequency of each amino acid (this IS protein conservation analysis)
        aa_counts = Counter(amino_acids)
        total_proteins = len(amino_acids)
        
        # Find the most common amino acid and its frequency
        most_common_aa, most_common_count = aa_counts.most_common(1)[0]
        consensus_fraction = most_common_count / total_proteins
        
        # Check if this meets the consensus threshold (high conservation)
        if consensus_fraction >= consensus_threshold:
            # Add information about the conservation pattern
            minority_proteins = []
            consensus_proteins = []
            
            for protein in protein_columns:
                if pd.notna(row[protein]) and row[protein] != '':
                    if row[protein] == most_common_aa:
                        consensus_proteins.append(protein)
                    else:
                        minority_proteins.append(f"{protein}({row[protein]})")
            
            # Create enhanced row with conservation information
            enhanced_row = row.copy()
            enhanced_row['consensus_aa'] = most_common_aa
            enhanced_row['consensus_fraction'] = consensus_fraction
            enhanced_row['consensus_proteins'] = ';'.join(consensus_proteins)
            enhanced_row['minority_proteins'] = ';'.join(minority_proteins)
            enhanced_row['total_variants'] = len(aa_counts)
            
            criteria3_positions.append(enhanced_row)
    
    criteria3_df = pd.DataFrame(criteria3_positions)
    
    print(f"📊 Positions meeting Criteria 3: {len(criteria3_df)}")
    
    if len(criteria3_df) > 0:
        print(f"📋 Preview of highly conserved positions:")
        for _, row in criteria3_df.head(10).iterrows():
            pos = int(row['Position'])
            consensus = row['consensus_aa']
            fraction = row['consensus_fraction']
            minority = row['minority_proteins']
            kcat = row['normalized_kcat_bc'] if pd.notna(row['normalized_kcat_bc']) else 'N/A'
            print(f"   Position {pos}: {fraction:.1%} have {consensus}, minorities: {minority}, kcat={kcat}")
    
    return criteria3_df

def export_results(criteria1_df, criteria2_df, criteria3_df, merged_df, output_base_path):
    """Export all results to a single comprehensive CSV file"""
    print(f"\n💾 Exporting results to single file...")
    
    # Get all protein columns
    protein_columns = [col for col in merged_df.columns if col not in ['Position', '#', 'kcat', 'normalized_kcat_bc', 'group', 'activity_label']]
    
    # Track which positions meet which criteria
    criteria1_positions = set(criteria1_df['Position'].tolist()) if len(criteria1_df) > 0 else set()
    criteria2_positions = set(criteria2_df['Position'].tolist()) if len(criteria2_df) > 0 else set()
    criteria3_positions = set(criteria3_df['Position'].tolist()) if len(criteria3_df) > 0 else set()
    
    # Combine all interesting positions
    all_interesting_positions = criteria1_positions.union(criteria2_positions).union(criteria3_positions)
    
    if not all_interesting_positions:
        print("⚠️ No positions found meeting any criteria")
        return 0
    
    # Filter merged data to only interesting positions
    interesting_df = merged_df[merged_df['Position'].isin(all_interesting_positions)].copy()
    
    # Add criteria flags
    interesting_df['Meets_Criteria_1_Low_Activity_Disagreement'] = interesting_df['Position'].isin(criteria1_positions)
    interesting_df['Meets_Criteria_2_Strong_Phenotype_Signal'] = interesting_df['Position'].isin(criteria2_positions)
    interesting_df['Meets_Criteria_3_Strong_Consensus'] = interesting_df['Position'].isin(criteria3_positions)
    
    # Add consensus information from criteria3
    consensus_info = {}
    if len(criteria3_df) > 0:
        for _, row in criteria3_df.iterrows():
            pos = row['Position']
            consensus_info[pos] = {
                'consensus_aa': row['consensus_aa'],
                'consensus_fraction': row['consensus_fraction'],
                'minority_proteins': row['minority_proteins'],
                'total_variants': row['total_variants']
            }
    
    # Add consensus columns
    interesting_df['Consensus_AA'] = interesting_df['Position'].map(
        lambda pos: consensus_info.get(pos, {}).get('consensus_aa', 'N/A')
    )
    interesting_df['Consensus_Fraction'] = interesting_df['Position'].map(
        lambda pos: consensus_info.get(pos, {}).get('consensus_fraction', '')
    )
    interesting_df['Minority_Proteins'] = interesting_df['Position'].map(
        lambda pos: consensus_info.get(pos, {}).get('minority_proteins', '')
    )
    interesting_df['Total_Variants'] = interesting_df['Position'].map(
        lambda pos: consensus_info.get(pos, {}).get('total_variants', '')
    )
    
    # Add summary columns
    interesting_df['Criteria_Met'] = interesting_df.apply(
        lambda row: ', '.join([
            'Low Activity + RBC4.10≠RBC4.9' if row['Meets_Criteria_1_Low_Activity_Disagreement'] else '',
            'Strong Phenotype Signal' if row['Meets_Criteria_2_Strong_Phenotype_Signal'] else '',
            'Strong Consensus' if row['Meets_Criteria_3_Strong_Consensus'] else ''
        ]).strip(', '), axis=1
    )
    
    # Add functional analysis columns
    interesting_df['Activity_Level'] = interesting_df['normalized_kcat_bc'].apply(
        lambda x: 'Low (<0.8)' if pd.notna(x) and x < 0.8 else 'Normal (≥0.8)' if pd.notna(x) else 'No Data'
    )
    
    # Enhanced sequence agreement analysis
    def analyze_sequence_agreement(row):
        amino_acids = {}
        for protein in protein_columns:
            amino_acids[protein] = row[protein]
        
        # Count unique amino acids
        unique_aas = set([aa for aa in amino_acids.values() if pd.notna(aa) and aa != ''])
        
        # If we have consensus info, use it
        if row['Consensus_Fraction'] != '':
            consensus_frac = float(row['Consensus_Fraction'])
            if consensus_frac >= 0.8:
                return f"Strong Consensus ({consensus_frac:.1%})"
            elif consensus_frac >= 0.6:
                return f"Moderate Consensus ({consensus_frac:.1%})"
            else:
                return f"Weak Consensus ({consensus_frac:.1%})"
        
        # Fallback to original logic
        if 'RBC4.9' in amino_acids:
            non_rbc49_aas = [amino_acids[p] for p in protein_columns if p != 'RBC4.9' and pd.notna(amino_acids[p])]
            if len(set(non_rbc49_aas)) == 1 and amino_acids['RBC4.9'] not in non_rbc49_aas:
                return 'RBC4.9 Unique'
            elif len(unique_aas) == 2:
                return 'Two Groups'
            elif len(unique_aas) > 2:
                return 'Multiple Variants'
            else:
                return 'All Same'
        return 'Complex'
    
    interesting_df['Sequence_Pattern'] = interesting_df.apply(analyze_sequence_agreement, axis=1)
    
    # Reorder columns for clarity
    final_columns = [
        'Position',
        'Criteria_Met',
        'Meets_Criteria_1_Low_Activity_Disagreement',
        'Meets_Criteria_2_Strong_Phenotype_Signal',
        'Meets_Criteria_3_Strong_Consensus',
        'normalized_kcat_bc',
        'Activity_Level',
        'Sequence_Pattern',
        'Consensus_AA',
        'Consensus_Fraction',
        'Minority_Proteins',
        'Total_Variants'
    ] + protein_columns + ['kcat', 'group', 'activity_label']
    
    # Keep only columns that exist in the dataframe
    final_columns = [col for col in final_columns if col in interesting_df.columns]
    interesting_df = interesting_df[final_columns]
    
    # Sort by position
    interesting_df = interesting_df.sort_values('Position')
    
    # Save to file
    output_file = f"{output_base_path}_all_results.csv"
    interesting_df.to_csv(output_file, index=False)
    
    print(f"✅ All results saved to: {output_file}")
    print(f"📊 Total positions in output: {len(interesting_df)}")
    
    # Print summary breakdown
    criteria1_count = interesting_df['Meets_Criteria_1_Low_Activity_Disagreement'].sum()
    criteria2_count = interesting_df['Meets_Criteria_2_Strong_Phenotype_Signal'].sum()
    criteria3_count = interesting_df['Meets_Criteria_3_Strong_Consensus'].sum()
    
    print(f"📋 Breakdown:")
    print(f"   • Criteria 1 (Low activity + disagreement): {criteria1_count}")
    print(f"   • Criteria 2 (Strong phenotype signal): {criteria2_count}")
    print(f"   • Criteria 3 (Strong consensus): {criteria3_count}")
    
    # Show preview
    print(f"\n📋 PREVIEW (first 5 positions):")
    preview_cols = ['Position', 'Criteria_Met', 'normalized_kcat_bc', 'Consensus_AA', 'RBC49_Unique_AA', 'Consensus_Fraction', 'RBC4.9', 'RBC4.10']
    preview_cols = [col for col in preview_cols if col in interesting_df.columns]
    print(interesting_df[preview_cols].head().to_string(index=False))
    
    return len(interesting_df)

def main():
    print("=" * 70)
    print("🧬 KINETIC DATA + ALIGNMENT ANALYZER")
    print("=" * 70)
    
    # =============================================================
    # FILE PATHS - UPDATE THESE FOR YOUR FILES:
    alignment_file = '/Users/leah-shihlab/Library/CloudStorage/GoogleDrive-l.taylorkearney@berkeley.edu/My Drive/RuBisCO/4_10 Fast Carboxyaltion Project/Faster RBC Designs/4_9 to 4_10 Speed Project/4_9x/Round 3 4.9x/clustalo-I20250904-171241-0868-39628282-p1m_differences.csv'
    kinetic_file = '/Users/leah-shihlab/Desktop/alanine_scan_normalized_output.csv'
    # =============================================================
    
    try:
        # Load data
        alignment_df = load_alignment_differences(alignment_file)
        kinetic_df = load_kinetic_data(kinetic_file)
        
        # Merge datasets
        merged_df = merge_data(alignment_df, kinetic_df)
        
        # Apply filtering criteria
        criteria1_results = apply_criteria_1(merged_df)
        criteria2_results = apply_criteria_2(merged_df)
        criteria3_results = apply_criteria_3(merged_df)  # New strong consensus criteria
        
        # Export results
        output_base = os.path.splitext(alignment_file)[0] + "_analysis"
        total_positions = export_results(criteria1_results, criteria2_results, criteria3_results, merged_df, output_base)
        
        # Final summary
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📊 Summary:")
        print(f"   • Total interesting positions found: {total_positions}")
        print(f"📂 Single output file created in same directory as alignment file")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


# In[14]:


# Converting xlsx format to csv for mutant sequence generation
import pandas as pd
import os
import re

def convert_excel_to_csv(excel_file_path, output_file_path=None):
    """
    Convert Excel file with protein mutations to CSV format for the protein mutator
    
    Expected Excel format:
    - Column A: Name (sequence names)
    - Column B: Template (protein sequence)  
    - Columns C onwards: Individual mutations (with blank cells)
    """
    
    # Read the Excel file
    print(f"Reading Excel file: {excel_file_path}")
    df = pd.read_excel(excel_file_path)
    
    # Get column names
    columns = df.columns.tolist()
    print(f"Found columns: {columns[:5]}...")  # Show first 5 columns
    
    # Prepare the output data
    csv_data = []
    
    for index, row in df.iterrows():
        # Get name and template from first two columns
        name = row.iloc[0]  # First column
        template = row.iloc[1]  # Second column
        
        # Collect mutations from remaining columns (skip first 2)
        mutations = []
        for col_idx in range(2, len(row)):
            value = row.iloc[col_idx]
            
            # Skip empty/NaN values
            if pd.notna(value) and str(value).strip() != '':
                mutation = str(value).strip()
                
                # Convert deletion format: "D461Delete" -> "461del"
                if 'Delete' in mutation:
                    # Extract position from mutations like "D461Delete"
                    match = re.search(r'([A-Z])(\d+)Delete', mutation)
                    if match:
                        position = match.group(2)
                        mutation = f"{position}del"
                
                mutations.append(mutation)
        
        # Join mutations with semicolons
        mutations_string = ';'.join(mutations)
        
        # Add to output data
        csv_data.append({
            'sequence_name': name,
            'template_sequence': template,
            'mutations': mutations_string
        })
        
        print(f"Processed {name}: {len(mutations)} mutations")
    
    # Create DataFrame and save
    output_df = pd.DataFrame(csv_data)
    
    # Set output file path
    if output_file_path is None:
        base_name = os.path.splitext(excel_file_path)[0]
        output_file_path = f"{base_name}_converted.csv"
    
    # Save to CSV
    output_df.to_csv(output_file_path, index=False)
    print(f"\n✅ Conversion complete!")
    print(f"📁 Output saved to: {output_file_path}")
    print(f"📊 Total sequences: {len(csv_data)}")
    
    # Show preview
    print(f"\n📋 Preview of mutations:")
    for i, row in enumerate(csv_data[:5]):  # Show first 5
        print(f"{row['sequence_name']}: {row['mutations']}")
    
    return output_file_path

def main():
    """Interactive version - asks for file path"""
    print("=== Excel to CSV Converter for Protein Mutations ===\n")
    
    # Get input file path
    excel_path = input("Enter the path to your Excel file: ").strip()
    
    # Remove quotes if present
    if excel_path.startswith('"') and excel_path.endswith('"'):
        excel_path = excel_path[1:-1]
    
    # Check if file exists
    if not os.path.exists(excel_path):
        print(f"❌ Error: File not found at '{excel_path}'")
        return
    
    try:
        # Convert the file
        output_path = convert_excel_to_csv(excel_path)
        
        print(f"\n🎉 Success! Your CSV file is ready to use with the protein mutator.")
        print(f"📍 File location: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("Make sure your Excel file has the correct format:")
        print("- Column A: sequence names")
        print("- Column B: template sequence")
        print("- Columns C+: individual mutations")

# Run directly with your specific file paths:
if __name__ == "__main__":
    excel_path = '/Users/leah-shihlab/Desktop/4_9x Design Round 3.xlsx'
    output_path = '/Users/leah-shihlab/Desktop/4_9x Design Round 3.csv'
    convert_excel_to_csv(excel_path, output_path)


# In[15]:


# Creating CSV with desired variants - OPTIMIZED VERSION
import csv
import os
import re

def apply_mutations_fast(sequence, mutations_str):
    """Apply mutations to a protein sequence - optimized version"""
    if not mutations_str.strip():
        return sequence
    
    mutations = [m.strip() for m in mutations_str.split(';') if m.strip()]
    if not mutations:
        return sequence
    
    # Parse all mutations at once
    substitutions = {}  # position: new_amino_acid
    deletions = set()   # positions to delete
    
    for mut in mutations:
        if 'del' in mut:
            if '-' in mut:  # Range deletion like "15-20del"
                range_part = mut.replace('del', '')
                start, end = map(int, range_part.split('-'))
                deletions.update(range(start-1, end))  # Convert to 0-based
            else:  # Single deletion like "30del"
                pos = int(re.findall(r'\d+', mut)[0]) - 1  # Convert to 0-based
                deletions.add(pos)
        else:  # Substitution like "S10A"
            # Extract position more efficiently
            match = re.match(r'[A-Z](\d+)[A-Z]', mut)
            if match:
                pos = int(match.group(1)) - 1  # Convert to 0-based
                new_aa = mut[-1]
                substitutions[pos] = new_aa
    
    # Apply all mutations in one pass
    result = []
    for i, amino_acid in enumerate(sequence):
        if i in deletions:
            continue  # Skip deleted positions
        elif i in substitutions:
            result.append(substitutions[i])  # Use substitution
        else:
            result.append(amino_acid)  # Keep original
    
    return ''.join(result)

def main():

    input_file = '/Users/leah-shihlab/Desktop/4_9x Design Round 3.csv'
    
    print(f"Processing file: {input_file}")
    print("Starting mutation processing...")
    
    results = []
    
    try:
        # Read input CSV with sequences and mutations
        with open(input_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)  # Read all at once
        
        print(f"Found {len(rows)} sequences to process")
        
        # Process each sequence
        for i, row in enumerate(rows, 1):
            name = row['sequence_name']
            template = row['template_sequence']
            mutations = row['mutations']
            
            print(f"Processing {i}/{len(rows)}: {name}...")
            
            mutated_seq = apply_mutations_fast(template, mutations)
            results.append([name, mutated_seq])
        
        # Create output filename in same directory as input
        output_file = os.path.join(os.path.dirname(input_file), 'mutated_sequences.csv')
        
        print("Writing results to CSV...")
        
        # Write output CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sequence_name', 'sequence'])
            writer.writerows(results)
        
        print(f"\n✅ Success!")
        print(f"Generated {len(results)} mutated sequences in '{output_file}'")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{input_file}'")
        print("Make sure the file path is correct and the file exists.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n⏱️ Total processing time: {end_time - start_time:.2f} seconds")


# In[ ]:
