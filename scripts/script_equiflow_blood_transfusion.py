import numpy as np
from equiflow import EquiFlow
import os
import matplotlib
import pandas as pd
matplotlib.use('Agg')

# Read the dataset
data = pd.read_csv("blood_transfusion.csv")

# Print initial cohort size
print(f"Initial cohort size: {len(data)}")


# Create output directory with absolute path
output_dir = os.path.abspath("test_output")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "imgs"), exist_ok=True)

# Convert max_troponin to numeric before creating EquiFlow instance
data_processed = data.copy()


# Check the data after each step
print(f"Initial data shape: {data_processed.shape}")

print(data_processed['race'].value_counts())

def recode_race(race_value):
    """
    Recode detailed race/ethnicity categories into 6 broad groups
    """
    if pd.isna(race_value):
        return 'Other/Unknown'
    
    race_upper = str(race_value).upper()
    
    # Caucasian/White
    if 'WHITE' in race_upper or 'PORTUGUESE' in race_upper:
        return 'Caucasian'
    
    # African American
    elif 'BLACK' in race_upper or 'AFRICAN' in race_upper or 'CAPE VERDEAN' in race_upper or 'CARIBBEAN' in race_upper:
        return 'African American'
    
    # Hispanic
    elif 'HISPANIC' in race_upper or 'LATINO' in race_upper or 'SOUTH AMERICAN' in race_upper:
        return 'Hispanic'
    
    # Asian
    elif 'ASIAN' in race_upper or 'CHINESE' in race_upper or 'KOREAN' in race_upper:
        return 'Asian'
    
    # Native American
    elif 'AMERICAN INDIAN' in race_upper or 'ALASKA NATIVE' in race_upper or 'NATIVE HAWAIIAN' in race_upper or 'PACIFIC ISLANDER' in race_upper:
        return 'Native American'
    
    # Other/Unknown
    else:
        return 'Other/Unknown'

# Apply the recoding
data_processed['race'] = data_processed['race'].apply(recode_race)

# Set categorical order
eth_order = ['Caucasian', 'African American', 'Other/Unknown', 'Hispanic', 'Asian', 'Native American']
data_processed['race'] = pd.Categorical(
    data_processed['race'],
    categories=eth_order,
    ordered=True
)

# Sort by race
data_processed = data_processed.sort_values('race', kind='mergesort')

# EXAMPLE 1: Full metrics version
print("\nCreating EquiFlow instance with ALL metrics...")
ef_full = EquiFlow(
    data=data_processed,
    initial_cohort_label="Initial Patient Cohort",
    categorical=['gender', 'race'],
    normal=['age', 'sofa_score'],
    format_cat='%'
)

# Add exclusion steps
print("Adding exclusion criteria...")

ef_full.add_exclusion(
    mask=~ef_full._dfs[-1]['baseline_bp_systolic'].isna(),
    exclusion_reason="missing BP Systolic data",
    new_cohort_label="Complete BP Systolic data"
)

ef_full.add_exclusion(
    mask=~ef_full._dfs[-1]['baseline_wbc'].isna(),
    exclusion_reason="missing WBC",
    new_cohort_label="Complete WBC"
)

# ef_full.add_exclusion(
#     mask=~ef_full._dfs[-1]['baseline_platelets'].isna(),
#     exclusion_reason="missing platelets",
#     new_cohort_label="Complete platelets"
# )

# ef_full.add_exclusion(
#     mask=~ef_full._dfs[-1]['baseline_hemoglobin'].isna(),
#     exclusion_reason="missing hemoglobin",
#     new_cohort_label="Complete hemoglobin"
# )

ef_full.add_exclusion(
    mask=~ef_full._dfs[-1]['pre_transfusion_hemoglobin'].isna(),
    exclusion_reason="missing pre transfusion hemoglobin",
    new_cohort_label="Complete pre transfusion hemoglobin"
)

ef_full.add_exclusion(
    mask=~ef_full._dfs[-1]['post_transfusion_hemoglobin'].isna(),
    exclusion_reason="missing post transfusion hemoglobin",
    new_cohort_label="Complete post transfusion hemoglobin"
)

ef_full.add_exclusion(
    mask=~ef_full._dfs[-1]['diuretic_type'].isna(),
    exclusion_reason="missing diuretic type",
    new_cohort_label="Complete diuretic type"
)

# Generate the full flow diagram
print("Generating flow diagram with ALL metrics...")
ef_full.plot_flows(
    output_folder=output_dir,
    output_file="v3_blood_transfusion_case_study",
    plot_dists=True,
    smds=True,
    legend=True,
    box_width=3.5,
    box_height=1.5,
    display_flow_diagram=True
)

print(f"Full metrics diagram saved to {output_dir}/v8_blood_tr ansfusion_case_study.pdf")