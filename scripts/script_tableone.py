import pandas as pd
import numpy as np
from tableone import TableOne

# Read the data
df = pd.read_csv('transfusion_data.csv')

# Create more readable labels for the grouping variable
df['transfusion_timing'] = df['early_transfusion'].map({
    0: 'Late (>6h)',
    1: 'Early (≤6h)'
})

# Group race/ethnicity into main categories
def group_race(race):
    if pd.isna(race):
        return 'Unknown'
    race = str(race).upper()
    if 'WHITE' in race:
        return 'White'
    elif 'BLACK' in race or 'AFRICAN' in race:
        return 'Black/African American'
    elif 'HISPANIC' in race or 'LATINO' in race:
        return 'Hispanic/Latino'
    elif 'ASIAN' in race:
        return 'Asian'
    elif 'UNKNOWN' in race or 'UNABLE' in race:
        return 'Unknown'
    else:
        return 'Other'

df['race_grouped'] = df['race'].apply(group_race)

# Simplify language to English vs Non-English
def group_language(lang):
    if pd.isna(lang):
        return 'Unknown'
    lang = str(lang).upper()
    if 'ENGLISH' in lang:
        return 'English'
    else:
        return 'Non-English'

df['language_grouped'] = df['language'].apply(group_language)

# Define columns for Table 1
columns = [
    # Demographics
    'age',
    'gender',
    'race_grouped',
    'weight',
    'insurance',
    'language_grouped',
    
    # Diagnosis & Severity
    'sofa_score',
    'admission_type',
    'ongoing_bleeding',
    
    # Comorbidities
    'heart_disease',
    'kidney_disease',
    'history_of_bleeding',
    'sepsis',
    
    # Baseline Labs - Hemoglobin
    'baseline_hemoglobin',
    'pre_transfusion_hemoglobin',
    'baseline_wbc',
    'baseline_platelets',
    'baseline_hematocrit',
    'baseline_creatinine',
    
    # Baseline Vitals
    'baseline_spo2',
    'baseline_sao2',
    'baseline_bp_systolic',
    'baseline_bp_diastolic',
    
    # Interventions
    'on_vasopressors',
    'vasopressor_type',
    'on_diuretics',
    
    # Transfusion variables
    'time_to_first_transfusion_hours',
    'number_of_transfusions',
    'units_first_transfusion',
    'total_units_transfused',
    
    # Hemolysis
    'possible_hemolysis',
    'ldh',
    'bilirubin_total',
    
    # Outcomes
    'in_hospital_mortality',
    'los_icu_days',
    'los_hospital_days'
]

# Define categorical variables
categorical = [
    'gender',
    'race_grouped',
    'insurance',
    'language_grouped',
    'admission_type',
    'ongoing_bleeding',
    'heart_disease',
    'kidney_disease',
    'history_of_bleeding',
    'sepsis',
    'on_vasopressors',
    'vasopressor_type',
    'on_diuretics',
    'possible_hemolysis',
    'in_hospital_mortality'
]

# Define variables that should be shown as non-normal (median [IQR])
nonnormal = [
    'sofa_score',
    'baseline_creatinine',
    'time_to_first_transfusion_hours',
    'number_of_transfusions',
    'units_first_transfusion',
    'total_units_transfused',
    'los_icu_days',
    'los_hospital_days',
    'ldh',
    'bilirubin_total'
]

# Rename columns for better display
rename = {
    'age': 'Age (years)',
    'gender': 'Gender',
    'race_grouped': 'Race/Ethnicity',
    'weight': 'Weight (kg)',
    'insurance': 'Insurance',
    'language_grouped': 'Primary Language',
    'sofa_score': 'SOFA Score',
    'admission_type': 'Admission Type',
    'ongoing_bleeding': 'Ongoing Bleeding',
    'heart_disease': 'Heart Disease',
    'kidney_disease': 'Chronic Kidney Disease',
    'history_of_bleeding': 'History of Bleeding',
    'sepsis': 'Sepsis',
    'baseline_hemoglobin': 'Baseline Hemoglobin (g/dL)',
    'pre_transfusion_hemoglobin': 'Pre-transfusion Hemoglobin (g/dL)',
    'baseline_wbc': 'White Blood Cell Count (K/uL)',
    'baseline_platelets': 'Platelet Count (K/uL)',
    'baseline_hematocrit': 'Hematocrit (%)',
    'baseline_creatinine': 'Creatinine (mg/dL)',
    'baseline_spo2': 'SpO2 (%)',
    'baseline_sao2': 'SaO2 from ABG (%)',
    'baseline_bp_systolic': 'Systolic BP (mmHg)',
    'baseline_bp_diastolic': 'Diastolic BP (mmHg)',
    'on_vasopressors': 'Vasopressor Use',
    'vasopressor_type': 'Vasopressor Type',
    'on_diuretics': 'Diuretic Use',
    'time_to_first_transfusion_hours': 'Time to First Transfusion (hours)',
    'number_of_transfusions': 'Number of Transfusions',
    'units_first_transfusion': 'Units in First Transfusion (mL)',
    'total_units_transfused': 'Total Units Transfused (mL)',
    'possible_hemolysis': 'Possible Hemolysis',
    'ldh': 'LDH (U/L)',
    'bilirubin_total': 'Total Bilirubin (mg/dL)',
    'in_hospital_mortality': 'In-Hospital Mortality',
    'los_icu_days': 'ICU Length of Stay (days)',
    'los_hospital_days': 'Hospital Length of Stay (days)'
}

# Create the table
print("=" * 80)
print("RACE/ETHNICITY AND LANGUAGE GROUPING")
print("=" * 80)
print("\nRace/Ethnicity Distribution:")
print(df['race_grouped'].value_counts().sort_index())
print(f"\nTotal: {len(df)}")

print("\n" + "-" * 80)
print("\nLanguage Distribution:")
print(df['language_grouped'].value_counts().sort_index())
print(f"\nTotal: {len(df)}")

print("\n" + "=" * 80)
print("TABLE 1: Baseline Characteristics by Transfusion Timing")
print("=" * 80)
print()

mytable = TableOne(
    df, 
    columns=columns, 
    categorical=categorical,
    nonnormal=nonnormal,
    groupby='transfusion_timing',
    rename=rename,
    pval=True,
    missing=False,
    overall=True
)

print(mytable.tabulate(tablefmt='fancy_grid'))

# Export to CSV
mytable.to_csv('table_one.csv')
print("\n✓ Table exported to: table_one.csv")

# Export to Excel with better formatting
mytable.to_excel('table_one.xlsx')
print("✓ Table exported to: table_one.xlsx")

# Export to LaTeX for publication
with open('table_one.tex', 'w') as f:
    f.write(mytable.tabulate(tablefmt='latex'))
print("✓ Table exported to: table_one.tex")

print("\n" + "=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)
print("""
For continuous variables:
- Normal distribution: Mean (SD)
- Non-normal distribution: Median [Q1, Q3]

For categorical variables:
- n (%)

P-values:
- Continuous normal: t-test
- Continuous non-normal: Mann-Whitney U test
- Categorical: Chi-square test (or Fisher's exact if small n)

Significance level: p < 0.05

The 'Overall' column shows statistics for the entire cohort.
The 'P-Value' column shows statistical comparison between Early and Late groups.
""")

# Create a summary of significant differences
print("\n" + "=" * 80)
print("SIGNIFICANT DIFFERENCES (p < 0.05)")
print("=" * 80)

# Get the p-values
pvals = mytable.tableone['P-Value']
pvals = pvals[pvals != '']  # Remove empty strings
pvals = pvals[pvals.notna()]  # Remove NaN

significant = []
for idx, pval in pvals.items():
    if isinstance(pval, str):
        try:
            p = float(pval.replace('<', '').replace('>', ''))
            if p < 0.05:
                significant.append((idx, pval))
        except:
            pass

if significant:
    print("\nVariables with statistically significant differences:")
    for var, pval in significant:
        print(f"  • {var}: p = {pval}")
else:
    print("\nNo variables showed statistically significant differences (p < 0.05)")

print("\n" + "=" * 80)
print("✓ Table One creation complete!")
print("=" * 80)