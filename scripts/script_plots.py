import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('transfusion_data.csv')

print("="*80)
print("TRANSFUSION TIMING STUDY - KEY VISUALIZATIONS")
print("="*80)
print(f"\nDataset: {len(df)} patients")
print(f"Early transfusion (≤6h): {df['early_transfusion'].sum()} ({df['early_transfusion'].mean()*100:.1f}%)")
print(f"Late transfusion (>6h): {(1-df['early_transfusion']).sum()} ({(1-df['early_transfusion'].mean())*100:.1f}%)")

# Create figure with 5 subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.delaxes(axes[1, 2])  # Remove the 6th subplot


# PLOT 1: AGE DISTRIBUTION

print("\n" + "="*80)
print("1. AGE DISTRIBUTION")
print("="*80)

age_comparison = df.groupby('early_transfusion')['age'].agg(['mean', 'std', 'median'])
print(f"\nEarly transfusion: {age_comparison.loc[1, 'mean']:.1f} ± {age_comparison.loc[1, 'std']:.1f} years (median: {age_comparison.loc[1, 'median']:.1f})")
print(f"Late transfusion: {age_comparison.loc[0, 'mean']:.1f} ± {age_comparison.loc[0, 'std']:.1f} years (median: {age_comparison.loc[0, 'median']:.1f})")

early_age = df[df['early_transfusion']==1]['age'].dropna()
late_age = df[df['early_transfusion']==0]['age'].dropna()
u_stat, p_val = mannwhitneyu(early_age, late_age, alternative='two-sided')
print(f"Mann-Whitney U test: p = {p_val:.4f}")

ax1 = axes[0, 0]
age_data = [df[df['early_transfusion']==0]['age'].dropna(),
            df[df['early_transfusion']==1]['age'].dropna()]
bp = ax1.boxplot(age_data, labels=['Late (>6h)', 'Early (≤6h)'], patch_artist=True,
                 showfliers=False, medianprops=dict(color='red', linewidth=2))
for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_ylabel('Age (years)', fontsize=13, fontweight='bold')
ax1.set_title('Age Distribution', fontsize=15, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3)


# PLOT 2: TIME TO FIRST TRANSFUSION

print("\n" + "="*80)
print("2. TIME TO FIRST TRANSFUSION")
print("="*80)

time_to_transfusion = df['time_to_first_transfusion_hours'].dropna()
print(f"\nMean: {time_to_transfusion.mean():.1f} ± {time_to_transfusion.std():.1f} hours")
print(f"Median: {time_to_transfusion.median():.1f} hours")
print(f"Range: {time_to_transfusion.min():.1f} - {time_to_transfusion.max():.1f} hours")

print(f"\nPatients transfused within:")
for hours in [1, 3, 6, 12, 24]:
    pct = (time_to_transfusion <= hours).sum() / len(time_to_transfusion) * 100
    print(f"  ≤{hours}h: {pct:.1f}%")

ax2 = axes[0, 1]
ax2.hist(time_to_transfusion, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
ax2.axvline(6, color='red', linestyle='--', linewidth=2.5, label='6-hour cutoff')
ax2.set_xlabel('Time to First Transfusion (hours)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Number of Patients', fontsize=13, fontweight='bold')
ax2.set_title('Distribution of Time to First Transfusion', fontsize=15, fontweight='bold', pad=15)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)


# PLOT 3: AVERAGE NUMBER OF TRANSFUSIONS PER PATIENT

print("\n" + "="*80)
print("3. TRANSFUSION VOLUME")
print("="*80)

early_vol = df[df['early_transfusion']==1]['number_of_transfusions'].dropna()
late_vol = df[df['early_transfusion']==0]['number_of_transfusions'].dropna()

print(f"\nNumber of transfusions:")
print(f"  Early: {early_vol.mean():.2f} ± {early_vol.std():.2f} (median: {early_vol.median():.1f})")
print(f"  Late: {late_vol.mean():.2f} ± {late_vol.std():.2f} (median: {late_vol.median():.1f})")

ax3 = axes[0, 2]
transfusion_vol = df.groupby('early_transfusion')['number_of_transfusions'].mean()
bars = ax3.bar(['Late (>6h)', 'Early (≤6h)'], transfusion_vol, 
               color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Number of Transfusions', fontsize=13, fontweight='bold')
ax3.set_title('Average Number of Transfusions per Patient', fontsize=15, fontweight='bold', pad=15)
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.15, f'{height:.2f}', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, max(transfusion_vol) * 1.2)


# PLOT 4: ICU LENGTH OF STAY

print("\n" + "="*80)
print("4. ICU LENGTH OF STAY")
print("="*80)

early_los = df[df['early_transfusion']==1]['los_icu_days'].dropna()
late_los = df[df['early_transfusion']==0]['los_icu_days'].dropna()

print(f"\nICU Length of Stay:")
print(f"Early transfusion: {early_los.mean():.1f} ± {early_los.std():.1f} days (median: {early_los.median():.1f})")
print(f"Late transfusion: {late_los.mean():.1f} ± {late_los.std():.1f} days (median: {late_los.median():.1f})")

u_stat, p_val = mannwhitneyu(early_los, late_los, alternative='two-sided')
print(f"Mann-Whitney U test: p = {p_val:.4f}")

ax4 = axes[1, 0]
los_data = [df[df['early_transfusion']==0]['los_icu_days'].dropna(),
            df[df['early_transfusion']==1]['los_icu_days'].dropna()]
bp2 = ax4.boxplot(los_data, labels=['Late (>6h)', 'Early (≤6h)'], patch_artist=True, 
                  showfliers=False, medianprops=dict(color='red', linewidth=2))
for patch, color in zip(bp2['boxes'], ['#2ecc71', '#e74c3c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('Days', fontsize=13, fontweight='bold')
ax4.set_title('ICU Length of Stay', fontsize=15, fontweight='bold', pad=15)
ax4.grid(axis='y', alpha=0.3)


# PLOT 5: IN-HOSPITAL MORTALITY BY TRANSFUSION TIMING

print("\n" + "="*80)
print("5. IN-HOSPITAL MORTALITY")
print("="*80)

mortality_by_timing = df.groupby('early_transfusion')['in_hospital_mortality'].agg(['sum', 'count', 'mean'])
mortality_by_timing['mortality_rate'] = mortality_by_timing['mean'] * 100

print(f"\nMortality Rates:")
print(f"Early transfusion: {mortality_by_timing.loc[1, 'mortality_rate']:.1f}% ({int(mortality_by_timing.loc[1, 'sum'])}/{int(mortality_by_timing.loc[1, 'count'])})")
print(f"Late transfusion: {mortality_by_timing.loc[0, 'mortality_rate']:.1f}% ({int(mortality_by_timing.loc[0, 'sum'])}/{int(mortality_by_timing.loc[0, 'count'])})")

contingency = pd.crosstab(df['early_transfusion'], df['in_hospital_mortality'])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"Chi-square test: χ² = {chi2:.3f}, p = {p_value:.4f}")

if p_value < 0.05:
    print("✓ Statistically significant difference in mortality!")
else:
    print("× No statistically significant difference in mortality")

ax5 = axes[1, 1]
groups = ['Late\n(>6h)', 'Early\n(≤6h)']
values = [mortality_by_timing.loc[0, 'mortality_rate'], mortality_by_timing.loc[1, 'mortality_rate']]
colors = ['#2ecc71', '#e74c3c']
bars = ax5.bar(groups, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.set_ylabel('Mortality Rate (%)', fontsize=13, fontweight='bold')
ax5.set_title('In-Hospital Mortality by Transfusion Timing', fontsize=15, fontweight='bold', pad=15)
ax5.set_ylim(0, max(values) * 1.3)
for i, (bar, val) in enumerate(zip(bars, values)):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
if p_value < 0.05:
    ax5.text(0.5, max(values) * 1.2, f'p = {p_value:.4f}*', 
             ha='center', fontsize=11, style='italic', fontweight='bold')
else:
    ax5.text(0.5, max(values) * 1.2, f'p = {p_value:.4f} (ns)', 
             ha='center', fontsize=11, style='italic')
ax5.grid(axis='y', alpha=0.3)


# SUMMARY

print("\n" + "="*80)
print("KEY FINDINGS SUMMARY")
print("="*80)

mortality_diff = mortality_by_timing.loc[1, 'mortality_rate'] - mortality_by_timing.loc[0, 'mortality_rate']
print(f"\n1. MORTALITY: Early transfusion shows {abs(mortality_diff):.1f} percentage point {'increase' if mortality_diff > 0 else 'decrease'}")
print(f"   Statistical significance: {'YES (p<0.05)' if p_value < 0.05 else f'NO (p={p_value:.4f})'}")

print(f"\n2. AGE: Groups are {'similar' if abs(age_comparison.loc[1, 'mean'] - age_comparison.loc[0, 'mean']) < 5 else 'different'} in age")
print(f"   Early: {age_comparison.loc[1, 'mean']:.1f} years, Late: {age_comparison.loc[0, 'mean']:.1f} years")

print(f"\n3. TRANSFUSION VOLUME: Early group receives {'more' if transfusion_vol[1] > transfusion_vol[0] else 'fewer'} transfusions on average")
print(f"   Early: {transfusion_vol[1]:.2f}, Late: {transfusion_vol[0]:.2f}")

print(f"\n4. ICU LOS: Early group has {'longer' if early_los.median() > late_los.median() else 'shorter'} ICU stay")
print(f"   Early: {early_los.median():.1f} days, Late: {late_los.median():.1f} days")

print(f"\n5. TIME TO TRANSFUSION: Median time is {time_to_transfusion.median():.1f} hours")
print(f"   {(time_to_transfusion <= 6).sum() / len(time_to_transfusion) * 100:.1f}% receive transfusion within 6 hours")

print("\n" + "="*80)

# Save figure
plt.tight_layout()
plt.savefig('transfusion_key_plots.png', dpi=300, bbox_inches='tight')
print("\n✓ Analysis complete! Figure saved as 'transfusion_key_plots.png'")
print("="*80)

plt.show()