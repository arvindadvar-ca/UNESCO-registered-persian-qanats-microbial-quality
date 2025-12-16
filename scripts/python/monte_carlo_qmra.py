import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import os

np.random.seed(0)
N = 10000
input_csv = 'qanats.csv'

os.makedirs('mc_outputs/samples', exist_ok=True)
os.makedirs('mc_outputs/plots', exist_ok=True)


median_V = 2.0     
GSD_V = 1.8
s_V = np.log(GSD_V)

GSD_C_default = 5.6  
f_fixed = 1.0
C_max = 100000.0     

 
alpha_fixed = 0.155
N50_fixed = 2.11e6


df = pd.read_csv(input_csv)
if 'exposure_days' not in df.columns:
    df['exposure_days'] = np.random.choice([365, 180, 60], len(df), p=[0.6, 0.3, 0.1])

results = []

for _, row in df.iterrows():
    site = row['site']

    for col, season in [('C_wet_MPN_per_L', 'wet'), ('C_dry_MPN_per_L', 'dry')]:
        C_obs = row.get(col, np.nan)
        if pd.isna(C_obs):
            continue

        
        s_C = np.log(GSD_C_default)
        C_samples = lognorm(s=s_C, scale=C_obs).rvs(N)
        V_samples = lognorm(s=s_V, scale=median_V).rvs(N)
        C_samples = np.minimum(C_samples, C_max)

        
        alpha_samples = np.full(N, alpha_fixed)
        N50_samples = np.full(N, N50_fixed)

        
        dose = C_samples * V_samples * f_fixed * 10
        term = 1 + dose * (2 ** (1 / alpha_samples) - 1) / N50_samples
        term = np.maximum(term, 1e-9)
        P_daily = 1 - term ** (-alpha_samples)

        
        median_days = row.get('exposure_days', 365)

        if median_days == 365:
            GSD_days = 1.2
        elif median_days == 180:
            GSD_days = 1.4
        else:
            GSD_days = 1.6

        s_days = np.log(GSD_days)
        days_samples = lognorm(s=s_days, scale=median_days).rvs(N)
        days_samples = np.clip(days_samples, 1, 365)
        days_samples = np.round(days_samples).astype(int)

        
        P_annual = 1 - (1 - P_daily) ** days_samples

        
        samples_df = pd.DataFrame({
            'site': site,
            'season': season,
            'C_sample': C_samples,
            'V_sample': V_samples,
            'alpha': alpha_samples,
            'N50': N50_samples,
            'days': days_samples,
            'dose': dose,
            'P_daily': P_daily,
            'P_annual': P_annual
        })
        samples_path = f"mc_outputs/samples/mc_samples_{site}_{season}.csv"
        samples_df.to_csv(samples_path, index=False)

        
        plt.figure(figsize=(6, 3))
        plt.hist(P_annual, bins=80)
        plt.xlabel('Annual infection probability')
        plt.title(f"{site} - {season} (Fixed α,N50)")
        plt.tight_layout()
        plt.savefig(f"mc_outputs/plots/hist_{site}_{season}.png", dpi=150)
        plt.close()

        
        sorted_vals = np.sort(P_annual)
        cdf = np.arange(1, N + 1) / N
        plt.figure(figsize=(6, 3))
        plt.plot(sorted_vals, cdf)
        plt.xlabel('Annual infection probability')
        plt.ylabel('CDF')
        plt.title(f"{site} - {season} (Fixed α,N50)")
        plt.tight_layout()
        plt.savefig(f"mc_outputs/plots/cdf_{site}_{season}.png", dpi=150)
        plt.close()

        
        results.append({
            'site': site,
            'season': season,
            'C_obs': C_obs,
            'exposure_days': median_days,
            'mean': P_annual.mean(),
            'median': np.median(P_annual),
            'P5': np.percentile(P_annual, 5),
            'P25': np.percentile(P_annual, 25),
            'P75': np.percentile(P_annual, 75),
            'P95': np.percentile(P_annual, 95),
            'prob_above_1e-4': np.mean(P_annual > 1e-4)
        })


summary_df = pd.DataFrame(results)
summary_df.to_csv('mc_outputs/mc_summary_full.csv', index=False)

print("✅ Monte Carlo completed using fixed α and N50 (MLE values).")
print("All outputs saved in folder: 'mc_outputs/'")
