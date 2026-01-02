import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, truncnorm, norm
import os, math

np.random.seed(0)
N = 10000
input_csv = 'qanats.csv'

os.makedirs('mc_outputs/samples', exist_ok=True)
os.makedirs('mc_outputs/plots', exist_ok=True)


def truncated_lognormal(mu, sigma, lower, upper, size):
    """ Draw samples from a truncated lognormal distribution: log(X) ~ N(mu, sigma^2), truncated to [log(lower), log(upper)]. """
    lower = max(lower, 1e-15)
    a = (np.log(lower) - mu) / sigma
    b = (np.log(upper) - mu) / sigma
    samples_log = truncnorm(a, b, loc=mu, scale=sigma).rvs(size)
    return np.exp(samples_log)


def apply_gaussian_copula(x, y, rho):
    """
    Impose correlation rho between x and y using Gaussian copula.
    Both x and y are assumed to be samples from truncated distributions (already in correct ranges).
    """
    
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    u_x = (rx + 1) / (len(x) + 1)
    u_y = (ry + 1) / (len(y) + 1)

    
    z_x = norm.ppf(u_x)
    z_y = norm.ppf(u_y)

    
    z_y_new = rho * z_x + np.sqrt(1 - rho**2) * z_y

    
    u_y_new = norm.cdf(z_y_new)
    y_corr = np.quantile(y, u_y_new)

    return x, y_corr


median_V = 2.0
GSD_V = 1.8
s_V = np.log(GSD_V)

GSD_C_default = 5.6
f_fixed = 1.0
C_max = 100000.0

scale_alpha = 0.155
scale_N50 = 2.11e6
GSD_alpha = math.exp((math.log(18.4) - math.log(0.0284)) / (2 * 1.96))
GSD_N50 = math.exp((math.log(7.85e8) - math.log(2.95e5)) / (2 * 1.96))
s_alpha = math.log(GSD_alpha)
s_N50 = math.log(GSD_N50)

alpha_min, alpha_max = 0.0284, 18.4
N50_min, N50_max = 2.95e5, 7.85e8


rho = -0.85


df = pd.read_csv(input_csv)
if 'exposure_days' not in df.columns:
    df['exposure_days'] = np.random.choice([365,180,60], len(df), p=[0.6,0.3,0.1])

results = []


for _, row in df.iterrows():
    site = row['site']

    for col, season in [('C_wet_MPN_per_L','wet'), ('C_dry_MPN_per_L','dry')]:
        C_obs = row.get(col, np.nan)
        if pd.isna(C_obs):
            continue

        C_obs_raw = C_obs if C_obs > 0 else 1e-6

        
        s_C = np.log(GSD_C_default)
        mu_C = np.log(C_obs_raw)

        
        C_samples = truncated_lognormal(mu=mu_C, sigma=s_C,
                                        lower=1e-12, upper=C_max, size=N)

        
        V_samples = lognorm(s=s_V, scale=median_V).rvs(N)

        
        mu_alpha = np.log(scale_alpha)
        mu_N50 = np.log(scale_N50)
        alpha_samples = truncated_lognormal(mu=mu_alpha, sigma=s_alpha,
                                            lower=alpha_min, upper=alpha_max, size=N)
        N50_samples = truncated_lognormal(mu=mu_N50, sigma=s_N50,
                                          lower=N50_min, upper=N50_max, size=N)

        
        alpha_samples, N50_samples = apply_gaussian_copula(alpha_samples, N50_samples, rho)

        
        dose = C_samples * V_samples * f_fixed * 10
        term = 1 + dose * (2**(1/alpha_samples) - 1) / N50_samples
        term = np.maximum(term, 1e-12)
        P_daily = 1 - term**(-alpha_samples)

       
        median_days = row.get('exposure_days', 365)
        if median_days == 365:
            GSD_days = 1.2
        elif median_days == 180:
            GSD_days = 1.4
        else:
            GSD_days = 1.6
        s_days = np.log(GSD_days)
        days_samples = lognorm(s=s_days, scale=median_days).rvs(N)
        days_samples = np.clip(days_samples, 1, 365).astype(int)

        
        P_annual = 1 - (1 - P_daily)**days_samples

        
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
        samples_df.to_csv(f"mc_outputs/samples/mc_samples_{site}_{season}.csv", index=False)

       
        plt.figure(figsize=(6,3))
        plt.hist(P_annual, bins=80)
        plt.xlabel('Annual infection probability')
        plt.title(f"{site} - {season} - Histogram (Copula α,N50)")
        plt.tight_layout()
        plt.savefig(f"mc_outputs/plots/hist_{site}_{season}.png", dpi=150)
        plt.close()

        sorted_vals = np.sort(P_annual)
        cdf = np.arange(1, N+1)/N
        plt.figure(figsize=(6,3))
        plt.plot(sorted_vals, cdf)
        plt.xlabel('Annual infection probability')
        plt.ylabel('CDF')
        plt.title(f"{site} - {season} - CDF (Copula α,N50)")
        plt.tight_layout()
        plt.savefig(f"mc_outputs/plots/cdf_{site}_{season}.png", dpi=150)
        plt.close()

        
        results.append({
            'site': site,
            'season': season,
            'C_obs': C_obs,
            'C_obs_used': C_obs_raw,
            'exposure_days': median_days,
            'mean': P_annual.mean(),
            'median': np.median(P_annual),
            'P5': np.percentile(P_annual,5),
            'P25': np.percentile(P_annual,25),
            'P75': np.percentile(P_annual,75),
            'P95': np.percentile(P_annual,95),
            'prob_above_1e-4': np.mean(P_annual > 1e-4)
        })


summary_df = pd.DataFrame(results)
summary_df.to_csv('mc_outputs/mc_summary_full.csv', index=False)
corr = np.corrcoef(np.log(samples_df['alpha']), np.log(samples_df['N50']))[0,1]

alpha_min_hit = np.sum(samples_df['alpha'] <= alpha_min)
alpha_max_hit = np.sum(samples_df['alpha'] >= alpha_max)
N50_min_hit = np.sum(samples_df['N50'] <= N50_min)
N50_max_hit = np.sum(samples_df['N50'] >= N50_max)

print("\n🔍 Validation results:")
print(f"   → Corr(log α, log N50) = {corr:.3f}")
print(f"   → α min hits: {alpha_min_hit}, max hits: {alpha_max_hit}")
print(f"   → N50 min hits: {N50_min_hit}, max hits: {N50_max_hit}")
print("----------------------------------------------------------")

print("✅ Monte Carlo completed using Gaussian Copula correlated α and N50 (ρ = -0.85).")
print("ℹ️ Zero C_obs values replaced with 1e-6.")
print("Outputs saved in 'mc_outputs/'.")
