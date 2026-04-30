import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.special import gamma as gamma_func

# Set random seed for reproducibility
np.random.seed(42)

# --- Constants ---
LAMBDA_EXP = 0.5
N_LARGE = 50000
N_REPLICATIONS = 500


# --- Cost Function ---
def calculate_cost(T):
    """Vectorized cost calculation"""
    C = np.where(T < 3, 200 * (3 - T), 0)
    return C


# --- Estimators ---

def crude_mc(N):
    """Crude Monte Carlo estimator"""
    T = np.random.exponential(scale=1 / LAMBDA_EXP, size=N)
    C = calculate_cost(T)
    return np.mean(C), np.std(C, ddof=1) / np.sqrt(N), C


def antithetic_mc(N):
    """Antithetic variates estimator"""
    N_half = N // 2
    U = np.random.uniform(0, 1, N_half)
    U_anti = 1 - U

    # Inverse CDF for Exponential(lambda=0.5)
    T1 = -2 * np.log(1 - U)
    T2 = -2 * np.log(1 - U_anti)

    C1 = calculate_cost(T1)
    C2 = calculate_cost(T2)

    C_combined = (C1 + C2) / 2
    return np.mean(C_combined), np.std(C_combined, ddof=1) / np.sqrt(N_half), np.concatenate([C1, C2])


def control_variate_mc(N):
    """Control variates estimator with T as control"""
    T = np.random.exponential(scale=1 / LAMBDA_EXP, size=N)
    C = calculate_cost(T)

    mu_T_true = 2.0
    cov_CT = np.cov(C, T)[0, 1]
    var_T = np.var(T, ddof=1)
    beta_hat = cov_CT / var_T

    estimate = np.mean(C) - beta_hat * (np.mean(T) - mu_T_true)

    adjusted_C = C - beta_hat * (T - mu_T_true)
    se = np.std(adjusted_C, ddof=1) / np.sqrt(N)

    return estimate, se, C


def importance_sampling_mc(N):
    """Importance sampling with proposal q(t) = e^-t"""
    # Proposal q(t) = e^-t (Exponential with lambda=1)
    T = np.random.exponential(scale=1.0, size=N)

    # Weight w(t) = f(t)/q(t) = 0.5 * e^(0.5t)
    weights = 0.5 * np.exp(0.5 * T)

    C = calculate_cost(T)
    weighted_C = C * weights

    estimate = np.mean(weighted_C)
    se = np.std(weighted_C, ddof=1) / np.sqrt(N)

    return estimate, se, weighted_C


# ==================== PART (b) ====================
print("=" * 70)
print("PART (b): Full Monte Carlo Pipeline")
print("=" * 70)

# --- Point Estimates with N=50,000 ---
print("\n(b.1) Point Estimates (N=50,000):")
print("-" * 70)

estimators = {
    "Crude MC": crude_mc,
    "Antithetic": antithetic_mc,
    "Control Variates": control_variate_mc,
    "Importance Sampling": importance_sampling_mc
}

results_point = {}
for name, func in estimators.items():
    mu, se, _ = func(N_LARGE)
    ci_low, ci_high = mu - 1.96 * se, mu + 1.96 * se
    results_point[name] = {'mean': mu, 'se': se, 'ci': (ci_low, ci_high)}
    print(f"{name:20s}: Mean = {mu:8.4f}, SE = {se:8.4f}, 95% CI = [{ci_low:8.4f}, {ci_high:8.4f}]")

# --- Convergence Analysis ---
print("\n(b.2) Convergence Analysis:")
print("-" * 70)

Ns = [int(10 ** p) for p in np.arange(2.0, 5.1, 0.5)]
rmses = {name: [] for name in estimators}

# Reference "true" value from large simulation
mu_true_ref, _, _ = crude_mc(2000000)
print(f"Reference true value (from 2M samples): {mu_true_ref:.4f}")

for N in Ns:
    for name, func in estimators.items():
        errors = []
        for _ in range(N_REPLICATIONS):
            mu_hat, _, _ = func(N)
            errors.append((mu_hat - mu_true_ref) ** 2)
        rmse = np.sqrt(np.mean(errors))
        rmses[name].append(rmse)

    print(f"N={N:6d}: Crude RMSE={rmses['Crude MC'][-1]:.2f}, " +
          f"CV RMSE={rmses['Control Variates'][-1]:.2f}")

# Calculate VRF
print("\n(b.3) Variance Reduction Factors (at N=10^5):")
print("-" * 70)
last_idx = -1
vrf_results = {}
for name in estimators:
    if name != "Crude MC":
        vrf = (rmses['Crude MC'][last_idx] / rmses[name][last_idx]) ** 2
        vrf_results[name] = vrf
        print(f"{name:20s}: VRF = {vrf:.2f}")

# Plot convergence
plt.figure(figsize=(10, 6))
for name, rmse_list in rmses.items():
    plt.loglog(Ns, rmse_list, marker='o', label=name, linewidth=2)
plt.xlabel('Sample Size N (log scale)', fontsize=12)
plt.ylabel('Empirical RMSE (log scale)', fontsize=12)
plt.title('Convergence of Monte Carlo Estimators', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="-", alpha=0.7)
plt.tight_layout()
plt.savefig('convergence_plot.png', dpi=300)
plt.show()

# --- VaR and CVaR ---
print("\n(b.4) Risk Metrics (Crude MC, N=50,000):")
print("-" * 70)
_, _, C_samples = crude_mc(N_LARGE)
VaR_95 = np.percentile(C_samples, 95)
CVaR_95 = np.mean(C_samples[C_samples >= VaR_95])

print(f"VaR_0.95:  ${VaR_95:.4f}")
print(f"CVaR_0.95: ${CVaR_95:.4f}")
print(f"Samples to halve SE of CVaR: {4 * N_LARGE:,} (4x current sample size)")

# ==================== PART (c) ====================
print("\n" + "=" * 70)
print("PART (c): Sensitivity Analysis")
print("=" * 70)

# Identify best estimator
best_method_name = min(results_point, key=lambda k: (results_point[k]['se'] ** 2))
print(f"\nBest performing estimator: {best_method_name}")


# Distribution samplers
def sample_weibull(N):
    """Weibull(k=1.5, lambda=2/Gamma(5/3)) with mean=2"""
    k_weibull = 1.5
    lambda_weibull = 2 / gamma_func(1 + 1 / k_weibull)
    return np.random.weibull(k_weibull, N) * lambda_weibull


def sample_gamma(N):
    """Gamma(k=2, theta=1) with mean=2"""
    return np.random.gamma(shape=2, scale=1, size=N)


def sample_exponential(N):
    """Exponential(lambda=0.5) with mean=2"""
    return np.random.exponential(scale=2, size=N)


distributions = {
    "Exponential": sample_exponential,
    "Weibull (k=1.5)": sample_weibull,
    "Gamma (k=2)": sample_gamma
}

# Run sensitivity analysis
results_sensitivity = {}

print("\n(c.1) Expected Cost and CVaR under Different Distributions:")
print("-" * 70)

for name, sampler in distributions.items():
    T = sampler(N_LARGE)
    C = calculate_cost(T)

    # Use control variates if it was best
    if best_method_name == "Control Variates":
        mu_T_true = 2.0
        cov_CT = np.cov(C, T)[0, 1]
        var_T = np.var(T, ddof=1)
        beta_hat = cov_CT / var_T
        estimate = np.mean(C) - beta_hat * (np.mean(T) - mu_T_true)
    else:
        estimate = np.mean(C)

    VaR = np.percentile(C, 95)
    CVaR = np.mean(C[C >= VaR])

    results_sensitivity[name] = {'E[C]': estimate, 'CVaR': CVaR}
    print(f"{name:15s}: E[C] = ${estimate:8.4f}, CVaR_0.95 = ${CVaR:8.4f}")

# Plot grouped bar chart
print("\n(c.2) Generating grouped bar chart...")
labels = list(results_sensitivity.keys())
E_C_vals = [results_sensitivity[l]['E[C]'] for l in labels]
CVaR_vals = [results_sensitivity[l]['CVaR'] for l in labels]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width / 2, E_C_vals, width, label='E[C] (Expected Cost)', color='steelblue')
rects2 = ax.bar(x + width / 2, CVaR_vals, width, label='CVaR_0.95', color='darkorange')

ax.set_ylabel('Cost ($)', fontsize=12)
ax.set_xlabel('Lifetime Distribution', fontsize=12)
ax.set_title('Warranty Cost Sensitivity to Lifetime Distribution', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'${height:.1f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('sensitivity_analysis.png', dpi=300)
plt.show()

print("\n" + "=" * 70)
print("Analysis complete. Figures saved as PNG files.")
print("=" * 70)