import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

# Data from Problem 3(a)
prices = np.array([12, 16, 20, 24, 28, 32])
demands = np.array([183, 153, 124, 100, 71, 41])
n = len(prices)

# Summary statistics
p_bar = 22
D_bar = 112
S_pp = 280
S_pD = -1960

# Cost
c = 8


def fit_demand_model(p, D):
    """Fit linear demand model D = beta0 + beta1*p using OLS"""
    p_bar = np.mean(p)
    D_bar = np.mean(D)
    S_pp = np.sum((p - p_bar) ** 2)
    S_pD = np.sum((p - p_bar) * (D - D_bar))

    beta1 = S_pD / S_pp
    beta0 = D_bar - beta1 * p_bar

    return beta0, beta1


def compute_optimal_price(beta0, beta1, c=8, p_min=18, p_max=22):
    """Compute constrained optimal price"""

    # Profit function
    def profit(p):
        return -(p - c) * (beta0 + beta1 * p)  # Negative for minimization

    # Optimize with bounds
    result = minimize_scalar(profit, bounds=(p_min, p_max), method='bounded')
    return result.x


def log_posterior_beta1(beta1):
    """Unnormalized log-posterior for beta1"""
    return -5 * (beta1 + 7) ** 2


# ==================== BOOTSTRAP ====================
print("=" * 70)
print("BOOTSTRAP ANALYSIS")
print("=" * 70)

B = 5000
bootstrap_prices = []

for b in range(B):
    # Resample with replacement
    indices = np.random.choice(n, size=n, replace=True)
    p_sample = prices[indices]
    D_sample = demands[indices]

    # Fit model
    beta0, beta1 = fit_demand_model(p_sample, D_sample)

    # Compute optimal price
    p_opt = compute_optimal_price(beta0, beta1, c)
    bootstrap_prices.append(p_opt)

bootstrap_prices = np.array(bootstrap_prices)

# Confidence interval
ci_lower = np.percentile(bootstrap_prices, 2.5)
ci_upper = np.percentile(bootstrap_prices, 97.5)

print(f"Bootstrap Optimal Prices Statistics:")
print(f"  Mean: {np.mean(bootstrap_prices):.4f}")
print(f"  Std Dev: {np.std(bootstrap_prices):.4f}")
print(f"  2.5th percentile: {ci_lower:.4f}")
print(f"  97.5th percentile: {ci_upper:.4f}")
print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_prices, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(ci_lower, color='red', linestyle='--', linewidth=2, label=f'2.5th: {ci_lower:.2f}')
plt.axvline(ci_upper, color='red', linestyle='--', linewidth=2, label=f'97.5th: {ci_upper:.2f}')
plt.axvline(np.mean(bootstrap_prices), color='blue', linestyle='-', linewidth=2,
            label=f'Mean: {np.mean(bootstrap_prices):.2f}')
plt.xlabel('Optimal Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Bootstrap Distribution of Optimal Prices (B=5000)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bootstrap_histogram.png', dpi=300)
plt.show()

# ==================== MCMC - Manual 3 Iterations ====================
print("\n" + "=" * 70)
print("MCMC - METROPOLIS-HASTINGS (3 Manual Iterations)")
print("=" * 70)

# Given values
beta1_current = -5.5
sigma_proposal = 0.5

# Random numbers provided
random_data = [
    (0, -1.2, 0.25),
    (1, +0.6, 0.70),
    (2, -0.4, 0.50)
]

chain = [beta1_current]

print(f"Starting value: β₁⁽⁾ = {beta1_current}")
print("-" * 70)
print(f"{'Step':<6} {'Proposal':<10} {'Δℓ':<12} {'α':<10} {'u':<8} {'Decision':<10} {'Chain'}")
print("-" * 70)

for k, epsilon, u in random_data:
    # Generate proposal
    beta1_proposal = beta1_current + epsilon

    # Compute log-posterior difference
    ell_current = log_posterior_beta1(beta1_current)
    ell_proposal = log_posterior_beta1(beta1_proposal)
    delta_ell = ell_proposal - ell_current

    # Acceptance probability
    alpha = min(1, np.exp(delta_ell))

    # Decision
    accept = u < alpha
    if accept:
        beta1_current = beta1_proposal
        decision = "Accept"
    else:
        decision = "Reject"

    chain.append(beta1_current)

    print(f"{k:<6} {beta1_proposal:<10.4f} {delta_ell:<12.4f} {alpha:<10.4f} {u:<8.2f} {decision:<10} {chain}")

print("-" * 70)
print(f"\nChain: β₁⁽⁰⁾={chain[0]:.1f}, β₁⁽¹⁾={chain[1]:.1f}, β₁⁽²⁾={chain[2]:.1f}, β₁⁽³⁾={chain[3]:.1f}")

# Qualitative behavior
print("\nQualitative Behavior:")
print(
    f"  - Chain is {'approaching' if abs(chain[-1] + 7) < abs(chain[0] + 7) else 'moving away from'} the mode at β₁ = -7")
print(f"  - Starting at {chain[0]:.1f}, ending at {chain[-1]:.1f}")

# ==================== MCMC - Full Sampler ====================
print("\n" + "=" * 70)
print("MCMC - FULL METROPOLIS-HASTINGS SAMPLER")
print("=" * 70)

# Full MCMC run
n_iterations = 50000
burn_in = 5000
thin = 5

# Storage
beta1_samples = []
beta1_current = -5.5

for i in range(n_iterations):
    # Propose
    epsilon = np.random.normal(0, sigma_proposal)
    beta1_proposal = beta1_current + epsilon

    # Compute acceptance probability
    delta_ell = log_posterior_beta1(beta1_proposal) - log_posterior_beta1(beta1_current)
    alpha = min(1, np.exp(delta_ell))

    # Accept/reject
    if np.random.uniform() < alpha:
        beta1_current = beta1_proposal

    # Store after burn-in and thinning
    if i >= burn_in and (i - burn_in) % thin == 0:
        beta1_samples.append(beta1_current)

beta1_samples = np.array(beta1_samples)

print(f"Total iterations: {n_iterations}")
print(f"Burn-in: {burn_in}")
print(f"Thinning: every {thin}th")
print(f"Posterior samples: {len(beta1_samples)}")
print(f"Posterior mean of β₁: {np.mean(beta1_samples):.4f}")
print(f"Posterior std of β₁: {np.std(beta1_samples):.4f}")

# Compute implied optimal prices
# We need beta0 for each beta1 sample
# Use the relationship: beta0 = D_bar - beta1 * p_bar
mcmc_optimal_prices = []
for beta1 in beta1_samples:
    beta0 = D_bar - beta1 * p_bar
    p_opt = compute_optimal_price(beta0, beta1, c)
    mcmc_optimal_prices.append(p_opt)

mcmc_optimal_prices = np.array(mcmc_optimal_prices)

# MCMC confidence interval
mcmc_ci_lower = np.percentile(mcmc_optimal_prices, 2.5)
mcmc_ci_upper = np.percentile(mcmc_optimal_prices, 97.5)

print(f"\nMCMC Optimal Prices:")
print(f"  Mean: {np.mean(mcmc_optimal_prices):.4f}")
print(f"  Std Dev: {np.std(mcmc_optimal_prices):.4f}")
print(f"  95% CI: [{mcmc_ci_lower:.4f}, {mcmc_ci_upper:.4f}]")

# Plot comparison
plt.figure(figsize=(12, 6))

# Bootstrap histogram
plt.hist(bootstrap_prices, bins=50, alpha=0.5, density=True,
         label='Bootstrap', color='blue', edgecolor='black')

# MCMC histogram
plt.hist(mcmc_optimal_prices, bins=50, alpha=0.5, density=True,
         label='MCMC', color='red', edgecolor='black')

plt.axvline(ci_lower, color='blue', linestyle='--', linewidth=2, alpha=0.7)
plt.axvline(ci_upper, color='blue', linestyle='--', linewidth=2, alpha=0.7)
plt.axvline(mcmc_ci_lower, color='red', linestyle=':', linewidth=2, alpha=0.7)
plt.axvline(mcmc_ci_upper, color='red', linestyle=':', linewidth=2, alpha=0.7)

plt.xlabel('Optimal Price', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Comparison: Bootstrap vs MCMC Distribution of Optimal Prices', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bootstrap_vs_mcmc.png', dpi=300)
plt.show()

# Discussion
print("\n" + "=" * 70)
print("COMPARISON DISCUSSION")
print("=" * 70)
print(f"Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"MCMC CI: [{mcmc_ci_lower:.4f}, {mcmc_ci_upper:.4f}]")
print(
    f"\nAgreement: {'Good' if abs(ci_lower - mcmc_ci_lower) < 0.5 and abs(ci_upper - mcmc_ci_upper) < 0.5 else 'Moderate'}")
print("\nWhich to trust more?")
print("  - MCMC incorporates prior information (mode at β₁ = -7)")
print("  - Bootstrap is purely data-driven (nonparametric)")
print("  - With limited data (n=6), MCMC with informative prior may be more reliable")
print("  - However, if prior is misspecified, bootstrap is more robust")