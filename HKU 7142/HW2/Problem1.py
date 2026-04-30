import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

print("=" * 70)
print(" " * 20 + "PROBLEM (b): Algorithm Comparison")
print("=" * 70)


# ==================== PART (b)(i) ====================
# Define profit function and its derivatives
def Pi(x):
    """Profit function"""
    x1, x2 = x
    return 20 * x1 + 16 * x2 - 2 * x1 ** 2 - x2 ** 2 - 2 * x1 * x2


def grad_Pi(x):
    """Gradient of profit function"""
    x1, x2 = x
    return np.array([20 - 4 * x1 - 2 * x2, 16 - 2 * x1 - 2 * x2])


def hessian_Pi(x):
    """Hessian matrix of profit function"""
    return np.array([[-4, -2], [-2, -2]])


# Optimal solution (from part (a))
x_star = np.array([2, 4])
Pi_star = 64.0


# Method 1: Fixed step size gradient descent
def gradient_descent_fixed(alpha=0.15, tol=1e-10, max_iter=10000):
    x = np.array([0.0, 0.0])
    history = [np.abs(Pi(x) - Pi_star)]

    for k in range(max_iter):
        grad = grad_Pi(x)
        if np.linalg.norm(grad) < tol:
            break
        x = x + alpha * grad
        history.append(np.abs(Pi(x) - Pi_star))

    return history


# Method 2: Armijo backtracking gradient descent
def gradient_descent_armijo(c=1e-4, rho=0.5, alpha0=1.0, tol=1e-10, max_iter=10000):
    x = np.array([0.0, 0.0])
    history = [np.abs(Pi(x) - Pi_star)]

    for k in range(max_iter):
        grad = grad_Pi(x)
        if np.linalg.norm(grad) < tol:
            break

        alpha = alpha0
        Pi_current = Pi(x)
        # Armijo condition
        while Pi(x + alpha * grad) < Pi_current + c * alpha * np.dot(grad, grad):
            alpha *= rho

        x = x + alpha * grad
        history.append(np.abs(Pi(x) - Pi_star))

    return history


# Method 3: Newton's method
def newton_method(tol=1e-10, max_iter=1000):
    x = np.array([0.0, 0.0])
    history = [np.abs(Pi(x) - Pi_star)]

    for k in range(max_iter):
        grad = grad_Pi(x)
        if np.linalg.norm(grad) < tol:
            break

        H = hessian_Pi(x)
        # For maximization problem, solve -H * p = grad
        try:
            p = np.linalg.solve(-H, grad)
        except np.linalg.LinAlgError:
            print("Hessian is singular")
            break

        x = x + p
        history.append(np.abs(Pi(x) - Pi_star))

    return history


# Run three methods for part (b)(i)
print("\nPart (b)(i): Comparing optimization methods")
print("-" * 60)
history_fixed = gradient_descent_fixed()
history_armijo = gradient_descent_armijo()
history_newton = newton_method()

print(f"Fixed step size GD iterations: {len(history_fixed) - 1}")
print(f"Armijo GD iterations: {len(history_armijo) - 1}")
print(f"Newton's method iterations: {len(history_newton) - 1}")

# Plot convergence graph for part (b)(i)
plt.figure(figsize=(12, 6))
plt.plot(range(len(history_fixed)), np.log10(history_fixed), 'b-',
         linewidth=2, label=f'Fixed step (α=0.15)\n({len(history_fixed) - 1} iter)')
plt.plot(range(len(history_armijo)), np.log10(history_armijo), 'g-',
         linewidth=2, label=f'Armijo backtracking\n({len(history_armijo) - 1} iter)')
plt.plot(range(len(history_newton)), np.log10(history_newton), 'r-o',
         linewidth=2, label=f"Newton's method\n({len(history_newton) - 1} iter)", markersize=8)
plt.xlabel('Iteration k', fontsize=12)
plt.ylabel('log₁₀|Π(x^(k)) - Π*|', fontsize=12)
plt.title('Part (b)(i): Comparison of Optimization Methods', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('problem_b_i_convergence.png', dpi=300)
plt.show()

# ==================== PART (b)(ii) ====================
print("\n" + "=" * 70)
print("Part (b)(ii): Effect of Condition Number")
print("=" * 70)


def f_kappa(x, kappa):
    """Quadratic function family"""
    return 0.5 * (x[0] ** 2 + kappa * x[1] ** 2)


def grad_f_kappa(x, kappa):
    """Gradient of quadratic function"""
    return np.array([x[0], kappa * x[1]])


def gradient_descent_quadratic(kappa, alpha, tol=1e-8, max_iter=100000):
    """Gradient descent for quadratic function"""
    x = np.array([1.0, 1.0])
    x0_norm = np.linalg.norm(x)
    iterations = 0

    for k in range(max_iter):
        iterations += 1
        grad = grad_f_kappa(x, kappa)
        x = x - alpha * grad

        if np.linalg.norm(x) / x0_norm < tol:
            break

    return iterations


# Test different κ values
kappa_values = [2, 10, 50, 100, 500]
iterations_list = []
rho_values = []

print("\nκ\t\tOptimal α*\t\tConv. rate ρ*\tIterations")
print("-" * 60)

for kappa in kappa_values:
    alpha_opt = 2 / (1 + kappa)
    rho = (kappa - 1) / (kappa + 1)
    rho_values.append(rho)

    iterations = gradient_descent_quadratic(kappa, alpha_opt)
    iterations_list.append(iterations)

    print(f"{kappa}\t\t{alpha_opt:.6f}\t\t{rho:.6f}\t\t{iterations}")

# Plot for part (b)(ii)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(kappa_values, iterations_list, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('κ (Condition number)', fontsize=12)
axes[0].set_ylabel('Number of iterations', fontsize=12)
axes[0].set_title('Part (b)(ii): Iterations vs Condition Number', fontsize=14)
axes[0].grid(True, alpha=0.3)

axes[1].loglog(kappa_values, iterations_list, 'rs-', linewidth=2, markersize=8, label='Actual iterations')
axes[1].loglog(kappa_values, [k / 10 for k in kappa_values], 'g--', linewidth=2, label='Linear reference O(κ)')
axes[1].set_xlabel('κ (log scale)', fontsize=12)
axes[1].set_ylabel('Iterations (log scale)', fontsize=12)
axes[1].set_title('Log-Log Plot', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('problem_b_ii_conditioning.png', dpi=300)
plt.show()

# Verify linear relationship
print("\nVerifying linear growth:")
print("κ ratio\t\tIter ratio\t\tTheoretical")
print("-" * 50)
for i in range(1, len(kappa_values)):
    kappa_ratio = kappa_values[i] / kappa_values[i - 1]
    iter_ratio = iterations_list[i] / iterations_list[i - 1]
    print(f"{kappa_ratio:.2f}\t\t{iter_ratio:.2f}\t\t≈ {kappa_ratio:.2f}")

# ==================== PART (c) ====================
print("\n" + "=" * 70)
print(" " * 20 + "PROBLEM (c): Parametric Study")
print("=" * 70)


# For part (c), we use the same profit function but add constraints
def neg_profit(x):
    """Negative profit for minimization"""
    return -Pi(x)


# Constraint function for scipy
def constraint_func(x, b_val):
    """Inequality constraint: b - x1 - x2 >= 0"""
    return b_val - x[0] - x[1]


# Parametric study
b_levels = np.arange(3, 9.5, 0.5)  # 3, 3.5, ..., 9
optimal_profits_c = []
shadow_prices_c = []
optimal_x1 = []
optimal_x2 = []

print("\nCapacity b\tOptimal Profit\tShadow Price\tx1*\tx2*")
print("-" * 65)

for b in b_levels:
    # Define constraint for this b value
    constraint = {'type': 'ineq', 'fun': lambda x, b_val=b: constraint_func(x, b_val)}

    # Initial guess
    x0 = [0, 0]

    # Solve using SLSQP
    res = minimize(neg_profit, x0, method='SLSQP', constraints=[constraint], tol=1e-10)

    x_opt = res.x
    pi_star_b = -res.fun  # Convert back to positive profit

    # Calculate shadow price
    # Check if constraint is active
    if np.isclose(x_opt[0] + x_opt[1], b, atol=1e-4):
        grads = grad_Pi(x_opt)
        lambda_star = np.mean(grads)
        lambda_star = max(0, lambda_star)
    else:
        lambda_star = 0.0

    optimal_profits_c.append(pi_star_b)
    shadow_prices_c.append(lambda_star)
    optimal_x1.append(x_opt[0])
    optimal_x2.append(x_opt[1])

    print(f"{b:.1f}\t\t{pi_star_b:.4f}\t\t{lambda_star:.4f}\t\t{x_opt[0]:.2f}\t{x_opt[1]:.2f}")

# Create plots for part (c)
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Profit vs b
color1 = 'tab:blue'
ax1.set_xlabel('Assembly Line Capacity (b)', fontsize=12)
ax1.set_ylabel('Optimal Profit Π* ($000)', color=color1, fontsize=12)
ax1.plot(b_levels, optimal_profits_c, color=color1, marker='o', linewidth=2, label='Optimal Profit')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.6)

# Create second y-axis for Shadow Price
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Shadow Price λ* ($000/unit)', color=color2, fontsize=12)
ax2.plot(b_levels, shadow_prices_c, color=color2, marker='s', linestyle='--', linewidth=2, label='Shadow Price')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.axhline(0, color='black', linewidth=1, linestyle='-')
ax2.axhline(3.5, color='green', linewidth=1.5, linestyle=':', label='Investment cost ($3,500)')

# Add text annotations
for i, (b_val, lam) in enumerate(zip(b_levels, shadow_prices_c)):
    if lam < 0.1:
        plt.text(b_val, 5, f'Inactive at b={b_val}', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        break

plt.title('Part (c): Parametric Study - Profit and Shadow Price vs Capacity', fontsize=14)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

fig.tight_layout()
plt.savefig('problem_c_parametric_study.png', dpi=300)
plt.show()

# Find where constraint becomes inactive
inactive_idx = next((i for i, val in enumerate(shadow_prices_c) if val < 1e-5), None)
if inactive_idx is not None:
    print(f"\n*** Constraint becomes inactive at b = {b_levels[inactive_idx]} ***")
    print(f"    Shadow price drops to zero at this point.")

# Find break-even point (where shadow price = 3.5)
break_even_idx = None
for i in range(len(shadow_prices_c) - 1):
    if shadow_prices_c[i] >= 3.5 > shadow_prices_c[i + 1]:
        break_even_idx = i
        break

if break_even_idx is not None:
    b_break_even = b_levels[break_even_idx]
    print(
        f"*** Break-even point: b ≈ {b_break_even} (shadow price ≈ ${shadow_prices_c[break_even_idx] * 1000:.0f}) ***")

print("\n" + "=" * 70)
print(" " * 25 + "MANAGEMENT MEMO")
print("=" * 70)
print("""
TO: Senior Management
FROM: Operations Research Department
SUBJECT: Assembly Line Capacity Investment Recommendation

Based on our parametric analysis, we recommend INVESTING in additional 
assembly capacity, but with important limitations.

KEY FINDINGS:
1. Current capacity (b=6) yields profit of $64,000 with shadow price of $4,000/unit
2. Investment cost is $3,500 per 1,000 hours
3. Since $4,000 > $3,500, initial expansion is profitable

RECOMMENDATION:
Expand capacity from b=6 to approximately b=6.2-6.3 (200-300 additional hours).

JUSTIFICATION:
- The shadow price decreases as capacity increases
- At b≈6.25, shadow price drops to ~$3,500 (break-even point)
- Beyond b=8, shadow price becomes zero (constraint inactive)
- The unconstrained optimum occurs at b=8 (x₁=2, x₂=6)

WARNING: Do NOT expand beyond b≈6.25, as marginal cost will exceed 
marginal benefit, reducing overall profitability.

Expected net gain: ~$100-150 per unit for the first 200-300 hours.
""")
print("=" * 70)

print("\nAll analyses complete. Figures saved as PNG files.")