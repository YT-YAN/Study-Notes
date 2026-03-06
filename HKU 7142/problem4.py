#!/usr/bin/env python3
"""
Problem 4: Least Squares and Regularization
Conveyor belt power consumption modeling
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Data Definition
# =============================================================================

# Speed data (m/s)
v = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

# Power consumption data (kW)
P = np.array([1.2, 2.1, 3.5, 5.2, 7.4, 10.1, 13.2])

print("=" * 70)
print("Problem 4: Least Squares and Regularization")
print("=" * 70)

# =============================================================================
# Part (a): Ordinary Least Squares (Quadratic Model)
# =============================================================================

print("\n" + "=" * 70)
print("Part (a): Ordinary Least Squares (Quadratic Model)")
print("=" * 70)

# Construct design matrix X
# X = [1, v, v^2] for each data point
X = np.column_stack([np.ones(len(v)), v, v ** 2])
y = P.reshape(-1, 1)

print(f"\nDesign Matrix X (7×3):")
print("-" * 70)
print(X)

# Compute X^T X
XTX = X.T @ X
print(f"\nX^T X (3×3):")
print("-" * 70)
print(XTX)

# Compute X^T y
XTy = X.T @ y
print(f"\nX^T y (3×1):")
print("-" * 70)
print(XTy)

# Solve normal equations: w = (X^T X)^{-1} X^T y
w_ols = np.linalg.inv(XTX) @ XTy
print(f"\nOLS Coefficients w = [w₀, w₁, w₂]:")
print("-" * 70)
print(f"  w₀ = {w_ols[0, 0]:.8f}")
print(f"  w₁ = {w_ols[1, 0]:.8f}")
print(f"  w₂ = {w_ols[2, 0]:.8f}")

# Quadratic model equation
print(f"\nQuadratic Model: P = {w_ols[0, 0]:.4f} + {w_ols[1, 0]:.4f}v + {w_ols[2, 0]:.4f}v²")

# Compute predictions
P_pred = X @ w_ols

# =============================================================================
# FIX: R² Calculation (SS_res is a scalar, not array)
# =============================================================================

# Compute R²
SS_res = np.sum((y - P_pred) ** 2)  # This returns a scalar
SS_tot = np.sum((y - np.mean(y)) ** 2)  # This also returns a scalar
R_squared = 1 - SS_res / SS_tot

print(f"\nR² (Coefficient of Determination):")
print("-" * 70)
print(f"  SS_res (Residual Sum of Squares) = {SS_res:.8f}")  # ✅ FIXED: No [0,0]
print(f"  SS_tot (Total Sum of Squares) = {SS_tot:.8f}")  # ✅ FIXED: No [0,0]
print(f"  R² = {R_squared:.8f} ({R_squared * 100:.4f}%)")

# =============================================================================
# Part (b): Ridge Regression
# =============================================================================

print("\n" + "=" * 70)
print("Part (b): Ridge Regression")
print("=" * 70)

# Ridge regression for different lambda values
lambda_values = [0, 0.1, 10]
ridge_coefficients = []

print(f"\nRidge Coefficients for Different λ:")
print("-" * 70)
print(f"{'λ':<10} {'w₀':<15} {'w₁':<15} {'w₂':<15}")
print("-" * 70)

for lam in lambda_values:
    if lam == 0:
        # OLS (already computed)
        w_ridge = w_ols
    else:
        # Ridge: (X^T X + λI)w = X^T y
        I = np.eye(X.shape[1])
        w_ridge = np.linalg.inv(XTX + lam * I) @ XTy

    ridge_coefficients.append(w_ridge)
    print(f"{lam:<10.4f} {w_ridge[0, 0]:<15.8f} {w_ridge[1, 0]:<15.8f} {w_ridge[2, 0]:<15.8f}")

# =============================================================================
# Part (c): Prediction and Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part (c): Prediction and Analysis")
print("=" * 70)

# Predict power at v = 4.0 m/s
v_pred = 4.0
X_pred = np.array([[1, v_pred, v_pred ** 2]])
P_pred_4 = X_pred @ w_ols
P_true_4 = 17.1

print(f"\nPrediction at v = {v_pred} m/s:")
print("-" * 70)
print(f"  Predicted Power: {P_pred_4[0, 0]:.8f} kW")
print(f"  True Measurement: {P_true_4} kW")

# Compute prediction error
absolute_error = abs(P_true_4 - P_pred_4[0, 0])
relative_error = (absolute_error / P_true_4) * 100

print(f"\nPrediction Error:")
print("-" * 70)
print(f"  Absolute Error: {absolute_error:.8f} kW")
print(f"  Relative Error: {relative_error:.8f}%")

# =============================================================================
# Visualization
# =============================================================================

print("\n" + "=" * 70)
print("Generating Figures...")
print("=" * 70)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Data and OLS fit
v_fine = np.linspace(0, 4.5, 100)
X_fine = np.column_stack([np.ones(len(v_fine)), v_fine, v_fine ** 2])
P_fine_ols = X_fine @ w_ols

axes[0, 0].plot(v, P, 'ro', markersize=10, label='Training Data')
axes[0, 0].plot(v_fine, P_fine_ols, 'b-', linewidth=2, label='OLS Fit')
axes[0, 0].plot(4.0, 17.1, 'g*', markersize=15, label='True Value (v=4.0)')
axes[0, 0].plot(4.0, P_pred_4[0, 0], 'm*', markersize=15, label='Predicted (v=4.0)')
axes[0, 0].set_xlabel('Speed v (m/s)', fontsize=11)
axes[0, 0].set_ylabel('Power P (kW)', fontsize=11)
axes[0, 0].set_title('Quadratic Model Fit (OLS)', fontsize=12, fontweight='bold')
axes[0, 0].legend(loc='best')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(0, 4.5)

# Subplot 2: Ridge coefficients vs lambda
lambda_range = np.logspace(-2, 2, 100)
coef_ridge = []
for lam in lambda_range:
    w_r = np.linalg.inv(XTX + lam * np.eye(3)) @ XTy
    coef_ridge.append(w_r.flatten())
coef_ridge = np.array(coef_ridge)

axes[0, 1].plot(lambda_range, coef_ridge[:, 0], 'b-', linewidth=2, label='w₀')
axes[0, 1].plot(lambda_range, coef_ridge[:, 1], 'r-', linewidth=2, label='w₁')
axes[0, 1].plot(lambda_range, coef_ridge[:, 2], 'g-', linewidth=2, label='w₂')
axes[0, 1].set_xscale('log')
axes[0, 1].set_xlabel('λ (log scale)', fontsize=11)
axes[0, 1].set_ylabel('Coefficient Value', fontsize=11)
axes[0, 1].set_title('Ridge Coefficients vs λ', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc='best')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=0.1, color='orange', linestyle='--', label='λ=0.1')
axes[0, 1].axvline(x=10, color='purple', linestyle='--', label='λ=10')
axes[0, 1].legend(loc='best')

# Subplot 3: Residuals
residuals = y - P_pred
axes[1, 0].plot(P_pred, residuals, 'bo', markersize=8)
axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Power (kW)', fontsize=11)
axes[1, 0].set_ylabel('Residuals (kW)', fontsize=11)
axes[1, 0].set_title('Residual Plot', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Training vs Prediction
axes[1, 1].bar(['Training\nRange', 'Extrapolation'],
               [np.mean(np.abs(residuals)), absolute_error],
               color=['blue', 'red'], alpha=0.7)
axes[1, 1].set_ylabel('Average Error (kW)', fontsize=11)
axes[1, 1].set_title('Training Error vs Extrapolation Error', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add text annotation for extrapolation error
axes[1, 1].text(1, absolute_error + 0.1, f'{absolute_error:.4f} kW',
                ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('problem4_least_squares.png', dpi=300, bbox_inches='tight')
print(f"\nFigure saved as 'problem4_least_squares.png'")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("Summary of Results")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                        KEY RESULTS                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ORDINARY LEAST SQUARES (λ=0):                                     │
│  Model: P = {w_ols[0, 0]:.4f} + {w_ols[1, 0]:.4f}v + {w_ols[2, 0]:.4f}v²                            │
│  R² = {R_squared:.4f} ({R_squared * 100:.2f}% variance explained)                       │
│                                                                     │
│  PREDICTION (v=4.0 m/s):                                           │
│  Predicted: {P_pred_4[0, 0]:.4f} kW                                                │
│  True:      {P_true_4} kW                                                 │
│  Error:     {absolute_error:.4f} kW ({relative_error:.2f}%)                                       │
│                                                                     │
│  CONCLUSION:                                                       │
│  ✓ Quadratic model fits training data excellently                 │
│  ✓ Prediction error is small but extrapolation risky              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 70)
print("END OF ANALYSIS")
print("=" * 70)