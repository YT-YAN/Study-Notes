import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Data Definition (from Problem 6c)
# =============================================================================

# Sample IDs
samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Predicted probabilities of being Defective
prob_defective = np.array([0.92, 0.85, 0.78, 0.65, 0.55, 0.45, 0.32, 0.25, 0.15, 0.08])

# True labels (1 = Defective, 0 = Good)
true_labels = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0])

print("=" * 70)
print("Problem 6: Classification Evaluation - Part (c)")
print("ROC Curve and AUC Computation")
print("=" * 70)

# =============================================================================
# Basic Statistics
# =============================================================================

n_samples = len(samples)
n_positives = np.sum(true_labels)
n_negatives = n_samples - n_positives

print(f"\nData Summary:")
print("-" * 70)
print(f"  Total Samples: {n_samples}")
print(f"  Positive Samples (Defective): {n_positives}")
print(f"  Negative Samples (Good): {n_negatives}")

# =============================================================================
# Compute TPR and FPR at Each Threshold
# =============================================================================

print(f"\n{'=' * 70}")
print("Step 1: Compute TPR and FPR at Each Threshold")
print("=" * 70)

# Get unique thresholds (sorted in descending order)
thresholds = np.unique(prob_defective)[::-1]

# Add a threshold above max to start at (0, 0)
thresholds = np.concatenate([[thresholds[0] + 0.01], thresholds])

# Lists to store TPR and FPR values
tpr_list = [0.0]
fpr_list = [0.0]
threshold_list = [thresholds[0]]

print(f"\n{'Threshold':<12} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<5} {'TPR':<10} {'FPR':<10}")
print("-" * 70)

# Initial point (threshold > max probability)
print(f"{thresholds[0]:<12.4f} {0:<5} {0:<5} {n_positives:<5} {n_negatives:<5} {0.0:<10.4f} {0.0:<10.4f}")

# Compute for each threshold
for thresh in thresholds[1:]:
    # Predictions at this threshold
    predictions = (prob_defective >= thresh).astype(int)

    # Confusion matrix components
    tp = np.sum((predictions == 1) & (true_labels == 1))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    tn = np.sum((predictions == 0) & (true_labels == 0))

    # TPR and FPR
    tpr = tp / n_positives if n_positives > 0 else 0
    fpr = fp / n_negatives if n_negatives > 0 else 0

    tpr_list.append(tpr)
    fpr_list.append(fpr)
    threshold_list.append(thresh)

    print(f"{thresh:<12.4f} {tp:<5} {fp:<5} {fn:<5} {tn:<5} {tpr:<10.4f} {fpr:<10.4f}")

# Convert to numpy arrays
tpr_array = np.array(tpr_list)
fpr_array = np.array(fpr_list)
threshold_array = np.array(threshold_list)

# =============================================================================
# Compute AUC Using Trapezoidal Rule (FIXED for NumPy 2.0+)
# =============================================================================

print(f"\n{'=' * 70}")
print("Step 2: Compute AUC (Area Under Curve)")
print("=" * 70)

# Method 1: Try NumPy's built-in function (handles both old and new versions)
try:
    # NumPy >= 2.0
    auc_numpy = np.trapezoid(tpr_array, fpr_array)
    print("  Using np.trapezoid (NumPy >= 2.0)")
except AttributeError:
    # NumPy < 2.0
    auc_numpy = np.trapz(tpr_array, fpr_array)
    print("  Using np.trapz (NumPy < 2.0)")

# Method 2: Manual calculation (for verification and compatibility)
auc_manual = 0.0
for i in range(1, len(fpr_array)):
    width = fpr_array[i] - fpr_array[i - 1]
    height = (tpr_array[i] + tpr_array[i - 1]) / 2
    auc_manual += width * height

print(f"\n  AUC (NumPy function): {auc_numpy:.8f}")
print(f"  AUC (Manual calculation): {auc_manual:.8f}")
print(f"\n  Final AUC = {auc_numpy:.4f}")

# Interpretation
print(f"\n  Interpretation:")
if auc_numpy >= 0.9:
    print(f"  - Excellent classifier (AUC >= 0.9)")
elif auc_numpy >= 0.8:
    print(f"  - Good classifier (0.8 <= AUC < 0.9)")
elif auc_numpy >= 0.7:
    print(f"  - Fair classifier (0.7 <= AUC < 0.8)")
elif auc_numpy >= 0.6:
    print(f"  - Poor classifier (0.6 <= AUC < 0.7)")
else:
    print(f"  - Very poor classifier (AUC < 0.6)")

# =============================================================================
# Find Optimal Threshold (Youden's J Statistic)
# =============================================================================

print(f"\n{'=' * 70}")
print("Step 3: Find Optimal Threshold")
print("=" * 70)

# Youden's J statistic: J = TPR - FPR
youden_j = tpr_array - fpr_array
optimal_idx = np.argmax(youden_j)
optimal_threshold = threshold_array[optimal_idx]
optimal_tpr = tpr_array[optimal_idx]
optimal_fpr = fpr_array[optimal_idx]

print(f"\n  Youden's J Statistic: J = TPR - FPR")
print(f"  Optimal Threshold: {optimal_threshold:.4f}")
print(f"  At this threshold:")
print(f"    TPR = {optimal_tpr:.4f}")
print(f"    FPR = {optimal_fpr:.4f}")
print(f"    J   = {youden_j[optimal_idx]:.4f}")

# =============================================================================
# Visualization: ROC Curve
# =============================================================================

print(f"\n{'=' * 70}")
print("Step 4: Generate ROC Curve Plot")
print("=" * 70)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: ROC Curve
axes[0].plot(fpr_array, tpr_array, 'b-', linewidth=2,
             label=f'ROC Curve (AUC = {auc_numpy:.4f})')
axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2,
             label='Random Classifier (AUC = 0.5)')
axes[0].scatter(fpr_array, tpr_array, c='blue', s=80, alpha=0.7,
                edgecolors='black', linewidth=1)

# Annotate each point with threshold
for i, (fpr, tpr, thresh) in enumerate(zip(fpr_array, tpr_array, threshold_array)):
    if i == 0:
        label = 'Start'
    else:
        label = f'{thresh:.2f}'
    axes[0].annotate(label, (fpr, tpr), textcoords="offset points",
                     xytext=(5, 5), ha='left', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                               alpha=0.5))

# Mark optimal threshold point
axes[0].scatter(optimal_fpr, optimal_tpr, c='green', s=150, marker='*',
                label=f'Optimal (t={optimal_threshold:.2f})',
                edgecolors='black', linewidth=2, zorder=5)

axes[0].set_xlabel('False Positive Rate (FPR)', fontsize=12)
axes[0].set_ylabel('True Positive Rate (TPR)', fontsize=12)
axes[0].set_title('ROC Curve - Quality Control Classifier', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])
axes[0].set_aspect('equal')

# Plot 2: Youden's J Statistic vs Threshold
axes[1].plot(threshold_array[1:], youden_j[1:], 'g-', linewidth=2, marker='o')
axes[1].axvline(x=optimal_threshold, color='red', linestyle='--',
                linewidth=2, label=f'Optimal Threshold = {optimal_threshold:.2f}')
axes[1].scatter(optimal_threshold, youden_j[optimal_idx], c='red', s=150,
                marker='*', zorder=5)
axes[1].set_xlabel('Threshold', fontsize=12)
axes[1].set_ylabel("Youden's J Statistic (TPR - FPR)", fontsize=12)
axes[1].set_title("Optimal Threshold Selection", fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 1])

plt.tight_layout()
plt.savefig('problem6_roc_curve.png', dpi=300, bbox_inches='tight')
print(f"\n  Figure saved as 'problem6_roc_curve.png'")

# =============================================================================
# Summary
# =============================================================================

print(f"\n{'=' * 70}")
print("Summary of Results")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                        KEY RESULTS                                  │
├─────────────────────────────────────────────────────────────────────┤
│  AUC = {auc_numpy:.4f} ({'Good' if auc_numpy >= 0.8 else 'Fair'} Classifier)                                    │
│  Optimal Threshold = {optimal_threshold:.4f}                                          │
│  TPR = {optimal_tpr:.4f}                                                              │
│  FPR = {optimal_fpr:.4f}                                                              │
└─────────────────────────────────────────────────────────────────────┘
""")

print(f"\n{'=' * 70}")
print("END OF ANALYSIS")
print("=" * 70)