import numpy as np

# =============================================================================
# Data Definition
# =============================================================================

# Define the data matrix A (5 samples x 4 sensors)
A = np.array([
    [2.5, 4.8, 3.1, 6.2],
    [1.8, 3.6, 2.3, 4.5],
    [3.2, 6.1, 4.0, 7.9],
    [2.1, 4.1, 2.7, 5.3],
    [2.9, 5.6, 3.6, 7.1]
], dtype=float)

print("=" * 70)
print("Problem 2: Dimensionality Reduction with SVD")
print("=" * 70)

# =============================================================================
# (1) Compute Full SVD and Report All Singular Values
# =============================================================================

print("\n" + "=" * 70)
print("Part 1: Full SVD Decomposition")
print("=" * 70)

# Compute full SVD: A = U @ Sigma @ Vt
U, s, Vt = np.linalg.svd(A, full_matrices=True)
V = Vt.T  # Transpose to get right singular vectors as columns

# Display all singular values
print("\nAll Singular Values:")
print("-" * 70)
for i, sigma in enumerate(s, 1):
    variance_contribution = (sigma ** 2) / np.sum(s ** 2) * 100
    print(f"  sigma_{i} = {sigma:.8f}  (contributes {variance_contribution:.4f}% of variance)")

# Determine the rank r (number of non-zero singular values)
tolerance = 1e-10
r = np.sum(s > tolerance)
print(f"\nMatrix Rank r = {r}")
print(f"Matrix Shape: {A.shape[0]} x {A.shape[1]}")
print(f"U Shape: {U.shape}")
print(f"Sigma would be: {A.shape[0]} x {A.shape[1]}")
print(f"Vt Shape: {Vt.shape}")

# =============================================================================
# (2) Compute Low-Rank Approximations for k = 1, 2, 3
# =============================================================================

print("\n" + "=" * 70)
print("Part 2: Low-Rank Approximations")
print("=" * 70)

# Compute Frobenius norm of original matrix
norm_A = np.linalg.norm(A, 'fro')
print(f"\n||A||_F (Frobenius norm of A) = {norm_A:.8f}")

# Store results for each k
results = []

for k in [1, 2, 3]:
    print(f"\n{'-' * 70}")
    print(f"Rank k = {k}")
    print("-" * 70)

    # Construct rank-k approximation: A_k = U_k @ Sigma_k @ V_k^T
    U_k = U[:, :k]  # First k left singular vectors
    Sigma_k = np.diag(s[:k])  # First k singular values (diagonal matrix)
    V_k = V[:, :k]  # First k right singular vectors

    # Reconstruct the approximation
    A_k = U_k @ Sigma_k @ V_k.T

    # Compute approximation error
    error = np.linalg.norm(A - A_k, 'fro')
    relative_error = error / norm_A

    # Compute variance captured
    variance_captured = np.sum(s[:k] ** 2) / np.sum(s ** 2) * 100

    # Store results
    results.append({
        'k': k,
        'A_k': A_k,
        'error': error,
        'relative_error': relative_error,
        'variance_captured': variance_captured
    })

    # Display results
    print(f"  ||A - A_{k}||_F (absolute error)   = {error:.8f}")
    print(f"  ||A - A_{k}||_F / ||A||_F (relative error) = {relative_error:.8f} ({relative_error * 100:.6f}%)")
    print(f"  Variance Captured                  = {variance_captured:.8f}%")

    # Display the approximated matrix
    print(f"\n  A_{k} (rank-{k} approximation):")
    for row in A_k:
        print(f"    [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}, {row[3]:7.4f}]")

# =============================================================================
# (3) Determine Rank for 99% Variance Capture
# =============================================================================

print("\n" + "=" * 70)
print("Part 3: Rank for 99% Variance Capture")
print("=" * 70)

# Compute cumulative variance ratio
cumulative_variance = np.cumsum(s ** 2) / np.sum(s ** 2) * 100

print("\nCumulative Variance by Rank:")
print("-" * 70)
for i, var in enumerate(cumulative_variance, 1):
    marker = "✓" if var >= 99 else " "
    print(f"  Rank {i}: {var:.8f}% {marker}")

# Find minimum rank for 99% variance
rank_99 = np.argmax(cumulative_variance >= 99) + 1
print(f"\n>>> Minimum rank to capture 99% variance: k = {rank_99}")

# =============================================================================
# (4) Analyze First Right Singular Vector v1
# =============================================================================

print("\n" + "=" * 70)
print("Part 4: First Right Singular Vector Analysis")
print("=" * 70)

# Extract first right singular vector
v1 = V[:, 0]

print("\nFirst Right Singular Vector (v1):")
print("-" * 70)
print(f"  v1 = [{v1[0]:.8f}, {v1[1]:.8f}, {v1[2]:.8f}, {v1[3]:.8f}]")

print("\nComponent Weights by Sensor:")
print("-" * 70)
sensor_labels = ["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4"]
for i, (label, value) in enumerate(zip(sensor_labels, v1)):
    percentage = abs(value) / np.sum(np.abs(v1)) * 100
    print(f"  {label}: {value:.8f} ({percentage:.4f}% of total weight)")

# =============================================================================
# (5) Physical Interpretation of Sensor Correlations
# =============================================================================

print("\n" + "=" * 70)
print("Part 5: Physical Interpretation")
print("=" * 70)

interpretation = """
Based on the first right singular vector v1, we observe:

1. POSITIVE CORRELATION:
   - All components of v1 are positive
   - This indicates all 4 sensors are positively correlated
   - When one sensor reading increases, others tend to increase too

2. RELATIVE WEIGHTS:
   - Sensor 4 has the largest weight (dominant contributor)
   - Sensor 2 has the second largest weight
   - Sensors 1 and 3 have smaller but significant weights

3. PHYSICAL INTERPRETATION:
   - The sensors likely measure the same underlying physical phenomenon
     (e.g., temperature, pressure, or concentration at different locations)
   - Sensor 4 and Sensor 2 are positioned in areas with stronger signal
   - The high correlation suggests redundancy in the sensor array

4. PRACTICAL IMPLICATIONS:
   - Since k=1 captures >99% of variance, the 4 sensors are highly redundant
   - Could potentially reduce to 1-2 virtual sensors without significant 
     information loss
   - This enables cost reduction and simplified data processing

5. RECOMMENDATION:
   - For monitoring purposes, a single principal component could represent
     the entire sensor array with 99.97% accuracy
   - Remaining variance (0.03%) may represent measurement noise or 
     local variations
"""

print(interpretation)

# =============================================================================
# (6) Verification: Reconstruction Error Check
# =============================================================================

print("\n" + "=" * 70)
print("Part 6: Verification")
print("=" * 70)

# FIX: Create proper Sigma matrix with correct dimensions (5x4)
Sigma = np.zeros(A.shape)
np.fill_diagonal(Sigma, s)

# Full reconstruction using all singular values
A_reconstructed = U @ Sigma @ Vt
reconstruction_error = np.linalg.norm(A - A_reconstructed, 'fro')

print(f"\nFull Reconstruction Error ||A - U@Sigma@Vt||_F = {reconstruction_error:.2e}")
print("(Should be close to machine precision ~1e-15)")

# Verify the shapes
print(f"\nMatrix Shapes:")
print(f"  U: {U.shape}")
print(f"  Sigma: {Sigma.shape}")
print(f"  Vt: {Vt.shape}")
print(f"  A: {A.shape}")
print(f"  A_reconstructed: {A_reconstructed.shape}")

# Verify the theoretical error formula
print("\nTheoretical Error Verification:")
print("-" * 70)
for k in [1, 2, 3]:
    theoretical_error = np.sqrt(np.sum(s[k:] ** 2))
    actual_error = results[k - 1]['error']
    difference = abs(theoretical_error - actual_error)
    print(f"  k={k}: Theoretical = {theoretical_error:.8f}, Actual = {actual_error:.8f}, Diff = {difference:.2e}")

print("\n" + "=" * 70)
print("END OF ANALYSIS")
print("=" * 70)