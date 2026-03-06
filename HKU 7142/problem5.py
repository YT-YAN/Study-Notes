import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# =============================================================================
# Data Definition
# =============================================================================

# Customer data: [monthly_shipments, average_package_weight_kg]
customers = np.array([
    [50, 2],  # Customer 1
    [60, 3],  # Customer 2
    [45, 2],  # Customer 3
    [200, 25],  # Customer 4
    [180, 30],  # Customer 5
    [220, 22],  # Customer 6
    [120, 12],  # Customer 7
    [140, 15],  # Customer 8
    [130, 10]  # Customer 9
])

customer_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]

print("=" * 70)
print("Problem 5: K-Means Clustering - Customer Segmentation")
print("=" * 70)


# =============================================================================
# K-Means Implementation from Scratch
# =============================================================================

def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.

    Parameters:
    -----------
    x1 : array-like
        First point
    x2 : array-like
        Second point

    Returns:
    --------
    distance : float
        Euclidean distance
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def initialize_centroids(X: np.ndarray, K: int, method: str = 'random') -> np.ndarray:
    """
    Initialize centroids for K-means.

    Parameters:
    -----------
    X : ndarray
        Data matrix (n_samples x n_features)
    K : int
        Number of clusters
    method : str
        Initialization method ('random' or 'first_k')

    Returns:
    --------
    centroids : ndarray
        Initial centroids (K x n_features)
    """
    n_samples = X.shape[0]

    if method == 'first_k':
        # Use first K data points as initial centroids
        centroids = X[:K].copy()
    else:
        # Random selection
        indices = np.random.choice(n_samples, K, replace=False)
        centroids = X[indices].copy()

    return centroids


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each data point to the nearest centroid.

    Parameters:
    -----------
    X : ndarray
        Data matrix (n_samples x n_features)
    centroids : ndarray
        Centroid matrix (K x n_features)

    Returns:
    --------
    labels : ndarray
        Cluster assignments for each data point
    """
    n_samples = X.shape[0]
    K = centroids.shape[0]
    labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        distances = np.zeros(K)
        for k in range(K):
            distances[k] = euclidean_distance(X[i], centroids[k])
        labels[i] = np.argmin(distances)

    return labels


def update_centroids(X: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """
    Recompute centroids as the mean of assigned points.

    Parameters:
    -----------
    X : ndarray
        Data matrix (n_samples x n_features)
    labels : ndarray
        Cluster assignments
    K : int
        Number of clusters

    Returns:
    --------
    centroids : ndarray
        Updated centroids (K x n_features)
    """
    n_features = X.shape[1]
    centroids = np.zeros((K, n_features))

    for k in range(K):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            centroids[k] = np.mean(cluster_points, axis=0)
        else:
            # If cluster is empty, keep previous centroid
            centroids[k] = np.zeros(n_features)

    return centroids


def compute_wcss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Compute Within-Cluster Sum of Squares (WCSS).

    WCSS = sum over k of sum over x in C_k of ||x - mu_k||^2

    Parameters:
    -----------
    X : ndarray
        Data matrix
    labels : ndarray
        Cluster assignments
    centroids : ndarray
        Centroid matrix

    Returns:
    --------
    wcss : float
        Within-Cluster Sum of Squares
    """
    wcss = 0.0
    n_samples = X.shape[0]

    for i in range(n_samples):
        cluster_k = labels[i]
        wcss += np.sum((X[i] - centroids[cluster_k]) ** 2)

    return wcss


def kmeans(X: np.ndarray, K: int, max_iters: int = 100,
           tol: float = 1e-6, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    K-Means clustering algorithm implemented from scratch.

    Parameters:
    -----------
    X : ndarray
        Data matrix (n_samples x n_features)
    K : int
        Number of clusters
    max_iters : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    labels : ndarray
        Cluster assignments
    centroids : ndarray
        Final centroids
    wcss : float
        Final WCSS value
    n_iters : int
        Number of iterations until convergence
    """
    np.random.seed(random_state)

    # Initialize centroids
    centroids = initialize_centroids(X, K, method='random')

    for iteration in range(max_iters):
        # Assignment step
        labels = assign_clusters(X, centroids)

        # Update step
        new_centroids = update_centroids(X, labels, K)

        # Check convergence
        centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids

        if centroid_shift < tol:
            break

    # Compute final WCSS
    wcss = compute_wcss(X, labels, centroids)
    n_iters = iteration + 1

    return labels, centroids, wcss, n_iters


# =============================================================================
# Part (b): Run K-Means for K = 1, 2, 3, 4, 5
# =============================================================================

print("\n" + "=" * 70)
print("Part (b): K-Means for Different K Values")
print("=" * 70)

K_values = [1, 2, 3, 4, 5]
wcss_values = []
results = {}

print(f"\n{'K':<5} {'WCSS':<15} {'Iterations':<15} {'Converged'}")
print("-" * 70)

for K in K_values:
    labels, centroids, wcss, n_iters = kmeans(customers, K, max_iters=100, random_state=42)
    wcss_values.append(wcss)
    results[K] = {
        'labels': labels,
        'centroids': centroids,
        'wcss': wcss,
        'n_iters': n_iters
    }

    converged = "Yes" if n_iters < 100 else "No"
    print(f"{K:<5} {wcss:<15.4f} {n_iters:<15} {converged}")

# =============================================================================
# Elbow Curve Plot
# =============================================================================

print("\n" + "=" * 70)
print("Elbow Method for Optimal K")
print("=" * 70)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Elbow Curve
axes[0].plot(K_values, wcss_values, 'bo-', linewidth=2, markersize=10)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Curve - Optimal K Determination', fontsize=14, fontweight='bold')
axes[0].set_xticks(K_values)
axes[0].grid(True, alpha=0.3)

# Annotate each point
for i, (K, wcss) in enumerate(zip(K_values, wcss_values)):
    axes[0].annotate(f'{wcss:.1f}', (K, wcss), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=10)

# Determine optimal K using elbow method
# Calculate the rate of decrease
wcss_diff = np.diff(wcss_values)
wcss_diff2 = np.diff(wcss_diff)

# Find the elbow point (maximum second derivative)
optimal_K_idx = np.argmax(wcss_diff2) + 2  # +2 because of two diff operations
optimal_K = K_values[optimal_K_idx]

axes[0].axvline(x=optimal_K, color='red', linestyle='--', linewidth=2,
                label=f'Optimal K = {optimal_K}')
axes[0].legend(loc='best')

# Plot 2: Percentage of Variance Explained
total_wcss = wcss_values[0]  # WCSS for K=1
variance_explained = [(1 - wcss / total_wcss) * 100 for wcss in wcss_values]

axes[1].plot(K_values, variance_explained, 'go-', linewidth=2, markersize=10)
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Variance Explained (%)', fontsize=12)
axes[1].set_title('Variance Explained by Number of Clusters', fontsize=14, fontweight='bold')
axes[1].set_xticks(K_values)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=90, color='orange', linestyle='--', label='90% Threshold')

# Annotate each point
for i, (K, var) in enumerate(zip(K_values, variance_explained)):
    axes[1].annotate(f'{var:.1f}%', (K, var), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=10)

axes[1].legend(loc='best')

plt.tight_layout()
plt.savefig('problem5_elbow_curve.png', dpi=300, bbox_inches='tight')
print(f"\nElbow curve saved as 'problem5_elbow_curve.png'")

print(f"\nOptimal K (Elbow Method): K = {optimal_K}")
print(f"Variance Explained at K={optimal_K}: {variance_explained[optimal_K - 1]:.2f}%")

# =============================================================================
# Part (c): Customer Segment Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Part (c): Customer Segment Analysis")
print("=" * 70)

# Use optimal K for final analysis
K_opt = optimal_K
final_labels = results[K_opt]['labels']
final_centroids = results[K_opt]['centroids']

print(f"\nUsing Optimal K = {K_opt} for Customer Segmentation")
print("-" * 70)

# Analyze each cluster
cluster_analysis = {}

for k in range(K_opt):
    cluster_members = [customer_ids[i] for i in range(len(customer_ids)) if final_labels[i] == k]
    cluster_data = customers[final_labels == k]

    centroid = final_centroids[k]
    avg_shipments = np.mean(cluster_data[:, 0])
    avg_weight = np.mean(cluster_data[:, 1])

    cluster_analysis[k] = {
        'members': cluster_members,
        'size': len(cluster_members),
        'avg_shipments': avg_shipments,
        'avg_weight': avg_weight,
        'centroid': centroid
    }

    print(f"\n{'=' * 70}")
    print(f"Cluster {k + 1}:")
    print(f"{'=' * 70}")
    print(f"  Customers: {cluster_members}")
    print(f"  Cluster Size: {len(cluster_members)} customers ({len(cluster_members) / len(customer_ids) * 100:.1f}%)")
    print(f"  Average Monthly Shipments: {avg_shipments:.1f}")
    print(f"  Average Package Weight: {avg_weight:.2f} kg")
    print(f"  Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")

# =============================================================================
# Business Interpretation and Strategy Recommendations
# =============================================================================

print("\n" + "=" * 70)
print("Business Interpretation and Logistics Strategy")
print("=" * 70)

# Create segment descriptions based on cluster characteristics
segment_descriptions = []
strategy_recommendations = []

for k in range(K_opt):
    info = cluster_analysis[k]

    # Determine segment type based on characteristics
    if info['avg_shipments'] < 100 and info['avg_weight'] < 10:
        segment_type = "Small Volume, Light Weight"
        segment_name = "Small Business / Individual"
    elif info['avg_shipments'] >= 100 and info['avg_weight'] < 15:
        segment_type = "Medium Volume, Medium Weight"
        segment_name = "Growing Business"
    else:
        segment_type = "Large Volume, Heavy Weight"
        segment_name = "Enterprise / Large Business"

    segment_descriptions.append({
        'cluster': k + 1,
        'type': segment_type,
        'name': segment_name
    })

    # Generate strategy recommendations
    if info['avg_shipments'] < 100:
        pricing = "Standard pricing with volume discounts for growth"
        service = "Basic service level with email support"
        delivery = "Standard delivery (3-5 business days)"
    elif info['avg_shipments'] < 200:
        pricing = "Tiered pricing with 10-15% discount"
        service = "Priority support with dedicated account manager"
        delivery = "Express delivery options (1-2 business days)"
    else:
        pricing = "Custom enterprise pricing with 20-30% discount"
        service = "Premium 24/7 support with SLA guarantees"
        delivery = "Same-day or next-day delivery options"

    strategy_recommendations.append({
        'cluster': k + 1,
        'pricing': pricing,
        'service': service,
        'delivery': delivery
    })

# Print detailed analysis
print("\n" + "-" * 70)
print("DETAILED SEGMENT ANALYSIS")
print("-" * 70)

for i in range(K_opt):
    print(f"\n{'=' * 70}")
    print(f"SEGMENT {i + 1}: {segment_descriptions[i]['name']}")
    print(f"{'=' * 70}")
    print(f"\nCharacteristics:")
    print(f"  - Type: {segment_descriptions[i]['type']}")
    print(f"  - Customers: {cluster_analysis[i]['members']}")
    print(f"  - Avg Shipments: {cluster_analysis[i]['avg_shipments']:.1f}/month")
    print(f"  - Avg Weight: {cluster_analysis[i]['avg_weight']:.2f} kg")
    print(f"\nWhat Distinguishes This Group:")

    if i == 0:
        print(f"  - Lowest shipping volume and package weight")
        print(f"  - Likely individual consumers or very small businesses")
        print(f"  - Price-sensitive, less frequent shippers")
    elif i == 1:
        print(f"  - Medium shipping volume with moderate package weights")
        print(f"  - Growing businesses with regular shipping needs")
        print(f"  - Value quality and reliability over lowest price")
    else:
        print(f"  - Highest shipping volume and package weight")
        print(f"  - Established enterprises with consistent high-volume needs")
        print(f"  - Priority is reliability and speed over cost")

    print(f"\nTailored Logistics Strategy:")
    print(f"  Pricing: {strategy_recommendations[i]['pricing']}")
    print(f"  Service Level: {strategy_recommendations[i]['service']}")
    print(f"  Delivery Options: {strategy_recommendations[i]['delivery']}")

# =============================================================================
# Visualization: Cluster Plot
# =============================================================================

print("\n" + "=" * 70)
print("Generating Cluster Visualization...")
print("=" * 70)

# Create scatter plot with clusters
fig2, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.rainbow(np.linspace(0, 1, K_opt))

for k in range(K_opt):
    cluster_points = customers[final_labels == k]
    cluster_ids = [customer_ids[i] for i in range(len(customer_ids)) if final_labels[i] == k]

    ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
               c=[colors[k]], s=150, alpha=0.7,
               label=f'Cluster {k + 1} (n={len(cluster_points)})',
               edgecolors='black', linewidth=1.5)

    # Annotate customer IDs
    for i, point in enumerate(cluster_points):
        ax.annotate(f'C{cluster_ids[i]}', (point[0], point[1]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=9, fontweight='bold')

# Plot centroids
ax.scatter(final_centroids[:, 0], final_centroids[:, 1],
           c='red', s=300, marker='X',
           label='Centroids', edgecolors='black', linewidth=2)

# Annotate centroids
for k in range(K_opt):
    ax.annotate(f'μ{k + 1}', (final_centroids[k, 0], final_centroids[k, 1]),
                textcoords="offset points", xytext=(10, -15),
                fontsize=12, fontweight='bold', color='red')

ax.set_xlabel('Monthly Shipments', fontsize=12)
ax.set_ylabel('Average Package Weight (kg)', fontsize=12)
ax.set_title(f'Customer Segmentation (K-Means, K={K_opt})', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('problem5_clusters.png', dpi=300, bbox_inches='tight')
print(f"\nCluster visualization saved as 'problem5_clusters.png'")

# =============================================================================
# Summary Table
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY OF RESULTS")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                        KEY FINDINGS                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  OPTIMAL NUMBER OF CLUSTERS:                                       │
│  ─────────────────────────────                                      │
│  K* = {optimal_K} (determined by elbow method)                            │
│  Variance Explained: {variance_explained[optimal_K - 1]:.2f}%                              │
│                                                                     │
│  CLUSTER SUMMARY:                                                   │
│  ────────────────                                                   │
│""")

for k in range(K_opt):
    print(f"│  Cluster {k + 1}: {cluster_analysis[k]['size']} customers, " +
          f"Avg Shipments={cluster_analysis[k]['avg_shipments']:.0f}, " +
          f"Avg Weight={cluster_analysis[k]['avg_weight']:.1f}kg".ljust(64) + "│")

print(f"""│                                                                     │
│  BUSINESS RECOMMENDATIONS:                                          │
│  ───────────────────────                                            │
│  ✓ Segment customers by volume and weight for targeted marketing   │
│  ✓ Offer tiered pricing based on cluster membership                │
│  ✓ Allocate resources proportionally to cluster value              │
│  ✓ Monitor cluster migration over time for business growth         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "=" * 70)
print("END OF ANALYSIS")
print("=" * 70)