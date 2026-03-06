import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# =============================================================================
# Data Definition
# =============================================================================

# Time points (seconds)
t = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5])

# Position data (cm)
x = np.array([0, 8, 22, 35, 42, 45])

# Boundary conditions: arm starts and ends at rest
# x'(0) = 0, x'(2.5) = 0 (clamped boundary conditions)
bc_type = ((1, 0.0), (1, 0.0))  # (derivative_order, derivative_value)

print("=" * 70)
print("Problem 3: Interpolation Methods - Robotic Arm Trajectory")
print("=" * 70)

# =============================================================================
# Part (a): Newton Divided Difference Interpolation
# =============================================================================

print("\n" + "=" * 70)
print("Part (a): Newton Divided Difference Interpolation")
print("=" * 70)

# Use first three data points
t_newton = t[:3]  # [0, 0.5, 1.0]
x_newton = x[:3]  # [0, 8, 22]

print(f"\nData points used: {len(t_newton)}")
for i in range(len(t_newton)):
    print(f"  (t{i}, x{i}) = ({t_newton[i]}, {x_newton[i]})")


# Construct divided difference table
def divided_difference_table(t, x):
    """
    Construct the divided difference table for Newton interpolation.

    Parameters:
    -----------
    t : array-like
        Time points
    x : array-like
        Position values

    Returns:
    --------
    dd : list
        Divided differences (diagonal elements for Newton form)
    table : 2D array
        Full divided difference table
    """
    n = len(t)
    table = np.zeros((n, n))
    table[:, 0] = x

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (t[i + j] - t[i])

    # Extract diagonal elements (coefficients for Newton form)
    dd = table[0, :]
    return dd, table


dd, dd_table = divided_difference_table(t_newton, x_newton)

print(f"\nDivided Difference Table:")
print("-" * 70)
print(f"{'t_i':<8} {'x_i':<10} {'1st DD':<12} {'2nd DD':<12}")
print("-" * 70)
for i in range(len(t_newton)):
    row = f"{t_newton[i]:<8.2f} {x_newton[i]:<10.2f}"
    if i < len(t_newton) - 1:
        row += f" {dd_table[i, 1]:<12.6f}"
    else:
        row += " " * 12
    if i < len(t_newton) - 2:
        row += f" {dd_table[i, 2]:<12.6f}"
    print(row)

print(f"\nNewton Form Coefficients:")
print(f"  f[t₀] = {dd[0]:.6f}")
print(f"  f[t₀,t₁] = {dd[1]:.6f}")
print(f"  f[t₀,t₁,t₂] = {dd[2]:.6f}")


# Newton polynomial evaluation
def newton_polynomial(t_eval, t_nodes, dd):
    """
    Evaluate Newton interpolating polynomial at given points.

    Parameters:
    -----------
    t_eval : array-like
        Points to evaluate
    t_nodes : array-like
        Interpolation nodes
    dd : array-like
        Divided differences (Newton coefficients)

    Returns:
    --------
    x_eval : array
        Interpolated values
    """
    n = len(dd)
    x_eval = np.zeros(len(t_eval))

    for i, t_val in enumerate(t_eval):
        # Horner's method for Newton form
        result = dd[n - 1]
        for j in range(n - 2, -1, -1):
            result = dd[j] + result * (t_val - t_nodes[j])
        x_eval[i] = result

    return x_eval


# Evaluate P₂(0.75)
t_eval_point = np.array([0.75])
x_eval_point = newton_polynomial(t_eval_point, t_newton, dd)

print(f"\nEvaluation:")
print(f"  P₂(0.75) = {x_eval_point[0]:.6f} cm")

# Verify with manual calculation
manual_calc = 12 * (0.75) ** 2 + 10 * (0.75)
print(f"  Manual calculation: 12(0.75)² + 10(0.75) = {manual_calc:.6f} cm")

# =============================================================================
# Part (b): Cubic Spline with Clamped Boundary Conditions
# =============================================================================

print("\n" + "=" * 70)
print("Part (b): Cubic Spline with Clamped Boundary Conditions")
print("=" * 70)

# Create cubic spline with clamped boundary conditions
# bc_type = ((1, 0.0), (1, 0.0)) means:
#   - First derivative at start = 0.0
#   - First derivative at end = 0.0
spline = CubicSpline(t, x, bc_type=bc_type)

print(f"\nSpline created successfully!")
print(f"  Number of data points: {len(t)}")
print(f"  Time range: [{t.min()}, {t.max()}] seconds")
print(f"  Position range: [{x.min()}, {x.max()}] cm")
print(f"  Boundary conditions: x'(0) = 0, x'(2.5) = 0 cm/s")

# Create fine grid for smooth plotting
t_fine = np.linspace(t.min(), t.max(), 500)

# Evaluate position, velocity, and acceleration
x_fine = spline(t_fine)  # Position x(t)
v_fine = spline(t_fine, 1)  # Velocity x'(t)
a_fine = spline(t_fine, 2)  # Acceleration x''(t)

# Find maximum acceleration
max_acceleration = np.max(np.abs(a_fine))
print(f"\nMaximum acceleration: {max_acceleration:.4f} cm/s²")

# Check feasibility
max_allowed_acceleration = 80.0  # cm/s²
is_feasible = max_acceleration <= max_allowed_acceleration

print(f"Motor's maximum allowed acceleration: {max_allowed_acceleration} cm/s²")
print(f"Trajectory feasible: {'YES ✓' if is_feasible else 'NO ✗'}")

# =============================================================================
# Create Figure with Three Subplots
# =============================================================================

print("\n" + "=" * 70)
print("Generating Figures...")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Subplot 1: Position
axes[0].plot(t_fine, x_fine, 'b-', linewidth=2, label='Cubic Spline')
axes[0].plot(t, x, 'ro', markersize=8, label='Data Points')
axes[0].set_ylabel('Position x (cm)', fontsize=11)
axes[0].set_title('Robotic Arm Trajectory - Position', fontsize=12, fontweight='bold')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(t.min(), t.max())

# Subplot 2: Velocity
axes[1].plot(t_fine, v_fine, 'g-', linewidth=2, label="Velocity x'(t)")
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1].set_ylabel('Velocity (cm/s)', fontsize=11)
axes[1].set_title('Robotic Arm Trajectory - Velocity', fontsize=12, fontweight='bold')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(t.min(), t.max())

# Mark boundary conditions
axes[1].plot([t[0], t[-1]], [0, 0], 'm^', markersize=10, label="x'(0)=x'(2.5)=0")
axes[1].legend(loc='best')

# Subplot 3: Acceleration
axes[2].plot(t_fine, a_fine, 'r-', linewidth=2, label="Acceleration x''(t)")
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[2].axhline(y=max_allowed_acceleration, color='orange', linestyle='-.',
                linewidth=2, label=f'Max Allowed ({max_allowed_acceleration} cm/s²)')
axes[2].axhline(y=-max_allowed_acceleration, color='orange', linestyle='-.', linewidth=2)
axes[2].set_ylabel('Acceleration (cm/s²)', fontsize=11)
axes[2].set_xlabel('Time t (s)', fontsize=11)
axes[2].set_title('Robotic Arm Trajectory - Acceleration', fontsize=12, fontweight='bold')
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(t.min(), t.max())

# Add feasibility annotation
if is_feasible:
    fig.suptitle(f'Trajectory FEASIBLE (max |a| = {max_acceleration:.2f} ≤ {max_allowed_acceleration} cm/s²)',
                 fontsize=14, fontweight='bold', color='green')
else:
    fig.suptitle(f'Trajectory NOT FEASIBLE (max |a| = {max_acceleration:.2f} > {max_allowed_acceleration} cm/s²)',
                 fontsize=14, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig('problem3_trajectory.png', dpi=300, bbox_inches='tight')
print(f"\nFigure saved as 'problem3_trajectory.png'")

# Additional analysis
print("\n" + "=" * 70)
print("Additional Analysis")
print("=" * 70)

# Find where maximum acceleration occurs
max_accel_idx = np.argmax(np.abs(a_fine))
t_max_accel = t_fine[max_accel_idx]
a_max_accel = a_fine[max_accel_idx]

print(f"\nMaximum acceleration occurs at t = {t_max_accel:.4f} s")
print(f"Maximum acceleration value = {a_max_accel:.4f} cm/s²")

# Velocity analysis
max_velocity = np.max(np.abs(v_fine))
t_max_vel = t_fine[np.argmax(np.abs(v_fine))]
print(f"\nMaximum velocity = {max_velocity:.4f} cm/s at t = {t_max_vel:.4f} s")

# Position at midpoint
x_midpoint = spline(1.25)
print(f"\nPosition at t = 1.25 s (midpoint): x = {x_midpoint:.4f} cm")

print("\n" + "=" * 70)
print("END OF ANALYSIS")
print("=" * 70)