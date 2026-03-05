import numpy as np

def lu_decomposition(A):
    """
    Performs LU decomposition using Gaussian elimination (Doolittle's method).
    Returns L (lower triangular) and U (upper triangular).
    """
    n = len(A)
    L = np.eye(n)  # Initialize L as identity matrix
    U = A.astype(float).copy() # Copy A to U to perform operations

    for k in range(n - 1):
        for i in range(k + 1, n):
            if U[k, k] == 0:
                raise ValueError("Zero pivot encountered. Partial pivoting required.")

            # Calculate multiplier
            factor = U[i, k] / U[k, k]
            L[i, k] = factor

            # Update row i of U
            U[i, k:] = U[i, k:] - factor * U[k, k:]

    return L, U

def forward_substitution(L, b):
    """
    Solves Ly = b for y.
    """
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def back_substitution(U, y):
    """
    Solves Ux = y for x.
    """
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def lu_solve(A, b):
    """
    Solves Ax = b using LU decomposition.
    """
    L, U = lu_decomposition(A)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x, L, U

# --- Main Execution ---
if __name__ == "__main__":
    # Define Matrix A and Vector b
    A = np.array([
        [3, 1, 2],
        [1, 4, 1],
        [2, 1, 5]
    ], dtype=float)

    b = np.array([50, 40, 60], dtype=float)

    # Solve the system
    x, L, U = lu_solve(A, b)

    # Print Results
    print("Matrix L:")
    print(L)
    print("\nMatrix U:")
    print(U)
    print("\nSolution x:")
    print(x)

    # Verification
    print("\n--- Verification ---")
    print("LU Product (should match A):")
    print(np.dot(L, U))

    # Calculate Residual Norm ||b - Ax||_2
    residual = b - np.dot(A, x)
    residual_norm = np.linalg.norm(residual, 2)

    print(f"\nResidual vector (b - Ax): {residual}")
    print(f"Residual Norm ||b - Ax||_2: {residual_norm}")