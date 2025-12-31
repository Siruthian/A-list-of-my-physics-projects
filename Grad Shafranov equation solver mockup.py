import numpy as np

# Setup
MaRadius = 1        # R0
MiRadius = 1
PhiFlux = 1         # ψ_axis
ToMag = 1
MagConst = 1        # μ₀
Paxi = 1            # p_axis
Baxi = 1            # b_axis
B0 = 1              # B₀

dr = 1
dz = 1

def TriGen(r, z, max_iter=10000, tol=1e-6):
    nr = r.size + 2
    nz = z.size + 2

    # Full grid with ghost/boundary points
    psi = np.zeros((nz, nr))

    # Precompute constant RHS factor
    rhs_coeff = (2 * MagConst * (r[:, None] ** 2) * Paxi + Baxi * MaRadius**2 * B0**2) / (PhiFlux**2)
    rhs_coeff = np.pad(rhs_coeff.T, pad_width=1, mode='constant')

    # Iterative Gauss-Seidel solver
    for iteration in range(max_iter):
        psi_old = psi.copy()
        for i in range(1, nz-1):
            for j in range(1, nr-1):
                r_val = j * dr
                term_r = (psi[i, j+1] - 2 * psi[i, j] + psi[i, j-1]) / dr**2
                term_r_cross = (psi[i, j+1] - psi[i, j-1]) / (2 * dr * r_val)
                term_z = (psi[i+1, j] - 2 * psi[i, j] + psi[i-1, j]) / dz**2
                rhs = -rhs_coeff[i, j] * psi[i, j]

                psi[i, j] = (term_r + term_z + term_r_cross - rhs) / (-2/dr**2 - 2/dz**2)

        # Convergence check
        if np.linalg.norm(psi - psi_old) < tol:
            print(f"Converged in {iteration} iterations")
            break

    return psi

# Test the function
r_vals = np.array([1, 2, 3])
z_vals = np.array([1, 2, 3])
solution = TriGen(r_vals, z_vals)
print(solution)