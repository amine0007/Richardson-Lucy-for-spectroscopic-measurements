"""
Richardson-Lucy avec régularisation intégrée et critère de convergence 

||SH[r] - SH[r-1]|| / ||SH[r-1]|| < ϵ (1e-6)

"""

import numpy as np
import matplotlib.pyplot as plt
import glob

from rldeconvolution.fix_inputs import fix_inputs

# Define paths to kernel and measurement data
psf_path = './data/kernel'
psf_files = sorted(glob.glob(psf_path + "/*.csv"))
orig_path = './data/original'
orig_files = glob.glob(orig_path + "/*.csv")

# Choose which file to use
file_index = 0

# Load kernel and measurement data
psf_data = np.loadtxt(psf_files[file_index], delimiter=';', skiprows=63)
orig_data = np.loadtxt(orig_files[file_index], delimiter=';', skiprows=63)

# Prepare inputs using the fix_inputs function
lb, b, lM, M = fix_inputs(
    psf_data[0, :], psf_data[1, :],
    orig_data[0, :], orig_data[1, :]
)

# Normalize the PSF
b = b / np.sum(b)

# Simulated measured signal
mes_sim = np.convolve(b, M, mode='same')

def rl_regularized(signal, psf, lambda_reg, max_iter=1000, epsilon=1e-6):
    """
    Regularized Richardson-Lucy deconvolution algorithm with convergence criterion.
    
    Parameters:
    - signal: Observed blurred signal (numpy array).
    - psf: Point Spread Function (numpy array).
    - lambda_reg: Regularization parameter (float).
    - max_iter: Maximum number of iterations (int).
    - epsilon: Convergence threshold (float).
    
    Returns:
    - s_k: Reconstructed signal after convergence (numpy array).
    - num_iter: Number of iterations performed (int).
    """
    # Initialize variables
    s_k = np.ones_like(signal)  # Initial estimate (uniform positive)
    psf_flip = psf[::-1]        # Flipped PSF for convolution
    
    for k in range(max_iter):
        # Convolve current estimate with PSF
        estimate = np.convolve(s_k, psf, mode='same')
        estimate = np.maximum(estimate, 1e-12)  # Avoid division by zero
        
        # Compute ratio of observed to estimated signal
        ratio = signal / estimate
        
        # Convolve ratio with flipped PSF
        correction = np.convolve(ratio, psf_flip, mode='same')
        
        # Update estimate with regularization
        s_k_new = s_k * correction
        s_k_new = s_k_new / (1 + lambda_reg * s_k)
        
        # Enforce non-negativity
        s_k_new = np.maximum(s_k_new, 0)
        
        # Check convergence
        relative_change = np.linalg.norm(s_k_new - s_k) / np.linalg.norm(s_k)
        if relative_change < epsilon:
            print(f'Converged at iteration {k+1} for lambda={lambda_reg}')
            break
        
        # Update estimates for next iteration
        s_k = s_k_new.copy()
    
    else:
        print(f'Max iterations reached for lambda={lambda_reg}')
    
    return s_k, k+1  # Return the solution and number of iterations

# Define regularization parameters to test
lambdas = np.array([1e-9, 1e-8, 1e-7, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])

# Lists to store norms
solution_norms = []
residual_norms = []

# Run the algorithm for each lambda
for lambda_reg in lambdas:
    s_k, num_iter = rl_regularized(
        signal=mes_sim,
        psf=b,
        lambda_reg=lambda_reg,
        max_iter=1000,
        epsilon=1e-6
    )
    
    # Compute residual norm ||m - b * s_k||
    estimate = np.convolve(s_k, b, mode='same')
    residual = mes_sim - estimate
    residual_norm = np.linalg.norm(residual)
    
    # Compute solution norm ||s_k||
    solution_norm = np.linalg.norm(s_k)
    
    # Append norms to lists
    residual_norms.append(residual_norm)
    solution_norms.append(solution_norm)

# Plot the L-curve
plt.figure(figsize=(8, 6))
plt.loglog(residual_norms, solution_norms, 'o-')
plt.xlabel('Residual Norm ||m - b * s_k||')
plt.ylabel('Solution Norm ||s_k||')
plt.title('L-curve')
for i, lambda_reg in enumerate(lambdas):
    plt.annotate(f'λ={lambda_reg}', (residual_norms[i], solution_norms[i]))
plt.grid(True, which='both', ls='--')
plt.show()


def compute_curvature(x_vals, y_vals):
    iters = np.arange(1, len(x_vals) + 1)
    log_iters = np.log(iters)
    
    dlog_iters = log_iters[1] - log_iters[0]
    
    dx = np.gradient(x_vals, dlog_iters)
    dy = np.gradient(y_vals, dlog_iters)
    
    d2x = np.gradient(dx, dlog_iters)
    d2y = np.gradient(dy, dlog_iters)
    
    denominator = np.abs((dx**2 + dy**2)**1.5)
    numerator = dx * d2y - dy * d2x
    #print(numerator.shape)
    #print(denominator.shape)
    curvature = np.where(denominator != 0, numerator / denominator, 0)
    
    return curvature

curvatures = compute_curvature(np.log(solution_norms), np.log(residual_norms))

# Plot courbure vs itération
iterations = np.arange(1, len(curvatures) + 1)
plt.figure()
plt.plot(iterations, curvatures)
plt.xlabel('lambda')
plt.ylabel('Curvature')
plt.title('Curvature vs Iteration')
plt.show()