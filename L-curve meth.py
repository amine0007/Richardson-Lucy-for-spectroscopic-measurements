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

lb, b, lM, M = fix_inputs(psf_data[0, :], psf_data[1, :], orig_data[0, :], orig_data[1, :])

# b = b/np.sum(b)

# signal mesuré optimal
mes_sim = np.convolve(b, M, mode='same')/np.sum(b)

def rlMeth(signal: np.ndarray, psf: np.ndarray, num_iter: int = 1500, autostop: bool = True):
    flipped_psf = psf[::-1]
    
    SH = np.zeros((num_iter + 1, signal.size))
    # SH[0] = signal
    SH[0] = np.ones_like(signal)
    
    residual = np.zeros((num_iter + 1, signal.size))
    residual[0] = signal - np.convolve(psf, SH[0], mode='same')/np.sum(psf)
    
    for r in range(1, num_iter + 1):
        temp1 = np.convolve(SH[r - 1], psf, mode='same')/np.sum(psf)
        temp1 = np.divide(signal, temp1, out=np.zeros_like(signal), where=temp1 != 0)
        
        # Check for zero division errors
        count_bad = np.count_nonzero(temp1 == 0)
        if count_bad > 0:
            print(f'After convolution in RL iteration {r}, {count_bad} estimated values were set to zero.')
        temp1[np.logical_or(np.isnan(temp1), np.isinf(temp1))] = 0

        temp2 = np.convolve(temp1, flipped_psf, mode='same')/np.sum(flipped_psf)
        tempSH = SH[r - 1] * temp2
        tempSH[np.logical_or(np.isnan(tempSH), np.isinf(tempSH))] = 0

        SH[r] = tempSH
        
        residual[r] = signal - np.convolve(psf, SH[r], mode='same')/np.sum(psf)
        """
        # Critère d'arrêt 
        if autostop and r > 1:
            delta = np.linalg.norm(SH[r] - SH[r - 1]) / np.linalg.norm(SH[r - 1])
            if delta < 1e-6:
                print(f'Convergence atteinte à l\'itération {r}')
                SH = SH[:r + 1]
                break
        """
        
    return SH, residual

def compute_L_curve(SH, residual):
    # num_iter = SH.shape[0] - 1

    solution_norms = np.linalg.norm(SH, ord=2, axis=1)
    residual_norms = np.linalg.norm(residual, ord=2, axis=1)
    
    """
    for k in range(num_iter + 1):
        solution_norms[k] = solution_norms.append(np.linalg.norm(SH[k], ord=2))
        residual_norms[k] = residual_norms.append(np.linalg.norm(residual[k], ord=2))
    
    # solution_norms = np.where(solution_norms > 0, solution_norms, 1e-10)
    # residual_norms = np.where(residual_norms > 0, residual_norms, 1e-10)
    """
    x_vals = np.log(solution_norms)
    y_vals = np.log(residual_norms)

    return solution_norms, residual_norms, x_vals, y_vals

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

# Richardson-Lucy
num_iterations = 2500
m = mes_sim  
SH, residual = rlMeth(signal=m, psf=b, num_iter=num_iterations, autostop=False)

# données L-curve
solution_norms, residual_norms, x_vals, y_vals  = compute_L_curve(SH, residual)


# courbure du L-curve
curvatures = compute_curvature(x_vals, y_vals)
"""
# itération optimal (maximum de courbure)
optimal_index = np.argmax(curvatures)
optimal_iteration = optimal_index + 1
print(f"Optimal iteration at: {optimal_iteration}")

# signal estimé optimal
optimal_spectrum = SH[optimal_iteration]

error_signal_2 = np.linalg.norm(optimal_spectrum - M)
relative_error_signal_2 = error_signal_2 / np.linalg.norm(M)
print(f"Error between optimal spectrum and original signal: {error_signal_2}")
print(f"Relative error: {relative_error_signal_2}")

# Plot L-curve et le point optimal
plt.figure()
plt.plot(x_vals, y_vals, label='L-curve')
plt.scatter(x_vals[optimal_iteration], y_vals[optimal_iteration], color='red', label='Optimal point')
plt.xlabel('log(||s_k||₂)')
plt.ylabel('log(||m - b * s_k||₂)')
plt.legend()
plt.title('L-curve with Optimal Point')
plt.show()
"""
# Plot courbure vs itération
iterations = np.arange(1, len(curvatures) + 1)
plt.figure()
plt.plot(iterations, curvatures)
plt.xlabel('Iteration')
plt.ylabel('Curvature')
plt.title('Curvature vs Iteration')
plt.show()
"""
# Plot du signal estimé à l'itération optimal
plt.figure()
plt.plot(lM, optimal_spectrum, label='Optimal Spectrum')
plt.xlabel('Wavelength')
plt.ylabel('Amplitude')
plt.title('Estimated Original Spectrum at Optimal Iteration')
plt.legend()
plt.show()
"""
# Calcul de la fonction objectif F(k, lambda) 

# les valeurs de lambda prises
lambdas = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]) 


# max_residual_norm = np.max(residual_norms)
# max_solution_norm = np.max(solution_norms)

# calcul F(k, lambda)
F_values = np.zeros((len(lambdas), num_iterations + 1))

for i, lam in enumerate(lambdas):
    F_values[i, :] = residual_norms + lam * solution_norms

# indices des F(k, lambda) minimal 
min_index = np.unravel_index(np.argmin(F_values, axis=None), F_values.shape)
optimal_lambda = lambdas[min_index[0]]
optimal_k = min_index[1]

print(f"Optimal lambda: {optimal_lambda}")
print(f"Optimal iteration (k): {optimal_k}")

# signal estimé optimal 
optimal_spectrum_lambda = SH[optimal_k]
"""
# error / erreur relative entre le signal optimal et le signal original 
error_signal = np.linalg.norm(optimal_spectrum_lambda - M)
relative_error_signal = error_signal / np.linalg.norm(M)
print(f"Error between optimal spectrum (lambda) and original signal: {error_signal}")
print(f"Relative error: {relative_error_signal}")
"""
# Plot du heatmap de F(k, lambda)
plt.figure(figsize=(10, 6))
plt.imshow(F_values, aspect='auto', origin='lower',
           extent=[0, num_iterations, lambdas[0], lambdas[-1]],
           cmap='viridis')
plt.colorbar(label='F(k, lambda)')
plt.xlabel('Iteration k')
plt.ylabel('Lambda') 
plt.title('Objective Function F(k, lambda)')
plt.scatter(optimal_k, optimal_lambda, color='red', label='Optimal (k, lambda)')
plt.legend()
plt.yscale('log')
plt.show()

# Plot du signal optimal
plt.figure(figsize=(10, 4))
plt.plot(lM, optimal_spectrum_lambda, label='Optimal Estimated Signal (lambda)')
plt.plot(lM, M, label='Original Signal', alpha=0.5)
plt.legend()
plt.title('Optimal Estimated Signal vs Original Signal')
plt.xlabel('Wavelength')
plt.ylabel('Amplitude')
plt.show()

# Plot des solutions / résidus vs itérations
plt.figure()
plt.plot(solution_norms, label='Norme de la solution ||s_k||₂')
plt.plot(residual_norms, label='Norme du résidu ||m - b * s_k||₂')
plt.xlabel('Itération k')
plt.ylabel('Norme L2')
plt.legend()
plt.title('Évolution des normes en fonction des itérations')
plt.show()

# Plot solution_norms vs residual_norms
plt.plot(residual_norms, solution_norms, 'o-')
plt.xlabel('residual_norms')
plt.ylabel('solution_norms')
plt.title('residual_norms vs solution_norms')
plt.show()

"""
lambda_values = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])

optimal_iterations = []
solution_norms_lambda = []
residual_norms_lambda = []

for lam in lambda_values:
    # Calculer F(k, lambda) pour toutes les itérations k
    F_lambda = (residual_norms / max_residual_norm) + lam * (solution_norms / max_solution_norm)
    
    # Trouver l'itération k qui minimise F(k, lambda) pour ce lambda
    k_opt = np.argmin(F_lambda)
    optimal_iterations.append(k_opt)
    
    # Récupérer les normes pour k_opt
    solution_norm = solution_norms[k_opt]
    residual_norm = residual_norms[k_opt]
    
    solution_norms_lambda.append(solution_norm)
    residual_norms_lambda.append(residual_norm)
    
    print(f"Lambda: {lam}, Optimal iteration: {k_opt}, Solution norm: {solution_norm}, Residual norm: {residual_norm}")


# Convertir les listes en tableaux numpy pour faciliter le traçage
solution_norms_lambda = np.array(solution_norms_lambda)
residual_norms_lambda = np.array(residual_norms_lambda)

plt.figure(figsize=(8, 6))
plt.plot(residual_norms_lambda, solution_norms_lambda, 'o-', label='Points (λ, k_opt)')
plt.xlabel('Norme L2 du résidu ||m - b * s_k||₂')
plt.ylabel('Norme L2 de la solution ||s_k||₂')
plt.title('Norme de la solution vs Norme du résidu pour différentes valeurs de λ')
plt.grid(True)
plt.legend()
plt.show()


error_signals = np.linalg.norm(SH - M, axis=1)
min_error = np.min(error_signals)
min_index = np.argmin(error_signals)
print(f"Erreur minimale : {min_error}")
print(f"Indice de l'erreur minimale : {min_index}")

"""
"""
def calculer_differences(SH):
    differences = []
    for k in range(len(SH) - 1):
        difference = SH[k+1] - SH[k]
        differences.append(difference)
    return differences
"""
"""
R = np.zeros((num_iterations + 1, SH.shape[1]))
cr = np.zeros(1499)

for i in range (num_iterations + 1):
    R[i] = m - SH[i]
    
for j in range(1, num_iterations - 1):
    cr[j] = (np.linalg.norm(R[j]) - np.linalg.norm(R[j+1])) / np.linalg.norm(R[j])

"""
"""
differences = []
for k in range(len(SH) - 1):
    difference = np.linalg.norm(SH[k+1] - SH[k])
    differences.append(difference)
"""


