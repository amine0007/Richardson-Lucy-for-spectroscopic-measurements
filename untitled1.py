import numpy as np
import matplotlib.pyplot as plt
import glob
#from scipy.interpolate import InterpolatedUnivariateSpline
#from rldeconvolution.method import rlMethod


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
orig_data = np.loadtxt(orig_files[file_index], delimiter=';', skiprows=61)



lb, b, lM, M = fix_inputs(psf_data[0, :], psf_data[1, :], orig_data[0, :], orig_data[1, :])

#b /= np.sum(b)
#M /= np.max(M)

# signal mesuré simulé
mes_sim = np.convolve(b, M, mode='same')

#mes_sim /= np.max(mes_sim)

def rlMeth(signal: np.ndarray, psf: np.ndarray, num_iter: int =1500, autostop: bool = True):
    
    flipped_psf = psf[::-1]
    
    SH = np.zeros((num_iter + 1, signal.size))
    SH[0] = signal
    
    residual = np.zeros((num_iter + 1, signal.size))
    residual[0] = signal - np.convolve(psf, SH[0], mode='same')
    
    for r in range(1, num_iter + 1):
        temp1 = np.convolve(SH[r - 1], psf, mode='same')
        temp1 = np.divide(signal, temp1, out=np.zeros_like(signal), where=temp1 != 0)
        
        # Check for zero division errors
        count_bad = np.count_nonzero(temp1 == 0)
        if count_bad > 0:
            print(f'After convolution in RL iteration {r}, {count_bad} estimated values were set to zero.')
        temp1[np.logical_or(np.isnan(temp1), np.isinf(temp1))] = 0

        temp2 = np.convolve(temp1, flipped_psf, mode='same')
        tempSH = SH[r - 1] * temp2
        tempSH[np.logical_or(np.isnan(tempSH), np.isinf(tempSH))] = 0

        SH[r] = tempSH
        
        residual[r] = signal - np.convolve(psf, SH[r], mode='same')
        
    return SH, residual

def compute_L_curve(SH, residual):
    num_iter = SH.shape[0] - 1

    solution_norms = np.zeros(num_iter + 1)
    residual_norms = np.zeros(num_iter + 1)

    for k in range(num_iter + 1):
        solution_norms[k] = np.linalg.norm(SH[k])
        residual_norms[k] = np.linalg.norm(residual[k])
    
    solution_norms = np.where(solution_norms > 0, solution_norms, 1e-10)
    residual_norms = np.where(residual_norms > 0, residual_norms, 1e-10)
    
    x_vals = np.log(solution_norms)
    y_vals = np.log(residual_norms)
    

    return x_vals, y_vals

def compute_curvature(x_vals, y_vals):
    iters = np.arange(1, len(x_vals) + 1)
    #curvatures = np.zeros(n - 2)
    log_iters = np.log(iters)
    
    dlog_iters = log_iters[1] - log_iters[0]
    
    dx = np.gradient(x_vals, dlog_iters)
    dy = np.gradient(y_vals, dlog_iters)
    
    d2x = np.gradient(dx, dlog_iters)
    d2y = np.gradient(dy, dlog_iters)
    
    denominator = np.abs((dx**2 + dy**2)**1.5)
    numerator = dx * d2y - dy * d2x
    print(numerator.shape)
    print(denominator.shape)
    curvature = np.where(denominator != 0, numerator / denominator, 0)
    
    return curvature

# Richardson-Lucy algorithm
num_iterations = 1500
SH, residual = rlMeth(signal=mes_sim, psf=b, num_iter=num_iterations, autostop=False)

# Calcul L-curve data
x_vals, y_vals = compute_L_curve(SH, residual)

# calcul de la courbure du L-curve
curvatures = compute_curvature(x_vals, y_vals)

# iteration optimal (maximum curvature)
optimal_index = np.argmax(curvatures)
optimal_iteration = optimal_index + 1
print(f"Optimal iteration at: {optimal_iteration}")

# signal estimé optimal
optimal_spectrum = SH[optimal_iteration]

error_signal_2 = np.linalg.norm(optimal_spectrum - M)

relative_error_signal_2 = np.linalg.norm(optimal_spectrum - M) / np.linalg.norm(M)


"""
num_estimations = SH.shape[0]  # Le nombre d'estimations dans SH
E = np.zeros(num_estimations)  # Tableau pour stocker les erreurs

# Calculer les erreurs pour chaque estimation dans SH
for i in range(num_estimations):
    # Calcul de la norme L2 de l'erreur entre le signal original et l'estimation actuelle
    E[i] = np.linalg.norm(SH[i] - M)

# Trouver l'indice de l'erreur minimale
min_idx = np.argmin(E)
"""

# Plot du L-curve point optimal
plt.figure()
plt.plot(x_vals, y_vals, label='L-curve')
plt.scatter(x_vals[optimal_iteration], y_vals[optimal_iteration], color='red', label='Optimal point')
plt.xlabel('log(||s_k||₂)')
plt.ylabel('log(||m - b * s_k||₂)')
plt.legend()
plt.title('L-curve with Optimal Point')
plt.show()

# Plot curvature vs iteration
iterations = np.arange(1, len(curvatures) + 1)
plt.figure()
plt.plot(iterations, curvatures)
plt.xlabel('Iteration')
plt.ylabel('Curvature')
plt.title('Curvature vs Iteration')
plt.show()

# plot du signal estimé
plt.figure()
plt.plot(lM, optimal_spectrum, label='Optimal Spectrum')
plt.xlabel('Wavelength')
plt.ylabel('Amplitude')
plt.title('Estimated Original Spectrum at Optimal Iteration')
plt.legend()
plt.show()


"""
S, res = rlMeth(M, b)

S_norms = np.linalg.norm(S, axis=1)

res_norms = np.linalg.norm(res, axis=1)

plt.scatter(res_norms, S_norms, color='purple', label='Normes de S vs. Normes de res')        
        

# Plotting the L-curve (Norms of residual vs Norms of solution)
plt.scatter(res_norms, S_norms, color='purple', label='Normes de S vs. Normes de res')        
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Norme du résidu (log)')
plt.ylabel('Norme de la solution (log)')
plt.title('L-curve: Normes du résidu vs. solution')
plt.show()

# Assuming residual_norms and solution_norms are arrays with the L-curve points
residual_norms = np.log10(res_norms)  # Log of residual norms (y = log residuals)
solution_norms = np.log10(S_norms)    # Log of solution norms (x = log solution norms)

# Interpolating the data using a spline
spline_residuals = InterpolatedUnivariateSpline(solution_norms, residual_norms)

# Generate a fine grid for calculating the curvature
q_solution_norms = np.linspace(solution_norms.min(), solution_norms.max(), 1000)
spline_residual_norms = spline_residuals(q_solution_norms)

# First and second derivatives
dq = q_solution_norms[1] - q_solution_norms[0]
first_derivative = np.gradient(spline_residual_norms, dq)
second_derivative = np.gradient(first_derivative, dq)

# Compute curvature
curvature = np.abs(second_derivative) / (1 + first_derivative**2)**1.5

# Find the point of maximum curvature (optimal trade-off)
max_curvature_idx = np.argmax(curvature)

# Retrieve the optimal solution and residual norms
optimal_solution_norm = 10**q_solution_norms[max_curvature_idx]
optimal_residual_norm = 10**spline_residual_norms[max_curvature_idx]

# Plotting the L-curve and the optimal point
plt.plot(10**solution_norms, 10**residual_norms, 'o-', label='L-curve')
plt.plot(optimal_solution_norm, optimal_residual_norm, 'ro', label='Optimal point')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Solution Norm (log)')
plt.ylabel('Residual Norm (log)')
plt.title('L-curve with Optimal Point (Maximum Curvature)')
plt.legend()
plt.show()

print(f"Optimal solution norm: {optimal_solution_norm}")
print(f"Optimal residual norm: {optimal_residual_norm}")
        


# Define paths to kernel and measurement data
psf_path = './data/kernel'
psf_files = sorted(glob.glob(psf_path + "/*.csv"))
orig_path = './data/original'
orig_files = glob.glob(orig_path + "/*.csv")

# Choose which file to use
file_index = 0

# Load kernel and measurement data
psf_data = np.loadtxt(psf_files[file_index], delimiter=';', skiprows=63)
orig_data = np.loadtxt(orig_files[file_index], delimiter=';', skiprows=61)


lpsf, psf, lorig, orig = fix_inputs(psf_data[0, :], psf_data[1, :], orig_data[0, :], orig_data[1, :])

# signal mesuré simulé
mes_sim = np.convolve(psf, orig, mode='same')


deconvolved, curvature_values, rms_values = rlMethod(mes_sim, psf)


sh, res = rlMeth(mes_sim, psf, 2000)

deconv = sh[10]
"""
        