import numpy as np
import matplotlib.pyplot as plt
import glob

#from scipy.linalg import toeplitz
#from scipy.linalg import solve

from scipy.sparse.linalg import LinearOperator, cg

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

# signal mesuré optimal
mes_sim = np.convolve(b, M, mode='same')/np.sum(b)


def center_psf(psf, N):
    K = len(psf)
    centered_psf = np.zeros(N)
    center = (K - 1) // 2
    centered_psf[:K] = psf
    centered_psf = np.roll(centered_psf, -center)
    return centered_psf


# Normalize and center the PSF
b = b / np.sum(b)
N = len(mes_sim)
b_centered = center_psf(b, N)

# Define the convolution operator using FFT for efficiency
def convolution_operator(psf, N):
    K = len(psf)
    L = N + K - 1  # Length for linear convolution
    psf_padded = np.zeros(L)
    psf_padded[:K] = psf
    psf_fft = np.fft.fft(psf_padded)
    
    def matvec(x):
        x_padded = np.zeros(L)
        x_padded[:N] = x
        y = np.fft.ifft(np.fft.fft(x_padded) * psf_fft)
        return np.real(y[:N])  # Truncate to original length
    
    def rmatvec(x):
        x_padded = np.zeros(L)
        x_padded[:N] = x
        y = np.fft.ifft(np.fft.fft(x_padded) * np.conj(psf_fft))
        return np.real(y[:N])  # Truncate to original length
    
    return LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec)


# Create the convolution operator with linear convolution
B_op = convolution_operator(b_centered, N)

# Regularization parameter
lambda_reg = 0.7906043210907702  

# Define the linear operator for the regularized system
def A_matvec(x):
    return B_op.rmatvec(B_op.matvec(x)) + lambda_reg * x

A = LinearOperator((N, N), matvec=A_matvec)

# Right-hand side
b_rhs = B_op.rmatvec(mes_sim)

# Solve the system using Conjugate Gradient
s_cg, info = cg(A, b_rhs, tol=1e-6)

if info == 0:
    print("Conjugate Gradient converged.")
else:
    print(f"Conjugate Gradient did not converge. Info: {info}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(s_cg, label='Deconvolved Signal')
plt.plot(M, label='Original Signal', alpha=0.7)
plt.legend()
plt.title('Deconvolved Signal vs Original Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()


"""
L-curve


lambdas = np.logspace(-5, 1, 50)

residual_norms = []
solution_norms = []

for lambda_reg in lambdas:
    # Define the regularized operator
    def A_matvec(x):
        return B_op.rmatvec(B_op.matvec(x)) + lambda_reg * x
    A = LinearOperator((N, N), matvec=A_matvec)

    # Right-hand side
    b_rhs = B_op.rmatvec(mes_sim)

    # Solve the system
    s_cg, info = cg(A, b_rhs, tol=1e-6, maxiter=1000)

    if info != 0:
        print(f"CG did not converge for lambda={lambda_reg}")
        continue

    # Compute norms
    residual = mes_sim - B_op.matvec(s_cg)
    residual_norms.append(np.linalg.norm(residual))
    solution_norms.append(np.linalg.norm(s_cg))



plt.figure(figsize=(8, 6))
plt.loglog(residual_norms, solution_norms, 'o-')
plt.xlabel('Residual Norm ||Bs - m||')
plt.ylabel('Solution Norm ||s||')
plt.title('L-curve for Regularization Parameter Selection')
plt.grid(True, which='both', ls='--')

# Annotate points with lambda values
for i, lambda_reg in enumerate(lambdas):
    plt.annotate(f'{lambda_reg:.1e}', (residual_norms[i], solution_norms[i]))

plt.show()


log_residual_norms = np.log(residual_norms)
log_solution_norms = np.log(solution_norms)

# Compute first derivatives
d_log_residual = np.gradient(log_residual_norms)
d_log_solution = np.gradient(log_solution_norms)

# Compute second derivatives
d2_log_residual = np.gradient(d_log_residual)
d2_log_solution = np.gradient(d_log_solution)

# Compute curvature
curvature = (d_log_solution * d2_log_residual - d_log_residual * d2_log_solution) / \
            (d_log_residual ** 2 + d_log_solution ** 2) ** (3 / 2)

# Find the index of maximum curvature
optimal_index = np.argmax(np.abs(curvature))
lambda_opt = lambdas[optimal_index]

print(f'Optimal lambda based on maximum curvature: {lambda_opt}')

# Plot curvature vs. lambda
plt.figure(figsize=(8, 6))
plt.semilogx(lambdas, curvature)
plt.xlabel('Lambda')
plt.ylabel('Curvature')
plt.title('Curvature vs Lambda')
plt.grid(True, which='both', ls='--')
plt.show()


# Solve with optimal lambda
def A_matvec_opt(x):
    return B_op.rmatvec(B_op.matvec(x)) + lambda_opt * x
A_opt = LinearOperator((N, N), matvec=A_matvec_opt)
b_rhs_opt = B_op.rmatvec(mes_sim)
s_opt, info = cg(A_opt, b_rhs_opt, tol=1e-6, maxiter=1000)

if info != 0:
    print(f"CG did not converge for optimal lambda={lambda_opt}")

# Plot the deconvolved signal
plt.figure(figsize=(10, 6))
plt.plot(s_opt, label='Deconvolved Signal')
plt.plot(M, label='Original Signal', alpha=0.7)
plt.legend()
plt.title(f'Deconvolved Signal with Optimal Lambda = {lambda_opt:.1e}')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

"""