import numpy as np
import matplotlib.pyplot as plt
import glob

from rldeconvolution.fix_inputs import fix_inputs
from rldeconvolution.method import rlMethod


# Define paths to kernel and measurement data
kernel_path = './data/kernel'
kernel_files = sorted(glob.glob(kernel_path + "/*.csv"))
measure_path = './data/measure'
measure_files = sorted(glob.glob(measure_path + "/*.csv"))

# Choose which file to use
file_index = 4

# Load kernel and measurement data
kernel_data = np.loadtxt(kernel_files[file_index], delimiter=';', skiprows=63)
measurement_data = np.loadtxt(measure_files[file_index], delimiter=';', skiprows=63)

# Fix inputs to ensure they have the same shape
lb, b, lM, M = fix_inputs(kernel_data[0, :], kernel_data[1, :], measurement_data[0, :], measurement_data[1, :])

# Deconvolve the data
deconvolved, curvature_values, rms_values = rlMethod(M, b)

# Plot the original and deconvolved spectra
plt.figure(figsize=(20, 12))
plt.plot(lM, M, label="Measured {}".format(measure_files[file_index]))
plt.plot(lM, deconvolved, label='Estimated {}'.format(measure_files[file_index]))
plt.xlabel("Wavelength / nm")
plt.ylabel("Spectrum amplitude / au")
plt.legend()
plt.grid()
plt.title('Comparison')

# Plot the curvature and RMS values
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(curvature_values, label="Curvature")
plt.grid(); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(rms_values, label="RMS")
plt.grid(); plt.legend()
