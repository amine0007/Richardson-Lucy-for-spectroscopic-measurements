from rldeconvolution.fix_inputs import fix_inputs
from rldeconvolution.method import rlMethod

import numpy as np
import matplotlib.pyplot as plt
import glob


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

# plot du psf

plt.plot(lpsf, psf, label='psf')

# plot du signal original

plt.plot(lorig, orig, label='signal original')

# plot du signal mesuré simulé
plt.plot(lorig, mes_sim, label='signal mesuré simulé')

deconv, curvature, rms = rlMethod(mes_sim, psf)

error_signal_1 = np.linalg.norm(deconv - orig)

relative_error_signal_1 = np.linalg.norm(deconv - orig) / np.linalg.norm(orig)

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(curvature, label="Curvature")
plt.grid(); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(rms, label="RMS")
plt.grid(); plt.legend()