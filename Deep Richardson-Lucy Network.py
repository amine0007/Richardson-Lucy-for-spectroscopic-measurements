import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Data generation functions
def generate_true_spectrum(length=1000, num_peaks=5, noise_level=0.0):
    x = np.linspace(0, 100, length)
    spectrum = np.zeros_like(x)
    for _ in range(num_peaks):
        # Random peak parameters
        peak_center = np.random.uniform(10, 90)
        peak_width = np.random.uniform(1, 5)
        peak_height = np.random.uniform(0.5, 1.0)
        # Gaussian peak
        peak = peak_height * np.exp(-((x - peak_center) ** 2) / (2 * peak_width ** 2))
        spectrum += peak
    # Add baseline noise if needed
    spectrum += noise_level * np.random.normal(size=length)
    return spectrum

def generate_psf(length=51, sigma=2.0):
    x = np.arange(-(length // 2), length // 2 + 1)
    psf = np.exp(-x ** 2 / (2 * sigma ** 2))
    psf /= np.sum(psf)  # Normalize the PSF
    return psf

def blur_spectrum(spectrum, psf):
    blurred_spectrum = convolve(spectrum, psf, mode='same')
    return blurred_spectrum

def add_noise(spectrum, noise_level=0.01):
    noisy_spectrum = spectrum + noise_level * np.random.normal(size=spectrum.size)
    return noisy_spectrum

def generate_dataset(num_samples=100, spectrum_length=1000, psf_length=51, psf_sigma=2.0, noise_level=0.01):
    spectra = []
    blurred_spectra = []
    psf = generate_psf(length=psf_length, sigma=psf_sigma)
    for _ in range(num_samples):
        true_spectrum = generate_true_spectrum(length=spectrum_length)
        blurred_spectrum = blur_spectrum(true_spectrum, psf)
        noisy_blurred_spectrum = add_noise(blurred_spectrum, noise_level=noise_level)
        spectra.append(true_spectrum)
        blurred_spectra.append(noisy_blurred_spectrum)
    return np.array(spectra), np.array(blurred_spectra), psf

# Generate synthetic data
spectra, blurred_spectra, psf = generate_dataset(num_samples=1000)

# Normalize data
spectra_max = spectra.max(axis=1, keepdims=True)
spectra = spectra / spectra_max
blurred_spectra = blurred_spectra / spectra_max  # Use the same scaling factor

# Split into training and validation sets
train_spectra = spectra[:800]
train_blurred = blurred_spectra[:800]
val_spectra = spectra[800:]
val_blurred = blurred_spectra[800:]

# PyTorch implementation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SpectrumDataset(Dataset):
    def __init__(self, spectra, blurred_spectra):
        self.spectra = spectra
        self.blurred_spectra = blurred_spectra

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        true_spectrum = self.spectra[idx]
        blurred_spectrum = self.blurred_spectra[idx]
        return torch.from_numpy(blurred_spectrum).float(), torch.from_numpy(true_spectrum).float()

class DeepRichardsonLucy(nn.Module):
    def __init__(self, psf, num_layers=10):
        super(DeepRichardsonLucy, self).__init__()
        self.num_layers = num_layers
        self.psf = torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, PSF_length)
        self.psf_flip = torch.flip(self.psf, dims=[2])  # Flipped PSF
        padding = (self.psf.size(2) - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=self.psf.size(2), padding=padding, bias=False)
        self.conv.weight = nn.Parameter(self.psf)
        self.conv_flip = nn.Conv1d(1, 1, kernel_size=self.psf_flip.size(2), padding=padding, bias=False)
        self.conv_flip.weight = nn.Parameter(self.psf_flip)
        # Freeze PSF convolutions
        self.conv.weight.requires_grad = False
        self.conv_flip.weight.requires_grad = False
        # Learnable parameters
        self.alpha = nn.Parameter(torch.ones(num_layers))
        
    def forward(self, m):
        # m: Input blurred spectrum, shape (batch_size, spectrum_length)
        batch_size, spectrum_length = m.size()
        m = m.unsqueeze(1)  # Shape: (batch_size, 1, spectrum_length)
        s = torch.full_like(m, 0.5)  # Initial estimate
        for k in range(self.num_layers):
            s_conv = self.conv(s) + 1e-6  # Avoid division by zero
            ratio = m / s_conv
            correction = self.conv_flip(ratio) + 1e-6  # Avoid negative or zero values
            correction = torch.clamp(correction, min=1e-6)
            # Update with learnable parameter alpha[k]
            s = s * (correction ** self.alpha[k])
            s = torch.clamp(s, min=1e-6)  # Enforce non-negativity
        s = s.squeeze(1)  # Shape: (batch_size, spectrum_length)
        return s

# Create datasets and dataloaders
train_dataset = SpectrumDataset(train_spectra, train_blurred)
val_dataset = SpectrumDataset(val_spectra, val_blurred)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepRichardsonLucy(psf=psf, num_layers=10).to(device)

# Define loss function
criterion = nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Reduced learning rate

num_epochs = 50  # Increased number of epochs

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for blurred_spectrum, true_spectrum in train_loader:
        blurred_spectrum = blurred_spectrum.to(device)
        true_spectrum = true_spectrum.to(device)
        optimizer.zero_grad()
        output = model(blurred_spectrum)
        loss = criterion(output, true_spectrum)
        if torch.isnan(loss):
            print("Encountered NaN loss at epoch", epoch+1)
            break
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * blurred_spectrum.size(0)
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for blurred_spectrum, true_spectrum in val_loader:
            blurred_spectrum = blurred_spectrum.to(device)
            true_spectrum = true_spectrum.to(device)
            output = model(blurred_spectrum)
            loss = criterion(output, true_spectrum)
            val_loss += loss.item() * blurred_spectrum.size(0)
    val_loss /= len(val_loader.dataset)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')

# Select a sample from the validation set
model.eval()
with torch.no_grad():
    sample_blurred_spectrum = val_blurred[0]
    sample_true_spectrum = val_spectra[0]
    input_spectrum = torch.from_numpy(sample_blurred_spectrum).float().unsqueeze(0).to(device)
    deconvolved_spectrum = model(input_spectrum).cpu().numpy().squeeze()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(sample_true_spectrum, label='True Spectrum')
plt.plot(sample_blurred_spectrum, label='Blurred Spectrum')
plt.plot(deconvolved_spectrum, label='Deconvolved Spectrum')
plt.legend()
plt.title('Deep Richardson-Lucy Deconvolution')
plt.xlabel('Wavelength Index')
plt.ylabel('Intensity')
plt.show()

"""
import glob
from rldeconvolution.fix_inputs import fix_inputs

# Define paths to kernel and measurement data
psf_path = './data/kernel'
psf_files = sorted(glob.glob(psf_path + "/*.csv"))
orig_path = './data/original'
orig_files = glob.glob(orig_path + "/*.csv")

# Choose which file to use
file_index = 0

# Load your PSF and measurement data
psf_data = np.loadtxt(psf_files[file_index], delimiter=';', skiprows=63)
orig_data = np.loadtxt(orig_files[file_index], delimiter=';', skiprows=63)

# Process the data to obtain the PSF and measured spectrum
lb, b, lM, M = fix_inputs(psf_data[0, :], psf_data[1, :], orig_data[0, :], orig_data[1, :])

# Ensure b and M are 1D NumPy arrays
b = np.array(b)
M = np.array(M)

# Normalize the PSF
b /= np.sum(b)

# Normalize the measured spectrum
M_max = np.max(M)
M_normalized = M / M_max


psf_length = len(b)
spectrum_length = len(M_normalized)


# Initialize the model with your PSF
class DeepRichardsonLucy(nn.Module):
    def __init__(self, psf, num_layers=10):
        super(DeepRichardsonLucy, self).__init__()
        self.num_layers = num_layers
        self.psf = torch.from_numpy(psf).float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, PSF_length)
        self.psf_flip = torch.flip(self.psf, dims=[2])  # Flipped PSF
        padding = (self.psf.size(2) - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=self.psf.size(2), padding=padding, bias=False)
        self.conv.weight = nn.Parameter(self.psf)
        self.conv_flip = nn.Conv1d(1, 1, kernel_size=self.psf_flip.size(2), padding=padding, bias=False)
        self.conv_flip.weight = nn.Parameter(self.psf_flip)
        # Freeze PSF convolutions
        self.conv.weight.requires_grad = False
        self.conv_flip.weight.requires_grad = False
        # Learnable parameters
        self.alpha = nn.Parameter(torch.ones(num_layers))
        
    def forward(self, m):
        batch_size, spectrum_length = m.size()
        m = m.unsqueeze(1)  # Shape: (batch_size, 1, spectrum_length)
        s = torch.full_like(m, 0.5)  # Initial estimate
        for k in range(self.num_layers):
            s_conv = self.conv(s) + 1e-6  # Avoid division by zero
            ratio = m / s_conv
            correction = self.conv_flip(ratio) + 1e-6  # Avoid negative or zero values
            correction = torch.clamp(correction, min=1e-6)
            # Update with learnable parameter alpha[k]
            s = s * (correction ** self.alpha[k])
            s = torch.clamp(s, min=1e-6)  # Enforce non-negativity
        s = s.squeeze(1)  # Shape: (batch_size, spectrum_length)
        return s

# Initialize the model with your PSF
model = DeepRichardsonLucy(psf=b, num_layers=10)


# Convert to PyTorch tensor
M_tensor = torch.from_numpy(M_normalized).float().unsqueeze(0)  # Shape: (1, spectrum_length)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
M_tensor = M_tensor.to(device)

model.eval()
with torch.no_grad():
    deconvolved_spectrum = model(M_tensor).cpu().numpy().squeeze()


# Rescale deconvolved spectrum
deconvolved_spectrum_rescaled = deconvolved_spectrum * M_max




# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(M, label='Measured Blurred Spectrum')
plt.plot(deconvolved_spectrum_rescaled, label='Deconvolved Spectrum')
plt.legend()
plt.title('Deconvolution of Measured Spectrum')
plt.xlabel('Wavelength Index')
plt.ylabel('Intensity')
plt.show()
"""
"""
import glob
from rldeconvolution.fix_inputs import fix_inputs

# Define paths to kernel and measurement data
psf_path = './data/kernel'
psf_files = sorted(glob.glob(psf_path + "/*.csv"))
orig_path = './data/original'
orig_files = glob.glob(orig_path + "/*.csv")

# Choose which file to use
file_index = 0

# Load your PSF and measurement data
psf_data = np.loadtxt(psf_files[file_index], delimiter=';', skiprows=63)
orig_data = np.loadtxt(orig_files[file_index], delimiter=';', skiprows=63)

# Process the data to obtain the PSF and measured spectrum
lb, b, lM, M = fix_inputs(psf_data[0, :], psf_data[1, :], orig_data[0, :], orig_data[1, :])

M = np.array(M)

M_max = np.max(M)
M_normalized = M / M_max

# Convert M to PyTorch tensor and move to the correct device
M_tensor = torch.from_numpy(M_normalized).float().unsqueeze(0).to(device)  # Shape: (1, spectrum_length)

# Set the model to evaluation mode
model.eval()

# Run the deconvolution on M
with torch.no_grad():
    deconvolved_spectrum = model(M_tensor).cpu().numpy().squeeze()  # Remove extra dimensions

# Rescale the deconvolved spectrum back to the original scale
deconvolved_spectrum_rescaled = deconvolved_spectrum * M_max

# Plot the original and deconvolved spectra for comparison
plt.figure(figsize=(12, 6))
plt.plot(M, label='Measured Blurred Spectrum')
plt.plot(deconvolved_spectrum_rescaled, label='Deconvolved Spectrum')
plt.legend()
plt.title('Deconvolution of Measured Spectrum using Deep Richardson-Lucy Network')
plt.xlabel('Wavelength Index')
plt.ylabel('Intensity')
plt.grid(True)
plt.show()

"""
