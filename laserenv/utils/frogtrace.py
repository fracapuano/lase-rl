import line_profiler
from typing import Union, Tuple

import torch
import numpy as np
from torch.fft import fft, ifft, fftshift, fftfreq, ifftshift
from scipy.interpolate import interp1d
from scipy.constants import c
import matplotlib.pyplot as plt
import pandas as pd

def spectral_phase(
        frequency:np.ndarray, 
        central_frequency:float, 
        gdd:float, 
        tod:float, 
        fod:float,
        gdd_scale:float=1e-24, 
        tod_scale:float=1e-36, 
        fod_scale:float=1e-48
    ) -> np.ndarray:
    """
    Compute the spectral phase from a set of dispersion parameters.

    Phase in the spectral domain is given by:
        phase = (1/2) * gdd * gdd_scale * (ω - ω₀)² +
                (1/6) * tod * tod_scale * (ω - ω₀)³ +
                (1/24) * fod * fod_scale * (ω - ω₀)⁴
    
    where ω = 2π * frequency and ω₀ = 2π * central_frequency.

    Args:
        frequency (np.ndarray): 1D array of frequencies in Hz.
        central_frequency (float): Central frequency in Hz.
        gdd (float): Group delay dispersion parameter.
        tod (float): Third order dispersion parameter.
        fod (float): Fourth order dispersion parameter.
        gdd_scale, tod_scale, fod_scale (float): Scaling factors for conversion.
        
    Returns:
        np.ndarray: Spectral phase (in radians) at each frequency.
    """
    central_omega = 2 * np.pi * central_frequency
    omega = 2 * np.pi * frequency
    
    phase = (1/2) * gdd * gdd_scale * (omega - central_omega) ** 2 + \
            (1/6) * tod * tod_scale * (omega - central_omega) ** 3 + \
            (1/24) * fod * fod_scale * (omega - central_omega) ** 4
    
    return phase

def compute_temporal_electrical_field(
        spectrum: torch.Tensor, 
        phase: torch.Tensor, 
        pad_width: int=10000
    ) -> torch.Tensor:
    """
    Compute the temporal electric field from a spectral amplitude and phase.

    The input spectrum and phase are zero padded for higher time resolution,
    and the inverse FFT is applied to recover the time-domain field.

    Args:
        spectrum (torch.Tensor): 1D spectral amplitude array.
        phase (torch.Tensor): Spectral phase array (radians).
        pad_width (int): Number of zeros to pad at each end.

    Returns:
        torch.Tensor: Normalized complex temporal electric field.
    """
    padded_spectrum = torch.nn.functional.pad(spectrum, (pad_width, pad_width), mode='constant', value=0)
    padded_phase = torch.nn.functional.pad(phase, (pad_width, pad_width), mode='constant', value=0)
    field_freq = padded_spectrum * torch.exp(1j * padded_phase)
    
    # The fftshift/ifftshift combination ensures the correct frequency ordering.
    field_time = ifftshift(ifft(fftshift(field_freq)))
    # Normalize to unit maximum amplitude
    field_time /= torch.abs(field_time).max()
    
    return field_time

@line_profiler.profile
def compute_frog_trace(
        E_time: torch.Tensor, 
        dt: float,
        trim_window: int=1000,
        pad_width: int=10_000,
        compute_axes: bool=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    GPU-accelerated FROG trace computation using PyTorch.
    """
    if trim_window is not None:
        max_index = E_time.shape[0]//2  # comes from padding!
        E_time = E_time[max_index-trim_window:max_index+trim_window]

    N = len(E_time)
    device = E_time.device
    
    # Create index tensors on GPU
    idx = torch.arange(N, device=device)
    delays = torch.arange(-N // 2, N // 2, device=device)
    
    # Build the rolling matrix using GPU operations
    indices = (idx[None, :] - delays[:, None]) % N
    E_roll = torch.index_select(E_time, 0, indices.flatten()).reshape(len(delays), -1)
    
    # Compute SHG signal
    product = E_time[None, :] * E_roll
    
    # FFT operations on GPU
    product_shifted = ifftshift(product, dim=1)
    F = fft(product_shifted, dim=1)
    F = fftshift(F, dim=1)
    
    # Compute and normalize intensity
    frog_intensity = torch.abs(F) ** 2
    frog_intensity = frog_intensity / frog_intensity.max()
    
    if compute_axes:
        # Create axes (can stay on CPU as they're only used for plotting)
        total_time = N * dt
        delay_axis = torch.linspace(-total_time / 2, total_time / 2, N)
        frequency_axis = fftshift(fftfreq(N, dt))
        
        return frog_intensity, delay_axis, frequency_axis
    else:
        return frog_intensity

def generate_frog_trace(
        freq: np.ndarray, 
        spectrum: np.ndarray, 
        gdd: float = 0, 
        tod: float = 0, 
        fod: float = 0,
        central_frequency: float = None, 
        pad_width: int = 10000,
        trim_window: int = 1000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an SHG-FROG trace from spectral data and phase parameters.

    This function (i) computes a spectral phase (using GDD, TOD, and FOD),
    (ii) computes the time-domain electric field via an inverse FFT with zero
    padding, and then (iii) computes the SHG-FROG trace by evaluating the signal
    intensity for a range of delays.

    Args:
        freq (np.ndarray): 1D array of frequencies in Hz on a uniform grid.
        spectrum (np.ndarray): 1D spectral amplitude array.
        gdd (float): Group delay dispersion parameter.
        tod (float): Third order dispersion parameter.
        fod (float): Fourth order dispersion parameter.
        central_frequency (float, optional): Central frequency in Hz. If None,
            it is set to the frequency corresponding to the maximum amplitude.
        pad_width (int, optional): Amount of zero padding on each end.
        trim_window (int, optional): Number of points (centered at the pulse) to keep in the final FROG trace.
        
    Returns:
        tuple:
            - np.ndarray: 2D normalized FROG trace intensity (possibly trimmed).
            - np.ndarray: Delay axis in picoseconds (possibly trimmed).
            - np.ndarray: SHG wavelength axis in nm (corresponding to the trimmed region).
    """
    # Determine the center frequency if not explicitly provided.
    if central_frequency is None:
        central_frequency = freq[np.argmax(spectrum)]

    # Compute the spectral phase.
    phase = spectral_phase(freq, central_frequency, gdd, tod, fod)
    # Compute the temporal field (with zero padding)
    E_time = compute_temporal_electrical_field(
        torch.from_numpy(spectrum), 
        torch.from_numpy(phase), 
        pad_width=pad_width
    )

    # Determine the frequency step from the original grid (assumed uniform).
    df = freq[1] - freq[0]
    N_pad = len(spectrum) + 2 * pad_width
    dt = 1 / (N_pad * df)  # time step in seconds

    # Compute the full FROG trace (delay vs. frequency).
    frog_trace, delay_axis, freq_axis = compute_frog_trace(
        E_time, 
        dt, 
        pad_width=pad_width,
        trim_window=trim_window,
        compute_axes=True
    )

    # Compute the wavelength axis for the SHG trace.
    # Convert frequency (Hz) to THz for plotting convenience.
    freq_axis_THz = freq_axis / 1e12
    central_frequency_THz = central_frequency / 1e12
    # For SHG, the effective frequency is shifted by 2x the central frequency.
    f_shg = freq_axis_THz + 2 * central_frequency_THz
    # Now convert back to wavelength (nm)
    wavelength_axis = (c / (f_shg * 1e12)) * 1e9

    return frog_trace, delay_axis, wavelength_axis

# Example usage:
if __name__ == "__main__":

    # Load spectral data from CSV.
    # The CSV is assumed to have columns:
    #   0: wavelength (nm), 1: frequency (in THz),
    #   2: intensity (normalized to have unit area)

    from pathlib import Path
    df = pd.read_csv(
        str(Path(__file__).parent.parent) + "/data/dira_spec.csv",
        header=0
    )
    # Convert columns to NumPy arrays.
    wavelength = df["Wavelength (nm)"].values
    intensity = df["Intensity"].values
    if "Frequency (THz)" not in df.columns:
        freq = c / (wavelength*1e-9)
    else:
        freq = df["Frequency (THz)"].values * 1e12

    # Use the square root of intensity (as in the original code) for the amplitude.
    amplitude = np.sqrt(np.abs(intensity))

    # Interpolate onto a uniform frequency grid.
    num_points = 1000
    freq_uniform = np.linspace(290.3*1e12, 291.4*1e12, num_points)
    interp_amplitude = interp1d(freq, amplitude, kind='cubic')
    spectrum_uniform = interp_amplitude(freq_uniform)

    # Set dispersion parameters (adjust these as needed).
    gdd = 0
    tod = 0
    fod = 0

    # # Generate the FROG trace.
    frog, delay_axis, wavelength_axis = generate_frog_trace(
        freq_uniform, spectrum_uniform, gdd=gdd, tod=tod, fod=fod, pad_width=10_000
    )

    # Plot the resulting SHG-FROG trace.
    plt.figure(figsize=(7, 7))
    plt.title("SHG FROG Trace")
    extent = [delay_axis[0], delay_axis[-1], wavelength_axis[0], wavelength_axis[-1]]
    plt.imshow(frog.T, extent=extent, aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel("Delay (ps)")
    plt.ylabel("Wavelength (nm)")
    plt.colorbar(label="Intensity (a.u.)")
    plt.show()

