import line_profiler

import numpy as np
from numpy.fft import fft, ifft, fftshift, fftfreq, ifftshift
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
        spectrum: np.ndarray, 
        phase: np.ndarray, 
        pad_width: int=10000
    ) -> np.ndarray:
    """
    Compute the temporal electric field from a spectral amplitude and phase.

    The input spectrum and phase are zero padded for higher time resolution,
    and the inverse FFT is applied to recover the time-domain field.

    Args:
        spectrum (np.ndarray): 1D spectral amplitude array.
        phase (np.ndarray): Spectral phase array (radians).
        pad_width (int): Number of zeros to pad at each end.

    Returns:
        np.ndarray: Normalized complex temporal electric field.
    """
    padded_spectrum = np.pad(spectrum, pad_width, mode='constant', constant_values=0)
    padded_phase = np.pad(phase, pad_width, mode='constant', constant_values=0)
    field_freq = padded_spectrum * np.exp(1j * padded_phase)
    
    # The fftshift/ifftshift combination ensures the correct frequency ordering.
    field_time = ifftshift(ifft(fftshift(field_freq)))
    # Normalize to unit maximum amplitude
    field_time /= np.abs(field_time).max()
    
    return field_time

@line_profiler.profile
def compute_frog_trace(
        E_time: np.ndarray, 
        dt: float,
        trim_window: int=1000,
        pad_width: int=10_000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the SHG-FROG trace from the time-domain electric field using a vectorized implementation.

    For each delay τ, this function computes the SHG signal as the product
        E(t) * E(t - τ)
    and then applies a Fourier transform along the time axis. Using vectorized
    operations over all delays (instead of a Python loop) makes this function much faster.
    
    Args:
        E_time (np.ndarray): 1D complex temporal electric field.
        dt (float): Time step in seconds.
        trim_window (int, optional): Number of points (centered at the pulse) to keep from the input signal.
        
    Returns:
        tuple:
            - np.ndarray: 2D (intensity-normalized) FROG trace.
            - np.ndarray: Delay axis, in seconds.
            - np.ndarray: Frequency axis, in Hz.
    """
    if trim_window is not None:
        E_time = E_time[pad_width-trim_window:pad_width+2*trim_window]
    
    N = len(E_time)
    # Create an index array for time and define delay values
    idx = np.arange(N)
    delays = np.arange(-N // 2, N // 2)

    # Build a matrix in which each row is a cyclic shift of E_time
    E_roll = E_time[(idx[None, :] - delays[:, None]) % N]

    # For each delay, compute the SHG signal (i.e. multiply the original field with the rolled field)
    product = E_time[None, :] * E_roll

    # Apply ifftshift before the FFT and fftshift after
    product_shifted = ifftshift(product, axes=1)
    F = fft(product_shifted, axis=1)
    F = fftshift(F, axes=1)

    # Compute and normalize the intensity
    frog_intensity = np.real(np.abs(F) ** 2)
    frog_intensity /= frog_intensity.max()

    # Create a delay axis, in picoseconds
    total_time = N * dt
    delay_axis = np.linspace(-total_time / 2, total_time / 2, N)

    # Frequency axis in Hz from the FFT
    frequency_axis = fftshift(fftfreq(N, d=np.array(dt)))
    
    return frog_intensity, delay_axis, frequency_axis

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
    E_time = compute_temporal_electrical_field(spectrum, phase, pad_width=pad_width)

    # Determine the frequency step from the original grid (assumed uniform).
    df = freq[1] - freq[0]
    N_pad = len(spectrum) + 2 * pad_width
    dt = 1 / (N_pad * df)  # time step in seconds

    # Compute the full FROG trace (delay vs. frequency).
    frog_trace, delay_axis, freq_axis = compute_frog_trace(
        E_time, 
        dt, 
        pad_width=pad_width,
        trim_window=trim_window
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

    df = pd.read_csv("/Users/fracapuano/Documents/Subs/RLC25/lase-rl/laserenv/data/dira_spec.csv", header=None)

    # Convert columns to NumPy arrays.
    wavelength = np.array(df[0])
    # Frequency in Hz (power spectrum are collected in THz, hence they need scaling)
    freq = np.array(df[1]) * 1e12
    # Use the square root of intensity (as in the original code) for the amplitude.
    intensity = np.array(df[2])
    amplitude = np.sqrt(np.abs(intensity))

    # Interpolate onto a uniform frequency grid.
    num_points = 1000
    freq_uniform = np.linspace(290.3*1e12, 291.4*1e12, num_points)
    interp_amplitude = interp1d(freq, amplitude, kind='cubic')
    spectrum_uniform = interp_amplitude(freq_uniform)

    # Set dispersion parameters (adjust these as needed).
    gdd = 10
    tod = 10
    fod = 1

    # Generate the FROG trace.
    frog, delay_axis, wavelength_axis = generate_frog_trace(
        freq_uniform, spectrum_uniform, gdd=gdd, tod=tod, fod=fod, pad_width=10_000
    )

    # Plot the resulting SHG-FROG trace.
    plt.figure(figsize=(7, 7))
    plt.title("SHG FROG Trace")
    extent = [delay_axis[0], delay_axis[-1], wavelength_axis[0], wavelength_axis[-1]]
    plt.imshow(frog, extent=extent, aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel("Delay (ps)")
    plt.ylabel("Wavelength (nm)")
    plt.colorbar(label="Intensity (a.u.)")
    plt.show()

