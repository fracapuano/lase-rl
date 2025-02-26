"""TODO reimplement in jax"""

import numpy as np
import pandas as pd
from scipy.constants import c
from typing import Tuple, Optional, Union, List

from laserenv.utils.funcs import get_project_root

class ComputationalLaserNumpy:
    def __init__(
                self, 
                frequency: np.ndarray, 
                field: np.ndarray, 
                compressor_params: Tuple[float, float, float],
                num_points_padding: int = int(5e4), 
                B: float = 2, 
                central_frequency: float = (c/(1030*1e-9)), 
                cristal_frequency: Optional[np.ndarray]=None, 
                cristal_intensity: Optional[np.ndarray]=None
                ) -> None:
        """Init function. 
        This model is initialized for a considered intensity in the frequency domain signal. The signal is assumed to be already cleaned. 

        Args:
            frequency (np.ndarray): Array of frequencies, measured in THz, already preprocessed.
            field (np.ndarray): Array of electrical field (measured with respect to the frequency).
            compressor_params (Tuple[float, float, float]): Compressor GDD, TOD and FOD in SI units.
            central_frequency (float, optional): Central frequency, may be derived from central wavelength. Defaults to (c/(1030*1e-9)) Hz.
            num_points_padding (int, optional): Number of points to be used to pad. Defaults to int(5e4).
            B (float, optional): B-integral value. Used to model the non-linear effects that DIRA has on the beam.
            cristal_frequency (np.ndarray, optional): Frequency (THz) of the amplification in the non-linear crystal at the beginning of DIRA.
            cristal_intensity (np.ndarray, optional): Intensity of the amplification in the non-linear crystal at the beginning of DIRA.
        """
        self.frequency = frequency * 1e12  # Convert THz to Hz
        self.field = field  # electric field is the square root of intensity
        self.central_frequency = central_frequency
        # number of points to be used in padding 
        self.pad_points = num_points_padding
        # hyperparameters - LASER parametrization
        self.compressor_params = compressor_params
        self.B = B
        # storing the original input
        self.input_frequency, self.input_field = frequency * 1e12, field
        # YB:Yab gain
        if cristal_intensity is not None and cristal_frequency is not None: 
            self.yb_frequency = cristal_frequency * 1e12  # THz to Hz
            self.yb_field = np.sqrt(cristal_intensity)
        else: 
            # reading the data with which to amplify the signal when non specific one is given
            cristal_path = str(get_project_root()) + "/data/cristal_gain.txt"
            gain_df = pd.read_csv(cristal_path, sep="  ", skiprows=2, header=None, engine="python")

            gain_df.columns = ["Wavelength (nm)", "Intensity"]
            gain_df["Intensity"] = gain_df["Intensity"] / gain_df["Intensity"].values.max()
            gain_df["Frequency (Hz)"] = gain_df["Wavelength (nm)"].apply(lambda wl: (c/((wl+1) * 1e-9)))  # 1nm rightwards shift

            gain_df.sort_values(by="Frequency (Hz)", inplace=True)
            yb_frequency, yb_field = gain_df["Frequency (Hz)"].values, np.sqrt(gain_df["Intensity"].values)
            
            # cutting the gain frequency accordingly
            from laserenv.utils import preprocessing
            yb_frequency, yb_field = preprocessing.cutoff_signal(
                frequency_cutoff=(self.frequency[0], self.frequency[-1]), 
                frequency=yb_frequency, 
                signal=yb_field)
            
            # augmenting the cut data
            yb_frequency, yb_field = preprocessing.equidistant_points(
                frequency=yb_frequency, 
                signal=yb_field, 
                num_points=len(self.frequency)
            )
            self.yb_frequency = yb_frequency
            self.yb_intensity = yb_field ** 2
            self.yb_field = yb_field

    def overwrite_B_integral(self, new_B: float) -> None: 
        """This function overwrites the B-integral value.

        Args:
            new_B (float): New B-integral value.
        """
        self.B = new_B

    def overwrite_compressor_params(
            self, 
            new_compressor_params: Tuple[float, float, float]
        ) -> None:
        """This function overwrites the compressor parameters.

        Args:
            new_compressor_params (Tuple[float, float, float]): New compressor parameters.
        """
        self.compressor_params = new_compressor_params
    
    def overwrite_single_compressor_params(
            self, 
            index: int,
            new_compressor_param: float
        ) -> None:
        """This function overwrites a single compressor parameter.
        """
        compressor_list = list(self.compressor_params)
        compressor_list[index] = new_compressor_param
        self.compressor_params = tuple(compressor_list)
    
    def translate_control(self, control: np.ndarray, verse: str = "to_gdd") -> np.ndarray: 
        """This function translates the control quantities either from Dispersion coefficients (the d_i's) to GDD, TOD and FOD using a system of linear equations 
        defined for this very scope or the other way around, according to the string "verse".  

        Args:
            control (np.ndarray): Control quantities (either the d_i's or delay information). Must be given in SI units.
            verse (str, optional): "to_gdd" to translate control from dispersion coefficients to (GDD, TOD and FOD), solving Ax = b.
                                     "to_disp" to translate (GDD, TOD and FOD) to dispersion coefficient left-multiplying the control by A. Defaults to "to_gdd". 

        Returns:
            np.ndarray: The control translated according to the verse considered.
        """
        # central wavelength (using c/f = lambda)
        central_wavelength = c / self.central_frequency

        a11 = (-2 * np.pi * c)/(central_wavelength ** 2) 
        a21 = (4 * np.pi * c)/(central_wavelength ** 3)
        a22 = ((2 * np.pi * c)/(central_wavelength ** 2))**2
        
        a31 = (-12 * np.pi * c)/(central_wavelength ** 4)
        a32 = -(24 * (np.pi * c) ** 2)/(central_wavelength ** 5)
        a33 = -((2 * np.pi * c) / (central_wavelength ** 2)) ** 3

        # conversion matrix
        A = np.array([
            [a11, 0, 0], 
            [a21, a22, 0], 
            [a31, a32, a33]
        ], dtype=np.float64)

        if verse.lower() == "to_gdd": 
            d2, d3, d4 = control
            # solving the conversion system using forward substitution
            GDD = d2 / A[0,0]
            TOD = (d3 - A[1,0] * GDD)/(A[1,1])
            FOD = (d4 - A[2,0] * GDD - A[2,1] * TOD)/(A[2,2])
            # grouping the values
            return np.array([GDD, TOD, FOD])

        elif verse.lower() == "to_disp": 
            return A @ control
        else: 
            raise ValueError('Control translation is either "to_gdd" or "to_disp"!')
    
    def emit_phase(self, control: np.ndarray) -> np.ndarray: 
        """This function returns the phase with respect to the frequency and some control parameters.

        Args:
            control (np.ndarray): Control parameters to be used to create the phase. 
                                    It contains GDD, TOD and FOD in s^2, s^3 and s^4.

        Returns:
            np.ndarray: The phase with respect to the frequency, measured in radians.
        """
        GDD, TOD, FOD = control
        # Convert frequency from Hz to THz for numerical stability
        frequency_thz = self.frequency * 1e-12
        central_frequency_thz = self.central_frequency * 1e-12
        
        phase = \
                (1/2) * GDD * (2*np.pi * (frequency_thz - central_frequency_thz))**2 + \
                (1/6) * TOD * (2*np.pi * (frequency_thz - central_frequency_thz))**3 + \
                (1/24) * FOD * (2*np.pi * (frequency_thz - central_frequency_thz))**4
        
        return phase
    
    def transform_limited(self, return_time: bool = True) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """This function returns the transform limited of the input spectrum.

        Args:
            return_time (bool, optional): Whether or not to return (also) the time-scale. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray]: Returns either (time, intensity) (with time measured in seconds) or intensity only.
        """
        step = np.diff(self.frequency)[0]
        Dt = 1 / step
        time = np.linspace(start=-Dt/2, stop=Dt/2, num=len(self.frequency) + self.pad_points)
        
        # transform limited of amplified spectrum
        amplified_field = self.yb_gain(self.field, np.sqrt(self.yb_intensity))
        
        # Pad the field for better resolution
        field_padded = np.pad(
            amplified_field,
            pad_width=(self.pad_points // 2, self.pad_points // 2),
            mode="constant",
            constant_values=0
        )

        # inverse FFT to go from frequency domain to temporal domain
        field_time = np.fft.ifftshift(np.fft.ifft(field_padded))
        intensity_time = np.abs(field_time)**2

        intensity_time = intensity_time / intensity_time.max()  # normalizing
        
        # either returning time or not according to return_time
        if not return_time: 
            return intensity_time
        else: 
            return time, intensity_time
    
    def yb_gain(self, signal: np.ndarray, intensity_yb: np.ndarray, n_passes: int = 50) -> np.ndarray:
        """This function models the passage of the signal in the crystal in which yb:yab gain is observed.
        
        Args: 
            signal (np.ndarray): The intensity signal that enters the system considered.
            intensity_yb (np.ndarray): The gain intensity of the crystal
            n_passes (int, optional): The number of times the beam passes through the crystal where spectrum narrowing is observed. 
            
        Returns: 
            np.ndarray: New spectrum, narrower because of the gain. 
        """
        return signal * (intensity_yb ** n_passes)
    
    def impose_phase(self, spectrum: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Apply a phase to a spectrum.
        
        Args:
            spectrum (np.ndarray): Complex spectrum to which the phase is applied.
            phase (np.ndarray): Phase to apply (in radians).
            
        Returns:
            np.ndarray: Complex spectrum with the applied phase.
        """
        return spectrum * np.exp(1j * phase)
    
    def pump_chain(self, control: np.ndarray) -> np.ndarray:
        """This function performs a forward pass in the model using control values stored in control.

        Args:
            control (np.ndarray): Control values to use in the forward pass. Must be dispersion coefficients.

        Returns:
            np.ndarray: The output spectrum of the pump chain.
        """
        # control quantities regulate the phase
        phi_stretcher = self.emit_phase(control=control)
        
        # phase imposed on the input field
        y1_frequency = self.impose_phase(
            spectrum=self.field, 
            phase=phi_stretcher
        )
        
        # spectrum amplified by DIRA crystal gain
        y1tilde_frequency = self.yb_gain(
            signal=y1_frequency, 
            intensity_yb=self.yb_field
        )
        
        # spectrum amplified in time domain, to apply non linear phase to it
        y1tilde_time = np.fft.ifftshift(
            np.fft.ifft(
                np.fft.fftshift(
                    y1tilde_frequency
                )
            )
        )
        
        # defining non-linear DIRA phase
        intensity = np.abs(y1tilde_time)**2
        phi_DIRA = (self.B / intensity.max()) * intensity
        
        # applying non-linear DIRA phase to the spectrum
        y2_time = self.impose_phase(spectrum=y1tilde_time, phase=phi_DIRA)
        
        # back to frequency domain
        y2_frequency = np.fft.fft(y2_time)
        
        # defining phase imposed by compressor
        scaling_factors = np.array([1e24, 1e36, 1e48], dtype=np.float64)
        scaled_compressor_params = np.array(self.compressor_params) * scaling_factors
        
        phi_compressor = self.emit_phase(control=scaled_compressor_params)
        
        # imposing compressor phase on spectrum
        y3_frequency = self.impose_phase(
            y2_frequency, 
            phase=phi_compressor
        )
        
        return y3_frequency
    
    def temporal_profile(self, frequency: np.ndarray, field: np.ndarray, npoints_pad: int = int(1e4)) -> Tuple[np.ndarray, np.ndarray]:
        """This function returns the temporal profile of a given signal represented in the frequency domain.
        
        Args:
            frequency (np.ndarray): Array of frequencies considered (measured in Hz)
            field (np.ndarray): Array of field measured in the frequency domain. 
            npoints_pad (int, optional): Number of points to be used in padding.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (time, intensity) with time measured in seconds.
        """
        # centering the array in its peak - padding the signal extremities to increase resolution
        field_padded = np.pad(
            field, 
            pad_width=(npoints_pad//2, npoints_pad//2), 
            mode="constant", 
            constant_values=0
        )
        
        # going from frequency to time
        field_time = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(field_padded)))
        
        # obtaining intensity
        intensity_time = np.abs(field_time) ** 2
        
        # normalizing the resulting signal
        intensity_time = intensity_time / intensity_time.max()
        
        # create the time array
        step = np.diff(frequency)[0]
        Dt = 1 / step
        time = np.linspace(
            start=-Dt/2, 
            stop=Dt/2, 
            num=len(frequency)+npoints_pad
        )
        
        return time, intensity_time
    
    def control_to_temporal(self, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
        """This function performs a forward pass in the model using control values stored in control.

        Args:
            control (np.ndarray): Control values to use in the forward pass. Must be dispersion coefficients, given in SI units 

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Time scale, Temporal profile of intensity for the given control).
        """
        y3_frequency = self.pump_chain(control=control)
        # return time scale and temporal profile of the (controlled) pulse
        return self.temporal_profile(
            frequency=self.frequency, 
            field=y3_frequency, 
            npoints_pad=self.pad_points
        )
    
    def control_to_frog(
            self, 
            control: np.ndarray, 
            return_axes: bool = False, 
            npoints_pad: int = int(1e4),
            trim_window: int = int(1e3)
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]: 
        """This function returns the FROG trace of the pulse for a given control.

        Args:
            control (np.ndarray): Control values to use in the forward pass. Must be dispersion coefficients, given in SI units 
            return_axes (bool, optional): Whether or not to return the axes of the FROG trace. Defaults to False.
            npoints_pad (int, optional): Number of points to pad the FROG trace. Defaults to 10000.
            trim_window (int, optional): Trim window size. Defaults to 1000.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]: The FROG trace and optionally time and wavelength axes.
        """
        # compute the temporal profile of the pulse
        _, y3_time = self.control_to_temporal(control=control)
        
        # compute the FROG trace
        from laserenv.utils import frogtrace
        frog_output = self.compute_frog_trace(
            E_time=y3_time, 
            dt=1/self.frequency[0],
            trim_window=trim_window,
            pad_width=npoints_pad,
            compute_axes=return_axes
        )
        
        if return_axes:
            frog, diff_time, diff_freq = frog_output
            # Compute the wavelength axis for the SHG trace
            # Convert frequency (Hz) to THz for plotting convenience
            diff_freq = diff_freq / 1e12
            # Convert time (s) to ps for plotting convenience
            diff_time = diff_time * 1e12
            central_frequency = self.central_frequency / 1e12
            
            # For SHG, the effective frequency is shifted by 2x the central frequency.
            f_shg = diff_freq + 2 * central_frequency  # [THz]
            # Now convert back to wavelength (nm)
            diff_wl = (c / (f_shg * 1e12)) * 1e9  # [nm]

            return frog, diff_time, diff_wl
        
        else:
            return frog_output
    
    def compute_frog_trace(self, E_time, dt, pad_width=10000, trim_window=1000, compute_axes=False):
        """
        Compute the FROG trace for a given complex electric field in the time domain.
        
        Args:
            E_time (np.ndarray): Complex electric field in time domain.
            dt (float): Time step between samples (in seconds).
            pad_width (int, optional): Number of zeros to pad signal with. Defaults to 10000.
            trim_window (int, optional): Number of points to keep in the final output. Defaults to 1000.
            compute_axes (bool, optional): Whether to compute and return time and frequency axes. Defaults to False.
            
        Returns:
            np.ndarray or tuple: FROG trace (intensity) or (FROG trace, time axis, frequency axis) if compute_axes=True.
        """
        # If trim_window is provided, trim the input signal first
        if trim_window is not None:
            N = len(E_time)
            max_index = N // 2
            if trim_window < N:
                E_time = E_time[max_index - trim_window // 2:max_index + trim_window // 2]
            
        # Get the length of the signal after optional trimming
        N = len(E_time)
        
        # Create index arrays for vectorized operations
        idx = np.arange(N)
        delays = np.arange(-N // 2, N // 2)
        
        # Create a 2D grid of indices: for each delay, compute indices for all time points
        # This creates a matrix of shape (len(delays), N)
        indices = (idx[None, :] - delays[:, None]) % N
        
        # Efficiently lookup all delayed fields at once
        # This gives us all E(t-τ) for all delays τ and all time points t
        E_roll = E_time[indices]
        
        # Apply the gating operation E(t) * E(t-τ) for all delays in one operation
        # Broadcasting expands E_time to match E_roll's shape
        gated_signal = E_time[None, :] * E_roll
        
        # Compute all spectra at once using FFT along the time axis (axis=1)
        spectra = np.fft.fftshift(np.fft.fft(gated_signal, axis=1), axes=1)
        
        # Compute intensity
        frog_intensity = np.abs(spectra) ** 2
        
        # Normalize
        frog_intensity = frog_intensity / np.max(frog_intensity)
        
        if compute_axes:
            # Create time and frequency axes
            total_time = N * dt
            time_axis = np.linspace(-total_time / 2, total_time / 2, N)
            freq_axis = np.fft.fftshift(np.fft.fftfreq(N, dt))
            
            return frog_intensity, time_axis, freq_axis
        else:
            return frog_intensity


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    # Define a simulation frequency axis (in THz)
    num_points = 1000
    freq_min = 280  # lower bound in THz
    freq_max = 300  # upper bound in THz
    frequency = np.linspace(freq_min, freq_max, num_points)  # in THz

    # Create a dummy electric field (Gaussian shape) centered at 290 THz.
    central_freq_sim = 290  # THz: center frequency for the simulation
    bandwidth = 5  # THz: standard deviation of the Gaussian
    field = np.exp(-0.5 * ((frequency - central_freq_sim) / bandwidth) ** 2)

    # Define compressor parameters as a tuple (GDD, TOD, FOD) in SI units.
    compressor_params = (1e-27, 1e-40, 1e-52)

    # Instantiate the ComputationalLaserNumpy object
    laser = ComputationalLaserNumpy(
        frequency=frequency,
        field=field,
        compressor_params=compressor_params,
        num_points_padding=int(5e4),
        B=2
    )

    # Define dummy control values (dispersion coefficients) for the forward pass, in ps^k
    control = np.array([265, 0, 0])

    def time_function(func, *args, **kwargs):
        """Time the execution of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    # Time the control_to_temporal function
    print("\nPerformance Timing:")
    print("-" * 50)
    
    n_runs = 5  # Number of runs to average
    
    # Time control_to_temporal
    total_time = 0
    for _ in range(n_runs):
        _, execution_time = time_function(laser.control_to_temporal, control)
        total_time += execution_time
    avg_time = total_time / n_runs
    print(f"control_to_temporal: {avg_time:.4f} seconds (average over {n_runs} runs)")
    
    # Time control_to_frog
    total_time = 0
    for _ in range(n_runs):
        _, execution_time = time_function(laser.control_to_frog, control, True)
        total_time += execution_time
    avg_time = total_time / n_runs
    print(f"control_to_frog (with axes): {avg_time:.4f} seconds (average over {n_runs} runs)")
    
    # Time control_to_frog without axes
    total_time = 0
    for _ in range(n_runs):
        _, execution_time = time_function(laser.control_to_frog, control, False)
        total_time += execution_time
    avg_time = total_time / n_runs
    print(f"control_to_frog (without axes): {avg_time:.4f} seconds (average over {n_runs} runs)")

    
    # Run the forward pass to compute the output temporal profile.
    time_forward, output_intensity = laser.control_to_temporal(control)
    frog, diff_time, diff_wl = laser.control_to_frog(control, return_axes=True)

    # Plot the output temporal profile from the forward pass.
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(time_forward, output_intensity)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Normalized Intensity")
    axs[0].set_title("Forward Pass Pulse Profile")
    axs[0].grid(True)

    axs[1].imshow(frog, cmap="gray")
    axs[1].set_title("FROG Trace")
    axs[1].axis("off")
    fig.tight_layout()
    plt.show()