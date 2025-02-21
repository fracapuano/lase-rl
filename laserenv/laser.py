import line_profiler

import torch
import numpy as np
import pandas as pd
from scipy.constants import c
from typing import Tuple, Optional

from laserenv.utils import (
    physics,
    preprocessing,
    frogtrace
)

from laserenv.utils.funcs import get_project_root

class ComputationalLaser: 
    def __init__(
                self, 
                frequency: torch.Tensor, 
                field: torch.Tensor, 
                compressor_params: Tuple[float, float, float],
                num_points_padding: int = int(3e4), 
                B: float = 2, 
                central_frequency: float = (c/(1030*1e-9)), 
                cristal_frequency: Optional[torch.Tensor]=None, 
                cristal_intensity: Optional[torch.Tensor]=None,
                device: str="cpu"
                ) -> None:
        """Init function. 
        This model is initialized for a considered intensity in the frequency domain signal. The signal is assumed to be already cleaned. 

        Args:
            frequency (torch.Tensor): Tensor of frequencies, measured in THz, already preprocessed.
            field (torch.Tensor): Tensor of electrical field (measured with respect to the frequency).
            compressor_params (Tuple[float, float, float]): Compressor GDD, TOD and FOD in SI units.
            central_frequency (float, optional): Central frequency, may be derived from central wavelength. Defaults to (c/(1030*1e-9)) Hz.
            num_points_padding (int, optional): Number of points to be used to pad. Defaults to int(3e4).
            B (float, optional): B-integral value. Used to model the non-linear effects that DIRA has on the beam.
            cristal_frequency (torch.Tensor, optional): Frequency (THz) of the amplification in the non-linear crystal at the beginning of DIRA.
            cristal_intensity (torch.Tensor, optional): Intensity of the amplification in the non-linear crystal at the beginning of DIRA.
            device (str, optional): Device to use for GPU operations. Defaults to "cpu".
        """
        self.device = device
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
            self.yb_field = torch.sqrt(cristal_intensity)
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
            yb_frequency, yb_field = preprocessing.cutoff_signal(
                frequency_cutoff=(self.frequency[0].item(), self.frequency[-1].item()), 
                frequency=yb_frequency, 
                signal=yb_field)
            
            # augmenting the cut data
            yb_frequency, yb_field = preprocessing.equidistant_points(
                frequency=yb_frequency, 
                signal=yb_field, 
                num_points=len(self.frequency)
            )
            self.yb_frequency = torch.from_numpy(yb_frequency)
            self.yb_intensity = torch.from_numpy(yb_field ** 2)
            self.yb_field = torch.from_numpy(yb_field)

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
    
    def translate_control(self, control: torch.Tensor, verse: str = "to_gdd") -> torch.Tensor: 
        """This function translates the control quantities either from Dispersion coefficients (the d_i's) to GDD, TOD and FOD using a system of linear equations 
        defined for this very scope or the other way around, according to the string "verse".  

        Args:
            control (torch.Tensor): Control quantities (either the d_i's or delay information). Must be given in SI units.
            verse (str, optional): "to_gdd" to translate control from dispersion coefficients to (GDD, TOD and FOD), solving Ax = b.
                                     "to_disp" to translate (GDD, TOD and FOD) to dispersion coefficient left-multiplying the control by A. Defaults to "to_gdd". 

        Returns:
            torch.Tensor: The control translated according to the verse considered.
        """
        return physics.translate_control(central_frequency=self.central_frequency, control=control, verse=verse)
    
    def emit_phase(self, control: torch.Tensor) -> torch.Tensor: 
        """This function returns the phase with respect to the frequency and some control parameters.
        Runs the phase calculation in float32 to allow for MPS compatibility. Running the operation in float32 requires
        to use non-SI units for the frequency (THz instead of Hz) and controls (ps^k instead of s^k).

        Args:
            control (torch.Tensor): Control parameters to be used to create the phase. 
                                    It contains GDD, TOD and FOD in s^2, s^3 and s^4.

        Returns:
            torch.Tensor: The phase with respect to the frequency, measured in radians.
        """
        return physics.phase_equation(
            frequency=(self.frequency * 1e-12).type(torch.float32).to(control.device), 
            central_frequency=torch.tensor(self.central_frequency * 1e-12, dtype=torch.float32).to(control.device), 
            control=control
        )
    
    def transform_limited(self, return_time: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """This function returns the transform limited of the input spectrum.

        Args:
            return_time (bool, optional): Whether or not to return (also) the time-scale. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns either (time, intensity) (with time measured in seconds) or intensity only.
        """
        step = torch.diff(self.frequency)[0]
        Dt = 1 / step
        time = torch.linspace(start=-Dt/2, end=Dt/2, steps=len(self.frequency) + self.pad_points)
        # transform limited of amplified spectrum
        field_padded = torch.nn.functional.pad(
            physics.yb_gain(self.field, torch.sqrt(self.yb_intensity)),
            pad=(self.pad_points // 2, self.pad_points // 2), 
            mode="constant", 
            value=0
        )

        # inverse FFT to go from frequency domain to temporal domain
        field_time = torch.fft.ifftshift(torch.fft.ifft(field_padded))
        intensity_time = torch.real(field_time * torch.conj(field_time))  # only for casting reasons

        intensity_time =  intensity_time / intensity_time.max()  # normalizing
        
        # either returning time or not according to return_time
        if not return_time: 
            return intensity_time
        else: 
            return time, intensity_time
        
    def control_to_temporal(self, control:torch.TensorType)->Tuple[torch.tensor, torch.tensor]: 
        """This function performs a forward pass in the model using control values stored in control.

        Args:
            control (torch.Tensor): Control values to use in the forward pass. Must be dispersion coefficients, given in SI units 

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Time scale, Temporal profile of intensity for the given control).
        """
        control = control.to(self.device)
        # control quantities regulate the phase
        phi_stretcher = self.emit_phase(control=control)
        # phase imposed on the input field
        y1_frequency = physics.impose_phase(
            spectrum=self.field.type(torch.float32).to(phi_stretcher.device), 
            phase=phi_stretcher
        )
        # spectrum amplified by DIRA crystal gain
        y1tilde_frequency = physics.yb_gain(
            signal=y1_frequency, 
            intensity_yb=self.yb_field.type(torch.float32).to(y1_frequency.device)
        )
        # spectrum amplified in time domain, to apply non linear phase to it
        y1tilde_time = torch.fft.ifft(y1tilde_frequency)
        # defining non-linear DIRA phase
        intensity = torch.real(y1tilde_time * torch.conj(y1tilde_time))
        phi_DIRA = (self.B / intensity.max()) * intensity
        # applying non-linear DIRA phase to the spectrum
        y2_time = physics.impose_phase(spectrum=y1tilde_time, phase=phi_DIRA)
        # back to frequency domain
        y2_frequency = torch.fft.fft(y2_time)
        # defining phase imposed by compressor
        phi_compressor = self.emit_phase(
            control=(
                self.compressor_params * torch.tensor([1e24, 1e36, 1e48], dtype=torch.float64)
            ).type(torch.float32)
        )
        # imposing compressor phase on spectrum
        y3_frequency = physics.impose_phase(
            y2_frequency, 
            phase=phi_compressor.to(y2_frequency.device)
        )
        # return time scale and temporal profile of the (controlled) pulse
        return physics.temporal_profile(frequency=self.frequency, field=y3_frequency, npoints_pad=self.pad_points)
    
    @line_profiler.profile
    def control_to_frog(
            self, 
            control:torch.Tensor, 
            return_axes: bool=False, 
            npoints_pad: int=int(1e4),
            trim_window: int=int(1e3)
        ) -> torch.Tensor: 
        """This function returns the FROG trace of the pulse for a given control.

        Args:
            control (torch.Tensor): Control values to use in the forward pass. Must be dispersion coefficients, given in SI units 
            return_axes (bool, optional): Whether or not to return the axes of the FROG trace. Defaults to False.

        Returns:
            torch.Tensor: The FROG trace of the pulse for the given control.
        """
        control = control.type(torch.float16).to(self.device)
        # control quantities regulate the phase
        phi_stretcher = self.emit_phase(control=control)
        # phase imposed on the input field
        y1_frequency = physics.impose_phase(
            spectrum=self.field.type(torch.float16).to(phi_stretcher.device), 
            phase=phi_stretcher
        )
        # spectrum amplified by DIRA crystal gain
        y1tilde_frequency = physics.yb_gain(
            signal=y1_frequency, 
            intensity_yb=self.yb_field.type(torch.float16).to(y1_frequency.device)
        )
        # spectrum amplified in time domain, to apply non linear phase to it
        y1tilde_time = torch.fft.ifft(y1tilde_frequency)
        # defining non-linear DIRA phase
        intensity = torch.real(y1tilde_time * torch.conj(y1tilde_time))
        phi_DIRA = (self.B / intensity.max()) * intensity
        # applying non-linear DIRA phase to the spectrum
        y2_time = physics.impose_phase(spectrum=y1tilde_time, phase=phi_DIRA)
        # back to frequency domain
        y2_frequency = torch.fft.fft(y2_time)
        # defining phase imposed by compressor
        phi_compressor = self.emit_phase(
            control=(
                self.compressor_params * torch.tensor([1e24, 1e36, 1e48], dtype=torch.float64)
            ).type(torch.float16)
        )
        # imposing compressor phase on spectrum
        y3_frequency = physics.impose_phase(
            y2_frequency, 
            phase=phi_compressor.to(y2_frequency.device)
        )

        y3_frequency = torch.nn.functional.pad(
            input=y3_frequency, 
            pad=(npoints_pad, npoints_pad), 
            mode="constant", 
            value=0
        )
        
        # compute the FROG trace
        frog_output = frogtrace.compute_frog_trace(
            E_time=y3_frequency, 
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define a simulation frequency axis (in THz)
    num_points = 1000
    freq_min = 280  # lower bound in THz
    freq_max = 300  # upper bound in THz
    frequency = torch.linspace(freq_min, freq_max, num_points)  # in THz

    # Create a dummy electric field (Gaussian shape) centered at 290 THz.
    central_freq_sim = 290  # THz: center frequency for the simulation
    bandwidth = 5  # THz: standard deviation of the Gaussian
    field = torch.exp(-0.5 * ((frequency - central_freq_sim) / bandwidth) ** 2)

    # Define compressor parameters as a tuple (GDD, TOD, FOD) in SI units.
    compressor_params = (1e-27, 1e-40, 1e-52)

    # Instantiate the ComputationalLaser object without any CUDA-specific code.
    laser = ComputationalLaser(
        frequency=frequency,
        field=field,
        compressor_params=compressor_params,
        num_points_padding=2048,
        B=2
    )

    # Define dummy control values (dispersion coefficients) for the forward pass.
    control = torch.tensor([1e-28, 1e-40, 1e-52], dtype=torch.float64)
    
    # Run the forward pass to compute the output temporal profile.
    time_forward, output_intensity = laser.control_to_temporal(control)
    frog, diff_time, diff_wl = laser.control_to_frog(control, return_axes=True)

    # Plot the output temporal profile from the forward pass.
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(time_forward.numpy(), output_intensity.numpy())
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Normalized Intensity")
    axs[0].set_title("Forward Pass Pulse Profile")
    axs[0].grid(True)

    axs[1].imshow(frog, cmap="gray")
    axs[1].set_title("FROG Trace")
    axs[1].axis("off")
    fig.tight_layout()
    plt.show()
    
