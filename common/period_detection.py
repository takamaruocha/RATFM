import numpy as np
from scipy.fft import fft, fftfreq


def detect_period_fourier(signal: np.ndarray, sampling_interval: float = 1.0) -> float:
    """
    Detect the dominant period in a univariate time series using Fourier Transform.

    Args:
        signal (np.ndarray): Input time series (1D array).
        sampling_interval (float): Time between two samples. Default is 1.0.

    Returns:
        float: Estimated dominant period in the time series.
    """
    n_samples = len(signal)

    # Compute FFT and corresponding frequencies
    fft_result = fft(signal)
    frequencies = fftfreq(n_samples, d=sampling_interval)

    # Compute amplitude spectrum
    amplitude_spectrum = np.abs(fft_result)

    # Filter for positive frequencies only
    positive_freqs = frequencies[frequencies > 0]
    positive_amplitudes = amplitude_spectrum[frequencies > 0]

    if len(positive_freqs) == 0:
        raise ValueError("No positive frequency component found. Check input signal length or content.")

    # Identify the peak frequency
    peak_index = np.argmax(positive_amplitudes)
    peak_frequency = positive_freqs[peak_index]

    # Convert frequency to period
    dominant_period = 1.0 / peak_frequency

    return dominant_period

