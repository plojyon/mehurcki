import numpy as np
import pywt


class ContinuousWaveletPreprocessor:
    """A preprocessor that applies wavelet transform."""
    display_name: str = "Continuous wavelet transform"

    def __init__(self, wavelet: str = "morl", level: int = 10, no_levels: int = 3):
        self.wavelet = wavelet
        self.level = level
        self.no_levels = no_levels

    def transform(self, sample):
        """Transform the sample."""
        # print(f"Applying wavelet transform on {type(sample)} of shape {sample.shape}")
        levels = np.arange(self.level - self.no_levels, self.level + 1)
        coeffs = pywt.cwt(sample, wavelet=self.wavelet, scales=levels)[0]
        # print(f"Wavelet transform result type: {type(coeffs)}, shape: {coeffs.shape}")
        return coeffs

    def transform_interval(self, interval):
        """Transform a specific interval of the sample."""
        return interval
