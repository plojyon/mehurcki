import numpy as np
import pywt


class ContinuousWaveletPreprocessor:
    """"A preprocessor that applies wavelet transform."""

    def __init__(self, wavelet: str = 'morl', level: int = 4):
        self.wavelet = wavelet
        self.level = level

    def transform(self, sample):
        """Transform the sample."""
        # print(f"Applying wavelet transform on {type(sample)} of shape {sample.shape}")
        coeffs = pywt.cwt(sample, wavelet=self.wavelet, scales=np.arange(1, self.level + 1))[0]
        # print(f"Wavelet transform result type: {type(coeffs)}, shape: {coeffs.shape}")
        return coeffs

    def transform_interval(self, interval):
        """Transform a specific interval of the sample."""
        return interval
