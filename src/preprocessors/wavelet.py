import numpy as np
import pywt


def upsample_to_length(c, factor, N):
    x = np.repeat(c, factor)
    if len(x) > N:
        return x[:N]
    elif len(x) < N:
        return np.pad(x, (0, N - len(x)))
    return x

class WaveletPreprocessor:
    """"A preprocessor that applies wavelet transform."""

    def __init__(self, wavelet: str = 'haar', level: int = 4):
        self.wavelet = wavelet
        self.level = level

    def transform(self, sample):
        """Transform the sample."""
        # print(f"Applying wavelet transform on {type(sample)} of shape {sample.shape}")
        coeffs = pywt.wavedec(sample, self.wavelet, level=self.level)
        # Upscale higher level coefficients to rectangularize the output
        rect = np.vstack([
            upsample_to_length(c, 2**k, len(sample))
            for k, c in enumerate(reversed(coeffs))
        ])
        # print(f"Wavelet transform result type: {type(rect)}, shape: {rect.shape}")
        return rect

    def transform_interval(self, interval):
        """Transform a specific interval of the sample."""
        return interval
