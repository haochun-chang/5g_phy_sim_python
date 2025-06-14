import numpy as np

class Equalizer:
    def __init__(self, method='zf'):
        assert method in ['zf', 'mmse'], "Unsupported equalization method"
        self.method = method

    def equalize(self, received: np.ndarray, channel: np.ndarray, noise_power: float = None) -> np.ndarray:
        if self.method == 'zf':
            return received / channel

        elif self.method == 'mmse':
            assert noise_power is not None, "Noise power required for MMSE equalizer"
            return (np.conj(channel) * received) / (np.abs(channel)**2 + noise_power)