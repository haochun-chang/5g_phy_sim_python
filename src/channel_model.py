import numpy as np

class ChannelModel:
    def __init__(self, snr_db: float, num_taps: int = 8, decay: float = 0.7, seed: int = None):
        self.snr_db = snr_db
        self.num_taps = num_taps
        self.decay = decay
        self.rng = np.random.default_rng(seed)
        self.h = self.generate_exponential_channel()

    def generate_exponential_channel(self):
        power_profile = np.exp(-self.decay * np.arange(self.num_taps))
        power_profile /= np.sum(power_profile)
        h = (self.rng.standard_normal(self.num_taps) + 1j * self.rng.standard_normal(self.num_taps))
        return h * np.sqrt(power_profile / 2)

    def apply(self, signal: np.ndarray) -> np.ndarray:
        return np.convolve(signal, self.h, mode='full')[:len(signal)]

    def add_awgn(self, signal: np.ndarray) -> np.ndarray:
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power/2) * (self.rng.standard_normal(len(signal)) + 1j * self.rng.standard_normal(len(signal)))
        return signal + noise