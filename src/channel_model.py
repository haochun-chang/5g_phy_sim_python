import numpy as np

class ChannelModel:
    ITU_MODELS = {
        'epa': {
            'delays': np.array([0, 30, 70, 90, 110, 190, 410]) * 1e-9,  # seconds
            'powers': np.array([0, -1, -2, -3, -8, -17.2, -20.8])      # dB
        },
        'eva': {
            'delays': np.array([0, 30, 150, 310, 370, 710, 1090, 1730, 2510]) * 1e-9,
            'powers': np.array([0, -1.5, -1.4, -3.6, -0.6, -9.1, -7, -12, -16.9])
        },
        'etu': {
            'delays': np.array([0, 50, 120, 200, 230, 5000]) * 1e-9,
            'powers': np.array([0, -1, -2, -3, -8, -17])
        }
    }

    def __init__(self, snr_db: float, fs: float = 15.36e6, model_type: str = 'exponential', num_taps: int = 8,
                 decay: float = 0.7, seed: int = None):
        self.snr_db = snr_db
        self.fs = fs  # sampling rate
        self.model_type = model_type
        self.num_taps = num_taps
        self.decay = decay
        self.rng = np.random.default_rng(seed)

        if model_type == 'exponential':
            self.h = self.generate_exponential_channel()
        elif model_type in self.ITU_MODELS:
            self.h, self.delay_samples = self.generate_itu_channel(model_type)
        else:
            raise ValueError(f"Unsupported channel model type: {model_type}")

    def generate_exponential_channel(self):
        power_profile = np.exp(-self.decay * np.arange(self.num_taps))
        power_profile /= np.sum(power_profile)
        h = (self.rng.standard_normal(self.num_taps) + 1j * self.rng.standard_normal(self.num_taps))
        return h * np.sqrt(power_profile / 2)

    def generate_itu_channel(self, model_name):
         # Load the delay and power profile from the ITU model
        model = self.ITU_MODELS[model_name]
        # Convert power profile from dB to linear scale
        powers = 10 ** (model['powers'] / 10)
        # Normalize total power to 1 to preserve total energy
        powers /= np.sum(powers)
        # Delays are originally in seconds (e.g., 30 ns), convert to integer sample index
        delays = model['delays']
        delay_samples = np.round(delays * self.fs).astype(int)
        # Find the maximum sample delay to determine the length of the channel vector
        max_delay = delay_samples.max()

        h = np.zeros(max_delay + 1, dtype=complex)
        # For each tap, generate a complex Gaussian sample with the corresponding power
        for i, d in enumerate(delay_samples):
            # Rayleigh fading: real and imaginary parts are drawn from N(0, 1)
            tap = (self.rng.standard_normal() + 1j * self.rng.standard_normal()) * np.sqrt(powers[i] / 2)
            # Assign the tap to the appropriate delayed sample location
            h[d] += tap

        return h, delay_samples

    def apply(self, signal: np.ndarray) -> np.ndarray:
        return np.convolve(signal, self.h, mode='full')[:len(signal)]

    def add_awgn(self, signal: np.ndarray) -> np.ndarray:
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power/2) * (self.rng.standard_normal(len(signal)) + 1j * self.rng.standard_normal(len(signal)))
        return signal + noise
