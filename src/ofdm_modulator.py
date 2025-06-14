import numpy as np

class OFDMModulator:
    def __init__(self, n_subcarriers=64, cp_len=16):
        self.n_subcarriers = n_subcarriers
        self.cp_len = cp_len

    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        symbols: (num_symbols, n_subcarriers) shape, complex
        Returns: time-domain signal with cyclic prefix, 1D array
        """
        num_symbols = symbols.shape[0]
        time_domain = np.fft.ifft(symbols, axis=1)
        with_cp = np.hstack([time_domain[:, -self.cp_len:], time_domain])
        return with_cp.reshape(-1)

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """
        signal: 1D time-domain signal with CP
        Returns: freq-domain symbols, shape (num_symbols, n_subcarriers)
        """
        symbol_len = self.n_subcarriers + self.cp_len
        num_symbols = len(signal) // symbol_len
        symbols = signal[:num_symbols * symbol_len].reshape((num_symbols, symbol_len))
        no_cp = symbols[:, self.cp_len:]
        return np.fft.fft(no_cp, axis=1)