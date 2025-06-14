import numpy as np
from pyldpc import decode, get_message

class LDPCDecoder:
    def __init__(self, G, H, maxiter=50):
        self.G = G
        self.H = H
        self.maxiter = maxiter

    def decode(self, llr: np.ndarray, snr) -> np.ndarray:
        """
        llr: log-likelihood ratios of received bits
        return: decoded bit array
        """
        codeword = decode(self.H, llr, maxiter = self.maxiter, snr = snr)
        return get_message(self.G, codeword)