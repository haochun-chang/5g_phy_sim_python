import numpy as np
from commpy.modulation import QAMModem

class Modulator:
    def __init__(self, mod_order: int = 4):
        self.mod_order = mod_order
        self.k = int(np.log2(mod_order))
        self.modem = QAMModem(mod_order)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        assert bits.size % self.k == 0, f"Input bits length must be multiple of {self.k}"
        return self.modem.modulate(bits)

    def demodulate(self, symbols: np.ndarray, hard=True) -> np.ndarray:
        return self.modem.demodulate(symbols, demod_type='hard' if hard else 'soft')