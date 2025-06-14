import numpy as np

class BitSource:
    def __init__(self, num_bits: int, seed: int = None):
        self.num_bits = num_bits
        self.rng = np.random.default_rng(seed)

    def generate(self) -> np.ndarray:
        """產生隨機 bitstream"""
        return self.rng.integers(0, 2, self.num_bits, dtype=np.uint8)
