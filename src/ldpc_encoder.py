import numpy as np
from scipy.sparse import csr_matrix
from pyldpc import make_ldpc, encode

class LDPCEncoder:
    def __init__(self, n: int = 128, d_v: int = 2, d_c: int = 4, seed: int = None):
        # n: codeword length
        # d_v: variable node degree, d_c: check node degree
        self.H, self.G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True, seed=seed)
        self.k = self.G.shape[1]  # message length
        self.n = n

    def encode(self, bits: np.ndarray) -> np.ndarray:
        assert bits.size == self.k, f"Input bits length must be {self.k}"
        codeword = encode(self.G, bits, snr=1000)   # 用pyldpc snr param為必填，先隨便填一個極大值
        codeword = (codeword + 1) // 2
        return codeword