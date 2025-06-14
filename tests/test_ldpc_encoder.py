import numpy as np
from src.ldpc_encoder import LDPCEncoder

def test_ldpc_encoder_encode_length():
    encoder = LDPCEncoder(n=64, d_v=2, d_c=4, seed=42)
    input_bits = np.random.randint(0, 2, encoder.k)
    codeword = encoder.encode(input_bits)
    assert len(codeword) == encoder.n
    assert set(codeword).issubset({0, 1})

def test_ldpc_encoder_repeatability():
    encoder1 = LDPCEncoder(n=64, d_v=2, d_c=4, seed=1)
    encoder2 = LDPCEncoder(n=64, d_v=2, d_c=4, seed=1)
    bits = np.random.randint(0, 2, encoder1.k)
    out1 = encoder1.encode(bits)
    out2 = encoder2.encode(bits)
    assert np.array_equal(out1, out2), "LDPC encoding must be deterministic with same seed"