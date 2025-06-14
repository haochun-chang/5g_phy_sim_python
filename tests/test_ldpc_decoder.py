import numpy as np
from src.ldpc_encoder import LDPCEncoder
from src.ldpc_decoder import LDPCDecoder

def test_ldpc_decode_recovery():
    encoder = LDPCEncoder(n=64, d_v=2, d_c=4, seed=42)
    input_bits = np.random.randint(0, 2, encoder.k)
    codeword = encoder.encode(input_bits)
    
    EbN0_dB = 10
    EbN0 = 10 ** (EbN0_dB / 10)
    noise_var = 1 / (2 * EbN0)  # assume signal power 1

    # BPSK modulation (0 -> -1, 1 -> +1), then add noise
    tx = 2 * codeword - 1
    rx = tx + noise_var * np.random.randn(*tx.shape)
    llr = 2 * rx / noise_var  # approximate LLR
    llr = llr.T

    decoder = LDPCDecoder(encoder.G, encoder.H)
    decoded_bits = decoder.decode(llr, snr=EbN0_dB).T

    assert np.array_equal(input_bits, decoded_bits), "LDPC decoding failed to recover original bits"