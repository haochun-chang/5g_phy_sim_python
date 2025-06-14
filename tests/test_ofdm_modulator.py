import numpy as np
from src.ofdm_modulator import OFDMModulator

def test_ofdm_modulate_demodulate_identity():
    ofdm = OFDMModulator(n_subcarriers=64, cp_len=16)
    symbols = np.random.randn(5, 64) + 1j * np.random.randn(5, 64)
    time_signal = ofdm.modulate(symbols)
    recovered = ofdm.demodulate(time_signal)
    np.testing.assert_allclose(recovered, symbols, rtol=1e-5, atol=1e-5)

def test_ofdm_output_length():
    ofdm = OFDMModulator(64, 16)
    symbols = np.random.randn(10, 64) + 1j * np.random.randn(10, 64)
    time_signal = ofdm.modulate(symbols)
    expected_len = 10 * (64 + 16)
    assert len(time_signal) == expected_len