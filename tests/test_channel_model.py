import numpy as np
from src.channel_model import ChannelModel

def test_channel_convolution_length():
    chan = ChannelModel(snr_db=20)
    sig = np.ones(100, dtype=complex)
    out = chan.apply(sig)
    assert len(out) == len(sig)

def test_awgn_noise_power():
    chan = ChannelModel(snr_db=10, seed=42)
    sig = np.ones(1000, dtype=complex)
    noisy = chan.add_awgn(sig)
    noise = noisy - sig
    measured_snr = 10 * np.log10(np.mean(np.abs(sig)**2) / np.mean(np.abs(noise)**2))
    assert 9 <= measured_snr <= 11, f"Measured SNR {measured_snr:.2f}dB not within expected range"