import numpy as np
from src.qam_modulator import Modulator

def test_modulate_demodulate_identity():
    mod = Modulator(mod_order=4)  # QPSK
    bits = np.random.randint(0, 2, 100 * mod.k)
    symbols = mod.modulate(bits)
    recovered = mod.demodulate(symbols, hard=True)
    assert np.array_equal(bits, recovered), "Demodulated bits should match original"

def test_modulator_shape():
    mod = Modulator(mod_order=16)
    bits = np.random.randint(0, 2, 160)
    symbols = mod.modulate(bits)
    assert symbols.shape[0] == 160 // mod.k
    assert np.iscomplexobj(symbols)