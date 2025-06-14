import numpy as np
from src.equalizer import Equalizer

def test_zf_equalizer():
    rx = np.array([2+2j, 4+4j])
    h = np.array([1+1j, 2+2j])
    eq = Equalizer('zf')
    eq_out = eq.equalize(rx, h)
    np.testing.assert_allclose(eq_out, np.array([2, 2]), atol=1e-6)

def test_mmse_equalizer():
    rx = np.array([2+2j])
    h = np.array([1+1j])
    noise_power = 0.01
    eq = Equalizer('mmse')
    out = eq.equalize(rx, h, noise_power=noise_power)
    assert out.shape == (1,)
    assert np.iscomplexobj(out)