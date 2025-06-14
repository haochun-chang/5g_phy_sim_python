import numpy as np
from src.bit_source import BitSource

def test_bit_source_fixed_seed():
    source1 = BitSource(10, seed=42)
    source2 = BitSource(10, seed=42)
    assert np.array_equal(source1.generate(), source2.generate()), "Same seed should produce same output"

def test_bit_source_random():
    source = BitSource(100)
    bits = source.generate()
    assert bits.dtype == np.uint8
    assert bits.shape == (100,)
    assert set(bits).issubset({0, 1}), "Bits must be 0 or 1 only"