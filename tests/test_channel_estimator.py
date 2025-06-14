import numpy as np
from src.channel_estimator import ChannelEstimator

def test_ls_estimation():
    pilots = np.array([1+1j, 2+2j, 3+3j, 4+4j])
    tx_pilots = np.array([1, 1, 1, 1])
    pilot_indices = np.array([0, 4, 8, 12])
    estimator = ChannelEstimator(method='ls', interp='linear')
    H = estimator.estimate(pilots, tx_pilots, pilot_indices, 16)
    assert H.shape == (16,)
    assert np.iscomplexobj(H)