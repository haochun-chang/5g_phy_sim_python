import numpy as np
from scipy.interpolate import CubicSpline

class ChannelEstimator:
    def __init__(self, method='ls', interp='cubic', num_taps=8):
        self.method = method  # 'ls', 'mmse', or 'dft'
        self.interp = interp  # 'linear' or 'cubic'
        self.num_taps = num_taps

    def estimate(self, Yp, Xp, pilot_indices, n_subcarriers, noise_power=None, h_true=None):
        H_est = np.zeros(n_subcarriers, dtype=complex)

        if self.method == 'ls':
            H_est[pilot_indices] = Yp / Xp

        elif self.method == 'mmse':
            assert noise_power is not None and h_true is not None
            sigma_h2 = np.mean(np.abs(h_true)**2)
            lambda_reg = noise_power / sigma_h2
            H_est[pilot_indices] = np.conj(Xp) * Yp / (np.abs(Xp)**2 + lambda_reg)

        elif self.method == 'dft':
            H_ls_pilot = Yp / Xp
            h_time = np.fft.ifft(H_ls_pilot, n=len(pilot_indices))
            h_trunc = np.copy(h_time)
            h_trunc[self.num_taps:] = 0
            H_dft_pilot = np.fft.fft(h_trunc, n=len(pilot_indices))
            H_est[pilot_indices] = H_dft_pilot

        else:
            raise ValueError("Unsupported estimation method")

        if self.interp == 'linear':
            H_est = np.interp(np.arange(n_subcarriers), pilot_indices, H_est[pilot_indices])
        elif self.interp == 'cubic':
            H_est = CubicSpline(pilot_indices, H_est[pilot_indices])(np.arange(n_subcarriers))

        return H_est
