import numpy as np
from scipy.interpolate import CubicSpline

class ChannelEstimator:
    def __init__(self, method='ls', interp='cubic', num_taps=8, adaptive_tap = False):
        self.method = method  # 'ls', 'mmse', or 'dft'
        self.interp = interp  # 'linear' or 'cubic'
        self.num_taps = num_taps
        self.adaptive_tap = adaptive_tap

    def estimate(self, Yp, Xp, pilot_indices, n_subcarriers, noise_power=None):
        H_est = np.zeros(n_subcarriers, dtype=complex)

        if self.method == 'ls':
            H_est[pilot_indices] = Yp / Xp

        elif self.method == 'mmse':
            assert noise_power is not None
            H_ls = Yp / Xp
            sigma_h2 = np.mean(np.abs(H_ls) ** 2)
            lambda_reg = noise_power / sigma_h2
            H_est[pilot_indices] = np.conj(Xp) * Yp / (np.abs(Xp)**2 + lambda_reg)

        elif self.method == 'dft':
            # LS on pilot positions
            H_est[pilot_indices] = Yp / Xp

            # Interpolation
            if self.interp == 'linear':
                H_est = np.interp(np.arange(n_subcarriers), pilot_indices, H_est[pilot_indices])
            elif self.interp == 'cubic':
                H_est = CubicSpline(pilot_indices, H_est[pilot_indices])(np.arange(n_subcarriers))

            # IFFT to time domain (pilot-limited)
            h_time = np.fft.ifft(H_est, n=n_subcarriers)

            # Energy-based adaptive tap truncation
            power = np.abs(h_time) ** 2
            total_power = np.sum(power)
            cum_power = np.cumsum(power)
            threshold = 0.99
            adaptive_taps = np.searchsorted(cum_power / total_power, threshold) + 1

            if self.adaptive_tap:
                # print(f"true tap num: {self.num_taps}, adaptive tap num: {adaptive_taps}")
                num_taps = adaptive_taps
            else:
                num_taps = self.num_taps

            # Soft truncation: hard cut-off equals to convolve sinc func in freq domain, which induces freq domain side-lobes. 
            # Use window func to cut-off to reduce first side-lobe level
            tap_len = min(num_taps, len(h_time))
            # window = np.hamming(len(h_time))
            # h_time[:tap_len] *= window
            h_time[tap_len:] = 0  # hard cut-off

            # FFT back to frequency domain
            H_est = np.fft.fft(h_time, n=n_subcarriers)

        else:
            raise ValueError("Unsupported estimation method")

        # Interpolation
        if self.method != "dft":
            if self.interp == 'linear':
                H_est = np.interp(np.arange(n_subcarriers), pilot_indices, H_est[pilot_indices])
            elif self.interp == 'cubic':
                H_est = CubicSpline(pilot_indices, H_est[pilot_indices])(np.arange(n_subcarriers))

        return H_est
