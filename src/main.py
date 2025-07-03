# main.py
import numpy as np
from bit_source import BitSource
from channel_estimator import ChannelEstimator
from channel_model import ChannelModel
from equalizer import Equalizer
from ldpc_decoder import LDPCDecoder
from ldpc_encoder import LDPCEncoder
from ofdm_modulator import OFDMModulator
from qam_modulator import Modulator

# 系統參數
mod_order = 4  # QPSK
n_subcarriers = 64
cp_len = 16
num_ofdm_symbols = 10 ** 4
pilot_spacing = 4
pilot_indices = np.arange(0, n_subcarriers, pilot_spacing)
data_indices = np.setdiff1d(np.arange(n_subcarriers), pilot_indices)
snr_db = 15

# 模組初始化
encoder = LDPCEncoder(n=64, d_v=2, d_c=4, seed=0)
H, G, k = encoder.H, encoder.G, encoder.k
# print(f'k = {k}')  # k = 33

# bit src len要滿足幾點條件: 
# 1. 必須是ldpc encoder k的倍數
# 2. ldpc encode之後，長度會變成n/k倍，此時必須是時域上一個symbol對應bit數mod_order * len(data_indices)的倍數
# 簡單的做法就是將num_ofdm_symbols變成k的倍數
num_ofdm_symbols = num_ofdm_symbols - (num_ofdm_symbols % k) if num_ofdm_symbols % k else num_ofdm_symbols
bit_source = BitSource(mod_order * len(data_indices) * num_ofdm_symbols)

modulator = Modulator(mod_order)
ofdm = OFDMModulator(n_subcarriers, cp_len)
channel = ChannelModel(snr_db=snr_db, num_taps=8, decay=0.7, seed=42)
estimator = ChannelEstimator(method='ls', interp='cubic', num_taps=8)
equalizer = Equalizer(method='mmse')
decoder = LDPCDecoder(G, H, 100)

# gen random bits
bits = bit_source.generate()

# LDPC 編碼
encoded = []
for i in range(0, len(bits), k):
    encoded.append(encoder.encode(bits[i:i+k]))
encoded = np.concatenate(encoded).astype(np.int64)
# print(type(encoded))
# print(encoded[:20])

# QAM 調變
symbols = modulator.modulate(encoded)
symbols = symbols.reshape((-1, len(data_indices)))
# print(symbols)
# print(symbols[0])

# gen zadoff-chu sequence for pilot data
def generate_zc_sequence(u, N_zc):
    n = np.arange(N_zc)
    zc_seq = np.exp(-1j * np.pi * u * n * (n + 1) / N_zc)
    return zc_seq
zc_seq = generate_zc_sequence(u=1, N_zc=len(pilot_indices))

# 插入 pilot並做ifft轉至時域
tx_signal = []
for i in range(num_ofdm_symbols):
    freq = np.zeros(n_subcarriers, dtype=complex)
    freq[data_indices] = symbols[i]
    freq[pilot_indices] = zc_seq
    time = np.fft.ifft(freq)
    tx_symbol = np.concatenate([time[-cp_len:], time])  # add cp
    tx_signal.append(tx_symbol)
tx_signal = np.concatenate(tx_signal)  # concat seperate symbols into a continual array

# 通道處理
rx_signal = channel.apply(tx_signal)  # through exponential PDP channel
rx_noisy = channel.add_awgn(rx_signal)  # add awgn noise
signal_power = np.mean(np.abs(rx_signal) ** 2)
noise_power = signal_power / (10 ** (snr_db / 10))

# ofdm 解調 --> 轉回頻域，shape (num_symbols, n_subcarriers)
rx_freq = ofdm.demodulate(rx_noisy)

# 通道估計並等化
rx_data = []
for i in range(num_ofdm_symbols):
    Yp = rx_freq[i, pilot_indices]
    Xp = zc_seq
    estimated_channel = estimator.estimate(Yp, Xp, pilot_indices, n_subcarriers)
    equalized = equalizer.equalize(rx_freq[i, data_indices], estimated_channel[data_indices], noise_power = noise_power)
    rx_data.append(equalized)
rx_symbols = np.concatenate(rx_data)

# QAM 解調 (soft decision)
# print(rx_symbols[:20])
rx_llr = modulator.demodulate(rx_symbols, hard=False, noise_var=noise_power / 2)
#rx_llr = modulator.demodulate(rx_symbols[:256], hard=False, noise_var=noise_power / 2)

# Debug: 檢查 LLR 統計量與範圍
#print(rx_llr[:20])
#print("LLR min/max/mean/std:", rx_llr.min(), rx_llr.max(), rx_llr.mean(), rx_llr.std())

# Clip 避免過大或過小值導致數值問題（根據 pyldpc 的建議範圍）
# rx_llr = np.clip(rx_llr, -20, 20)

# LDPC 解碼
decoded = []
for i in range(0, len(rx_llr), 64):
    decoded_bits = decoder.decode(rx_llr[i:i+64], snr=1000)
    decoded.append(decoded_bits)
decoded = np.concatenate(decoded)

# 計算 BER 並輸出比對前後
original = bits[:len(decoded)]
ber = np.mean(original != decoded)
print(original[:20])
print(decoded[:20])
print(f"BER: {ber:.4e}")
