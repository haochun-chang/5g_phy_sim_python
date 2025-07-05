import numpy as np
from commpy.modulation import QAMModem

class Modulator:
    def __init__(self, mod_order: int = 4):
        self.mod_order = mod_order
        self.num_bits_symbol = int(np.log2(mod_order))
        self.modem = QAMModem(mod_order)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        assert bits.size % self.num_bits_symbol == 0, f"Input bits length must be multiple of {self.num_bits_symbol}"
        return self.modem.modulate(bits)

    def demodulate(self, symbols: np.ndarray, hard=True, noise_var=None) -> np.ndarray:
        if hard:
            return self.modem.demodulate(symbols, demod_type='hard')
        else:
            assert noise_var is not None and noise_var > 0, "Need noise variance for soft decision"
            return self.demodulate_soft(symbols, noise_var=noise_var)

    # add patch to org commpy code
    def demodulate_soft(self, input_symbols, noise_var=0):
        # print(f"noise_var = {noise_var}")
        demod_bits = np.zeros(len(input_symbols) * self.num_bits_symbol)
        
        for i in range(len(input_symbols)):
            current_symbol = input_symbols[i]
            
            for bit_index in range(self.num_bits_symbol):
                llr_num = []  # list of log-likelihoods where bit = 1
                llr_den = []  # list of log-likelihoods where bit = 0
                
                for bit_value, symbol in enumerate(self.modem._constellation):
                    dist = abs(current_symbol - symbol) ** 2
                    log_metric = -dist / noise_var if noise_var > 0 else -dist  # avoid divide by 0
                    
                    if ((bit_value >> bit_index) & 1):
                        llr_num.append(log_metric)
                    else:
                        llr_den.append(log_metric)
                
                # Log-sum-exp trick for numerical stability
                def log_sum_exp(log_vals):
                    a_max = np.max(log_vals)
                    return a_max + np.log(np.sum(np.exp(log_vals - a_max)))
                
                llr_numerator = log_sum_exp(llr_num)
                llr_denominator = log_sum_exp(llr_den)
                
                llr = llr_numerator - llr_denominator  # log(P(bit=1)/P(bit=0))
                demod_bits[i * self.num_bits_symbol + self.num_bits_symbol - 1 - bit_index] = llr

        return demod_bits
