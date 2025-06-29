# reweighting functions
import numpy as np
from scipy.fft import fft, ifft, fftfreq

def rweight(r, r0, delta=0.5):
    weight = 1 / 2 * (1 + np.tanh((r0 - r) / delta))
    return weight

def rweight_back(r, r0, delta=0.5):
    weight = 1 / 2 * (1 - np.tanh((r0 - r) / delta))
    return weight

def lowpass(U, r, cutoff):
    U_k = fft(U)
    freqs = fftfreq(len(U), r[1] - r[0])
    U_k[np.abs(freqs) > cutoff] = 0
    return np.real(ifft(U_k))