import numpy as np
from numpy import pi as pi
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from benchmark_demo.utilstf import *
from benchmark_demo.SignalBank import SignalBank
from methods.method_delaunay_triangulation import delaunay_triangulation_denoising

# A test for new methods
# np.random.seed(0) 

# signal parameters
SNRin = 20
N = 2**9

sbank = SignalBank(N=N)
# s = sbank.signal_linear_chirp()
# s = sbank.signal_exp_chirp()
s = sbank.signal_mc_multi_linear()
# s = sbank.signal_cos_chirp()
# s = sbank.signal_mc_double_cos_chirp()
# s = sbank.signal_mc_modulated_tones()
# s = sbank.signal_mc_on_off_tones()
# s = sbank.signal_mc_synthetic_mixture()
# s = sbank.signal_mc_synthetic_mixture_2()
# s = sbank.signal_mc_impulses()

signal = add_snr(s,SNRin)

Nfft = N
g,T = get_round_window(Nfft)
Lx = Nfft/T
S, stft, stft_padded, Npad = get_spectrogram(signal, window = g)

output = delaunay_triangulation_denoising(signal, return_dic = True)

s_r, mask, tri, tri_select, zeros = ( output[key] for key in ('s_r','mask','tri','tri_select','zeros'))

print(10*np.log10((np.sum(s**2))/(np.sum((s-s_r)**2))))

# a_method_instance = instantiate_method()
# print(a_method_instance.get_method_id())
# xr= a_method_instance.method(signal)

print(S.shape)
fig, ax = plt.subplots(2,2,figsize = (10,10))
ax[0,0].imshow(np.log10(S), origin='lower', cmap=cmocean.cm.deep)
ax[0,1].imshow(np.log10(S), origin='lower', cmap=cmocean.cm.deep)
ax[0,1].triplot(zeros[:, 1], zeros[:, 0], tri, color = 'b')
ax[0,1].triplot(zeros[:, 1], zeros[:, 0], tri_select, color = 'g')
ax[1,0].imshow(mask, origin='lower')
plt.show()

