import numpy as np
from numpy import pi as pi
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from benchmark_demo.utilstf import *
from benchmark_demo.SignalBank import SignalBank
from methods.method_delaunay_triangulation import *
from methods.contours_utils import max_finder, zeros_finder

# A test for new methods
# np.random.seed(0) 

# signal parameters
SNRin = 0
N = 2**9

sbank = SignalBank(N=N, Nsub = 2**8)
s = sbank.signal_linear_chirp()
# s = sbank.signal_cos_chirp()
# s = sbank.signal_mc_double_cos_chirp()
# s = sbank.signal_mc_modulated_tones()
signal = add_snr(s,SNRin)

Nfft = len(signal)
g, T = get_round_window(Nfft)
stft, stft_padded, Npad = get_stft(signal,g)
margin = 2

S = np.abs(stft)
maxima = max_finder(S)
zeros = zeros_finder(S)
vertices = zeros/T # Normalize
delaunay_graph = Delaunay(zeros)
tri = delaunay_graph.simplices
sides, max_sides = counting_edges(delaunay_graph,vertices)

# for i,_ in enumerate(tri):
#     valid_tri[i] = np.all(valid_ceros[tri[i]])
#     side = max_sides[i]
#     selection[i] = np.any(LB < side) & np.all(UB > side) & valid_tri[i]


# # selection = np.where((LB < max_side) & (UB > max_side))
# # selection =  & valid_tri
# tri_select = tri[selection]
# mask = mask_triangles(stft, delaunay_graph, np.where(selection))  
# signal_r, t = reconstruct_signal_2(mask, stft_padded, Npad)

fig, ax = plt.subplots(1,2,figsize = (10,10))
ax[0].imshow((S), origin='lower')
ax[1].imshow((S), origin='lower')
ax[1].plot(zeros[:, 1], zeros[:, 0],'ro', ms = 1)
ax[1].plot(maxima[:, 1], maxima[:, 0],'y*')
# ax[1].triplot(zeros[:, 1], zeros[:, 0], tri, color = 'b')
plt.show()

