import numpy as np
import scipy.stats as spst
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean
import scipy.signal as sg
import utilstf as utils
from benchmark_demo.SignalBank import SignalBank

N = 256

bank = SignalBank(N = 128)
signal = bank.signal_linear_chirp()

#signal = np.random.randn(N)


Ni = len(signal)
Npad = Ni//2

signal_pad = np.zeros(Ni+2*Npad)
signal_pad[Npad:Npad+Ni] = signal
Nfft = Ni

# analysis window
g = sg.gaussian(Nfft, np.sqrt((Nfft)/2/np.pi))
g = g/g.sum()

# computing STFT
_, _, stft = sg.stft(signal_pad, window=g, nperseg=Nfft, noverlap = Nfft-1)
Sww = np.abs(stft)**2
Sww = np.abs(stft[:,Npad:Npad+Ni])**2

fig,axs = plt.subplots(1,1)
axs.imshow(Sww, origin = 'lower')

plt.show()