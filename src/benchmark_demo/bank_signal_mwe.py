import numpy as np
from numpy import pi as pi
# import pandas as pd
# import scipy.signal as sg
# import seaborn as sns
import matplotlib.pyplot as plt
# import cmocean
from benchmark_demo.utilstf import *
# from benchmark_demo.Benchmark import Benchmark
# from benchmark_demo.Benchmark import dic2df
from benchmark_demo.SignalBank import SignalBank

if __name__ == "__main__":
    N = 2**9
    signal_bank = SignalBank(N=N,)
    fmin = signal_bank.fmin
    fmax = signal_bank.fmax
    tmin = signal_bank.tmin
    tmax = signal_bank.tmax

    signals_dic = signal_bank.signalDict
    number_of_signals = len(signals_dic.keys())
    # nplots = int(np.ceil(np.sqrt(number_of_signals)))
    # print(nplots)
    fig, ax = plt.subplots(7, 3, figsize = (30,20))

    for i,signal_id in enumerate(signals_dic):
        signal = signals_dic[signal_id]()
        S, _, _, _ = get_spectrogram(signal)
        idx = np.unravel_index(i, ax.shape)
        # print(idx)
        ax[idx].imshow(S, origin = 'lower')
        ax[idx].set_title('signal_id = '+ signal_id)
        ax[idx].set_xticks([],[])
        # ax[idx].set_xlabel('time')
        ax[idx].set_yticks([])
        ax[idx].set_ylabel('frequency')

        ax[idx].plot([tmin, tmin],[fmin*N, fmax*N],'--w')
        ax[idx].plot([tmax, tmax],[fmin*N, fmax*N],'--w')
        ax[idx].plot([tmin, tmax],[fmin*N, fmin*N],'--w')
        ax[idx].plot([tmin, tmax],[fmax*N, fmax*N],'--w')

    plt.show()
