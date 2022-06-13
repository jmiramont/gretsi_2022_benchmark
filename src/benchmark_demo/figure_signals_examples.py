import numpy as np
from numpy import pi as pi
import pandas as pd
import scipy.signal as sg
import seaborn as sns
import matplotlib.pyplot as plt
from benchmark_demo.utilstf import *
from benchmark_demo.SignalBank import SignalBank
from mpl_toolkits.axes_grid1 import ImageGrid

if __name__ == "__main__":
    N = 2**9
    signal_bank = SignalBank(N=N)
    fmin = signal_bank.fmin
    fmax = signal_bank.fmax
    tmin = signal_bank.tmin
    tmax = signal_bank.tmax

    def set_size(w,h, ax=None):
        """ w, h: width, height in inches """
        if not ax: ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)

    signals_dic = signal_bank.signalDict
    number_of_signals = len(signals_dic.keys())
    # nplots = int(np.ceil(np.sqrt(number_of_signals)))
    # print(nplots)
    # signals = [signals_dic[signal_id]() for signal_id in ('LinearChirp', 'CosChirp', 'ExpChirp',
    #                                                     'McModulatedTones2', 'McOnOffTones', 'McCosPlusTone',
    #                                                     'McSyntheticMixture', 'McSyntheticMixture2', 'HermiteFunction')]

    signals = [signals_dic[signal_id]() for signal_id in ('CosChirp', 'McModulatedTones',
                                                        'McMultiLinear', 'McTripleImpulse')]

    titles = ['Cos. Chirp', 'Multi. Cos.', 'Multi. Linear', 'Impulses']


    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1,4),  # creates 2x2 grid of axes
                    axes_pad=0.025,  # pad between axes in inch.
                    )

    for signal, ax, title in zip(signals, grid, titles):
        window, _ = get_round_window(2*N)
        S, _, _, _ = get_spectrogram(signal, window=window)
        # print(idx)
        ax.imshow(S, origin = 'lower')# cmap = plt.cm.jet)
        ax.set_title(title, fontsize = '7')
        ax.set_xticks([],[])
        ax.set_xlabel('time', fontsize = '6')
        ax.set_yticks([])
        ax.set_ylabel('frequency', fontsize = '6')

        # ax.plot([tmin, tmin],[fmin*N, fmax*N],'w')
        # ax.plot([tmax, tmax],[fmin*N, fmax*N],'w')
        # ax.plot([tmin, tmax],[fmin*N, fmin*N],'w')
        # ax.plot([tmin, tmax],[fmax*N, fmax*N],'w')

    plt.show()
    # fig.set_size_inches((3.5,1))
    # plt.savefig('results/figure_signals_example.pdf',bbox_inches='tight')
