import numpy as np
from numpy import pi as pi
import pandas as pd
import scipy.signal as sg
import seaborn as sns
import matplotlib.pyplot as plt
from benchmark_demo.utilstf import *
from benchmark_demo.SignalBank import SignalBank
from mpl_toolkits.axes_grid1 import ImageGrid
import cmocean


from methods.method_empty_space import empty_space_denoising
from methods.method_delaunay_triangulation import delaunay_triangulation_denoising

from benchmark_demo.SignalBank import SignalBank

if __name__ == "__main__":
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


    # A test for new methods
    np.random.seed(0) 

    # signal parameters
    SNRin = 20
    N = 2**9

    sbank = SignalBank(N=N)
    s = sbank.signal_mc_multi_linear()
    signal = add_snr(s,SNRin)

    output = delaunay_triangulation_denoising(signal, return_dic = True)
    s_r, mask_DT, tri, tri_select, zeros = ( output[key] for key in ('s_r','mask','tri','tri_select','zeros'))

    output = empty_space_denoising(signal,return_dic=True)
    s_r, mask_ES = (output[key] for key in ('s_r','mask')) 

    # titles = ('Harmonic','Delaunay Tri.','Empty Space')
    titles = ('Delaunay Tri.','Empty Space')

    Nfft = N
    g,T = get_round_window(Nfft)
    Lx = Nfft/T
    S, stft, stft_padded, Npad = get_spectrogram(signal, window = g)

    signals = (mask_DT, mask_ES)


    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1,2),  # creates 2x2 grid of axes
                    axes_pad=0.025,  # pad between axes in inch.
                    )
    k =1
    for signal, ax, title in zip(signals, grid, titles):
        if k==0:
            ax.imshow(np.log10(signal), origin='lower', cmap=cmocean.cm.deep)# cmap = plt.cm.jet)
        else:    
            ax.imshow(signal, origin = 'lower')# cmap = plt.cm.je
        k += 1
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
    # fig.set_size_inches((3.5,1.5))
    # plt.savefig('results/figure_masks.pdf',bbox_inches='tight')
