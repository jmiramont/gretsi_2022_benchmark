import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cmocean
from benchmark_demo.utilstf import *
from methods.spatstats_utils import *
from math import atan2
from benchmark_demo.SignalBank import SignalBank
import multiprocessing


# Graphics and plot
def label_line(line, label_text, near_i=None, near_x=None, near_y=None, rotation_offset=0, offset=(0,0)):
    """call
        l, = plt.loglog(x, y)
        label_line(l, "text", near_x=0.32)
    """
    def put_label(i, axis):
        """put label at given index"""
        i = min(i, len(x)-2)
        dx = sx[i+1] - sx[i]
        dy = sy[i+1] - sy[i]
        rotation = (np.rad2deg(atan2(dy, dx)) + rotation_offset)*0
        pos = [(x[i] + x[i+1])/2. + offset[0], (y[i] + y[i+1])/2 + offset[1]]
        axis.text(pos[0], pos[1], label_text, size=12, rotation=rotation, color = line.get_color(),
        ha="center", va="center", bbox = dict(ec='1',fc='1', alpha=1., pad=0))

    x = line.get_xdata()
    y = line.get_ydata()
    ax = line.axes
    if ax.get_xscale() == 'log':
        sx = np.log10(x)    # screen space
    else:
        sx = x
    if ax.get_yscale() == 'log':
        sy = np.log10(y)
    else:
        sy = y

    # find index
    if near_i is not None:
        i = near_i
        if i < 0: # sanitize negative i
            i = len(x) + i
        put_label(i, ax)
    elif near_x is not None:
        for i in range(len(x)-2):
            if (x[i] < near_x and x[i+1] >= near_x) or (x[i+1] < near_x and x[i] >= near_x):
                put_label(i, ax)
    elif near_y is not None:
        for i in range(len(y)-2):
            if (y[i] < near_y and y[i+1] >= near_y) or (y[i+1] < near_y and y[i] >= near_y):
                put_label(i, ax)
    else:
        raise ValueError("Need one of near_i, near_x, near_y")

def plotRankEnvRes(radius, k, t2, t2_exp): #, tinfty, t2_exp, tinfty_exp):
    lsize=16 # labelsize

    fig, ax = plt.subplots(figsize=(4, 4))

    lk, = ax.plot(radius, t2[k, :], color='g', alpha=1)
    ax.fill_between(radius, 0, t2[k, :], color='g', alpha=.8)
    label_line(lk, r'$t_k$', near_i=49, offset=(0.3, 0))
    t2_exp.resize((t2_exp.size,))
    lexp, = ax.plot(radius, t2_exp, color='k')
    label_line(lexp, r'$t_{\mathrm{exp}}$', near_i=49, offset=(0.3, 0))

    ax.set_ylabel(r'$T_2$' + r'$\mathrm{-statistic}$', fontsize=lsize)
    ax.set_xlabel(r'$r_{\mathrm{max}}$', fontsize=lsize)
    ax.set_xlim([0, 4])
    ax.set_yticks(np.linspace(0, 0.20, 5))
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    sns.despine(offset=10)
    fig.subplots_adjust(left=.25, right=0.9, bottom=0.2, top=0.97)


def par_loop(params):
    chirp, radius, statistic, pnorm, j = params
    print(j)
    output, radius = compute_mc_sim(chirp, Nfft, MC_reps = 199,
                                         statistic=('L','Frs', 'Fcs','Fkm'), 
                                         pnorm = pnorm, radius = radius)
    reject_H0 = np.zeros(tm.shape[1])
    reject_H0[np.where(t_exp > tm[k, :])] = 1
    # plotRankEnvRes(radius, k, tm, t_exp)
    # plt.show()
    return reject_H0


if __name__ == '__main__':
    N = 2**8
    SNRin = (0, 5, 10, 15, 20)
    Nfft = N
    sbank = SignalBank(N = N)
    chirp = sbank.signal_linear_chirp()
    # chirp = sbank.signal_mc_harmonic()
    
    g,_ = get_round_window(Nfft)
    S, stft, stft_padded, Npad = get_spectrogram(chirp, window = g)
    # plt.figure()
    # plt.imshow(S)
    # plt.show()

    radius = np.arange(0.0, 6.0, 0.020)
    # radius = np.linspace(0.0, 4.0)
    reps = 200
    output = np.zeros((reps,len(radius)))

    statistics_names = ()
    pnorms = {'2': 2, 'inf': np.inf}
    combinations_of_parameters = [(snr,pnorm) for pnorm in pnorms for snr in SNRin]
    print(combinations_of_parameters)

    for snr, pnorm in combinations_of_parameters:
        signal = add_snr(chirp,SNRin)
        print('outputmat_'+pnorm+'_SNRin_'+str(snr)+'.npy')
        par_params = [(signal, radius, pnorms[pnorm], i) for i in range(reps)]
        pool = multiprocessing.Pool(processes=12) 
        output = pool.map(par_loop, par_params) 
        pool.close() 
        pool.join()
        np.save('outputmat_'+pnorm+'_SNRin_'+str(snr)+'.npy', output)
        
        

    # output = np.load('outputmat_F_rs_inf.npy')
    # output = np.mean(output, axis=0)
    # plt.figure()
    # plt.plot(radius,output)
    
    #