import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from ceciestunepipe.util.sound import spectral as sp


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
        
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = int((valid) // ss)
    out = np.ndarray((nw,ws),dtype = a.dtype)

    for i in range(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start : stop]

    return out


def stft(X, fftsize=128, step=65, mean_normalize=True, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()

    X = overlap(X, fftsize, step)
    
    size = fftsize
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X


def pretty_spectrogram(signal, log=True, thresh=5, fft_size=512, step_size=64):
    """
    creates a spectrogram
    log: take the log of the spectrgram
    thresh: threshold minimum power for log spectrogram
    """
    specgram = np.abs(stft(signal, fftsize=fft_size, step=step_size, real=False,
        compute_onesided=True))
  
    if log == True:
        specgram /= specgram.max() # volume normalize to max 1
        specgram = np.log10(specgram) # take log
        specgram[specgram < -thresh] = -thresh # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh # set anything less than the threshold as the threshold
    
    return specgram
    

def plot_spectrogram(signal, sr, ax=None, n_window=512, step_ms=1, f_min=1000, f_max=8000, cut_off=5.5e-05):
    '''
    Plot spectrogram given a signal and sampling rate (and figure axis).
    '''
    
    f, tm, sxx = sp.ms_spectrogram(signal, sr, n_window=n_window, step_ms=step_ms, f_min=f_min, f_max=f_max, cut_off=cut_off)

    if ax == None:
        fig, ax = plt.subplots()
        
    ax.pcolormesh(tm, f, np.log(sxx), cmap='inferno')
    ax.set_xlabel('time (s)', fontsize=20)
    # ax.set_ylabel('f (kHz)', fontsize=20)
    # ax.set_yticklabels(['1', '2', '4', '6', '8'], fontsize=14)
    ax.set_yticklabels([])


def plot_spectrogram_sxx(sxx, f=None, tm=None, ax=None):
    """
        Plot spectrogram given a signal converted to its frequecy response (timestamps, frequencies, sxx).
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    if f is None:
        print('vector of frequencies not provided, plotting linearly...')
        f = list(range(sxx.shape[0]))   
        
    if tm is None:
        print('timestamps not provided, plotting linearly...')
        tm = list(range(sxx.shape[1]))
         
    cmes = ax.pcolormesh(tm, f, np.log(sxx), cmap='inferno')
    cbar = plt.colorbar(cmes, ax=ax)
    ax.set_ylabel('f (kHz)', fontsize=20)
    ax.set_yticklabels(['1', '2', '4', '6', '8'], fontsize=14)
    
    
def save_signal_as_wav(signal, fs, file_path='test.wav'):
    """
        Save a signal to a .wav file given its sampling rate.
    """
    wavfile.write(file_path, fs, signal)
