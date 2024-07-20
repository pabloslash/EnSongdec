import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from ceciestunepipe.util.sound import spectral as sp
from scipy import signal


def compute_custom_spectrogram(x, fs, n_window=512, step_ms=1, f_min=250, f_max=8000, cut_off=0.0001):
    """
    Computes a spectrogram from an audio waveform according for the FALCON benchmark.
    To be consistent with the FALCON challenge, DO NOT MODIFY DEFAULTS.

    Args:
        x (np.ndarray): Input audio signal.
        fs (int): Sampling frequency of the audio signal.
        n_window (int, optional): Number of samples per window segment. Defaults to 512.
        step_ms (int, optional): Step size in milliseconds between successive windows. Defaults to 1.
        f_min (int, optional): Minimum frequency to be included in the output. Defaults to 250 Hz.
        f_max (int, optional): Maximum frequency to be included in the output. Defaults to 8000 Hz.
        cut_off (float, optional): Threshold relative to the maximum of the spectrum, below which values are set to 1. Defaults to 0.0001.

    Returns:
        tuple:
            - np.ndarray: Array of time bins for the spectrogram.
            - np.ndarray: Array of frequencies between `f_min` and `f_max`.
            - np.ndarray: Spectrogram of the frequencies between `f_min` and `f_max`.
    """
    # Overlap: Window size minus smples in a millisec
    msec_samples = int(fs * 0.001)
    n_overlap = n_window - msec_samples * step_ms
    sigma = 1 / 200. * fs

    # Spectrogram computation
    f, t, Sxx = signal.spectrogram(x, fs,
                                   nperseg=n_window,
                                   noverlap=n_overlap,
                                   window=signal.windows.gaussian(n_window, sigma),
                                   scaling='spectrum')

    if cut_off > 0:
        Sxx[Sxx < np.max((Sxx) * cut_off)] = 1
    Sxx[f<f_min, :] = 1

    return t, f[(f > f_min) & (f < f_max)], Sxx[(f > f_min) & (f < f_max)]
    

def plot_spectrogram(signal, sr, ax=None, n_window=512, step_ms=1, f_min=250, f_max=8000, cut_off=0.0001, xlabel=False):
    '''
    Plot spectrogram given a signal and sampling rate (and figure axis).
    '''
    
    t, f, sxx = compute_custom_spectrogram(signal, sr, n_window=n_window, step_ms=step_ms, f_min=f_min, f_max=f_max, cut_off=cut_off)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.pcolormesh(t, f, np.log(sxx), cmap='inferno', shading='auto')
    if xlabel:
        ax.set_xlabel('Time (s)', fontsize=35)
    return ax
    

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
        Convert signal to 16-bit format and save to a .wav file given its sampling rate.
    """
    wavfile.write(file_path, fs, np.int16(signal))
