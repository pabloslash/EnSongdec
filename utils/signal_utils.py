import numpy as np
from ceciestunepipe.util.sound import spectral as sp


def resample_by_interpolation_2d(dat, input_fs, output_fs):
    """"
     Linearly resample each entry in a 2D-list from input_fs to output_fs.

     Input: dat[n x Samples]
     input_fs = original sampling frequency.
     output_fs: target sampling frequency.

     Output: [n x (re)Samples]
    """
    resampled_dat = []
    for i in range(np.array(dat).shape[0]):
        resampled_dat.append(resample_by_interpolation_1d(dat[i], input_fs, output_fs))

    return np.array(resampled_dat)
    

def resample_by_interpolation_1d(signal, input_fs, output_fs):
    """"
     Linearly resample an array from input_fs to output_fs.

     Input: signal[1 x Samples]
     input_fs = original sampling frequency.
     output_fs: target sampling frequency.

     Output: [1 x (re)Samples]
    """

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal


def resample_by_furier_1d(signal, num):
    """"
     Resample x to num samples using Fourier method along the given axis.

     Input: signal[1 x Samples]
     num: number. ofsamples in the resampled signal

     Output: [1 x (re)Samples]
    """
    return sp.signal.resample(x, num, t=None, axis=0, window=None, domain='time')


# def normalize_array_2d(X):
#     """"
#     Normalizes array X between 0 and 1.
#     """
#     for x in range()
#     return (x-np.min(x))/(np.max(x)-np.min(x))

def normalize_signal(x):
    """"
    Normalizes signal x between 0 and 1.
    """
    return (x-np.min(x))/(np.max(x)-np.min(x))