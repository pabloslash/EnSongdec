# General imports
import os
import glob
import random
import copy
import neo
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import interp1d
import warnings

# For audio
from scipy.io import wavfile
import librosa
import librosa.display

# For data handling
import numpy as np
import pandas as pd
import quantities as pq
import json 
import pickle as pkl
from importlib import reload
import pathlib
import typing as tp

# For plotting
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Utils
import audio_utils as au
import encodec_utils as eu

# Import SONGBIRDCORE
import songbirdcore as sbc
from songbirdcore import speech_bci_struct as fs
import songbirdcore.spikefinder.spike_analysis_helper as sh
import songbirdcore.spikefinder.filtering_helper as fh
import songbirdcore.util.audio_spectrogram_utils as auts
import songbirdcore.util.plot_utils as puts
import songbirdcore.util.label_utils as luts
import songbirdcore.dataframes_package as dfpkg
# Songbirdcore para,s
from songbirdcore.params import BirdSpecificParams as BSP
from songbirdcore.params import GlobalParams as params
# My state space analysis models
from songbirdcore.statespace_analysis.gpfa_songbirdcore import GPFACore
from songbirdcore.statespace_analysis.pca_songbirdcore import PCACore
from songbirdcore.statespace_analysis.statespace_analysis_utils import convert_to_neo_spike_trains, convert_to_neo_spike_trains_3d
from songbirdcore.statespace_analysis.statespace_analysis_utils import permute_array_rows_independently, permute_array_cols_independently
from songbirdcore.style_params import syl_colors, style_dict

# Tim S. noise reduce
import noisereduce as nr

# EncoDec
from encodec import EncodecModel
from encodec.utils import convert_audio

# Torch
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Other
import math
from collections import Counter

from ceciestunepipe.util.sound import spectral as sp