import os
import sys
import json
from matplotlib.backends.backend_pdf import PdfPages
import pickle as pkl
import numpy as np
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter, gaussian_filter1d

def add_to_sys_path(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        sys.path.append(dirpath)        
root_dir = '/home/jovyan/pablo_tostado/bird_song/enSongDec/'
add_to_sys_path(root_dir)

# sklearn
from sklearn.metrics import mean_squared_error

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ensongdec
from FFNNmodel import FeedforwardNeuralNetwork, ffnn_predict
from neural_audio_dataset import NeuralAudioDataset
import utils.audio_utils as au
import utils.signal_utils as su
import utils.encodec_utils as eu
import utils.train_utils as tu
from utils.evaluation_utils import load_experiment_metadata, load_model

# songbirdcore
import songbirdcore.spikefinder.spike_analysis_helper as sh
import songbirdcore.spikefinder.filtering_helper as fh
import songbirdcore.utils.label_utils as luts
import songbirdcore.utils.plot_utils as puts
import songbirdcore.utils.audio_spectrogram_utils as auts

# EncoDec
from encodec import EncodecModel
from encodec.utils import convert_audio

# Tim S. noise reduce
import noisereduce as nr

# CCA alignment
from cca_utilities import *



def load_model_statedict(models_checkpoints_dir, dataset_dir, model_filename, shuffle_inputs=False):
    
    # --------- LOAD EXPERIMENT --------- #
    experiment_metadata = load_experiment_metadata(models_checkpoints_dir, model_filename)
    
    # Experiment params
    dataset_filename = experiment_metadata['dataset_filename']
    train_idxs = experiment_metadata['train_idxs']
    test_idxs = experiment_metadata['test_idxs']
    model_layers = experiment_metadata['layers']
    neural_mode = experiment_metadata['neural_mode']
    neural_key = experiment_metadata['neural_key']    
    neural_history_ms = experiment_metadata["neural_history_ms"]
    gaussian_smoothing_sigma = experiment_metadata["gaussian_smoothing_sigma"]    
    max_temporal_shift_ms = experiment_metadata["max_temporal_shift_ms"]
    noise_level = experiment_metadata["noise_level"]
    transform_probability = experiment_metadata["transform_probability"]    
    dropout_prob = experiment_metadata["dropout_prob"]
    batch_size = experiment_metadata["batch_size"]
    learning_rate = experiment_metadata["learning_rate"]
    
    print('Loaded model: train_idxs: ', train_idxs, ' - test_ixs: ', test_idxs, ' - model_layers: ', model_layers)    

    # --------- LOAD NEURAL & AUDIO DATA --------- #
    # Open the file in binary read mode ('rb') and load the content using pickle
    with open(dataset_dir + dataset_filename + '.pkl', 'rb') as file:
        data_dict = pkl.load(file)
    
    # Extarct data from dataset
    if neural_mode == 'RAW' or neural_mode == 'TX':
        neural_array = data_dict['neural_dict'][neural_key]
    elif neural_mode == 'TRAJECTORIES':
        try:
            latent_mode = experiment_metadata.get('latent_mode', 'gpfa') # GPFA by default
            dimensionality = experiment_metadata.get('dimensionality', 12) # 12 latent dimensions by default
            neural_array = data_dict['neural_dict'][neural_key][latent_mode][neural_key+'_dim'+str(dimensionality)]['trajectories']
        # Legacy (tofix): Old datasets have different data format for trajectories
        except: 
            neural_array = data_dict['neural_dict'][neural_key]
    audio_motifs = data_dict['audio_motifs']
    fs_audio = data_dict['fs_audio']
    fs_neural = data_dict['fs_neural']

    tu.check_experiment_duration(neural_array, audio_motifs, fs_neural, fs_audio)

    # --------- PROCESS AUDIO --------- #
    audio_motifs = tu.preprocess_audio(audio_motifs, fs_audio)

    # --------- INSTANTIATE ENCODEC --------- #
    encodec_model = eu.instantiate_encodec()
    # Embed motifs (warning: slow!)
    audio_embeddings, audio_codes, scales = eu.encodec_encode_audio_array_2d(audio_motifs, fs_audio, encodec_model)
    
    # --------- PROCESS NEURAL --------- #
    # Resample neural data to match audio embeddings
    original_samples = neural_array.shape[-1]
    target_samples = audio_embeddings.shape[-1]

    print(f'WARNING: Neural samples should be greater than embedding samples! Downsampling neural data from {original_samples} samples to match audio embedding samples {target_samples}.')
    neural_array = tu.process_neural_data(neural_array, neural_mode, gaussian_smoothing_sigma, original_samples, target_samples)
    
    bin_length = ((original_samples / fs_neural)*1000) / neural_array.shape[-1] # ms
    history_size = int(neural_history_ms // bin_length) # Must be minimum 1
    print('Using {} bins of neural data history.'.format(history_size))


    # --------- PREPARE DATALOADERS --------- #   
    train_neural = neural_array[train_idxs]  
    train_audio = audio_embeddings[train_idxs]
    test_neural = neural_array[test_idxs]  
    test_audio = audio_embeddings[test_idxs]

    if shuffle_inputs:
        print(f'CONTROL: Shuffling model inputs in the test set along the second dimension!')
        test_neural = np.apply_along_axis(np.random.permutation, axis=2, arr=test_neural)
    
    # Create dataset objects
    max_temporal_shift_bins = int(max_temporal_shift_ms // bin_length) # Temporal jitter for data augmentation

    train_dataset, train_dataloader = tu.prepare_dataloader(train_neural, 
                                                         train_audio, 
                                                         batch_size, 
                                                         history_size, 
                                                         max_temporal_shift_bins=max_temporal_shift_bins,
                                                         noise_level=noise_level,
                                                         transform_probability=transform_probability, 
                                                         shuffle_samples = True)
    
    test_dataset, test_loader = tu.prepare_dataloader(test_neural, 
                                                   test_audio, 
                                                   batch_size, 
                                                   history_size, 
                                                   max_temporal_shift_bins=0,
                                                   noise_level=0,
                                                   transform_probability=0, 
                                                   shuffle_samples = False)

    print('Train samples: ', len(train_dataset), 'Test samples: ', len(test_dataset))
    

     # --------- LOAD MODEL & OPTIMIZER STATE DICT --------- #
    ffnn_model, optimizer = load_model(models_checkpoints_dir, model_filename, model_layers, learning_rate)
    
    return ffnn_model, test_dataset, test_loader, fs_audio, encodec_model, scales




def plot_spectrogram(audio, fs_audio, plot_samples, ax=None, xlabel=True, ylabel=True):
    
    if ax == None:
        fig, ax = plt.subplots(nrows=1, figsize=(10, 3))
    auts.plot_spectrogram(audio[:plot_samples], fs_audio, ax, f_min=500, f_max=8500)
    
    if xlabel:
        ax.set_xlabel('time (s)', fontsize=30)
        ax.tick_params(axis='x', labelsize=25)    
    else:
        ax.set_xlabel('')  
        ax.set_xticks([])  
    if ylabel:
        ax.set_ylabel('f (kHz)', fontsize=30)
        ax.set_yticklabels(['0', '2', '4', '6', '8'], fontsize=25)
    else:
        ax.set_ylabel('')  
        ax.set_yticks([])  