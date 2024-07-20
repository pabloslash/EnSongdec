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


def load_model_statedict_align_complimentary_space(models_checkpoints_dir, dataset_dir, model_filename, shuffle_inputs=False):
    
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
    
    # Extract data and align spaces
    if neural_mode == 'RAW' or neural_mode == 'TX':
        neural_original_traces = data_dict['neural_dict']['original_neural_traces'][neural_key]
        neural_complimentary_traces = data_dict['neural_dict']['complimentary_neural_traces'][neural_key]
    elif neural_mode == 'TRAJECTORIES':
        latent_mode = experiment_metadata['latent_mode']
        dimensionality = experiment_metadata['dimensionality']
        neural_original_traces = data_dict['neural_dict']['original_neural_traces'][neural_key][latent_mode][neural_key+'_dim'+str(dimensionality)]['trajectories']
        neural_complimentary_traces = data_dict['neural_dict']['complimentary_neural_traces'][neural_key][latent_mode][neural_key+'_dim'+str(dimensionality)]['trajectories']
    audio_motifs = data_dict['audio_motifs']
    fs_audio = data_dict['fs_audio']
    fs_neural = data_dict['fs_neural']

    tu.check_experiment_duration(neural_original_traces, audio_motifs, fs_neural, fs_audio)
    
    # --------- FIT CCA: ALIGN NEURAL SPACES --------- #
    print(f'Fitting CCA model to align neural spaces.')
    # Use neural trajetcories/spiketrains around stereotyped motif (skipping t_pre and t_post) to fit CCA model to dimensions
    n_dims = neural_original_traces.shape[1]
    print('n_dims: ', n_dims)

    # Reshape to fit model to all concatenated samples only in the stereotyped part of the motif
    from_samples = int(data_dict['t_pre'] * data_dict['fs_neural'])
    to_samples = int(data_dict['t_post'] * data_dict['fs_neural'])
    reshaped_original_traces = neural_original_traces[:, :, from_samples:to_samples].transpose(1, 2, 0).reshape(n_dims, -1).T
    reshaped_complimentary_traces = neural_complimentary_traces[:, :, from_samples:to_samples].transpose(1, 2, 0).reshape(n_dims, -1).T
    print(reshaped_original_traces.shape, reshaped_complimentary_traces.shape)

    # Fit CCA
    print('fitting CCA model')
    cca = fit_CCA_model(reshaped_original_traces, reshaped_complimentary_traces, n_components=n_dims)
    print('CCA fitted!')
    # Align complimentary trajectories to original trajectorie's space
    neural_array = np.array([Y_transform_to_X(Y.T, cca).T for Y in neural_complimentary_traces])

    tu.check_experiment_duration(neural_original_traces, audio_motifs, fs_neural, fs_audio)
    
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
    