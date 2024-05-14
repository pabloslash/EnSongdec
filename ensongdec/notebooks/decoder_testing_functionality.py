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

    check_experiment_duration(neural_array, audio_motifs, fs_neural, fs_audio)

    # --------- PROCESS AUDIO --------- #
    audio_motifs = preprocess_audio(audio_motifs, fs_audio)

    # --------- INSTANTIATE ENCODEC --------- #
    encodec_model = instantiate_encodec()
    # Embed motifs (warning: slow!)
    audio_embeddings, audio_codes, scales = eu.encodec_encode_audio_array_2d(audio_motifs, fs_audio, encodec_model)
    
    # --------- PROCESS NEURAL --------- #
    # Resample neural data to match audio embeddings
    original_samples = neural_array.shape[-1]
    target_samples = audio_embeddings.shape[-1]

    print(f'WARNING: Neural samples should be greater than embedding samples! Downsampling neural data from {original_samples} samples to match audio embedding samples {target_samples}.')
    neural_array = process_neural_data(neural_array, neural_mode, gaussian_smoothing_sigma, original_samples, target_samples)
    
    bin_length = ((original_samples / fs_neural)*1000) / neural_array.shape[-1] # ms
    history_size = int(neural_history_ms // bin_length) # Must be minimum 1
    print('Using {} bins of neural data history.'.format(history_size))


    # --------- PREPARE DATALOADERS --------- #
    max_temporal_shift_bins = int(max_temporal_shift_ms // bin_length) # Temporal jitter for data augmentation
    train_dataset, test_dataset, train_loader, test_loader = prepare_dataloaders(neural_array, 
                                                                                 audio_embeddings, 
                                                                                 train_idxs, 
                                                                                 test_idxs, 
                                                                                 batch_size, 
                                                                                 history_size, 
                                                                                 max_temporal_shift_bins=max_temporal_shift_bins, 
                                                                                 noise_level=noise_level, 
                                                                                 transform_probability=transform_probability, 
                                                                                 shuffle_inputs=shuffle_inputs)

     # --------- LOAD MODEL & OPTIMIZER STATE DICT --------- #
    ffnn_model, optimizer = load_model(models_checkpoints_dir, model_filename, model_layers, learning_rate)
    
    return ffnn_model, test_dataset, test_loader, fs_audio, encodec_model, scales
    


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

    check_experiment_duration(neural_original_traces, audio_motifs, fs_neural, fs_audio)
    
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

    check_experiment_duration(neural_original_traces, audio_motifs, fs_neural, fs_audio)
    
    # --------- PROCESS AUDIO --------- #
    audio_motifs = preprocess_audio(audio_motifs, fs_audio)

    # --------- INSTANTIATE ENCODEC --------- #
    encodec_model = instantiate_encodec()
    # Embed motifs (warning: slow!)
    audio_embeddings, audio_codes, scales = eu.encodec_encode_audio_array_2d(audio_motifs, fs_audio, encodec_model)
    
    
    # --------- PROCESS NEURAL --------- #
    # Resample neural data to match audio embeddings
    original_samples = neural_array.shape[-1]
    target_samples = audio_embeddings.shape[-1]

    print(f'WARNING: Neural samples should be greater than embedding samples! Downsampling neural data from {original_samples} samples to match audio embedding samples {target_samples}.')
    neural_array = process_neural_data(neural_array, neural_mode, gaussian_smoothing_sigma, original_samples, target_samples)
    
    bin_length = ((original_samples / fs_neural)*1000) / neural_array.shape[-1] # ms
    history_size = int(neural_history_ms // bin_length) # Must be minimum 1
    print('Using {} bins of neural data history.'.format(history_size))
    
    # --------- PREPARE DATALOADERS --------- #
    max_temporal_shift_bins = int(max_temporal_shift_ms // bin_length) # Temporal jitter for data augmentation
    train_dataset, test_dataset, train_loader, test_loader = prepare_dataloaders(neural_array, 
                                                                                 audio_embeddings, 
                                                                                 train_idxs, 
                                                                                 test_idxs, 
                                                                                 batch_size, 
                                                                                 history_size, 
                                                                                 max_temporal_shift_bins=max_temporal_shift_bins, 
                                                                                 noise_level=noise_level, 
                                                                                 transform_probability=transform_probability, 
                                                                                 shuffle_inputs=shuffle_inputs)
    
     # --------- LOAD MODEL & OPTIMIZER STATE DICT --------- #
    ffnn_model, optimizer = load_model(models_checkpoints_dir, model_filename, model_layers, learning_rate)
    
    return ffnn_model, test_dataset, test_loader, fs_audio, encodec_model, scales
    


def load_experiment_metadata(directory, filename):
    """
    Loads the experiment metadata from a JSON file in the specified directory that matches the filename.
    Args:
        directory (str): The directory containing the metadata files.
        filename (str): The base name of the metadata file to find, without its extension.
    Returns:
        dict: A dictionary containing the metadata loaded from the JSON file.
    """
    metadata_extension = '.json'
    metadata_files = [file for file in os.listdir(directory) if file.endswith(metadata_extension)]
    metadata_file = next((file for file in metadata_files if filename[:-3] in file), None)
    print(f'Loading {metadata_file}')
    
    with open(os.path.join(directory, metadata_file), 'rb') as file:
        return json.load(file)
        

def preprocess_audio(audio_data, fs_audio, filter_path):
    """
    Applies a series of preprocessing steps to audio data.

    Args:
        audio_data (np.array): The raw audio data.
        fs_audio (int): The sampling frequency of the audio data.
        filter_path (str): Path to the filter coefficients file used for audio filtering.

    Returns:
        list: A list of noise-reduced audio motifs.
    """
    b, a = fh.load_filter_coefficients_matlab(filter_path)
    filtered_audio = fh.noncausal_filter_2d(audio_data, b=b, a=a)
    noise_reduced_audio = [nr.reduce_noise(motif, sr=fs_audio) for motif in filtered_audio]
    return noise_reduced_audio


def check_experiment_duration(neural_array, audio_motifs, fs_neural, fs_audio):
    # Calculate the duration of the experiment based on the last dimension of the arrays and their sampling rates
    trial_length_neural = (neural_array.shape[-1] / fs_neural)*1000
    trial_length_audio = (audio_motifs.shape[-1] / fs_audio)*1000
    
    # Check the durations of the neural/audio data are equal and raise a warning if they aren't
    print('Length of neural trials: {} ms, length of audio trials: {} ms. '.format(trial_length_neural, trial_length_audio))
    if trial_length_neural != trial_length_audio:
        warnings.warn("WARNING: Neural data duration and audio motifs duration are different in this dataset!")


def preprocess_audio(audio_data, fs_audio):
    """
    Applies a series of preprocessing steps to audio data.

    Args:
        audio_data (np.array): The raw audio data.
        fs_audio (int): The sampling frequency of the audio data.
        filter_path (str): Path to the filter coefficients file used for audio filtering.

    Returns:
        list: A list of noise-reduced audio motifs.
    """
    filter_path = '/home/jovyan/pablo_tostado/repos/songbirdcore/songbirdcore/filters/butter_bp_250Hz-8000hz_order4_sr25000.mat'
    b, a = fh.load_filter_coefficients_matlab(filter_path)
    filtered_audio = fh.noncausal_filter_2d(audio_data, b=b, a=a)
    noise_reduced_audio = [nr.reduce_noise(motif, sr=fs_audio) for motif in filtered_audio]
    return noise_reduced_audio
    

def instantiate_encodec():
    """
    Instantiates a model of the Encodec with preset configurations.
    Returns:
        EncodecModel: A configured Encodec model instance.
    """
    model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(24.0)
    return model


def process_neural_data(neural_array, neural_mode, sigma, original_samples, target_samples):
    """
    Processes neural data array based on the specified mode.
    Args:
        neural_array (np.array): The array of neural data.
        neural_mode (str): The processing mode ('RAW', 'TX' or 'TRAJECTORIES').
        sigma (float): The sigma value for Gaussian filtering, used in 'RAW' mode.
        resample_params (dict): Parameters for resampling, including 'history_size', 'orig_samples', and 'target_samples'.
    Returns:
        np.array: The processed neural data.
    """
    if neural_mode == 'RAW' or neural_mode == 'TX':
        neural_processed = gaussian_filter1d(neural_array, sigma=sigma, axis=2)
        neural_processed = sh.downsample_list_3d(neural_processed, original_samples//target_samples, mode='sum')
    elif neural_mode == 'TRAJECTORIES':
        neural_processed = np.array([su.resample_by_interpolation_2d(neural_trace, original_samples, target_samples) for neural_trace in neural_array])
    else:
        raise ValueError("Neural mode must be 'RAW', 'TX' or 'TRAJECTORIES'")
    return neural_processed


def prepare_dataloaders(neural_data, audio_data, train_indices, test_indices, batch_size, history_size, max_temporal_shift_bins=0, noise_level=0, transform_probability=0.5, shuffle_inputs=False):
    """
    Prepares dataloaders for training and testing datasets based on the specified parameters.
    Args:
        neural_data (np.array): The neural data.
        audio_data (np.array): The corresponding audio data.
        train_indices (np.array): Indices for training data.
        test_indices (np.array): Indices for testing data.
        batch_size (int): The batch size for the dataloaders.
        history_size (int): The size of historical data to consider.
        max_temporal_shift_bins (int): Maximum temporal shift (in bins) applied to the training data.
        noise_level (float): Noise level to be added to the training data.
        transform_probability (float): Probability of applying transformations to training data.
        shuffle_inputs (bool): Whether to shuffle inputs in the test data.
    Returns:
        tuple: A tuple containing the training and testing DataLoader instances.
    """
    train_neural, test_neural = neural_data[train_indices], neural_data[test_indices]
    train_audio, test_audio = audio_data[train_indices], audio_data[test_indices]
    
    if shuffle_inputs:
        print(f'CONTROL: Shuffling model inputs in the test set along the second dimension!')
        test_neural = np.apply_along_axis(np.random.permutation, axis=2, arr=test_neural)
    
    train_dataset = NeuralAudioDataset(train_neural, train_audio, history_size, max_temporal_shift=max_temporal_shift_bins, noise_level=noise_level, transform_probability=transform_probability)
    test_dataset = NeuralAudioDataset(test_neural, test_audio, history_size, max_temporal_shift=0, noise_level=0, transform_probability=0)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f'Data loaders created. Training samples: {len(train_dataset)}, Testing sample: {len(test_dataset)}')
    
    return train_dataset, test_dataset, train_loader, test_loader


def load_model(model_directory, model_filename, model_layers, learning_rate):
    """
    Loads a model and its optimizer from checkpoints.

    Args:
        model_directory (str): Directory where the model checkpoints are stored.
        model_filename (str): The name of the model checkpoint file.
        model_layers (list of int): List specifying the number of units in each layer of the model.
        learning_rate (float): Learning rate for the optimizer.
    Returns:
        tuple: A tuple containing the loaded model and its optimizer.
    """
    checkpoint = torch.load(os.path.join(model_directory, model_filename))
    model = FeedforwardNeuralNetwork(model_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


def get_original_reconstructed_audio(ffnn_model, dataset, loader, fs_audio, encodec_model, scale):
    
    # ORIGINAL AUDIO
    it = iter(dataset)
    samples = []
    for _ in range(len(dataset)):
        sample = next(it)
        samples.append(sample[1])  
    original_embeddings = torch.stack(samples, dim=0).permute(1,0)
    original_audio = eu.audio_from_embedding(original_embeddings, scale, encodec_model, fs_audio).squeeze(0).squeeze(0)

    # DECODED AUDIO
    decoded_embeddings, error = ffnn_predict(ffnn_model, loader)
    decoded_embeddings = decoded_embeddings.permute(1, 0)
    
    decoded_embeddings = decoded_embeddings.to(scale.device)
    decoded_audio = eu.audio_from_embedding(decoded_embeddings, scale, encodec_model, fs_audio).squeeze(0).squeeze(0)

    return original_audio.detach().numpy(), decoded_audio.detach().numpy()


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