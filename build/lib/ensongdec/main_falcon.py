import os
import sys
import argparse
import json
from matplotlib.backends.backend_pdf import PdfPages
import wandb
import datetime
from scipy.ndimage import gaussian_filter, gaussian_filter1d

def add_to_sys_path(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        sys.path.append(dirpath)        
ensongdec_dir = '/home/jovyan/pablo_tostado/bird_song/enSongDec/'
add_to_sys_path(ensongdec_dir)
falconb1 = '/home/jovyan/pablo_tostado/repos/falcon_b1/'
add_to_sys_path(falconb1)

import pickle as pkl
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from FFNNmodel import FeedforwardNeuralNetwork, ffnn_train, ffnn_evaluate, ffnn_predict
from neural_audio_dataset import NeuralAudioDataset
import utils.audio_utils as au
import utils.encodec_utils as eu
import utils.signal_utils as su
import utils.train_utils as tu
import utils.visualization_utils as vu

import songbirdcore.spikefinder.spike_analysis_helper as sh
import songbirdcore.spikefinder.filtering_helper as fh

# EncoDec
from encodec import EncodecModel
from encodec.utils import convert_audio

# Tim S. noise reduce
import noisereduce as nr

from nwb_utils import load_nwb


def flatten_dict(d):
    """
    Flatten a nested dictionary into a flat dictionary.
    """
    return {k: v for key, val in d.items() for k, v in (flatten_dict(val).items() if isinstance(val, dict) else [(key, val)])}


def main(config_filepath, nwb_filepath, override_dict=None):

    # --------- EXPERIMENT CONFIG --------- #
    # Extract experiment params from JSON config file
    with open(config_filepath, 'r') as file:
        config = json.load(file)
    experiment_metadata = flatten_dict(config)

    # GRID SEARCH: Override param
    if override_dict:
        for k, v in override_dict.items():
            override_experiment_metadata = experiment_metadata.copy()
            if k in experiment_metadata:
                for k_val in v:
                    override_experiment_metadata[k] = k_val
                    print(f'Updated {k} to {k_val}')
                    run_experiment(override_experiment_metadata, config_filepath, nwb_filepath)
            else:
                print(f'{k} not found in experiment_metadata. Skipping override.')
    else:
        run_experiment(experiment_metadata, config_filepath, nwb_filepath)


def run_experiment(experiment_metadata, config_filepath, nwb_file_path):

    # --------- EXTRACT CONFIG INFO --------- #
    # Directories of interest
    models_checkpoints_dir   = experiment_metadata['models_checkpoints_dir']
    train_figures_dir        = experiment_metadata['train_figures_dir']
    # Experiment params
    neural_mode              = experiment_metadata['neural_mode']
    bird                     = experiment_metadata['bird']
    # Config params
    config_id                = experiment_metadata['config_id']
    # Data_processing_params
    neural_history_ms        = experiment_metadata["neural_history_ms"]
    gaussian_smoothing_sigma = experiment_metadata["gaussian_smoothing_sigma"]
    # Data_augmentation_params
    max_temporal_shift_ms    = experiment_metadata["max_temporal_shift_ms"]
    noise_level              = experiment_metadata["noise_level"]
    transform_probability    = experiment_metadata["transform_probability"]
    # Model_params
    network                  = experiment_metadata['network']
    hidden_layer_sizes       = experiment_metadata["hidden_layer_sizes"]
    dropout_prob             = experiment_metadata["dropout_prob"]
    # Training_params
    percent_validation       = experiment_metadata["percent_validation"]
    batch_size               = experiment_metadata["batch_size"]
    learning_rate            = experiment_metadata["learning_rate"]
    num_epochs               = experiment_metadata["num_epochs"]

    print(hidden_layer_sizes)

    # Define experiment name to save output files
    experiment_name = "_".join([os.path.basename(nwb_file_path), network, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')])

    # Figure List to save to pdf
    figures_list = []  
    
    # --------- LOAD DATA --------- #
    trial_info, neural_array, fs_neural, audio_motifs, fs_audio = load_nwb(nwb_file_path)
    tu.check_experiment_duration(neural_array, audio_motifs, fs_neural, fs_audio)

    # Plot and save figures for raw neural_array and audio_motifs
    title = 'Raw Neural Traces'
    figures_list.append(vu.visualize_neural(neural_array, title=title, neural_channel=10, offset=1))
    title = 'Raw Audio Motifs'
    figures_list.append(vu.visualize_audio(audio_motifs, title=title, offset=40000))
    
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

    print(f'Neural array shape {neural_array.shape}, Audio embeddings shape {audio_embeddings.shape}.')
    assert neural_array.shape[-1] == audio_embeddings.shape[-1], "Mismatch Error: The length of 'neural_array' does not match the length of 'audio_embeddings'."
    
    bin_length = ((original_samples / fs_neural)*1000) / neural_array.shape[-1] # ms
    decoder_history = int(neural_history_ms // bin_length) # Must be minimum 1
    print('Using {} bins of neural data history for decoding.'.format(decoder_history))
    
    # --------- PREPARE DATALOADERS --------- #
    num_motifs = len(neural_array)
    num_train_examples = int(num_motifs * (1-percent_validation))
    print(f'Out of the {num_motifs} total motifs, the first {num_train_examples} are used for training. {num_motifs - num_train_examples} for validation.')
    
    # Split the data into train and test sets
    train_idxs, validation_idxs = list(range(0, num_train_examples)), list(range(num_train_examples, num_motifs))
    train_neural, train_audio = neural_array[train_idxs], audio_embeddings[train_idxs]
    validation_neural, validation_audio = neural_array[validation_idxs], audio_embeddings[validation_idxs]
    
    # Create dataset objects
    max_temporal_shift_bins = int(max_temporal_shift_ms // bin_length) # Temporal jitter for data augmentation

    train_dataset, train_dataloader = tu.prepare_dataloader(train_neural, 
                                                     train_audio, 
                                                     batch_size, 
                                                     decoder_history, 
                                                     max_temporal_shift_bins=max_temporal_shift_bins,
                                                     noise_level=noise_level,
                                                     transform_probability=transform_probability, 
                                                     shuffle_samples = True)

    if validation_idxs:
        validation_dataset, validation_dataloader = tu.prepare_dataloader(validation_neural, 
                                                                       validation_audio, 
                                                                       batch_size, 
                                                                       decoder_history, 
                                                                       max_temporal_shift_bins=0,
                                                                       noise_level=0,
                                                                       transform_probability=0, 
                                                                       shuffle_samples = False)
    else: 
        validation_dataloader = None
    
    print(f'Train samples: {len(train_dataset)}') 
    if validation_dataloader: print(f'Validation samples: {len(validation_dataset)}')


    # --------- TRAIN MODEL --------- #
    # Initialize the neural network and optimizer
    input, target = next(iter(train_dataset))
    input_dim = input.shape[0]
    output_dim = target.shape[0]
    print('Input_dim: ', input_dim, ' -  Output dim: ', output_dim)
    
    # Instantiate model
    layers = [input_dim] + hidden_layer_sizes + [output_dim]
    ffnn_model = FeedforwardNeuralNetwork(layers, dropout_prob=dropout_prob)
    total_params = tu.compute_num_model_params(ffnn_model)
    
    # Loss function and optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(ffnn_model.parameters(), lr=learning_rate)
    
    # Expand experiment metadata
    experiment_metadata['config_filepath'] = config_filepath
    experiment_metadata['experiment_name'] = experiment_name
    experiment_metadata['layers'] = layers
    experiment_metadata['total_params'] = total_params
    
    # TRACK with WANDB
    wandb.init(
        project = "_".join([bird, neural_mode]),
        name = experiment_name,
        config = experiment_metadata # track hyperparameters and run metadata
    )
    
    # Train
    tot_train_loss, tot_train_err, tot_val_loss, tot_val_err = ffnn_train(ffnn_model, 
                                                                          train_dataloader, 
                                                                          optimizer, 
                                                                          criterion, 
                                                                          num_epochs, 
                                                                          val_dataloader=validation_dataloader)
    print('Done training!')
    wandb.finish()

    # Plot and save figures for loss and error
    title = 'Loss'
    figures_list.append(vu.visualize_loss_error(tot_train_loss, tot_val_loss, title=title))
    title = 'Embeddings Reconstruction Error'
    figures_list.append(vu.visualize_loss_error(tot_train_err, tot_val_err, title=title))

    # Initialize PdfPages for saving figures to PDF
    try:
        print('figures ', len(figures_list))
        figures_filename = os.path.join(train_figures_dir, f'{experiment_name}_figures.pdf') 
        pdf_pages = PdfPages(figures_filename)
        for fig in figures_list:
            pdf_pages.savefig(fig)
        pdf_pages.close()
    except Exception as e:
        if pdf_pages is not None:
            pdf_pages.close()  

    # --------- SAVE MODEL --------- #
    if not os.path.exists(models_checkpoints_dir):
        os.makedirs(models_checkpoints_dir)

    # Expand experiment metadata
    experiment_metadata['tot_train_loss'] = tot_train_loss
    experiment_metadata['tot_train_err'] = tot_train_err
    experiment_metadata['tot_val_loss'] = tot_val_loss
    experiment_metadata['tot_val_err'] = tot_val_err
    
    tu.save_model(models_checkpoints_dir, experiment_name, ffnn_model, optimizer, experiment_metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a birdsong decoding experiment.")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to experiment's .JSON config file.")
    parser.add_argument("--nwb_filepath", type=str, required=True, help="Path to the NWB file containing neural and audio data.")
    parser.add_argument("--override_dict", type=str, required=False, help="Dictionary params to override in the config file one at a time, e.g. {'learning_rate': [0.02, 0.07], 'dropout_prob': [0.25]}")
    args = parser.parse_args()

    override_dict = None
    if args.override_dict:
        override_dict = json.loads(args.override_dict)

    main(args.config_filepath, args.nwb_filepath, override_dict=override_dict)
    

