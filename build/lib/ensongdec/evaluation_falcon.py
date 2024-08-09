import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ensongdec.utils import encodec_utils as eu
import ensongdec.utils.train_utils as tu
from ensongdec.utils.evaluation_utils import load_experiment_metadata

# From falcon-challenge
from preproc.b1_nwb_utils import load_nwb_b1

def prepare_eval_dataloader_from_nwb(models_checkpoints_dir, model_filename, nwb_file_path):
    
    # --------- LOAD EXPERIMENT --------- #
    experiment_metadata = load_experiment_metadata(models_checkpoints_dir, model_filename)
    
    # Experiment params
    model_layers = experiment_metadata['layers']
    neural_mode = experiment_metadata['neural_mode']
    neural_history_ms = experiment_metadata["neural_history_ms"]
    gaussian_smoothing_sigma = experiment_metadata["gaussian_smoothing_sigma"]    
    # batch_size = experiment_metadata["batch_size"]
    batch_size = 1 # Predict one batch at a time

    # --------- LOAD DATA --------- #
    trial_info, neural_array, fs_neural, audio_motifs, fs_audio = load_nwb_b1(nwb_file_path)
    tu.check_experiment_duration(neural_array, audio_motifs, fs_neural, fs_audio)

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
    decoder_history_bins = int(neural_history_ms // bin_length) # Must be minimum 1
    print('Using {} bins of neural data history for decoding.'.format(decoder_history_bins))

    # --------- PREPARE EVAL DATALOADER --------- #
    
    test_dataset, test_loader = tu.prepare_dataloader(neural_array, 
                                                   audio_embeddings, 
                                                   batch_size, 
                                                   decoder_history_bins, 
                                                   max_temporal_shift_bins=0,
                                                   noise_level=0,
                                                   transform_probability=0, 
                                                   shuffle_samples = False)
    
    return test_dataset, test_loader, fs_audio, encodec_model, scales