import pandas as pd
import os
import json
import numpy as np
import warnings

from scipy.ndimage import gaussian_filter, gaussian_filter1d

import torch
from torch.utils.data import DataLoader

# Ensure you import your custom modules here if they are part of your project
import songbirdcore.spikefinder.spike_analysis_helper as sh
import songbirdcore.spikefinder.filtering_helper as fh
import ensongdec.utils.signal_utils as su

# EnSongdec
from neural_audio_dataset import NeuralAudioDataset

# Tim S. noise reduce
import noisereduce as nr


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
    

def check_experiment_duration(neural_array, audio_motifs, fs_neural, fs_audio):
    # Calculate the duration of the experiment based on the last dimension of the arrays and their sampling rates
    trial_length_neural = (neural_array.shape[-1] / fs_neural)*1000
    trial_length_audio = (audio_motifs.shape[-1] / fs_audio)*1000
    
    # Check the durations of the neural/audio data are equal and raise a warning if they aren't
    print('Length of neural trials: {} ms, length of audio trials: {} ms. '.format(trial_length_neural, trial_length_audio))
    if trial_length_neural != trial_length_audio:
        warnings.warn("WARNING: Neural data duration and audio motifs duration are different in this dataset!")


def prepare_dataloader(neural_data, audio_data, batch_size, decoder_history_bins, max_temporal_shift_bins=0, noise_level=0, transform_probability=0.5, shuffle_samples=False):
    """
    Prepares dataloaders for training and testing datasets based on the specified parameters.
    Args:
        neural_data (np.array): The neural data.
        audio_data (np.array): The corresponding audio data.
        batch_size (int): The batch size for the dataloaders.
        decoder_history_bins (int): The number of bins historical data to consider.
        max_temporal_shift_bins (int): Maximum temporal shift (in bins) applied to the training data.
        noise_level (float): Noise level to be added to the training data.
        transform_probability (float): Probability of applying transformations to training data.
        shuffle_samples (bool): Whether to shuffle dataloader samples (usually True for training, False for testing).
    Returns:
        tuple: A tuple containing the dataset and DataLoader instances.
    """
    
    dataset = NeuralAudioDataset(neural_data, audio_data, 
                                 decoder_history_bins, max_temporal_shift=max_temporal_shift_bins, 
                                 noise_level=noise_level, transform_probability=transform_probability)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_samples)
    print(f'Dataset and Dataloader created. Samples: {len(dataset)}')
    
    return dataset, dataloader
    

def compute_num_model_params(model):
    """
    Computes and prints the number of parameters for each layer in a model, as well as the total number of parameters.
    Args:
        model (nn.Module): The neural network model for which parameters are to be counted.
    Returns:
        int: The total number of parameters in the model.
    """
    total_params = 0
    for name, parameter in model.named_parameters():
        print(f"Layer: {name} | Size: {parameter.size()} | Number of Parameters: {parameter.numel()}")
        total_params += parameter.numel()
    print(f"Total number of parameters in the model: {total_params}")
    return total_params


def save_model(models_checkpoints_dir, experiment_name, model, optimizer, experiment_metadata):
    """
    Saves the model and optimizer state dictionaries to a file and appends training 
    information to a CSV file.

    If the CSV file already exists, it appends the new training information as a new row.
    If any new fields are present in the training_info dictionary, they are added as new 
    columns in the CSV file. Missing data for these new columns in previous rows are filled
    with NA values to maintain consistency.

    Args:
        models_checkpoints_dir (str): The directory path where the model and CSV files 
            will be saved.
        experiment_name (str): The base name for the files to be saved. This will be used 
            to name both the model's state dictionary file and the CSV file.
        model (torch.nn.Module): The model instance whose state dictionary will be saved.
        optimizer (torch.optim.Optimizer): The optimizer instance whose state dictionary 
            will be saved.
        experiment_metadata (dict): A dictionary containing various pieces of training 
            information to be logged in the CSV file. E.g.
            
            experiment_metadata = {
                'model_name': experiment_name,
                'dataset_filename': dataset_filename,
                'bird': bird,
                'neural_key': neural_key,
                'train_idxs': train_idxs,
                'test_idxs': test_idxs,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'layers': layers,
                'neural_history_ms': neural_history_ms,
                'num_epochs': num_epochs,
                'num_params': total_params,
                'tot_train_loss': tot_train_loss,
                'tot_val_loss': tot_val_loss,
                'tot_train_err': tot_train_err,
                'tot_val_err': tot_val_err,
                'config_path': config_path,
                'config_id': config_id,
            }

    Returns:
        None: This function does not return a value but prints out confirmation of the 
        saved files.
    """
    # Save the model and optimizer state dictionaries
    model_path = os.path.join(models_checkpoints_dir, f'{experiment_name}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    
    # Save the experiment_metadata dictionary to a JSON file
    metadata_path = os.path.join(models_checkpoints_dir, f'{experiment_name}_metadata.json') 
    with open(metadata_path, 'w') as output_file:
        json.dump(experiment_metadata, output_file, indent=4)
    
    print(f"Model statedict saved to {model_path}", f"\nExperiment metadata saved to {metadata_path}")

    # Path for the CSV file
    info_path = os.path.join(models_checkpoints_dir, 'models_info.csv')
    
    # Create a DataFrame for the new training information
    df_new = pd.DataFrame([experiment_metadata])

    # Check if the file exists and append if it does, otherwise write a new file
    if os.path.isfile(info_path):
        df_existing = pd.read_csv(info_path)

        # Ensure all columns exist in both DataFrames to avoid alignment issues
        for column in df_new.columns.difference(df_existing.columns):
            df_existing[column] = pd.NA  # Add missing column as NA in existing DataFrame
        for column in df_existing.columns.difference(df_new.columns):
            df_new[column] = pd.NA  # Add missing column as NA in new DataFrame
            
        idx = df_existing[df_existing['experiment_name'] == experiment_name].index
        if not idx.empty:
            df_existing.loc[idx] = df_new.values
            df_combined = df_existing
        else:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(info_path, index=False)
    print('Training session info saved: ', info_path)

