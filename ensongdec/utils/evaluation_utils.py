import os
import sys
import json
import numpy as np

import torch
import torch.optim as optim

# ensongdec
from FFNNmodel import FeedforwardNeuralNetwork, ffnn_predict
import utils.encodec_utils as eu

def add_to_sys_path(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        sys.path.append(dirpath)        
root_dir = '/home/jovyan/pablo_tostado/repos/falcon_b1/'
add_to_sys_path(root_dir)
from nwb_utils import load_nwb
    

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
    metadata_file = next((file for file in metadata_files if os.path.splitext(filename)[0] in file), None)
    print(f'Loading {metadata_file}')
    
    with open(os.path.join(directory, metadata_file), 'rb') as file:
        return json.load(file)


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



def generate_trialized_original_and_reconstructed_audio(ffnn_model, dataset, loader, fs_audio, encodec_model, scale, num_trials):
    '''
        This function has the advantage (over generate_original_and_reconstructed_audio) of not losing samples when resampling to original sample rate in eu.audio_from_embedding.
    '''
    # ORIGINAL AUDIO
    it = iter(dataset)
    samples = []
    for _ in range(len(dataset)):
        sample = next(it)
        samples.append(sample[1])  
    
    original_embeddings = torch.stack(samples, dim=0).permute(1,0)
    original_embeddings = original_embeddings.reshape(original_embeddings.shape[0], num_trials, -1) # Embedding_dim x Trials x Samples
    original_audio = [eu.audio_from_embedding(original_embeddings[:,i,:], scale, encodec_model, fs_audio).squeeze(0).squeeze(0).detach() for i in range(num_trials)]
    
    # DECODED AUDIO
    decoded_embeddings, error = ffnn_predict(ffnn_model, loader)
    decoded_embeddings = decoded_embeddings.permute(1, 0)
    decoded_embeddings = decoded_embeddings.reshape(decoded_embeddings.shape[0], num_trials, -1) # Embedding_dim x Trials x Samples
    
    decoded_embeddings = decoded_embeddings.to(scale.device)
    decoded_audio = [eu.audio_from_embedding(decoded_embeddings[:,i,:], scale, encodec_model, fs_audio).squeeze(0).squeeze(0).detach() for i in range(num_trials)]
    
    return np.array(original_audio), np.array(decoded_audio)


def generate_original_and_reconstructed_audio(ffnn_model, dataset, loader, fs_audio, encodec_model, scale):
    
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
