import datetime
import pickle
from typing import Dict, Union
import numpy as np

def save_dataset(
    neural_dict: Dict,
    audio_array: np.ndarray,
    labels_array: np.ndarray,
    fs_ap: float,
    fs_audio: float,
    t_pre: float,
    t_post: float,
    sess_par: Dict,
    file_type: str, 
    dir_path: str
) -> None:
    """
    Saves the dataset consisting of neural data, audio motifs, and their labels to a file with a timestamp in the filename.
    The data is saved in pickle format in the specified directory.

    Parameters:
        neural_dict (Dict): Dictionary containing neural data, with keys and values representing different aspects of neural activity.
        audio_array (np.ndarray): Array containing audio motifs, represented as either floating-point or integer values.
        labels_array (np.ndarray): Array containing labels for each audio motif, represented as either floating-point or integer values.
        fs_ap (float): Sampling rate of the neural data, indicating the frequency at which the neural data was captured.
        fs_audio (float): Sampling rate of the audio data, indicating the frequency at which the audio data was captured.
        t_pre (float): Time interval before the event of interest, used to specify the duration of data to include prior to the event.
        t_post (float): Time interval after the event of interest, used to specify the duration of data to include following the event.
        sess_par (Dict): Dictionary containing session parameters, with keys and values detailing various aspects of the recording session.
        file_type (str): Type of the data file to be saved, which should be either 'RAW' or 'TRAJECTORIES', used as a prefix in the filename.
        dir_path (str): The directory path where the data file will be saved, indicating the location for storing the output file.

    Returns:
        None: This function does not return a value but saves a file to the specified location.

    Raises:
        ValueError: If the provided `file_type` is not 'RAW', 'TX' or 'TRAJECTORIES'.

    The function constructs a comprehensive dataset from the provided parameters, generates a filename with a timestamp,
    and saves the dataset as a pickle file in the specified directory. It also validates the `file_type` and raises an
    exception if the type is incorrect.
    """
    
    if file_type not in ['RAW', 'TX', 'TRAJECTORIES']:
        raise ValueError("file_type must be 'RAW', 'TX' or 'TRAJECTORIES'")

    data = {
        'neural_dict': neural_dict,
        'audio_motifs': audio_array,
        'audio_labels': labels_array,
        'fs_neural': fs_ap,
        'fs_audio': fs_audio,
        't_pre': t_pre,
        't_post': t_post,
        'sess_params': sess_par
    }

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bird_name = sess_par.get('bird', 'UnknownBird')
    file_name = f"{file_type}_{bird_name}_{timestamp}.pkl"
    file_path = dir_path + file_name

    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
        print(f"Dictionary saved as {file_path}")


    