import os
import sys
import argparse
import json
from matplotlib.backends.backend_pdf import PdfPages
import wandb
import datetime

def add_to_sys_path(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        sys.path.append(dirpath)        
root_dir = '/home/jovyan/pablo_tostado/bird_song/enSongDec/'
add_to_sys_path(root_dir)


from FFNNmodel import FeedforwardNeuralNetwork, ffnn_train, ffnn_evaluate, ffnn_predict
from neural_audio_dataset import NeuralAudioDataset
import utils.audio_utils as au
import utils.encodec_utils as eu
import utils.signal_utils as su
import utils.train_utils as tu
import utils.visualization_utils as vu

exec(open('/home/jovyan/pablo_tostado/bird_song/enSongDec/utils/all_imports.py').read())


def flatten_dict(d):
    """
    Flatten a nested dictionary into a flat dictionary.
    """
    return {k: v for key, val in d.items() for k, v in (flatten_dict(val).items() if isinstance(val, dict) else [(key, val)])}

    
def main(config_filepath, override_dict=None):

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
                    run_experiment(override_experiment_metadata, config_filepath)
            else:
                print(f'{k} not found in experiment_metadata. Skipping override.')
    else:
        run_experiment(experiment_metadata, config_filepath)

def run_experiment(experiment_metadata, config_filepath):

    # --------- EXTRACT CONFIG INFO --------- #
    # Directories of interest
    dataset_dir              = experiment_metadata['dataset_dir']
    models_checkpoints_dir   = experiment_metadata['models_checkpoints_dir']
    train_figures_dir        = experiment_metadata['train_figures_dir']
    # Experiment params
    dataset_filename         = experiment_metadata['dataset_filename']
    neural_mode              = experiment_metadata['neural_mode']
    neural_key               = experiment_metadata['neural_key']
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
    percent_train            = experiment_metadata["percent_train"]
    percent_test             = experiment_metadata["percent_test"]
    batch_size               = experiment_metadata["batch_size"]
    learning_rate            = experiment_metadata["learning_rate"]
    num_epochs               = experiment_metadata["num_epochs"]

    # Define experiment name to save output files
    experiment_name = "_".join([bird, neural_mode, neural_key, network, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')])

    # Figure List to save to pdf
    figures_list = []  
    
    # --------- LOAD DATA --------- #
    
    # Open the file in binary read mode ('rb') and load the content using pickle
    with open(dataset_dir + dataset_filename + '.pkl', 'rb') as file:
        data_dict = pkl.load(file)
    
    # Extarct data from dataset
    neural_array = data_dict['neural_dict'][neural_key]
    audio_motifs = data_dict['audio_motifs']
    fs_audio = data_dict['fs_audio']
    fs_neural = data_dict['fs_neural']
    
    # Calculate the duration based on the last dimension of the arrays and their sampling rates
    trial_length_neural = (neural_array.shape[-1] / fs_neural)*1000
    trial_length_audio = (audio_motifs.shape[-1] / fs_audio)*1000
    
    # Check the durations of the neural/audio data are equal and raise a warning if they aren't
    print('Length of neural trials: {} ms, length of audio trials: {} ms. '.format(trial_length_neural, trial_length_audio))
    if trial_length_neural != trial_length_audio:
        warnings.warn("WARNING: Neural data duration and audio motifs duration are different in this dataset!")

    # Plot and save figures for raw neural_array and audio_motifs
    title = 'Raw Neural Traces'
    figures_list.append(vu.visualize_neural(neural_array, title=title, neural_channel=10, offset=1))
    title = 'Raw Audio Motifs'
    figures_list.append(vu.visualize_audio(audio_motifs, title=title, offset=40000))

    
    # --------- PROCESS AUDIO --------- #
    
    b, a = fh.load_filter_coefficients_matlab(
        '/home/jovyan/pablo_tostado/repos/songbirdcore/songbirdcore/filters/butter_bp_250Hz-8000hz_order4_sr25000.mat')
    audio_motifs = fh.noncausal_filter_2d(audio_motifs, b=b, a=a)
    
    # Reduce noise
    for m in range(len(audio_motifs)):
        audio_motifs[m] = nr.reduce_noise(audio_motifs[m], sr=fs_audio)
    
    
    # --------- INSTANTIATE ENCODEC --------- #
    
    # Instantiate a pretrained EnCodec model
    encodec_model = EncodecModel.encodec_model_48khz()
    # bandwidth = 24kbps for 48kHz model (n_q=16)
    encodec_model.set_target_bandwidth(24.0)
    
    # Embed motifs
    audio_embeddings, audio_codes, scales = eu.encodec_encode_audio_array_2d(audio_motifs, fs_audio, encodec_model)
    
    
    # --------- PROCESS NEURAL --------- #
    
    # Resample neural datato match audio embeddings
    samples_neural = neural_array.shape[2]
    samples_embeddings = audio_embeddings.shape[2]
    history_size = samples_neural//samples_embeddings
    
    # Match neural to audio samples (! Different for raw spiketrains vs trajectories)
    if neural_mode == 'RAW' or neural_mode == 'THRESHOLDS':
        print(f'Pre-processing neural data as {neural_mode}')
        # Gaussian kernel along the temporal dimension of the spiketrains
        neural_array = gaussian_filter1d(neural_array, sigma=gaussian_smoothing_sigma, axis=2) 
        # Downsample to spikerate at given bin_size
        neural_array = sh.downsample_list_3d(neural_array, history_size, mode='sum')  
    elif neural_mode == 'TRAJECTORIES':
        print(f'Pre-processing neural data as {neural_mode}')
        # Downsample by interpolation
        neural_array = np.array([su.resample_by_interpolation_2d(n, samples_neural, samples_embeddings) for n in neural_array])
    else:
        raise ValueError("Neural mode must be 'RAW', 'THRESHOLDS' or 'TRAJECTORIES'")
    
    bin_length = trial_length_neural / neural_array.shape[2] # ms
    history_size = int(neural_history_ms // bin_length) # Must be minimum 1
    print('Using {} bins of neural data history.'.format(history_size))

    # Plot and save figures for resampled neural_array and audio_embeddings
    title = 'Resampled Neural Traces {}'.format(neural_array.shape)
    figures_list.append(vu.visualize_neural(neural_array, title=title, neural_channel=10, offset=1))
    title = 'Audio Embeddings {}'.format(audio_embeddings.shape)
    figures_list.append(vu.visualize_audio_embeddings(audio_embeddings, title=title, embedding_dim=0, offset=10))

    
    # --------- PREPARE DATALOADERS --------- #
    
    num_motifs = len(neural_array)
    num_train_examples = int(num_motifs * percent_train)
    print(f'Out of the {num_motifs} total motifs, the first {num_train_examples} are used for training. The rest, for testing.')
    
    # Split the data into train and test sets
    train_idxs = list(range(0, num_train_examples))
    test_idxs = list(range(num_train_examples, num_motifs))
    train_neural = neural_array[train_idxs]  
    train_audio = audio_embeddings[train_idxs]
    test_neural = neural_array[test_idxs]  
    test_audio = audio_embeddings[test_idxs]
    
    # Create dataset objects
    max_temporal_shift_bins = int(max_temporal_shift_ms // bin_length) # Temporal jitter for data augmentation
    
    train_dataset = NeuralAudioDataset(train_neural, 
                                       train_audio, 
                                       history_size, 
                                       max_temporal_shift=max_temporal_shift_bins,
                                       noise_level=noise_level,
                                       transform_probability=transform_probability)
    
    test_dataset = NeuralAudioDataset(test_neural, 
                                      test_audio, 
                                      history_size, 
                                      max_temporal_shift=0,
                                      noise_level=0,
                                      transform_probability=0)
    
    print('Train samples: ', len(train_dataset))
    print('Test samples: ', len(test_dataset))
    
    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
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
    experiment_metadata['train_idxs'] = train_idxs
    experiment_metadata['test_idxs'] = test_idxs
    
    # TRACK with WANDB
    wandb.init(
        project = "_".join([bird, neural_mode, neural_key]),
        name = experiment_name, 
        config = experiment_metadata # track hyperparameters and run metadata
    )
    
    # Train
    tot_train_loss, tot_train_err, tot_val_loss, tot_val_err = ffnn_train(ffnn_model, 
                                                                          train_loader, 
                                                                          optimizer, 
                                                                          criterion, 
                                                                          num_epochs, 
                                                                          val_dataloader=test_loader)
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
    parser = argparse.ArgumentParser(description="Run the bird song decoding experiment.")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to JSON config file of the experiment.")
    parser.add_argument("--override_dict", type=str, required=False, help="Dictionary params to override in the config file one at a time, e.g. {'learning_rate': [0.02, 0.07], 'dropout_prob': [0.25]}")
    args = parser.parse_args()

    # Parse the override_dict JSON string into a dictionary
    override_dict = None
    if args.override_dict:
        override_dict = json.loads(args.override_dict)
    
    main(args.config_filepath, override_dict=override_dict)

    

