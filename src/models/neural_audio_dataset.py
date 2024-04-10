import torch
from torch.utils.data import Dataset
import numpy as np

class NeuralAudioDataset(Dataset):
    def __init__(self, 
                 neural_data, 
                 audio_data, 
                 history_samples, 
                 max_temporal_shift=0, 
                 noise_level=0, 
                 transform_probability=0.5):
        '''
        neural_data = [trials x neural_channels x samples]
        audio_data = [trials x audio_features x samples]
        history_samples = Number of neural samples used to predict each audio sample
        max_temporal_shift = Maximum allowed temporal jitter (in samples) for data augmentation
        noise_level = Standard deviation of Gaussian noise added to neural data for augmentation
        transform_probability = Probability of applying temporal shift augmentation and/or noise augmentation
        '''
        
        self.neural_data = neural_data
        self.audio_data = audio_data
        self.history_size = history_samples
        self.max_temporal_shift = max_temporal_shift
        self.noise_level = noise_level
        self.p = transform_probability

        assert self.neural_data.shape[0] == self.audio_data.shape[0], "Number of trials must match in both datasets"
        assert self.neural_data.shape[-1] == self.audio_data.shape[-1], "Number of samples must match in both datasets"
        
        self.inputs, self.targets = [], []
        for trial in range(self.neural_data.shape[0]):
            # Skip history_size bins. 
            for sample in range(self.history_size+self.max_temporal_shift, self.neural_data.shape[2]-self.max_temporal_shift): 
                # Store indices of neural data corresponding to each audio sample to later retrieve neural data w/out temporal jitter.
                self.inputs.append((trial, sample))  
                self.targets.append(self.audio_data[trial, :, sample])

        # Total number of samples
        assert len(self.inputs) == len(self.targets), "Number of inputs and targets must match"
        self.total_samples = len(self.inputs)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        trial, sample = self.inputs[idx]
        neural_sample = self.neural_data[trial, :, sample - self.history_size:sample]
        audio_sample = self.targets[idx]  # Corresponding audio sample

        # --- DATA AUGMENTATION --- #
        # Apply temporal shift augmentation with probability p
        if np.random.rand() < self.p:
            # ! WARNING: keep in mind that this will result in neural traces with all zeros if at the start/end of the trial.
            # To prevent this behavior, add self.max_temporal_shift to the constructor of the self.inputs, 
            # although it will result in less data for training.
            shift = np.random.randint(-self.max_temporal_shift, self.max_temporal_shift + 1)
            start_idx = max(0, sample - self.history_size + shift)
            end_idx = start_idx + self.history_size
            neural_sample = self.neural_data[trial, :, start_idx:end_idx]

        # Flatten for FFNN
        neural_sample = neural_sample.reshape(-1)

        # Apply noise augmentation with probability p
        if np.random.rand() < self.p:
            noise = np.random.normal(0, self.noise_level, size=neural_sample.shape)
            neural_sample += noise

        return torch.as_tensor(neural_sample, dtype=torch.float32), torch.as_tensor(audio_sample, dtype=torch.float32)