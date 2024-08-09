import numpy as np
import torch
from encodec.utils import convert_audio
import typing as tp

# EncoDec
from encodec import EncodecModel


def instantiate_encodec():
    """
    Instantiates a model of the Encodec with preset configurations.
    Returns:
        EncodecModel: A configured Encodec model instance.
    """
    model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(24.0)
    return model
    

def audio_from_embedding(emb, emb_scale, encodec_model, fs_audio):
    """
    Converts an audio embedding into an audio signal using the specified encoder-decoder model,
    then resamples the resulting audio to the desired sampling rate.

    Args:
        emb (torch.Tensor): The audio embedding tensor to be converted into audio.
        emb_scale (float): The scaling factor applied to the embedding.
        encodec_model (object): The encoder-decoder model used for generating the audio signal from the embedding.
                                This model should have `quantizer`, `frame_rate`, `bandwidth`, `decode`,
                                and `sample_rate` attributes or methods as required by the function.
        fs_audio (int): The target sampling rate for the output audio signal.

    Returns:
        torch.Tensor: The resampled audio signal as a tensor, with the sampling rate adjusted to `fs_audio`.

    Note:
        The function assumes that the `encodec_model` has methods for encoding and decoding the audio,
        and attributes defining its internal sample rate and other necessary parameters. The function
        also handles the transposition of codes to match the expected input shape for decoding.
    """
    # Original embedding
    EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]
    encoded_frames: tp.List[EncodedFrame] = []

    # Pass quantized values through quantizer and decoder
    codes = encodec_model.quantizer.encode(emb.unsqueeze(0), encodec_model.frame_rate, encodec_model.bandwidth)
    codes = codes.transpose(0, 1)
    encoded_frames.append((codes, emb_scale))

    audio_signal = encodec_model.decode(encoded_frames)

    # Reconvert decoded audio to original sampling rate
    model_sr = encodec_model.sample_rate
    resamp_audio_signal = convert_audio(audio_signal, model_sr, fs_audio, target_channels=1) #[2xN] if model 48kHz

    return resamp_audio_signal


def numpy_to_torch(X: np.array):
    '''
    Convert numpy array (N,) to expanded float torch tensor [1xN]. E.g. to feed to EncoDec 
    '''
    
    X = torch.from_numpy(X)
    X = X.expand(1, -1)
    X = X.to(torch.float)
    return X


def encodec_encode(audio: torch.Tensor, sr, model):
    '''
    Encode an audio sample into its embeddings and quantized codes.
    
    inputs:
        audio: torch.Tensor (torch.Size([1, samples]))
        sr: audio sampling rate
        model: encodec.model.EncodecModel
    '''
    
    # Interpolate audio to desired model sample_rate / n_channels:
    original_sr, target_sr = sr, model.sample_rate
    audio = convert_audio(audio, original_sr, target_sr, model.channels) #[2xN] if model 48kHz
    audio = audio.unsqueeze(0) # [1x2xN]

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(audio) 
   
    return encoded_frames, audio   


def encodec_encode_audio_array_2d(audio_array: np.ndarray, sr, model):
    '''
    Encode an audio sample into its embeddings and quantized codes.
    
    inputs:
        audio_array: np.array (audio_trials, samples)
        sr: audio sampling rate
        model: encodec.model.EncodecModel
    '''
    
    # Embed motifs
    audio_embeddings, audio_codes, scales = [], [], []
    
    for m in audio_array:
        
        # Extract embeddings, codes, scale
        audio_embeddings.append(encodec_encode(numpy_to_torch(m), sr, model)[0][0][0]) # embeddings 
        audio_codes.append(encodec_encode(numpy_to_torch(m), sr, model)[0][0][1]) # codes
        scales.append(encodec_encode(numpy_to_torch(m), sr, model)[0][0][2]) # scales

    # Convert to torch tensors
    audio_embeddings = torch.squeeze(torch.stack(audio_embeddings), 1)
    audio_codes = torch.squeeze(torch.stack(audio_codes), 1)
    scales = torch.squeeze(torch.stack(scales), [1,2])

    return audio_embeddings, audio_codes, scales
