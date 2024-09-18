import os
import numpy as np
import soundfile as sf
from scipy import signal
import random

def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = np.sqrt(np.mean(y ** 2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y, rms, scalar


def is_clipped(y, clipping_threshold=0.999):
    return (np.abs(y) > clipping_threshold).any()


def subsample(data, sub_sample_length, nchannels):
    """
    Subsamples or pads the input data to a fixed length.

    Parameters:
    - data: Input numpy array. Can be 1D (samples,) or 2D (channels, samples)
    - sub_sample_length: Target length of the samples axis
    - nchannels: Number of channels (used when padding 2D data)

    Returns:
    - data: Subsampled or padded data of length sub_sample_length
    """
    ndim = data.ndim
    if ndim == 1:
        length = data.shape[0]
    elif ndim == 2:
        length = data.shape[1]
    else:
        raise ValueError(f"Only support 1D and 2D data. The dim is {ndim}")

    if length > sub_sample_length:
        start = np.random.randint(length - sub_sample_length)
        end = start + sub_sample_length
        if ndim == 1:
            data = data[start:end]
        else:
            data = data[:, start:end]
        assert data.shape[-1] == sub_sample_length
    elif length < sub_sample_length:
        pad_width = sub_sample_length - length
        if ndim == 1:
            data = np.pad(data, (0, pad_width), mode='constant')
        else:
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
    # If length == sub_sample_length, data remains unchanged

    return data

# Function to extract direct path from RIR
def get_direct_path_rir(rir):
    if rir.ndim == 1:
        rir = rir[:, np.newaxis]
    num_samples, num_channels = rir.shape
    direct_rir = np.zeros_like(rir)
    for ch in range(num_channels):
        channel_rir = rir[:, ch]
        max_idx = np.argmax(np.abs(channel_rir))
        direct_rir[max_idx, ch] = channel_rir[max_idx]
    return direct_rir

# Function to apply multi-channel RIR to clean audio
def apply_multi_channel_rir(clean_y, rir):
    """
    Apply a multi-channel Room Impulse Response (RIR) to a single-channel clean signal.
    Outputs a multi-channel signal where each output channel is the convolution of the clean signal with each channel of the RIR.
    """
    if rir.ndim == 1:
        # If RIR is mono, simply convolve it
        return signal.fftconvolve(clean_y, rir, mode='full')[:len(clean_y)]

    # Initialize an empty array to store the convolved multi-channel output
    num_channels = rir.shape[1]
    output_length = len(clean_y) + len(rir) - 1  # Length of the output from full convolution
    convolved_signals = np.zeros((output_length, num_channels))

    # Convolve each channel of the RIR with the clean signal
    for channel in range(num_channels):
        convolved_signals[:, channel] = signal.fftconvolve(clean_y, rir[:, channel], mode='full')[:output_length]

    # Optionally, trim the convolved signal to the original length of clean_y
    convolved_signals = convolved_signals[:len(clean_y), :].T

    return convolved_signals

# Function to select noise segments of appropriate length
def select_noise_y(noise_dataset_list, target_length, sr, silence_length=0.2, nchannels=6):
    """
    Selects noise segments and appends silence between them to create a noise signal of the desired length.
    
    Parameters:
    - noise_dataset_list: List of paths to noise audio files.
    - target_length: Desired length of the output noise signal in samples.
    - sr: Sampling rate.
    - silence_length: Length of the silence segments to insert between noise segments (in seconds).
    - nchannels: Number of channels in the audio data.
    
    Returns:
    - noise_y: The constructed noise signal of shape (nchannels, target_length).
    """
    noise_y = np.zeros((nchannels, 0), dtype=np.float32)
    silence = np.zeros((nchannels, int(sr * silence_length)), dtype=np.float32)
    remaining_length = target_length

    while remaining_length > 0:
        # Randomly select a noise file
        noise_file = random.choice(noise_dataset_list)
        noise_new_added, sr_ = sf.read(noise_file, dtype='float32')
        assert sr == sr_
        noise_new_added = noise_new_added.T  # Shape: (channels, samples)
        
        noise_y = np.append(noise_y, noise_new_added, axis=1)
        remaining_length -= noise_new_added.shape[1]

        # If more noise is needed, insert a silence segment
        if remaining_length > 0:
            silence_len = min(remaining_length, silence.shape[1])
            noise_y = np.append(noise_y, silence[:, :silence_len], axis=1)
            remaining_length -= silence_len

    # Trim or randomly crop the noise_y to the target_length
    total_noise_length = noise_y.shape[1]
    if total_noise_length > target_length:
        idx_start = np.random.randint(total_noise_length - target_length)
        noise_y = noise_y[:, idx_start:idx_start + target_length]

    return noise_y


# Function to mix clean and noise audio at a given SNR
def snr_mix(clean_y, noise_y, snr, target_dB_FS, target_dB_FS_floating_value, rir_folder=None, eps=1e-6):
    if rir_folder is not None:
        rir_files = [os.path.join(rir_folder, f) for f in os.listdir(rir_folder) if f.endswith('.wav')]
        rir_file = random.choice(rir_files)
        rir, sr = sf.read(rir_file)
        direct_rir = get_direct_path_rir(rir)
        clean_direct = apply_multi_channel_rir(clean_y, direct_rir)
        clean_y = apply_multi_channel_rir(clean_y, rir)
    else:
        clean_direct = clean_y

    clean_y, _ = norm_amplitude(clean_y)
    clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
    clean_rms = np.sqrt((clean_y ** 2).mean())
    
    clean_direct, _ = norm_amplitude(clean_direct)
    clean_direct, _, _ = tailor_dB_FS(clean_direct, target_dB_FS)
    
    noise_y, _ = norm_amplitude(noise_y)
    noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
    noise_rms = np.sqrt((noise_y ** 2).mean())

    snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
    noise_y *= snr_scalar
    noisy_y = clean_y + noise_y

    # Randomly select RMS value of dBFS between target_dB_FS +- target_dB_FS_floating_value
    noisy_target_dB_FS = random.randint(target_dB_FS - target_dB_FS_floating_value, target_dB_FS + target_dB_FS_floating_value)
    
    noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
    clean_direct *= noisy_scalar

    # Check for clipping
    if is_clipped(noisy_y):
        noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)
        noisy_y = noisy_y / noisy_y_scalar
        clean_direct = clean_direct / noisy_y_scalar

    return noisy_y, clean_direct

# Function to generate the noisy dataset
def generate_noisy_dataset(
    clean_dataset_dir,
    noise_dataset_dir,
    output_dir,
    snr_range=(-3, 10),
    rir_folder=None,
    target_dB_FS=-25,
    target_dB_FS_floating_value=10,
    sub_sample_length=3,
    sr=16000,
    nchannels=6,
    ref_channel=4,
    selected_channels=[0, 1 ,2, 3, 4, 5]
):
    # Prepare dataset lists
    clean_dataset_list = os.listdir(os.path.abspath(os.path.expanduser(clean_dataset_dir)))
    noise_dataset_list = os.listdir(os.path.abspath(os.path.expanduser(noise_dataset_dir)))
    clean_dataset_list.sort()
    noise_dataset_list.sort()
    
    clean_dataset_list = [os.path.join(os.path.abspath(os.path.expanduser(clean_dataset_dir)), cp) for cp in clean_dataset_list]
    noise_dataset_list = [os.path.join(os.path.abspath(os.path.expanduser(noise_dataset_dir)), np) for np in noise_dataset_list]
    
    snr_list = list(range(snr_range[0], snr_range[1]+1))
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over the dataset
    for i in range(len(clean_dataset_list)):
        clean_file = clean_dataset_list[i]
        clean_y, sr_ = sf.read(clean_file, dtype='float32')
        clean_y = clean_y.T
        assert sr == sr_
        
        #clean_y = clean_y.T  # Shape: (channels, samples)
        
        # Subsample or pad the clean audio
        clean_y = subsample(clean_y, sub_sample_length=int(sub_sample_length * sr), nchannels=nchannels)
        
        # Get corresponding noise segment
        target_length = clean_y.shape[0]
        noise_y = select_noise_y(noise_dataset_list, target_length=target_length, sr=sr, silence_length=0.2, nchannels=nchannels)
        
        # Randomly select an SNR value
        snr = random.choice(snr_list)
        
        # Mix clean and noise audio
        noisy_y, clean_y_proc = snr_mix(
            clean_y=clean_y,
            noise_y=noise_y,
            snr=snr,
            target_dB_FS=target_dB_FS,
            target_dB_FS_floating_value=target_dB_FS_floating_value,
            rir_folder=rir_folder  
        )
        
        # Select the desired channels
        noisy_y_selected = noisy_y[selected_channels].astype(np.float32)
        # For clean_y, select the reference channel
        if clean_y_proc.ndim == 1:
            clean_y_proc = clean_y_proc[np.newaxis, :]
        else:
            clean_y_proc = clean_y_proc[ref_channel, :].astype(np.float32)
            clean_y_proc = clean_y_proc[np.newaxis, :]
        
        # Get the file name without extension
        file_name = os.path.splitext(os.path.basename(clean_file))[0]
        
        # Save the noisy signal to the output directory
        sf.write(os.path.join(output_dir, f"{file_name}.wav"), noisy_y_selected.T, sr)

# Parameters (adjust these as needed)
clean_dataset_dir = ""
noise_dataset_dir = ""
output_dir = ""
snr_range = (-3, 10)
rir_folder = ""  # Set to your RIR folder
target_dB_FS = -25
target_dB_FS_floating_value = 10
sub_sample_length = 3
sr = 16000
nchannels = 6
ref_channel = 4
selected_channels=[0, 1 ,2, 3, 4, 5]

if __name__ == "__main__":
    generate_noisy_dataset(
        clean_dataset_dir=clean_dataset_dir,
        noise_dataset_dir=noise_dataset_dir,
        output_dir=output_dir,
        snr_range=snr_range,
        rir_folder=rir_folder,
        target_dB_FS=target_dB_FS,
        target_dB_FS_floating_value=target_dB_FS_floating_value,
        sub_sample_length=sub_sample_length,
        sr=sr,
        nchannels=nchannels,
        ref_channel=ref_channel,
        selected_channels=selected_channels
    )
