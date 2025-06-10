import os
from torch.nn import functional as F
import torch
import torchaudio
import torchaudio.transforms as T
import glob, os, random
import numpy as np
import librosa
import io
import random
import string

"""DAC Codec
github: https://github.com/descriptinc/descript-audio-codec
install: pip install descript-audio-codec
usage:
"""
import dac
from audiomentations import (
    Compose, 
    PitchShift, 
    SevenBandParametricEQ, 
    HighPassFilter, 
    LowPassFilter, 
    )
from pedalboard import (
    Pedalboard, 
    Compressor,
    MP3Compressor,
    )
import sys
sys.path.append("/mnt/md0/user_max/toolkit/Chiahua_BCM/toolkit/DeepFilterNet/DeepFilterNet")
from df import enhance, init_df
import noisereduce as nr
# from noisereduce.generate_noise import band_limited_noise

dac_model_path = dac.utils.download(model_type="16khz")
dac_model = dac.DAC.load(dac_model_path)
dac_model.cuda()
dac_model.eval()
def pert_dac(waveform):
    # waveform [1, T]
    waveform = waveform.to('cuda')

    x = dac_model.preprocess(waveform.audio_data, waveform.sample_rate)
    z, codes, latents, _, _ = dac_model.encode(x)

    codec_waveform = dac_model.decode(z)
    # codec_waveform [1, 1, T] -> [1, T]
    codec_waveform = codec_waveform.squeeze(0)
    # if codec_waveform.shape[-1] < waveform.shape[-1]:
    #     codec_waveform = F.pad(codec_waveform,\
    #                                        (0, waveform.shape[-1]\
    #                                        - codec_waveform.shape[-1]))
    return codec_waveform
    # waveform = waveform.cpu()
    # x = dac_model.compress(waveform)
    # # Decompress it back to an AudioSignal
    # y = dac_model.decompress(x)
    # return y

"""Encodec
github: https://github.com/facebookresearch/encodec
install: pip install git+https://github.com/huggingface/transformers.git@main
usage:
"""
from transformers import EncodecModel, AutoProcessor
model_encodec = None
processor_encodec = None


def pert_encodec(
    waveform, 
    bandwidth,
):
    """Do encodec attack on any (but should be watermarked) waveform.

    Args:
        waveform:
            The input waveform, shape will be forced to [T,].
        bandwidth:
            Compress the input waveform to [1.5, 3, 6, 12, 24] kbps,
            must choose numbers listed here.
    Returns:
        encodec_waveform:
            Waveform reconstructed by the encodec model, [1, T] tensor.
    """
    # Encodec model only works for 24kHz
    sample_rate = 24000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the models only if they are not already initialized
    global model_encodec, processor_encodec
    if model_encodec is None or processor_encodec is None:
        model_encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
        processor_encodec = AutoProcessor.from_pretrained("facebook/encodec_24khz")


    model_encodec.eval()
    waveform = waveform.squeeze()

    inputs = processor_encodec(raw_audio=waveform, sampling_rate=24000, return_tensors="pt")
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    encoder_outputs = model_encodec.encode(inputs["input_values"], inputs["padding_mask"], bandwidth)
    encodec_waveform = model_encodec.decode(encoder_outputs.audio_codes,\
                                            encoder_outputs.audio_scales,\
                                            inputs["padding_mask"],)[0]

    # encodec_waveform = encodec_waveform.squeeze(0)

    return encodec_waveform

"""WavTokenizer
github: https://github.com/jishengpeng/WavTokenizer
install: Try your own env first; they have requirement.txt 
usage:
tested under ../WavTokenizer/
"""


import sys
sys.path.append("/mnt/md0/user_max/toolkit/Chiahua_BCM/toolkit/WavTokenizer")
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer
device=torch.device('cpu')
model_path = '/mnt/md0/user_max/toolkit/Chiahua_BCM/model_zoo/wavtokenizer/WavTokenizer_small_600_24k_4096.ckpt'
config_path = '/mnt/md0/user_max/toolkit/Chiahua_BCM/model_zoo/wavtokenizer/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml'
wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)
wavtokenizer = wavtokenizer.to(device)

def pert_wav_tokenizer(
    waveform,
    original_sr
):
    """Do encodec attack using WavTokenizer

    Args:
        waveform:
            The input waveform, should be [1, T]
        original_sr:
            Original sampling rate, since the input waveform will be forced to 24kHz
    Returns:
        audio_out:
            Waveform reconstructed by WavTokenizer, [1, T] tensor.
    """
    waveform = convert_audio(waveform, original_sr, 24000, 1)
    bandwidth_id = torch.tensor([0])
    waveform = waveform.to(device)
    features,discrete_code = wavtokenizer.encode_infer(waveform, bandwidth_id=bandwidth_id)
    audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)

    

    return audio_out

"""OPUS
github: https://github.com/moyangkuo/AudioMarkBench/blob/main/no-box/nobox_audioseal_audiomarkdata.py
install: pip3 install opuspy
usage:
bitrate_list = [1, 2, 4, 8 ,16, 31]
"""
import opuspy
def pert_opus(
    waveform,
    bitrate,
):
    """Save audio into a .opus file and read it back

    Args:
        waveform:
            The input waveform, should be [1, T]
        bitrate:
            Setting for opus write, the ones used in AudioMarkbench is [1, 2, 4, 8 ,16, 31]
    Returns:
        audio_out:
            Waveform compressed by opus, [1, T] tensor.
    """

    bitrate = 1000 * bitrate
    waveform_scaled = waveform * 32768
    waveform_scaled = waveform_scaled.reshape(-1,1).numpy()
    random_string = ''.join(random.choices(string.ascii_letters, k=10))
    temp_file = '/tmp/'+random_string+".opus"
    opuspy.write(temp_file, waveform_scaled, sample_rate = 16000, 
                bitrate = bitrate, signal_type = 0, encoder_complexity = 1)    
    pert_waveform, sampling_rate = opuspy.read(temp_file)
    os.remove(temp_file)
    pert_waveform = torch.tensor(pert_waveform, dtype=torch.float32).reshape(1,-1)
    pert_waveform /= 32768
    pert_waveform = torchaudio.functional.resample(
        pert_waveform,
        orig_freq=48000,
        new_freq=16000,
    )
    return pert_waveform


rir_path = "/mnt/md0/user_max/toolkit/Chiahua_BCM/dataset/noise_dataset/RIRS_NOISES/simulated_rirs"
rir_files = glob.glob(os.path.join(rir_path, "*/*/*.wav"))
musan_path = "/mnt/md0/user_max/toolkit/Chiahua_BCM/dataset/noise_dataset/musan/noise"
musan_files = glob.glob(os.path.join(musan_path,'*/*.wav'))
noiselist = {}
for file in musan_files:
    if file.split('/')[-4] not in noiselist:
        noiselist[file.split('/')[-3]] = []
    noiselist[file.split('/')[-3]].append(file)

def pert_rir(
    waveform,
    sample_rate
):
    rir_file = random.choice(rir_files) 
    rir, sr_rir = torchaudio.load(rir_file)
    if sample_rate != sr_rir:
        raise ValueError("Sample rates of audio and RIR must match.")

    # rir = rir / torch.sqrt(torch.sum(rir ** 2))
    # return torchaudio.functional.convolve(waveform.cuda(), rir.cuda(), mode="same")
    rir = rir / (torch.sqrt(torch.sum(rir ** 2)) + 1e-8)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # 添加 channel 維度
    if rir.ndim == 1:
        rir = rir.unsqueeze(0)  # 添加 channel 維度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    waveform = waveform.to(device)
    rir = rir.to(device)
    # print(f"Waveform device: {waveform.device}, RIR device: {rir.device}")

    result = torchaudio.functional.convolve(waveform, rir)

    result = result[..., :waveform.size(-1)]

    return result

def pert_musan(
    waveform,
    snr_value,
):
    noise_file = random.choice(musan_files)
    # snr = random.randint(1, 10) 
    snr = torch.tensor([snr_value], dtype=torch.float32)
    noise, sr = torchaudio.load(noise_file)
    while noise.shape[-1] < waveform.shape[-1]:
        noise = torch.cat([noise, noise], dim=-1)

    wav_len = waveform.shape[-1]
    noise_len = noise.shape[-1]
    if noise_len > wav_len:
        start = random.randint(0, noise_len - wav_len)
    else:
        start = 0
    noise = noise[:, start:start + wav_len]
    waveform = waveform / torch.max(torch.abs(waveform))
    noise = noise / torch.max(torch.abs(noise))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    waveform = waveform.to(device)
    noise = noise.to(device)
    snr = snr.to(device)

    result = torchaudio.functional.add_noise(waveform, noise, snr)

    return result

def pert_time_stretch(
    audio, 
    speed_factor,
):
    """Do time stretch attack on any (but should be watermarked) waveform.

    Args:
        waveform:
            The input waveform, shape will be forced to [T,].
        speed_factor:
            A factor to control the rate - speed up (>1)or slow down (<1).
    """
    waveform_stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
    result = torch.from_numpy(waveform_stretched).unsqueeze(0)

    return result


# Used in pert_encodec()
model_encodec = None
processor_encodec = None



def pert_gaussian(
    waveform, 
    snr_db,
):
    """Add gaussian noise to the given waveform.

    Args:
        waveform:
            The input waveform, doesn't need specify the shape.
        snr_db:
            The target signal-to-noise ratio in db.
    Returns:
        noisy_waveform:
            Input waveform added with gaussian noise, should have same shape
            as the input.
    """
    waveform_power = torch.mean(waveform**2).to(device=waveform.device)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = waveform_power / snr_linear
    noise = torch.randn(waveform.size()).to(device=waveform.device) * torch.sqrt(noise_power)
    noisy_waveform = waveform + noise
    snr = compute_snr(waveform, noisy_waveform)
    #     print(f"target snr:{snr_db}, calculated snr:{snr:.2f}")

    return noisy_waveform


def pert_quantization(
    waveform, 
    quantization_bit,
):
    """Perform quantization to the given waveform, i.e., mapping continuous
       waveform values to a smaller set of discrete values.

    Args:
        waveform:
            The input waveform, doesn't need specify the shape.
        quantization_bit:
            Size of the targeting quantized value set, larger bit gives relatively
            less rounding errors.
    Returns:
        rescaled_waveform:
            The quantized waveform, should have same shape as the input.
    """

    # Normalize the waveform to the range of the quantization levels
    min_val, max_val = waveform.min(), waveform.max()
    normalized_waveform = (waveform - min_val) / (max_val - min_val)

    # Quantize the normalized waveform
    quantized_waveform = torch.round(normalized_waveform * (quantization_bit - 1))

    # Rescale the quantized waveform back to the original range
    rescaled_waveform = (quantized_waveform / (quantization_bit - 1)) \
                        * (max_val - min_val) + min_val

    return rescaled_waveform

def pert_smooth(waveform, window_size):
    """Perform moving average smoothing to the input waveform

    Args:
        waveform:
            The input waveform, will be forced to [1, 1, T] if not
        window_size:
            The length of the window
    Returns:
        smoothed:
            The smoothed waveform, should be [1, T] 
    """
    # Ensure waveform has the correct shape: (batch_size=1, channels=1, length)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    
    window_size = int(window_size)
    # Create a uniform smoothing kernel
    kernel = torch.ones(1, 1, window_size, dtype=waveform.dtype, \
                        device=waveform.device) / window_size

    # Compute appropriate padding to maintain the same output length
    padding = (window_size - 1) // 2

    # Apply convolution with padding
    smoothed = F.conv1d(waveform, kernel, padding=padding)

    # If window_size is even, output length might be off by one, 
    # so we trim or pad as necessary
    if smoothed.shape[-1] > waveform.shape[-1]:
        smoothed = smoothed[..., :waveform.shape[-1]]
    elif smoothed.shape[-1] < waveform.shape[-1]:
        smoothed = F.pad(smoothed, (0, waveform.shape[-1] - smoothed.shape[-1]))

    # Make sure the output shape is [1, T]
    smoothed = smoothed.squeeze(0)

    return smoothed

def pert_highpass(waveform, sample_rate, min_cutoff_freq, max_cutoff_freq, p=1.0):
    """Pass the input waveform to a highpass filter conditioned by a float ratio,
       this function uses the julius package

    Args:
        waveform:
            The input waveform
        cutoff_ratio:
            A float number between [0, 0.5], expressed as f/fs, where f is the cutoff
            frequency and fs is the sampling rate.
            Example:
            ratio = 0.1 to cut frequency bins below 1600Hz of a 16kHz sampled waveform
            ratio = 0.5 to cut frequency bins below 8000Hz, i.e., all frequency. 
    Returns:
        The processed waveform by julius.highpass_filter()
    """
    highpass_filter = HighPassFilter(
        min_cutoff_freq=min_cutoff_freq, 
        max_cutoff_freq=max_cutoff_freq, 
        p=p
        )
    filtered_audio = highpass_filter(waveform, sample_rate=sample_rate)
    processed = torch.from_numpy(filtered_audio).unsqueeze(0)
    return processed

def pert_lowpass(waveform, sample_rate, min_cutoff_freq, max_cutoff_freq, p=1.0):
    """Similar to pert_highpass() but uses a lowpass filter

    """
    lowpass_filter = LowPassFilter(
        min_cutoff_freq=min_cutoff_freq,
        max_cutoff_freq=max_cutoff_freq, 
        p=p
        )
    filtered_audio = lowpass_filter(waveform, sample_rate=sample_rate)
    processed = torch.from_numpy(filtered_audio).unsqueeze(0)

    return processed


def pert_pitchshift(audio, sample_rate, step):
    """Perform pitch shift to the input waveform

    Args:
        waveform:
            The input waveform
        step:
            The steps to shift waveform
    Returns:
        The shifted waveform by torchaudio
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # waveform = waveform.to(device)
    audio = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio
    processed = torchaudio.functional.pitch_shift(audio, sample_rate, n_steps=step)

    return processed

def pert_downsample(waveform, intermediate_freq):
    """Resample the input waveform to a new sampling rate, idk why I named it downsample

    Args:
        waveform:
            The input waveform
        intermediate_freq:
            The target sampling rate
    Returns:
        down_sampled_waveform:
            The resampled waveform but padded or cut to the same length as the input
    """
    temp_waveform = torchaudio.functional.resample(waveform,\
                                                   orig_freq=16000,\
                                                   new_freq=intermediate_freq)    
    down_sampled_waveform = torchaudio.functional.resample(temp_waveform,\
                                                           orig_freq=intermediate_freq,\
                                                           new_freq=16000)
    # if down_sampled_waveform.shape[1] < waveform.shape[1]:
    #     down_sampled_waveform = F.pad(down_sampled_waveform,\
    #                                        (0, waveform.shape[1]\
    #                                        - down_sampled_waveform.shape[1]))
    # elif down_sampled_waveform.shape[1] > waveform.shape[1]:
    #     down_sampled_waveform = down_sampled_waveform[:, : waveform.shape[1]]
    
    return down_sampled_waveform

def pert_none(waveform, dummy):
    """Return the unprocessed waveform

    """
    return waveform

def pert_hifigan(waveform, dummy):
    """Resynthesize the input waveform using a 16kHz version HIFIGAN Vocoder

    Args:
        waveform:
            The input waveform
        dummy:
            Does nothing, just pass a parameter for logging and saving in the main file
    Returns:
        new_waveform:
            The resynthesized waveform with same length as the input one
    """
    # Load a pretrained HIFIGAN Vocoder
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",\
                    savedir="/home/smg/gewanying/Log/aw-spk/timbre/pretrained_models/tts-hifigan-libritts-16kHz",\
                    run_opts={"device":"cuda"})
    # Get the mel spec
    spectrogram, _ = mel_spectogram(audio=waveform.squeeze(), sample_rate=16000,\
                        hop_length=256, win_length=1024, n_mels=80, n_fft=1024,\
                        f_min=0.0, f_max=8000.0, power=1, normalized=False,\
                        min_max_energy_norm=True, norm="slaney", mel_scale="slaney",\
                        compression=True)
    # Pass to vocoder
    new_waveform = hifi_gan.decode_batch(spectrogram)
    if new_waveform.shape[-1] > waveform.shape[-1]:
        new_waveform = new_waveform[..., :waveform.shape[-1]]
    return new_waveform


def pert_echo(waveform, duration, volume=0.4, sample_rate=16000):
    """Add echo noise to the input waveform where the echo starts after <duration> sec
       It's not technically reverberation, more like two identical but overlapped voices

    Args:
        waveform:
            The input waveform
        duration:
            A float number to control how many seconds the 2nd voice starts after the
            1st voice started
        volume:
            The volume of the 2nd voice relatively to the 1st voice
        sample_rate:
            The sampling rate of the input waveform, used for converting duration to
            the actual number of waveform samples
    Returns:
        reverbed_signal:
            The mixed waveform        
    """
    # Ensure tensor has shape [batch_size=1, channels=1, length]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, length)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # Shape: (1, channels, length)
    else:
        raise ValueError("Input tensor must be 1D or 2D")

    batch_size, channels, length = waveform.shape
    
    # Calculate the number of samples for the delay
    n_delay_samples = int(sample_rate * duration)

    # Create the impulse response for the echo
    impulse_response = torch.zeros(n_delay_samples + 1, dtype=waveform.dtype,\
                                   device=waveform.device)
    impulse_response[0] = 1.0  # Original signal
    impulse_response[-1] = volume  # Echo signal delayed by `duration`

    # Reshape impulse_response to match conv1d weight shape
    # conv1d expects weight of shape (out_channels, in_channels/groups, kernel_size)
    impulse_response = impulse_response.view(1, 1, -1)  # Shape: (1, 1, kernel_size)

    # Apply convolution to add echo effect
    # Since we're convolving the signal with the impulse response, we need to pad the input signal
    padded_input = F.pad(waveform, (0, impulse_response.shape[-1] - 1))  # Pad at the end
    reverbed_signal = F.conv1d(padded_input, impulse_response, groups=channels)

    # Ensure the output length matches the original input length
    reverbed_signal = reverbed_signal[..., :length]

    # Remove added dimensions to return to original shape
    reverbed_signal = reverbed_signal.squeeze(0)  # Shape: (channels, length)

    return reverbed_signal

def compute_snr(signal, noisy_signal):
    signal_power = torch.mean(signal**2)
    noise = noisy_signal - signal
    noise_power = torch.mean(noise**2)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr

def pert_mp3compression(audio, sample_rate, vbr_quality):

    # https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.MP3Compressor

    board = Pedalboard([MP3Compressor(vbr_quality)])
    processed = board(audio, sample_rate)
    processed = torch.from_numpy(processed).unsqueeze(0)
    return processed

def pert_amplification(audio, gain_factor):

    amplified_audio = audio * gain_factor
    max_val = np.max(np.abs(amplified_audio))
    if max_val > 1.0:
        amplified_audio = amplified_audio / max_val  # Normalize to [-1, 1]
    processed = torch.from_numpy(amplified_audio).unsqueeze(0)
    return processed

def pert_random_trimming(audio, max_duration_ms=None):
    if max_duration_ms is None:
        max_duration_ms = len(audio)
    min_duration_ms = 1000

    max_start_time_ms = len(audio) - min_duration_ms  # Ensure we have enough space for the trimming
    if len(audio) < min_duration_ms:
        original_audio = np.array(audio.get_array_of_samples()).astype(np.float32) / (2 ** (audio.sample_width * 8 - 1))
        original_audio = torch.tensor(original_audio)
        original_audio = original_audio.unsqueeze(0)
        return original_audio

    start_time_ms = random.randint(0, max(0, len(audio) - min_duration_ms))
    duration_ms = random.randint(min_duration_ms, min(max_duration_ms, len(audio) - start_time_ms))

    trimmed_audio = audio[start_time_ms:start_time_ms + duration_ms]
    # Convert to NumPy array and normalize to [-1, 1]
    trimmed_audio = np.array(trimmed_audio.get_array_of_samples()).astype(np.float32) / (2 ** (audio.sample_width * 8 - 1))
    trimmed_audio = torch.tensor(trimmed_audio)
    trimmed_audio = trimmed_audio.unsqueeze(0)
    return trimmed_audio

def pert_frequency_mask(audio, freq_mask_param):

    # Compute STFT and magnitude-phase separation
    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
    magnitude, phase = np.abs(stft), np.angle(stft)

    # Apply frequency masking
    db_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    spectrogram = torch.tensor(db_spectrogram, dtype=torch.float32).unsqueeze(0)  # shape: (1, freq, time)
    transform = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
    masked_spectrogram = transform(spectrogram).squeeze(0).numpy()

    # Reconstruct waveform
    masked_magnitude = librosa.db_to_amplitude(masked_spectrogram)
    processed = librosa.istft(masked_magnitude * np.exp(1j * phase), hop_length=512)
    # Normalize to match original peak
    processed = processed / np.max(np.abs(processed))  # peak normalize
    processed = torch.from_numpy(processed).unsqueeze(0)

    return processed

def pert_clipping(audio, lower_bound, upper_bound):
    processed = np.clip(audio, lower_bound, upper_bound)
    return processed

def pert_overdrive(audio, gain, colour):
    # https://pytorch.org/audio/main/generated/torchaudio.functional.overdrive.html
    processed =  torchaudio.functional.overdrive(audio, gain=gain, colour=colour)
    return processed 

def pert_eq(audio, sample_rate, min_gain_db, max_gain_db, p):
    # Define the EQ parameters (adjust as needed)
    augmenter = Compose([
        SevenBandParametricEQ(
            min_gain_db=min_gain_db,  # Minimum gain (dB) for each band
            max_gain_db=max_gain_db,   # Maximum gain (dB) for each band
            p=p  # Probability of applying the effect
        )
    ])
    processed = augmenter(audio, sample_rate=sample_rate)
    processed = torch.from_numpy(processed).unsqueeze(0)
    return processed

def pert_compressor(audio, sample_rate, ratio, threshold_db):

    board = Pedalboard([Compressor(threshold_db=threshold_db, ratio=ratio),])
    processed = board(audio, sample_rate)
    processed = torch.from_numpy(processed).unsqueeze(0)
    return processed

def pert_noisegate(audio, sample_rate, threshold):
    # pip install noisereduce
    # Noisereduce is a noise reduction algorithm in python that reduces noise in time-domain signals
    processed = nr.reduce_noise(y = audio, sr=sample_rate, n_std_thresh_stationary=threshold, stationary=True)
    processed = torch.from_numpy(processed).unsqueeze(0)
    return processed
    
def pert_noisereduction(audio):
    # pip install deepfilternet
    # https://github.com/Rikorose/DeepFilterNet/blob/main/README.md
    model, df_state, _ = init_df()  # Load default model
    processed = enhance(model, df_state, audio)
    return processed
    