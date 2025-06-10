import glob, os, random
import numpy as np
import librosa
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from audio_processing.audio_io import get_waveform, load_audio, save_waveform
# from torchaudio.io import CodecConfig, AudioEffector
import soundfile as sf
from typing import Optional
from audio_processing.perturbation import (
    pert_time_stretch,
    pert_encodec,
    pert_gaussian,
    pert_quantization,
    pert_smooth,
    pert_highpass,
    pert_lowpass,
    pert_echo,
    pert_hifigan,
    pert_pitchshift,
    pert_downsample,
    pert_none,
    pert_rir,
    pert_musan,
    pert_dac,
    pert_opus,
    pert_wav_tokenizer,
    pert_mp3compression,
    pert_amplification,
    pert_random_trimming,
    pert_frequency_mask,
    pert_clipping,
    pert_overdrive,
    pert_eq,
    pert_compressor,
    pert_noisegate,
    pert_noisereduction
)
# from audiomentations import (
#     Compose, 
#     PitchShift, 
#     SevenBandParametricEQ, 
#     HighPassFilter, 
#     LowPassFilter, 
#     )
# from pedalboard import (
#     Pedalboard, 
#     Compressor,
#     MP3Compressor,
#     )
# from audiotools import AudioSignal
# from pydub import AudioSegment
# import sys
# sys.path.append("/mnt/md0/user_max/toolkit/Chiahua_BCM/toolkit/DeepFilterNet/DeepFilterNet")
# from df import enhance, init_df
# import noisereduce as nr
# from noisereduce.generate_noise import band_limited_noise
from pathlib import Path
from torchtyping import TensorType
audio_file = "LA_E_5954030.wav"

# ====== Tunable Parameters (Extracted from Magic Numbers) ======
DEFAULT_SAMPLE_RATE = 16000
MUSAN_SNR_VALUES = [5, 10, 15]
GAUSSIAN_SNR_VALUES = [5, 10, 15]
QUANTIZATION_BITS = [8, 16, 24, 32]
DOWNSAMPLE_FREQS = [4000, 8000]
UPSAMPLE_FREQS = [22050, 24000, 44100, 48000]
MP3_VBR_QUALITIES = [0.0, 0.5, 1]
OPUS_BITRATES = [1, 2, 4, 8, 16, 31]
ENCODEC_BANDWIDTH_LIST = [1.5, 3, 6, 12, 24]
GAIN_VALUES = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
TIME_STRETCH_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
PITCHSHIFT_STEPS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
WINDOW_SIZES = [4, 8, 16]
HIGHPASS_CUTOFF = (20.0, 2400.0)
LOWPASS_CUTOFF = (1000.0, 5000.0)
DEFAULT_PROBABILITY = 1.0
FREQ_MASK_VALUES = [10, 20, 30, 40, 50, 60, 70, 80]
GAIN_RANGE = (0.0, 50.0)
COLOUR_RANGE = (0.0, 50.0)
EQ_GAIN_DB_RANGE = (-12, 12)
COMPRESSOR_RATIO_RANGE = (2.0, 10.0)
COMPRESSOR_THRESHOLD_DB_RANGE = (-50.0, -10.0)
NOISE_GATE_STD_THRESHOLDS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
# ===============================================================


def ensure_directory_exists(directory_path):
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")


def make_none(
    audio_file: str, 
    output_dir: str = None, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "None"
    ) -> TensorType["channels", "samples"]:
    # v
    processed = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")

    if output_dir is not None:
        output_path = Path(output_dir,Path(audio_file).stem+"_"+attack_name+".wav")
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed

def make_rir(
    audio_file: str, 
    output_dir: str = None, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "rir"
    ) -> TensorType["channels", "samples"]:
    
    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")
    processed = pert_rir(audio, sample_rate)

    if output_dir is not None:
        output_path = Path(output_dir,Path(audio_file).stem+"_"+attack_name+".wav")
        save_waveform(processed, output_path, sample_rate=sample_rate)

    return processed


def make_musan(
    audio_file: str,
    snr_value: Optional[int] = None,
    output_dir: Optional[str] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    attack_name: str = "musan"
) -> TensorType["channels", "samples"]:

    if snr_value is None:
        snr_value = random.choice(MUSAN_SNR_VALUES)
    
    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")
    processed = pert_musan(audio, snr_value)

    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}_snr{snr_value}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)

    return processed

def make_gaussian_noise(
    audio_file: str,
    snr: Optional[int] = None,
    output_dir: Optional[str] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    attack_name: str = "gaussian_noise"
) -> TensorType["channels", "samples"]:

    if snr is None:
        snr = random.choice(GAUSSIAN_SNR_VALUES)

    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")
    processed = pert_gaussian(audio, snr)

    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}_snr{snr}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)

    return processed


def make_quantization(
    audio_file: str,
    output_dir: Optional[str] = None,
    q_bit: Optional[int] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    attack_name: str = "quantization",
    ) -> TensorType["channels", "samples"]:
    
    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")
    
    if q_bit is None:
        q_bit = random.choice(QUANTIZATION_BITS)
    
    processed = pert_quantization(audio, q_bit)
    
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}_q{q_bit}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)

    return processed


def make_downsample(
    audio_file: str,
    output_dir: Optional[str] = None,
    intermediate_freq: Optional[int] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    attack_name: str = "downsample"
    ) -> TensorType["channels", "samples"]:

    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")
    if intermediate_freq is None:
        intermediate_freq = random.choice(DOWNSAMPLE_FREQS)

    processed = pert_downsample(audio, intermediate_freq)

    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}_ds{intermediate_freq}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    
    return processed

def make_upsample(
    audio_file: str,
    output_dir: Optional[str] = None,
    intermediate_freq: Optional[int] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    attack_name: str = "upsample"
    ) -> TensorType["channels", "samples"]:
    
    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")

    if intermediate_freq is None:
        intermediate_freq = random.choice(UPSAMPLE_FREQS)
    
    processed = pert_downsample(audio, intermediate_freq)

    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}_to{intermediate_freq}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    
    return processed

def make_mp3compression(
    audio_file: str,
    output_dir: Optional[str] = None,
    vbr_quality: Optional[float] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    attack_name: str = "mp3compression"
    ) -> TensorType["channels", "samples"]:
    
    if vbr_quality is None:
        vbr_quality = random.choice(MP3_VBR_QUALITIES)

    audio = load_audio(audio_file, target_sr=sample_rate, backend="librosa")
    processed = pert_mp3compression(audio, sample_rate, vbr_quality)

    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}_{vbr_quality}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    
    return processed


def make_opus(
    audio_file: str,
    output_dir: Optional[str] = None,
    q_bit: Optional[int] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    attack_name: str = "opus"
    ) -> TensorType["channels", "samples"]:
    
    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")

    if q_bit is None:
        q_bit = random.choice(OPUS_BITRATES)

    processed = pert_opus(audio, q_bit)

    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}_q{q_bit}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)

    return processed

# ogg_stream.close()  # Release memory allocated to ogg_stream

def make_encodec(
    audio_file: str, 
    output_dir: Optional[str] = None,
    bandwidth: Optional[int] = None,
    encodec_sampling_rate: int = 24000, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "encodec"
    ) -> TensorType["channels", "samples"]:

    bandwidth = random.choice(ENCODEC_BANDWIDTH_LIST)
    audio = load_audio(audio_file, target_sr=encodec_sampling_rate, backend="librosa")

    encodec_waveform = pert_encodec(audio, bandwidth)
    encodec_waveform = encodec_waveform.cpu().detach().numpy()
    if encodec_sampling_rate!=sample_rate:
        encodec_waveform = librosa.resample(encodec_waveform, orig_sr=encodec_sampling_rate, target_sr=sample_rate)
    
    processed = torch.from_numpy(encodec_waveform).squeeze(0)
    
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed

def make_dac(
    audio_file: str, 
    output_dir: str = None, 
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    dac_sampling_rate: int = 16000,
    attack_name:str = "dac"
    ) -> TensorType["channels", "samples"]:
    # v
    audio = load_audio(audio_file, backend="audiotools")
    processed = pert_dac(audio)
    if dac_sampling_rate != sample_rate:
        processed = torchaudio.functional.resample(processed, orig_freq=dac_sampling_rate, new_freq=sample_rate)
    
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed


def make_wavtokenizer(
    audio_file: str, 
    output_dir: str = None, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    wavtokenizer_sr:int = 24000,
    attack_name:str = "wavtokenizer"
    ) -> TensorType["channels", "samples"]:

    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")
    processed = pert_wav_tokenizer(audio, sample_rate)
    if wavtokenizer_sr != sample_rate:
        processed = torchaudio.functional.resample(processed, orig_freq=wavtokenizer_sr, new_freq=sample_rate)
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed


def make_amplification(
    audio_file: str, 
    output_dir: str = None, 
    gain_factor: float = 3.0, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "amplification"
    ) -> TensorType["channels", "samples"]:
    # v
    gain_factor = random.choice(GAIN_VALUES)
    audio = load_audio(audio_file, target_sr=sample_rate, backend="librosa")

    processed = pert_amplification(audio, gain_factor)

    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed


def make_time_stretch(
    audio_file: str, 
    output_dir: str = None, 
    rate: float = 0.5, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "time_stretch"
    ) -> TensorType["channels", "samples"]:

    speed_factor = random.choice(TIME_STRETCH_VALUES)

    audio = load_audio(audio_file, target_sr=sample_rate, backend="librosa")
    processed = pert_time_stretch(audio, speed_factor)

    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed

def make_pitchshift(
    audio_file: str, 
    output_dir: str = None, 
    step: int = 5, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "pitchshift"
    ) -> TensorType["channels", "samples"]:

    step = random.choice(PITCHSHIFT_STEPS)

    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")
    processed =pert_pitchshift(audio, sample_rate, step)
    
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed


def make_smooth(
    audio_file: str, 
    output_dir: str = None, 
    window_size: int = 8, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "smooth",
    window_list: list = WINDOW_SIZES,
    random_window: bool = True
    ) -> TensorType["channels", "samples"]:

    if random_window:
        window_size = random.choice(window_list)

    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")
    processed = pert_smooth(audio, window_size)

    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed

def make_highpass_filtering(
    audio_file: str, 
    output_dir: str = None, 
    min_cutoff_freq: float = HIGHPASS_CUTOFF[0],
    max_cutoff_freq: float = HIGHPASS_CUTOFF[1],
    p: float = DEFAULT_PROBABILITY,
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "highpass_filtering"
    ) -> TensorType["channels", "samples"]:
    
    # https://iver56.github.io/audiomentations/waveform_transforms/high_pass_filter
    audio = load_audio(audio_file, target_sr=sample_rate, backend="librosa")
    processed = pert_highpass(audio, sample_rate, min_cutoff_freq, max_cutoff_freq, p)
    
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed

def make_lowpass_filtering(
    audio_file: str, 
    output_dir: str = None, 
    min_cutoff_freq: float = LOWPASS_CUTOFF[0],
    max_cutoff_freq: float = LOWPASS_CUTOFF[1],
    p: float = DEFAULT_PROBABILITY,
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "lowpass_filtering"
    ) -> TensorType["channels", "samples"]:
    # v
    audio = load_audio(audio_file, target_sr=sample_rate, backend="librosa")
    processed = pert_lowpass(audio, sample_rate, min_cutoff_freq, max_cutoff_freq, p)
    
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed
    

def make_random_trimming(
    audio_file: str, 
    output_dir: str = None, 
    sample_rate:int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "random_trimming"
    ) -> TensorType["channels", "samples"]:

    audio = load_audio(audio_file, target_sr=sample_rate, backend="pydub")
    processed = pert_random_trimming(audio)
  
    if output_dir is not None :
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
        # processed.export(output_file, format="wav", parameters=["-acodec", "pcm_s16le"])
    return processed
    


def make_frequency_masking(
    audio_file: str, 
    output_dir: str = None, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "frequency_masking"
    ) -> TensorType["channels", "samples"]:

    audio = load_audio(audio_file, target_sr=sample_rate, backend="librosa")
    freq_mask_param = random.choice(FREQ_MASK_VALUES)
    processed = pert_frequency_mask(audio,freq_mask_param)

    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed
    
def make_clipping(
    audio_file: str, 
    output_dir: str = None, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "clipping"
    ) -> TensorType["channels", "samples"]:

    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")
    lower_bound = np.percentile(audio, 1)
    upper_bound = np.percentile(audio, 99)

    processed = pert_clipping(audio, lower_bound, upper_bound)
    
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed
    
def make_overdrive(
    audio_file: str, 
    output_dir: str = None, 
    gain: float = 20.0, 
    colour: float = 0.5, 
    gain_range: tuple = GAIN_RANGE,
    colour_range: tuple = COLOUR_RANGE,
    sample_rate:int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "overdrive",
    random_params: bool = True
    ) -> TensorType["channels", "samples"]:
    # v
    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")
    # Randomly generate gain and colour
    if random_params:
        gain = random.uniform(*gain_range)
        colour = random.uniform(*colour_range)

    processed = pert_overdrive(audio, gain, colour)
    
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed
    
def make_eq(
    audio_file: str, 
    output_dir: str = None, 
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    min_gain_db: float = EQ_GAIN_DB_RANGE[0],
    max_gain_db: float = EQ_GAIN_DB_RANGE[1],
    p: float = DEFAULT_PROBABILITY,
    attack_name:str = "eq"
    ) -> TensorType["channels", "samples"]:
    audio = load_audio(audio_file, target_sr=sample_rate, backend="librosa")

    processed = pert_eq(audio, sample_rate, min_gain_db, max_gain_db, p)
    
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed
    
    

def make_compressor(
    audio_file: str, 
    output_dir: str = None, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    ratio_range: tuple = COMPRESSOR_RATIO_RANGE,
    threshold_db_range: tuple = COMPRESSOR_THRESHOLD_DB_RANGE,
    attack_name:str = "compressor"
    ) -> TensorType["channels", "samples"]:

    audio = load_audio(audio_file, target_sr=sample_rate, backend="librosa")
    ratio = random.uniform(*ratio_range)
    threshold_db = random.uniform(*threshold_db_range)
    processed = pert_compressor(audio, sample_rate, ratio, threshold_db)

    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed
    

def make_noise_gate(
    audio_file: str, 
    output_dir: str = None, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "noise_gate"
    ) -> TensorType["channels", "samples"]:

    audio = load_audio(audio_file, target_sr=sample_rate, backend="librosa")
    # Randomly generate n_std_thresh_stationary within a range
    threshold = random.choice(NOISE_GATE_STD_THRESHOLDS)
    processed = pert_noisegate(audio, sample_rate, threshold)
    
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed
    

def make_noise_reduction(
    audio_file: str, 
    output_dir: str = None, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    attack_name:str = "noise_reduction"
    ) -> TensorType["channels", "samples"]:
    audio = load_audio(audio_file, target_sr=sample_rate, backend="torchaudio")
    processed = pert_noisereduction(audio)
    
    if output_dir is not None:
        output_path = Path(output_dir) / f"{Path(audio_file).stem}_{attack_name}.wav"
        save_waveform(processed, output_path, sample_rate=sample_rate)
    return processed
    


def main():
    
    dir_path = "output2"
    ensure_directory_exists(dir_path)
    print("Start attacking audio files !")
    make_rir(audio_file, output_dir=dir_path)
    print("make_rir")
    make_musan(audio_file, output_dir=dir_path)
    print("make_musan")
    make_gaussian_noise(audio_file, output_dir=dir_path)
    print("make_gaussian_noise")
    make_quantization(audio_file, output_dir=dir_path)
    print("make_quantization")
    make_downsample(audio_file, output_dir=dir_path)
    print("make_downsample")
    make_upsample(audio_file, output_dir=dir_path)
    print("make_upsample")
    make_mp3compression(audio_file,output_dir=dir_path)
    print("make_mp3compression")
    make_opus(audio_file, output_dir=dir_path)
    print("make_opus")
    make_encodec(audio_file, output_dir=dir_path)
    print("make_encodec")
    make_dac(audio_file, output_dir=dir_path)
    print("make_dac")
    make_wavtokenizer(audio_file, output_dir=dir_path)
    print("make_wavtokenizer")
    make_amplification(audio_file, output_dir=dir_path)
    print("make_amplification")
    make_time_stretch(audio_file, output_dir=dir_path)
    print("make_time_stretch")
    make_pitchshift(audio_file, output_dir=dir_path)
    print("make_pitchshift")
    make_smooth(audio_file, output_dir=dir_path)
    print("make_smooth")
    make_highpass_filtering(audio_file, output_dir=dir_path)
    print("make_highpass_filtering")
    make_lowpass_filtering(audio_file, output_dir=dir_path)
    print("make_lowpass_filtering")
    make_random_trimming(audio_file, output_dir=dir_path)
    print("make_random_trimming")
    make_frequency_masking(audio_file, output_dir=dir_path)
    print("make_frequency_masking")
    make_clipping(audio_file, output_dir=dir_path)
    print("make_clipping")
    make_overdrive(audio_file, output_dir=dir_path)
    print("make_overdrive")
    make_eq(audio_file, output_dir=dir_path)
    print("make_eq")
    make_compressor(audio_file, output_dir=dir_path)
    print("make_compressor")
    make_noise_gate(audio_file, output_dir=dir_path)
    print("make_noise_gate")
    make_noise_reduction(audio_file, output_dir=dir_path)
    print("make_noise_reduction")
if __name__ == "__main__":
    main()



