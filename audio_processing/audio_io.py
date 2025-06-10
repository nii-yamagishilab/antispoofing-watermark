# io.py
from typing import Union, Literal
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
from audiotools import AudioSignal
from pydub import AudioSegment
from torch import Tensor
from torchtyping import TensorType
import soundfile as sf

def get_waveform(
    audio_file: str,
    target_sr: int = 16000,
    device: str = "cpu",
    ) -> TensorType["channels", "samples"]:
    if not Path(audio_file).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    audio, orig_sr = torchaudio.load(audio_file)
    audio = audio.to(device)

    if orig_sr != target_sr:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr).to(device)
        audio = resampler(audio)
    
    return audio

def load_audio(
    audio_file: str,
    target_sr: int = 16000,
    backend: Literal["torchaudio", "librosa", "audiotools", "pydub"] = "torchaudio",
    device: str = "cpu"
    ) -> Union[torch.Tensor, AudioSignal, AudioSegment]:
    """
    Unified audio loader supporting multiple backends.

    Returns:
        - torch.Tensor for torchaudio/librosa
        - AudioSignal for audiotools
        - AudioSegment for pydub
    """
    if not Path(audio_file).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    if backend == "torchaudio":
        audio, orig_sr = torchaudio.load(audio_file)
        audio = audio.to(device)
        if orig_sr != target_sr:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr).to(device)
            audio = resampler(audio)
        return audio

    elif backend == "librosa":
        audio_np, _ = librosa.load(audio_file, sr=target_sr)
        return audio_np
    
    elif backend == "audiotools":
        signal = AudioSignal(audio_file)
        return signal

    elif backend == "pydub":
        return AudioSegment.from_file(str(audio_file)).set_frame_rate(target_sr)

    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
def save_waveform(
    audio: Tensor,
    output_path: Path,
    sample_rate: int = 16000
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio.T.cpu().detach().numpy(), sample_rate, subtype='PCM_16')

