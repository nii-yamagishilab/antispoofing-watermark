import json
import sys
from typing import Dict

import torch
import torchaudio

# Add project path
sys.path.append('./SSL_Anti-spoofing')
from model import Model 

class SSLASVWrapper:
    def __init__(self, model_path: str, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.model = Model(args, self.device).to(self.device)
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load model weights."""
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def infer(self, audio: torch.Tensor) -> float:
        """Run inference on input audio and return the score."""
        audio = audio.to(self.device)
        output = self.model(audio)
        score = output[:, 1].data.cpu().numpy().ravel()
        return score[0]

    def test(self, wav_path: str):
        """Test function using a local WAV file."""
        audio, _ = torchaudio.load(wav_path)
        print(f"Input shape: {audio.shape}")
        score = self.infer(audio)
        print(f"Score: {score}")


if __name__ == "__main__":
    model_path = "/mnt/md0/user_max/toolkit/Chiahua_BCM/model_zoo/SSL_Anti-spoofing/Pre_trained_SSL_anti-spoofing_models/LA_model.pth"
    wrapper = SSLASVWrapper(model_path)
    audio_path = "LA_E_5954030.wav"
    wrapper.test(audio_path)
