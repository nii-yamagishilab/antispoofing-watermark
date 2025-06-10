import json
from typing import Dict
import torch
import torchaudio
import sys
import models
from importlib import import_module


class aasistWrapper:
    def __init__(self, config_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._load_config(config_path)
        self.model = self._load_model_from_config(self.config, self.device)

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, "r") as f:
            return json.load(f)

    def _get_model(self, model_config: Dict) -> torch.nn.Module:
        module = import_module(f"models.{model_config['architecture']}")
        ModelClass = getattr(module, "Model")
        model = ModelClass(model_config).to(self.device)
        return model

    def _load_model_from_config(self, config: Dict, device: torch.device) -> torch.nn.Module:
        model = self._get_model(config["model_config"])
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        model.eval()
        return model

    def infer(self, audio_tensor: torch.Tensor) -> float:
        audio_tensor = audio_tensor.to(self.device)
        _, batch_out = self.model(audio_tensor)
        batch_score = batch_out[:, 1].data.cpu().numpy().ravel()
        return batch_score[0]

    def test(self, audio_path: str) -> float:
        audio_tensor, _ = torchaudio.load(audio_path)
        return self.infer(audio_tensor)



if __name__ == "__main__":
    config = "config/AASIST.conf"
    wrapper = Wrapper(config)
    audio_path = "/mnt/md0/user_max/toolkit/Chiahua_BCM/toolkit/exp/ASVspoof2019_LA/wav/LA_E_2323050.wav"
    sco = wrapper.test(audio_path)
    print(sco)
