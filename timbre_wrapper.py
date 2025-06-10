import sys
import yaml
import torch
import torchaudio
import numpy as np
import soundfile
import logging

# Path to model
sys.path.append("watermarking_model")
from watermarking_model.model.conv2_mel_modules import Encoder, Decoder

class TimbreWatermarkWrapper:
    def __init__(self,
                 config_dir: str,
                 checkpoint_path: str,
                 device: str = "cuda"):

        self.device = device

        process_config = yaml.load(open(f"{config_dir}/process.yaml"), Loader=yaml.FullLoader)
        model_config = yaml.load(open(f"{config_dir}/model.yaml"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(f"{config_dir}/train.yaml"), Loader=yaml.FullLoader)

        self.msg_length = train_config["watermark"]["length"]
        win_dim = process_config["audio"]["win_len"]
        embedding_dim = model_config["dim"]["embedding"]

        encoder_layers = model_config["layer"]["nlayers_encoder"]
        decoder_layers = model_config["layer"]["nlayers_decoder"]
        encoder_heads = model_config["layer"]["attention_heads_encoder"]
        decoder_heads = model_config["layer"]["attention_heads_decoder"]

        self.encoder = Encoder(process_config, model_config, self.msg_length, win_dim, embedding_dim,
                               nlayers_encoder=encoder_layers, attention_heads=encoder_heads).to(self.device)

        self.decoder = Decoder(process_config, model_config, self.msg_length, win_dim, embedding_dim,
                               nlayers_decoder=decoder_layers, attention_heads=decoder_heads).to(self.device)

        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"], strict=False)
        self.encoder.eval()
        self.decoder.eval()

    def embed(self, audio: torch.Tensor, watermark_bits: list) -> torch.Tensor:
        audio = audio.unsqueeze(0).to(self.device)
        msg = torch.tensor([[watermark_bits]], dtype=torch.float32).to(self.device) * 2 - 1
        with torch.no_grad():
            encoded_audio, _ = self.encoder.test_forward(audio, msg)
        return encoded_audio.squeeze(0).cpu()

    def detect(self, audio: torch.Tensor, watermark_bits: list = None):
        audio = audio.unsqueeze(0).to(self.device)
        with torch.no_grad():
            decoded = self.decoder.test_forward(audio)

        detected_bits = (decoded > 0).int().view(-1)

        if watermark_bits:
            gt = torch.tensor(watermark_bits, dtype=torch.float32).to(self.device) * 2 - 1
            gt = gt.view(decoded.shape)
            acc = (decoded >= 0).eq(gt >= 0).sum().float() / gt.numel()
            score = (decoded * gt).mean()
            return detected_bits.cpu(), acc.item(), score.item()
        else:
            return detected_bits.cpu(), None, None


    def infer(self, audio: torch.Tensor, watermark_bits: list = None):
        audio = audio.unsqueeze(0).to(self.device)
        with torch.no_grad():
            decoded = self.decoder.test_forward(audio)
        return "/".join(map(str, decoded.squeeze(0).squeeze(0).cpu().numpy()))


    def save_audio(self, audio: torch.Tensor, path: str, sample_rate: int = 16000):
        audio_np = audio.squeeze().detach().numpy()
        soundfile.write(path, audio_np, samplerate=sample_rate)

    def load_audio(self, path: str) -> torch.Tensor:
        audio, _ = torchaudio.load(path)
        return audio

# Example usage
if __name__ == "__main__":
    config_dir = "config/Timbre"
    ckpt_path = "../model_zoo/timbre_16khz_ASVspoof2019LA_16bits/ckpt_ASVsoof_20epoch/none-conv2_ep_20_2024-11-29_06_52_39.pth.tar"
    
    wrapper = TimbreWatermarkWrapper(config_dir, ckpt_path)

    audio, _ = torchaudio.load("LA_E_5954030.wav")
    wm = [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
    
    embedded_audio = wrapper.embed(audio, wm)
    wrapper.save_audio(embedded_audio, "output_timbre.wav")

    loaded_audio = wrapper.load_audio("output_timbre.wav")
    bits = wrapper.infer(loaded_audio, watermark_bits=wm)

    print(f"Detected bits: {bits}")
    # if acc is not None:
    #     print(f"Accuracy: {acc}, Score: {score}")
