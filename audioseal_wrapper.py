import torch
import torchaudio
import numpy as np
from audioseal import AudioSeal

class AudioSealWrapper:
    def __init__(self,
                 generator_ckpt: str,
                 detector_ckpt: str,
                 sample_rate: int = 16000,
                 nbits: int = 16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        self.nbits = nbits

        self.generator = AudioSeal.load_generator(generator_ckpt, nbits=nbits).to(self.device)
        self.detector = AudioSeal.load_detector(detector_ckpt, nbits=nbits).to(self.device)

    def embed(self, audio: torch.Tensor, watermark_bits: list, alpha: float = 1.0) -> torch.Tensor:
        audio = audio.unsqueeze(0).to(self.device)
        watermark_tensor = torch.tensor([watermark_bits], dtype=torch.int32).to(self.device)

        with torch.no_grad():
            watermarked_audio = self.generator(audio, message=watermark_tensor,
                                               sample_rate=self.sample_rate, alpha=alpha)
        return watermarked_audio.squeeze(0).cpu()

    def infer(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.unsqueeze(0).to(self.device)
        with torch.no_grad():
            result, message = self.detector(audio, self.sample_rate)
        return "/".join(map(str, message.squeeze(0).cpu().numpy()))

    def detect_from_file(self, file_path: str, watermark_bits: list = None, threshold: float = 0.5):
        audio, sample_rate = torchaudio.load(file_path)
        audio = audio.to(self.device).unsqueeze(0)

        result, message = self.detector.detect_watermark(audio, sample_rate=sample_rate, message_threshold=threshold)
        message_tensor = torch.tensor(message, dtype=torch.float32)[0]

        print(f"Detected bits: {message_tensor}")

        if watermark_bits:
            target_bits = torch.tensor(watermark_bits, dtype=torch.float32).to(message_tensor.device)
            score = (message_tensor * target_bits).mean()
            print(f"Matching score: {score}")
            message_tensor = torch.gt(message_tensor, threshold).int()
            print(f"Binarized bits: {message_tensor}")
        return message_tensor

# Example usage
if __name__ == "__main__":
    generator_ckpt = "/mnt/md0/user_max/toolkit/Chiahua_BCM/model_zoo/audioseal_16khz_asvspoof2019_models/checkpoint_generator_16khz_asvspoof2019.pth"
    detector_ckpt = "/mnt/md0/user_max/toolkit/Chiahua_BCM/model_zoo/audioseal_16khz_asvspoof2019_models/checkpoint_detector_16khz_asvspoof2019.pth"

    audioseal = AudioSealWrapper(generator_ckpt, detector_ckpt)

    # audio, _ = torchaudio.load("audio/input.wav")
    # watermark_bits = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]
    # watermarked_audio = audioseal.embed(audio, watermark_bits)
    # torchaudio.save("audio/output_seal.wav", watermarked_audio.unsqueeze(0), 16000)

    # audioseal.detect_from_file("audio/output_seal.wav", watermark_bits=watermark_bits)
