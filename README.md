# A Comparative Study on Proactive and Passive Detection of Deepfake Speech
This repository contains our implementation of the paper to be published at the Interspeech 2025 

```bibtex
A Comparative Study on Proactive and Passive Detection of Deepfake Speech,
Chia-Hua Wu, Wanying Ge, Xin Wang, Junichi Yamagishi, Yu Tsao, Hsin-Min Wang
Interspeech 2025 (accepted)
```
Links (to be added)

## Key Features

### Anti-spoofing & Watermarking Models
- **[AASIST](https://arxiv.org/abs/2110.01200)**: Passive approach: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
- **[SSL-AASIST](https://arxiv.org/abs/2202.12233)**: Passive approach: Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation
- **[AudioSeal](https://arxiv.org/abs/2401.17264)**: Proactive approach: Detection of Voice Cloning with Localized Watermarking
- **[Timbre](https://arxiv.org/abs/2312.03410)**: Proactive approach: Detecting Voice Cloning Attacks via Timbre Watermarking
  
### Supported Attacks
- None (no attack applied)
- Gaussian Noise
- [MUSAN](https://arxiv.org/abs/1510.08484) (Only the noise set is used)
- RIRs (Room Impulse Responses)
- Quantization
- Compressor
- Opus
- [DAC](https://arxiv.org/abs/2306.06546)
- [WavTokenizer](https://arxiv.org/abs/2408.16532)
- [Encodec](https://arxiv.org/abs/2210.13438)
- Clipping
- Overdrive
- Random Trimming
- Equalizer
- Frequency Masking
- Noise Gate
- Noise Reduction
- Time Stretch
- Pitch Shift
- Downsample
- Upsample
- Mp3compression
- Amplification
- Smooth
- Highpass filtering
- Lowpass filtering

## Requirements and Installation

### Environment Requirements

- **Python**: 3.10  
- **CUDA**: 12.6 (Make sure your GPU driver is compatible with this version)  
- It is recommended to use a virtual environment (e.g., `venv` or `conda`) to avoid dependency conflicts.

> ⚠️ Some external tools (e.g., AASIST, SSL-AASIST) may have their own CUDA or PyTorch requirements. Please check their official repositories for compatibility.


### Step 1: [Optional] Install spoofing & watermarking tools

These tools are optional, but required for specific spoofing/watermarking tasks.

1. **AASIST**  
   - Ensure AASIST is properly installed before use.  
   - See the [AASIST GitHub repository](https://github.com/clovaai/aasist) for installation details.

2. **SSL-AASIST**  
   - Ensure SSL-AASIST is properly installed before use.  
   - See the [SSL_Anti-spoofing GitHub repository](https://github.com/TakHemlata/SSL_Anti-spoofing) for installation details.

3. **AudioSeal**  
   - Ensure AudioSeal is properly installed before use.  
   - See the [AudioSeal GitHub repository](https://github.com/facebookresearch/audioseal) for installation details.

4. **Timbre**  
   - Ensure Timbre is properly installed before use.  
   - See the [Timbre GitHub repository](https://github.com/TimbreWatermarking/TimbreWatermarking)  for installation details.

---

### Step 2: Install antispoofing-watermark

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/nii-yamagishilab-visitors/antispoofing-watermark.git
   cd antispoofing-watermark
   pip install -r requirements.txt
    ```
    
2. [Optional] Put spoofing & watermarking tools under `antispoofing-watermark`
    ```bash
    cd <antispoofing-watermark-root>
    ln -s <AASIST-root> .
    ln -s <SSL_Anti-spoofing-root> .
    ln -s <TimbreWatermarking-root>/watermarking_model .
    ```
### Dataset
- 19LA: [ASVspoof2019 LA](https://www.asvspoof.org/index2019.html)
- 21LA & 21DF: [ASVspoof2021 LA & DF](https://www.asvspoof.org/index2021.html)
   - For both datasets, only the 'nocodec' subset from the evaluation partition is used.

### Fairseq Pre-trained wav2vec 2.0 XLSR (300M)
Download the XLSR models from [Fairseq official website](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)

## Usage
### Example: Test AASIST Model (Using Pretrained Checkpoint)

This example demonstrates how to use our code to run the pretrained AASIST model on RIR-attacked audio files.
```bash 
# python evaluate.py --test_set <test_set> --model_name <model> --attack_name <attack_method> --n_job <number_of_jobs>
python evaluate.py --test_set ASVspoof2021_LA --model_name AASIST --attack_name rir --n_job 20
```

## Reference Repo
Thanks for the following repos:
1. [AASIST GitHub repository](https://github.com/clovaai/aasist)
2. [SSL_Anti-spoofing GitHub repository](https://github.com/TakHemlata/SSL_Anti-spoofing)
3. [AudioSeal GitHub repository](https://github.com/facebookresearch/audioseal)
4. [Timbre  GitHub repository](https://github.com/TimbreWatermarking/TimbreWatermarking)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
This work was conducted during the first author’s internship at the National Institute of Informatics (NII), Japan. This study was partially supported by JST AIP Acceleration Research (JPMJCR24U3), MEXT KAKENHI Grant (24H00732), and JST PRESTO (JPMJPR23P9).

## Citation
If you use this code in your research please use the following citation:
```bibtex
@inproceedings{antispoofing_watermark,
  author={Chia-Hua Wu, Wanying Ge, Xin Wang, Junichi Yamagishi, Yu Tsao, Hsin-Min Wang},
  title={A Comparative Study on Proactive and Passive Detection of Deepfake Speech},
  year=2025,
  booktitle={Proc. Interspeech (to appear)},
}

```
