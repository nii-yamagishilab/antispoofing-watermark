#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
import os
import torch
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from watermarking_score import produce_wm_score_file

from ssl_aasist_wrapper import SSLASVWrapper
from aasist_wrapper import aasistWrapper
from timbre_wrapper import TimbreWatermarkWrapper
from audioseal_wrapper import AudioSealWrapper

from aasist.evaluation import calculate_tDCF_EER
import subprocess
import re
import random
import argparse
from audio_processing.pipeline import (
    make_none,
    make_rir,
    make_musan,
    make_gaussian_noise,
    make_quantization,
    make_downsample,
    make_upsample,
    make_mp3compression,
    make_opus,
    make_encodec,
    make_dac,
    make_wavtokenizer,
    make_amplification,
    make_time_stretch,
    make_pitchshift,
    make_smooth,
    make_highpass_filtering,
    make_lowpass_filtering,
    make_random_trimming,
    make_frequency_masking,
    make_clipping,
    make_overdrive,
    make_eq,
    make_compressor,
    make_noise_gate,
    make_noise_reduction,
    )

def aasist_data_preprocessing(x: torch.Tensor, max_len: int = 64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len // x_len) + 1
    padded_x = x.repeat(num_repeats)[:max_len]
    return padded_x

def timbre_data_preprocessing(wav: torch.Tensor, sr: int = 16000, max_len = 176000):
    if wav.shape[1] > max_len:
        cuted_len = random.randint(5*sr, max_len)
        wav = wav[:, :cuted_len]
    return wav

def attack_func(
        audio_file: str, 
        attack_name: str) -> torch.Tensor:
    perturbations = {
        "none": make_none,
        "rir": make_rir,
        "musan": make_musan,
        "gaussian_noise": make_gaussian_noise,
        "quantization": make_quantization,
        "downsample":make_downsample,
        "upsample": make_upsample,
        "mp3compression": make_mp3compression,
        "opus": make_opus,
        "encodec": make_encodec,
        "dac": make_dac,
        "wavtokenizer": make_wavtokenizer,
        "amplification": make_amplification,
        "time_stretch": make_time_stretch,
        "pitchshift": make_pitchshift,
        "smooth": make_smooth,
        "highpass_filtering": make_highpass_filtering,
        "lowpass_filtering": make_lowpass_filtering,
        "random_trimming": make_random_trimming, 
        "frequency_masking": make_frequency_masking,
        "clipping": make_clipping,
        "overdrive": make_overdrive,
        "eq": make_eq,
        "compressor": make_compressor,
        "noise_gate": make_noise_gate,
        "noise_reduction": make_noise_reduction,
    }

    try:
        perturbation = perturbations[attack_name]
    except KeyError:
        raise ValueError(f"Unsupported noise_name: {attack_name}. Supported values are: {list(perturbations.keys())}")

    processed = perturbation(audio_file)
    if isinstance(processed, np.ndarray):
        processed = torch.from_numpy(processed)
    return processed

def preprocess_waveform(
        model_tag: str, 
        waveform: torch.Tensor) -> torch.Tensor:
    
    if model_tag in {"AASIST", "SSL-AASIST"}:
        waveform = aasist_data_preprocessing(waveform.squeeze(0), max_len=64600).unsqueeze(0)
    elif model_tag == "Timbre":
        waveform = timbre_data_preprocessing(waveform)
    return waveform


_model_cache = {}
def get_model(model_name: str):
    if model_name not in _model_cache:
        # print(f"Loading model: {model_name}")
        if model_name == "AASIST":
            _model_cache[model_name] = aasistWrapper(
                config_path="config/AASIST.conf"
            )
        elif model_name == "SSL-AASIST":
            _model_cache[model_name] = SSLASVWrapper(
                model_path = "../model_zoo/ssl_aasist/Pre_trained_SSL_anti-spoofing_models/LA_model.pth",
            )
        elif model_name == "AudioSeal":
            _model_cache[model_name] = AudioSealWrapper(
                generator_ckpt = "../model_zoo/AudioSeal_16khz_ASVspoof2019LA_16bits/checkpoint_generator_16khz_asvspoof2019.pth", 
                detector_ckpt = "../model_zoo/AudioSeal_16khz_ASVspoof2019LA_16bits/checkpoint_detector_16khz_asvspoof2019.pth"
            )
        elif model_name == "Timbre":
            _model_cache[model_name] = TimbreWatermarkWrapper(
                config_dir = "config/Timbre",
                checkpoint_path = "../model_zoo/timbre_16khz_ASVspoof2019LA_16bits/ckpt_ASVsoof_20epoch/none-conv2_ep_20_2024-11-29_06_52_39.pth.tar"
            )
    return _model_cache[model_name]

def detection_func(
        model_tag: str, 
        waveform: torch.Tensor) -> float:
    waveform = preprocess_waveform(model_tag, waveform)
    try:
        with torch.no_grad():
            model = get_model(model_tag)
            predict_score = model.infer(waveform)

    except KeyError:
        raise ValueError(f"Unsupported model_tag: {model_tag}. Supported values are: {list(models.keys())}")

    return predict_score



def evaluation(
        test_set: str, 
        model_tag:str, 
        attack_name: str,
        audio_dir:str,
        trl: str) -> dict:
    
    utt_id, src, key = None, None, None
    if test_set =="ASVspoof2019_LA":
        _, utt_id, _, src, key = trl.strip().split(' ')
    else:
        # 'LA_0009 LA_E_5524346 ulaw loc_tx A16 spoof notrim eval\n',
        utt_id =  trl.strip().split()[1]

    
    if utt_id != None:
        
        if model_tag in {"Timbre", "AudioSeal"}:
            audio_file = Path(audio_dir+"_"+model_tag, utt_id+".wav")
        else:
            audio_file = Path(audio_dir, utt_id+".wav")

        processed = attack_func(audio_file, attack_name) # torch.Size([channels, samples])
        predict_score = detection_func(model_tag, processed)
        score_dict = {
            "utt_id" : utt_id,
            "src": src,
            "key": key,
            "score": predict_score,
        }
        return score_dict
    else:
        return None

def filtered_data(
        test_set: str, 
        trial_lines:list) -> list:
    if test_set == "ASVspoof2021_LA":
        trial_lines = [item for item in trial_lines if item.split()[7] == "eval"]
    elif test_set == "ASVspoof2021_DF":
        trial_lines = [item for item in trial_lines if item.split()[2] == "nocodec" and item.split()[7] == "eval"]
    return trial_lines


def produce_evaluation_file(
        model_tag: str,
        test_set: str,
        attack_name: str,
        n_jobs: int, 
        audio_dir: str,
        eval_score_file: str,
        trial_path: str,
        wm_score_file: str = None) -> None:
    """Perform evaluation and save the score to a file"""

    with open(trial_path, "r") as rf:
        trial_lines = rf.readlines()
    trial_lines = filtered_data(test_set, trial_lines)
    trial_lines = trial_lines

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluation)(test_set, model_tag, attack_name, audio_dir, trl) for trl in tqdm(trial_lines, desc="Computing")
    )

    if model_tag in {"Timbre", "AudioSeal"}:
        eval_score_file = wm_score_file
    if test_set =="ASVspoof2019_LA":
        with open(eval_score_file, "w") as wf:
            print("Number of results:", len(results))
            for score_dict in results:
                print(score_dict)
                wf.write("{} {} {} {}\n".format(score_dict["utt_id"], score_dict["src"], score_dict["key"], score_dict["score"]))
                #output format
                # LA_E_2834763 A11 spoof 0.33809256670549875
                # LA_E_8877452 A14 spoof 0.5215931230986878
                # LA_E_6828287 A16 spoof 0.5185240396082264
                print("Scores saved to {}".format(eval_score_file))
    else:
        with open(eval_score_file, "w") as wf:
            for score_dict in results:
                wf.write("{} {}\n".format(score_dict["utt_id"], score_dict["score"]))

        print("Scores saved to {}".format(eval_score_file))

def main():
    parser = argparse.ArgumentParser(description="Provide test set, model name, attack mathod, and number of jobs.")
    parser.add_argument('--test_set', type=str, help="Name of the dataset")
    parser.add_argument('--model_name', type=str, help="Name of the model")
    parser.add_argument('--attack_name', type=str, help="Name of the attack")
    parser.add_argument('--n_jobs', type=int, default=1, help="Number of jobs to run in parallel.")

    args = parser.parse_args()

    model_tag = args.model_name
    attack_name = args.attack_name
    n_jobs = args.n_jobs
    test_set = args.test_set

    trial_file = {
        "ASVspoof2019_LA": "/mnt/md0/user_max/toolkit/Chiahua_BCM/ASVspoof2019_key/ASVspoof2019.LA.cm.eval.trl.txt",
        "ASVspoof2021_LA": "/mnt/md0/user_max/toolkit/Chiahua_BCM/ASVspoof2021_keys/LA/CM/trial_metadata.txt",
        "ASVspoof2021_DF": "/mnt/md0/user_max/toolkit/Chiahua_BCM/ASVspoof2021_keys/DF/CM/trial_metadata.txt"
    }
    audio_data = {
        "ASVspoof2019_LA": "/mnt/md0/user_max/toolkit/Chiahua_BCM/toolkit/exp/ASVspoof2019_LA/wav",
        "ASVspoof2021_LA": "/mnt/md0/user_max/toolkit/Chiahua_BCM/toolkit/exp/corpus/ASVspoof2021_LA_eval/wav",
        "ASVspoof2021_DF": "/mnt/md0/user_max/toolkit/Chiahua_BCM/toolkit/exp/corpus/ASVspoof2021_DF_eval/wav"
    }

    eval_trial_path = trial_file[test_set]
    audio_dir = audio_data[test_set]

    asv_score_file="/mnt/md0/user_max/toolkit/Chiahua_BCM/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    output_dir = "exp_results"
    output_EER_file = Path(output_dir) / f"{test_set}_{model_tag}_{attack_name}_t-DCF_EER.txt"
    eval_score_file = Path(output_dir) / f"{test_set}_{model_tag}_{attack_name}.txt"
    wm_score_file = str(Path(eval_score_file).with_name(Path(eval_score_file).stem + "_output" + Path(eval_score_file).suffix))
    parent_eval_trial_directory = os.path.dirname(os.path.dirname(eval_trial_path))

    print(f"Model: {model_tag}, Attack: {attack_name}, Dataset: {test_set}")

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_tag in {"Timbre", "AudioSeal"}:
        produce_evaluation_file(model_tag, test_set, attack_name, n_jobs, audio_dir, eval_score_file, eval_trial_path, wm_score_file)
        if os.path.exists(wm_score_file):
            produce_wm_score_file(
                test_set = test_set,
                model_tag = model_tag,
                wm_score_file = wm_score_file,
                eval_score_file = eval_score_file,
                eval_trial_path = eval_trial_path,
                wm_file = "wmpool.txt"
            )
    else:
        produce_evaluation_file(model_tag, test_set, attack_name, n_jobs, audio_dir, eval_score_file, eval_trial_path)

    # Compute the evaluation metrics, including min-tDCF and EER
    if test_set == "ASVspoof2019_LA":
        calculate_tDCF_EER(
            cm_scores_file = eval_score_file,
            asv_score_file = asv_score_file,
            output_file = output_EER_file)
    elif test_set =="ASVspoof2021_LA":
        try:
            if os.path.exists(eval_score_file):
                # python evaluate_2021_LA.py Score_LA.txt ./keys eval
                command = ["python", "evaluate_2021_LA.py", eval_score_file, parent_eval_trial_directory, "eval"]
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                stdout_content = result.stdout
                min_tDCF_match = re.search(r"min_tDCF:\s*([0-9.]+)", stdout_content)
                min_tDCF = float(min_tDCF_match.group(1)) if min_tDCF_match else None

                eer_match = re.search(r"eer:\s*([0-9.]+)", stdout_content)
                eer = float(eer_match.group(1)) if eer_match else None
                print("min_tDCF:", min_tDCF)
                print("eer:", eer)
            else:
                print(eval_score_file, " does not exist.")

        except subprocess.CalledProcessError as e:
            print("An error occurred:")
            print(e.stderr)

    print("DONE.")

if __name__ == "__main__":
    main()
