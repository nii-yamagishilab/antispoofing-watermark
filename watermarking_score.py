import numpy as np
import math

def compute_watermark_score(model_tag: str, decoded: list, message: dict) -> float:
    def sigmoid_inverse(y):
        return -math.log((1 - y) / y)

    def preprocess(msg):
        arr = np.array(list(map(float, msg)))
        arr[arr == 0] = -1
        return np.expand_dims(arr, axis=0)

    bonafide = preprocess(message["bonafide"])
    spoof = preprocess(message["spoof"])

    scores = np.array(list(map(float, decoded)))
    
    if model_tag.lower() == "audioseal":
        scores = np.array([sigmoid_inverse(y) for y in scores])

    cm_scores = np.mean(bonafide * scores, axis=1) - np.mean(spoof * scores, axis=1)
    return cm_scores[0]



def filtered_data(test_set: str, trial_lines:list) -> list:
    if test_set == "ASVspoof2021_LA":
        trial_lines = [item for item in trial_lines if item.split()[7] == "eval"]
    elif test_set == "ASVspoof2021_DF":
        trial_lines = [item for item in trial_lines if item.split()[2] == "nocodec" and item.split()[7] == "eval"]
    return trial_lines

def load_wm(wm_file: str = "wmpool.txt") -> dict:
    with open(wm_file, 'r') as f:
        lines = f.readlines()
    return {
        "bonafide": eval(lines[1]),
        "spoof": eval(lines[0])
    }


def produce_wm_score_file(
    test_set: str,
    model_tag: str, 
    eval_score_file: str,
    wm_score_file: str,
    eval_trial_path: str = None,
    wm_file: str = "wmpool.txt"
) -> None:
    
    wm = load_wm(wm_file)

    if test_set == "ASVspoof2019_LA":
        with open(eval_score_file, "w") as wf:    
            with open(wm_score_file) as rf:
                for line in rf:
                    utt_id, src, key, decoded = line.strip().split()
                    decoded = decoded.split("/")
                    if model_tag in {"Timbre", "AudioSeal"}:
                        sco = compute_watermark_score(model_tag, decoded, wm)
                    else:
                        sco = 0.0  # default or fallback score
                    wf.write(f"{utt_id} {src} {key} {sco}\n")

    else:
        # ASVspoof2021 format
        if eval_trial_path is None:
            raise ValueError("eval_trial_path must be provided for ASVspoof2021 format.")

        with open(eval_trial_path, "r") as rf:
            trial_lines = rf.readlines()
        trial_lines = filtered_data(test_set, trial_lines)

        with open(eval_score_file, "w") as wf:    
            with open(wm_score_file) as rf:
                for line in rf:
                    utt_id, decoded = line.strip().split()
                    decoded = decoded.split("/")
                    if model_tag in {"Timbre", "AudioSeal"}:
                        sco = compute_watermark_score(model_tag, decoded, wm)
                    else:
                        sco = 0.0
                    wf.write(f"{utt_id} {sco}\n")
