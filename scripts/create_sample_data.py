import os
import json
import pandas as pd
import numpy as np
import soundfile as sf

def create_sample_dataset():
    sample_dir = "data/sample"
    os.makedirs(f"{sample_dir}/eatd", exist_ok=True)
    os.makedirs(f"{sample_dir}/daic", exist_ok=True)

    # 1. EATD Mock Sample (3 participants: 2 control, 1 depressed)
    eatd_records = []
    sr = 16000
    for idx, (pid, lbl, score) in enumerate([("sample_1", 0, 42.0), ("sample_2", 0, 48.5), ("sample_3", 1, 62.0)]):
        p_dir = f"{sample_dir}/eatd/{pid}"
        os.makedirs(p_dir, exist_ok=True)
        # Create small dummy audio WAV files (1.0 sec sine wave)
        for tone_type, freq in [("positive", 440), ("neutral", 550), ("negative", 330)]:
            wav_path = f"{p_dir}/{tone_type}.wav"
            t = np.linspace(0, 1.0, sr, endpoint=False)
            audio = 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
            sf.write(wav_path, audio, sr)
        
        eatd_records.append({
            "participant_id": pid,
            "label": lbl,
            "sds_score": score,
            "audio_positive_path": f"{p_dir}/positive.wav",
            "audio_positive_dur": 1.0,
            "audio_positive_dropped": False,
            "audio_neutral_path": f"{p_dir}/neutral.wav",
            "audio_neutral_dur": 1.0,
            "audio_neutral_dropped": False,
            "audio_negative_path": f"{p_dir}/negative.wav",
            "audio_negative_dur": 1.0,
            "audio_negative_dropped": False,
            "num_valid_audio": 3,
            "text_combined_clean": "Sample synthetic prompt response for testing audio text alignment."
        })
    
    pd.DataFrame(eatd_records).to_csv(f"{sample_dir}/eatd/sample_manifest.csv", index=False)
    print("Created synthetic EATD sample in data/sample/eatd/")

    # 2. DAIC Mock Sample (3 participants)
    daic_records = []
    for pid, lbl, phq in [("sample_301", 0, 3), ("sample_302", 0, 6), ("sample_303", 1, 15)]:
        p_dir = f"{sample_dir}/daic/{pid}_P"
        os.makedirs(p_dir, exist_ok=True)
        # Mock 2-turn transcript CSV
        transcript_df = pd.DataFrame([
            {"start_time": 62.0, "stop_time": 65.5, "speaker": "Ellie", "value": "How are you doing today?"},
            {"start_time": 66.0, "stop_time": 72.0, "speaker": "Participant", "value": "I am feeling okay, just getting by."},
            {"start_time": 73.0, "stop_time": 76.0, "speaker": "Ellie", "value": "Tell me about your sleep."},
            {"start_time": 77.0, "stop_time": 84.0, "speaker": "Participant", "value": "It has been difficult falling asleep at night."}
        ])
        transcript_df.to_csv(f"{p_dir}/{pid}_TRANSCRIPT.csv", sep="\t", index=False)
        
        # Mock 20-second audio
        wav_path = f"{p_dir}/{pid}_AUDIO.wav"
        t = np.linspace(0, 20.0, sr * 20, endpoint=False)
        audio = 0.05 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(wav_path, audio, sr)
        
        daic_records.append({
            "Participant_ID": pid,
            "PHQ8_Binary": lbl,
            "PHQ8_Score": phq,
            "Gender": 1,
            "Split": "dev"
        })
    pd.DataFrame(daic_records).to_csv(f"{sample_dir}/daic/sample_dev_split.csv", index=False)
    print("Created synthetic DAIC sample in data/sample/daic/")

if __name__ == "__main__":
    create_sample_dataset()
