import whisper
import librosa
import numpy as np
import os
import re
from datetime import datetime
import json


output_dir = "./output_transcriptions"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# load model whisper
model = whisper.load_model("small")

def transcribe_audio(file_path: str) -> dict:
    target_sr = 16000

    # preprocess audio
    audio, sr = librosa.load(file_path, sr=None, mono=False)
    audio = audio.mean(axis=0) if audio.ndim > 1 else audio

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    max_v = np.max(np.abs(audio))
    audio = (audio / max_v).astype(np.float32) if max_v > 0 else audio.astype(np.float32)
    
    # transkripsi audio
    result = model.transcribe(
        audio,
        language="id",
        task="transcribe",
        beam_size=5,
        temperature=0.0,
        initial_prompt="Berikut pesanan sembako: beras 2 kilo harga 25.000. Total belanja 1.500.000 atau 150.000."
    )

    clean_text = result["text"].strip() 
    clean_text = re.sub(r'\s+', ' ', clean_text)

    segments = result.get("segments", [])

    if segments:
        avg_logprob = sum(seg["avg_logprob"] for seg in segments) / len(segments)
        no_speech_prob = sum(seg["no_speech_prob"] for seg in segments) / len(segments)
    else:
        avg_logprob = 0.0
        no_speech_prob = 0.0

    output_data = {
        "text"      : clean_text,
        "confidence": round(avg_logprob, 4),     
        "no_speech" : round(no_speech_prob, 4),  
    }

    # Save the transcription to a text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"transcript_{timestamp}.json"
    json_path = os.path.join(output_dir, json_filename)

    output_data = {
        "text"      : clean_text,
        "confidence": round(avg_logprob, 4),
        "no_speech" : round(no_speech_prob, 4),
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        # Save the output data as JSON
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    output_data["json_file"] = json_path

    return output_data