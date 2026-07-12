import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import json
import os

fs = 44100
seconds = 10
filename = "test_rekaman.wav"

print(f"Starting recording for {seconds} seconds")

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
print("Recording finished")

wav.write(filename, fs, myrecording)

base_url = "http://127.0.0.1:8000"

with open(filename, 'rb') as f:
    files = {'file': (filename, f, 'audio/wav')}
    resp = requests.post(f"{base_url}/transcribe", files=files)

if resp.status_code == 200:
    data = resp.json()
    print("\nTranscription + Extraction")
    print(json.dumps(data, indent=4, ensure_ascii=False))

    # detail prediction
    pred_file = os.path.basename(data["json_file"])
    resp_pred = requests.get(f"{base_url}/predictions/{pred_file}")
    if resp_pred.status_code == 200:
        print(f"\nPrediction Detail: {pred_file}")
        print(json.dumps(resp_pred.json(), indent=4, ensure_ascii=False))
else:
    print(f"\nTranscribe error {resp.status_code}: {resp.text}")

# list semua predictions
resp_list = requests.get(f"{base_url}/predictions")
if resp_list.status_code == 200:
    print(f"\nAll Predictions ")
    print(json.dumps(resp_list.json(), indent=4, ensure_ascii=False))

if os.path.exists(filename):
    os.remove(filename)