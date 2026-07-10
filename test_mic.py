import sounddevice as sd
import scipy.io.wavfile as wav
import requests
import os

fs = 44100  
seconds = 5 
filename = "test_rekaman.wav"

print(f"Starting recording for {seconds} seconds")

# Start recording audio
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
print("Recording finished")

# Save as .wav
wav.write(filename, fs, myrecording)

url = "http://127.0.0.1:8000/transcribe" 

with open(filename, 'rb') as f:
    files = {'file': (filename, f, 'audio/wav')}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print("\nTranscription Result:")
    import json
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
else:
    print(f"\nError {response.status_code}: {response.text}")

if os.path.exists(filename):
    os.remove(filename)