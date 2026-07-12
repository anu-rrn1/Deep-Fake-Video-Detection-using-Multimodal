#test_audio.py:
import torch
import torchaudio
import os

# Find a sample file
DATASET_PATH = r"c:\Users\CSE-312-01\Downloads\Capstone Project 1992\archive (15)\FakeAVCeleb_v1.2"
sample_path = None

for root, dirs, files in os.walk(DATASET_PATH):
    for f in files:
        if f.endswith(".mp4"):
            sample_path = os.path.join(root, f)
            break
    if sample_path: break

if not sample_path:
    print("No video found to test.")
    exit()

print(f"Testing torchaudio on: {sample_path}")

try:
    # Requires 'soundfile' or 'ffmpeg'
    waveform, sample_rate = torchaudio.load(sample_path)
    print(f"Success!")
    print(f"Shape: {waveform.shape}")
    print(f"Sample Rate: {sample_rate}")
    print(f"Max Value: {waveform.max()}")
    
    # Test MFCC transform availability
    transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=40)
    mfcc = transform(waveform)
    print(f"MFCC Shape: {mfcc.shape}")

except Exception as e:
    print(f"FAILED: {e}")
    # Check backends
    print(f"Available backends: {torchaudio.list_audio_backends()}")
