#verify_audio_dump.py:
 import os
import numpy as np
import warnings
from moviepy import AudioFileClip
import soundfile as sf

# Config
DATASET_PATH = r"c:\Users\CSE-312-01\Downloads\Capstone Project 1992\archive (15)\FakeAVCeleb_v1.2"

def test_extraction():
    print("Searching for a video...")
    target_path = None
    for root, dirs, files in os.walk(DATASET_PATH):
        for f in files:
            if f.endswith(".mp4"):
                target_path = os.path.join(root, f)
                break
        if target_path: break
    
    if not target_path:
        print("No video found.")
        return

    print(f"Target: {target_path}")
    
    try:
        print("Attempting to load with MoviePy...")
        audioclip = AudioFileClip(target_path)
        print(f"Duration: {audioclip.duration}s")
        
        # Extract 2 seconds
        y = audioclip.to_soundarray(fps=16000, nbytes=4)
        y = y[:16000*2]
        
        print(f"Raw shape: {y.shape}")
        
        if y.ndim > 1:
            y = y.mean(axis=1)
            
        print(f"Mono shape: {y.shape}")
        print(f"Max Amplitude: {np.max(np.abs(y))}")
        print(f"Mean Amplitude: {np.mean(np.abs(y))}")
        
        if np.max(np.abs(y)) < 0.001:
            print("❌ SILENCE DETECTED! Audio extraction failed.")
        else:
            print("✅ Audio seems valid.")
            
        # Save for manual inspection? (Can't hear it, but file size proves content)
        sf.write('debug_audio_dump.wav', y, 16000)
        print("Saved 'debug_audio_dump.wav'")

        audioclip.close()
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_extraction()
