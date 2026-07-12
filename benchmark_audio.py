#benchmark_audio.py:
import time
import os
import imageio
from moviepy import AudioFileClip
import numpy as np
import imageio_ffmpeg
import subprocess
import io
import scipy.io.wavfile

DATASET_PATH = r"c:\Users\CSE-312-01\Downloads\Capstone Project 1992\archive (15)\FakeAVCeleb_v1.2"

def get_sample():
    for root, dirs, files in os.walk(DATASET_PATH):
        for f in files:
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    return None

sample_path = get_sample()
print(f"Sample: {sample_path}")
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
print(f"FFmpeg Binary: {ffmpeg_exe}")

# Benchmark MoviePy (Baseline)
t0 = time.time()
try:
    audioclip = AudioFileClip(sample_path)
    y = audioclip.to_soundarray(fps=16000, nbytes=4)
    audioclip.close()
    print(f"MoviePy Time: {time.time()-t0:.4f}s")
except Exception as e:
    print(f"MoviePy Failed: {e}")

# Benchmark Subprocess FFmpeg (Optimized)
t0 = time.time()
try:
    # Read 3 seconds max, resample to 16k, mono, wav format to pipe
    command = [
        ffmpeg_exe, 
        '-i', sample_path, 
        '-t', '3.0',           # Limit duration here (ffmpeg does it fast)
        '-vn',                 # No video
        '-f', 'wav',           # WAV format
        '-ar', '16000',        # 16k Hz
        '-ac', '1',            # Mono
        '-'                    # Pipe to stdout
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    
    if process.returncode != 0:
        raise Exception(f"FFmpeg Error: {err.decode()}")
        
    rate, data = scipy.io.wavfile.read(io.BytesIO(out))
    print(f"FFmpeg Subprocess Time: {time.time()-t0:.4f}s")
    print(f"Data Shape: {data.shape}")
    
except Exception as e:
    print(f"FFmpeg Subprocess Failed: {e}")
