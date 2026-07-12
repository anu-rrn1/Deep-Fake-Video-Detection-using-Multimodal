#convert_script.py:
import os
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import imageio_ffmpeg

DATASET_PATH = r"c:\Users\CSE-312-01\Downloads\Capstone Project 1992\archive (15)\FakeAVCeleb_v1.2"
FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()

def convert_one(file_path):
    try:
        wav_path = file_path.replace(".mp4", ".wav")
        if os.path.exists(wav_path):
            return # Skip existing
            
        command = [
            FFMPEG_BINARY,
            '-y', # Overwrite
            '-i', file_path,
            '-vn',
            '-ac', '1',
            '-ar', '16000',
            '-f', 'wav',
            '-loglevel', 'error',
            wav_path
        ]
        subprocess.run(command, check=True)
    except Exception as e:
        pass # print(f"Error {file_path}: {e}")

def main():
    print("Scanning for MP4 files...")
    # Recursive glob
    files = glob.glob(os.path.join(DATASET_PATH, "**/*.mp4"), recursive=True)
    print(f"Found {len(files)} videos.")
    
    print("Starting Parallel Conversion (16 threads)...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(tqdm(executor.map(convert_one, files), total=len(files), unit="file"))
        
    print("Conversion Complete!")

if __name__ == "__main__":
    main()
