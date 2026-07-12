#diagnose_data.py:
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18
from tqdm import tqdm
import warnings
import scipy.io.wavfile
import librosa

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = r"c:\Users\CSE-312-01\Downloads\Capstone Project 1992\archive (15)\FakeAVCeleb_v1.2"
WEIGHTS_PATH = r"c:\Users\CSE-312-01\Desktop\Capstone Project 1992\archive (16)\resnet18-f37072fd.pth"
EPOCHS = 15
BATCH_SIZE = 32
NUM_WORKERS = 8
MFCC_DIM = 40
NUM_CLASSES = 2 

# ==========================================
# SMART AUDIO FEATURE EXTRACTION
# ==========================================

def extract_audio_features(video_path, n_mfcc=40, max_len=120):
    """
    Implements Smart Tri-Segment Logic:
    1. Early Transition (10-25%)
    2. Middle Stable (30-60%)
    3. High Energy (Loudest 20% window)
    """
    try:
        # Load audio at 16kHz
        y, sr = librosa.load(video_path, sr=16000)
        L = len(y)
        
        if L < 16000: # If less than 1s, just pad the original
            y_smart = y
        else:
            # 1. Early Transition Segment
            seg_trans = y[int(0.10 * L) : int(0.25 * L)]
            
            # 2. Middle Stable Segment
            seg_stable = y[int(0.30 * L) : int(0.60 * L)]
            
            # 3. High Energy Segment (Loudest 20% window)
            window_size = int(0.20 * L)
            energy = np.abs(y)
            # Find the loudest continuous block
            energies = np.convolve(energy, np.ones(window_size), mode='valid')
            max_start = np.argmax(energies)
            seg_energy = y[max_start : max_start + window_size]

            # Combine the danger zones
            y_smart = np.concatenate([seg_trans, seg_stable, seg_energy])

        # Normalize volume
        if np.max(np.abs(y_smart)) > 0:
            y_smart = y_smart / (np.max(np.abs(y_smart)) + 1e-9)
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y_smart, sr=sr, n_mfcc=n_mfcc).T 
        
        # Consistent shape for the Bi-LSTM (Batch, Time, Features)
        if mfcc.shape[0] > max_len:
            mfcc = mfcc[:max_len, :]
        else:
            pad_width = max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
            
        return mfcc

    except Exception:
        return np.zeros((max_len, n_mfcc))

# ==========================================
# REST OF THE MODEL (Dataset, Encoder, Main)
# ==========================================

class FakeAVDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Video frames (using your existing extract_frames)
        # Note: Ensure extract_frames is defined in your script
        from __main__ import extract_frames 
        frames = extract_frames(sample["path"])
        
        # NEW Smart Audio Features
        audio = extract_audio_features(sample["path"])
        
        # Data Augmentation (Horizontal Flip)
        if np.random.rand() > 0.5:
             frames = np.flip(frames, axis=2).copy()
             
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
        audio = torch.tensor(audio).float()
        
        # Binary label mapping
        label = 0 if sample["label"] == 0 else 1
        return frames, audio, torch.tensor(label).long()

def main():
    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Dataset Discovery
    label_map = {"RealVideo-RealAudio": 0, "RealVideo-FakeAudio": 1, "FakeVideo-RealAudio": 2, "FakeVideo-FakeAudio": 3}
    video_samples = []
    for folder, label in label_map.items():
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.exists(folder_path): continue
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".mp4"):
                    video_samples.append({"path": os.path.join(root, file), "label": label})
    
    # 2. Split (70/30)
    labels = [s["label"] for s in video_samples]
    train_data, test_data, train_labels, _ = train_test_split(
        video_samples, labels, test_size=0.30, stratify=labels, random_state=42
    )

    # Print Ratio
    total = len(video_samples)
    print(f"\n--- DATA SPLIT SUMMARY ---")
    print(f"Total Videos: {total}")
    print(f"Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"Test:  {len(test_data)} ({len(test_data)/total*100:.1f}%)\n")

    # 3. Weighted Sampler (To handle Real/Fake imbalance)
    binary_labels = [0 if l == 0 else 1 for l in train_labels]
    class_counts = np.bincount(binary_labels)
    sample_weights = [1./class_counts[l] for l in binary_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_data), replacement=True)

    # 4. DataLoaders
    train_loader = DataLoader(FakeAVDataset(train_data), batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(FakeAVDataset(test_data), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ... [Insert your DeepTraceModel definition and training loop here] ...
    print("Ready to start training with Smart Tri-Segment Audio features.")

if __name__ == "__main__":
    main()
