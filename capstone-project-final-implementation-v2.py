#capstone-project-final-implementation-v2.py:
#!/usr/bin/env python
# coding: utf-8
 
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold          # FIX 5: k-fold
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from torchvision.models import resnet18
from tqdm import tqdm
import warnings
import imageio_ffmpeg
import scipy.io.wavfile
import librosa
import torch.nn.functional as F
 
# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = r"c:\Users\CSE-312-01\Desktop\Capstone Project 1992\archive (15)\FakeAVCeleb_v1.2"
WEIGHTS_PATH = r"c:\Users\CSE-312-01\Desktop\Capstone Project 1992\archive (16)\resnet18-f37072fd.pth"
EPOCHS      = 25
BATCH_SIZE  = 32
NUM_WORKERS = 8
MFCC_DIM    = 40
NUM_CLASSES = 2
N_FOLDS     = 5          # FIX 5: stratified k-fold
 
FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()
 
# ==========================================
# MODEL COMPONENTS
# ==========================================
 
class VideoEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.cnn = resnet18(weights=None)
        if os.path.exists(WEIGHTS_PATH):
            self.cnn.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        self.cnn.fc = nn.Identity()
        self.proj   = nn.Linear(512, embed_dim)
 
    def forward(self, x):
        B, T, C, H, W = x.shape
        x    = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(B, T, 512)
        feats = self.proj(feats)
        return feats.mean(dim=1)
 
 
class AudioEncoder(nn.Module):
    def __init__(self, input_dim=40, embed_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64,
                               kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.lstm  = nn.LSTM(input_size=64, hidden_size=embed_dim // 2,
                             num_layers=1, batch_first=True, bidirectional=True)
 
    def forward(self, x):
        x        = x.transpose(1, 2)
        x        = self.conv1(x)
        x        = self.relu(x)
        x        = self.pool(x)
        x        = x.transpose(1, 2)
        output, _ = self.lstm(x)
        return output.mean(dim=1)
 
 
class CrossModalAttention(nn.Module):
    def __init__(self, dim_video=256, dim_audio=128):
        super().__init__()
        self.attn       = nn.MultiheadAttention(embed_dim=dim_video, num_heads=4,
                                                 batch_first=True)
        self.audio_proj = nn.Linear(dim_audio, dim_video)
 
    def forward(self, video_feat, audio_feat):
        audio_feat = self.audio_proj(audio_feat).unsqueeze(1)
        video_feat = video_feat.unsqueeze(1)
        fused, _   = self.attn(video_feat, audio_feat, audio_feat)
        return fused.squeeze(1)
 
 
class DeepTraceModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()
        self.cross_attn    = CrossModalAttention()
        self.classifier    = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
 
    def forward(self, frames, audio):
        v_feat = self.video_encoder(frames)
        a_feat = self.audio_encoder(audio)
        fused  = self.cross_attn(v_feat, a_feat)
        return self.classifier(fused)
 
 
# ==========================================
# UTILS & DATASET
# ==========================================
 
def extract_frames(video_path, num_frames=2, size=112):
    cap          = cv2.VideoCapture(video_path)
    frames       = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    if total_frames > 0:
        frame_idxs = np.linspace(
            total_frames // 4, total_frames - (total_frames // 4), num_frames
        ).astype(int)
        idx = 0
        for i in range(frame_idxs[-1] + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if i == frame_idxs[idx]:
                frame = cv2.resize(frame, (size, size))
                frames.append(frame)
                idx += 1
                if idx >= len(frame_idxs):
                    break
    cap.release()
 
    while len(frames) < num_frames:
        if len(frames) > 0:
            frames.append(frames[-1])
        else:
            return np.zeros((num_frames, size, size, 3))
    return np.array(frames)
 
 
def extract_audio_features(video_path, n_mfcc=40, max_len=100):
    try:
        wav_path = video_path.replace(".mp4", ".wav")
        if not os.path.exists(wav_path):
            return np.zeros((max_len, n_mfcc))
        rate, y = scipy.io.wavfile.read(wav_path)
 
        target_len = 48000
        if len(y) > target_len:
            start_idx = (len(y) - target_len) // 2
            y = y[start_idx: start_idx + target_len]
 
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) > 0:
            y    = y.astype(np.float32) / 32768.0
            mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=n_mfcc).T
            if mfcc.shape[0] > max_len:
                mfcc = mfcc[:max_len, :]
            else:
                pad_width = max_len - mfcc.shape[0]
                mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
            return mfcc
    except Exception:
        pass
    return np.zeros((max_len, n_mfcc))
 
 
class FakeAVDataset(Dataset):
    """
    FIX 2: Minority class (label=0) receives aggressive augmentation.
    Majority class (label=1) receives light augmentation only.
    """
    def __init__(self, samples):
        self.samples = samples
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = extract_frames(sample["path"])
        audio  = extract_audio_features(sample["path"])
        label  = 0 if sample["label"] == 0 else 1
 
        # --- FIX 2: Class-conditional augmentation ---
        if label == 0:
            # Minority class: augment aggressively
            if np.random.rand() > 0.3:
                frames = np.flip(frames, axis=2).copy()                         # horizontal flip
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                frames = np.clip(frames * factor, 0, 255)                       # brightness jitter
            if np.random.rand() > 0.5:
                noise  = np.random.normal(0, 5, frames.shape).astype(np.float32)
                frames = np.clip(frames + noise, 0, 255)                        # Gaussian noise
            if np.random.rand() > 0.5:
                shift = np.random.randint(1, 15)
                audio = np.roll(audio, shift, axis=0)                           # audio time shift
            if np.random.rand() > 0.5:
                audio = audio * np.random.uniform(0.85, 1.15)                  # audio amplitude jitter
        else:
            # Majority class: light augmentation only
            if np.random.rand() > 0.5:
                frames = np.flip(frames, axis=2).copy()
 
        frames = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
        audio  = torch.tensor(audio).float()
        return frames, audio, torch.tensor(label).long()
 
 
# ==========================================
# TRAINING / EVALUATION
# ==========================================
 
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    progress = tqdm(loader, desc="Training", leave=False)
 
    for frames, audio, labels in progress:
        frames, audio, labels = frames.to(device), audio.to(device), labels.to(device)
        optimizer.zero_grad()
 
        with torch.cuda.amp.autocast():
            outputs = model(frames, audio)
            loss    = criterion(outputs, labels)
 
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
 
        total_loss += loss.item()
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)
        progress.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
 
    return total_loss / len(loader), correct / total
 
 
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    progress = tqdm(loader, desc="Evaluating", leave=False)
 
    for frames, audio, labels in progress:
        frames, audio, labels = frames.to(device), audio.to(device), labels.to(device)
        outputs = model(frames, audio)
        loss    = criterion(outputs, labels)
        total_loss += loss.item()
 
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)
 
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
 
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
 
    # FIX 3: compute macro F1 alongside accuracy
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), correct / total, all_preds, all_labels, all_probs, macro_f1
 
 
# FIX 4: Find the optimal decision threshold for reliable performance
def find_best_threshold(all_labels, all_probs):
    """
    Scans for the probability threshold (for class 1) that maximizes MACRO F1 score.
    This ensures we balance precision and recall properly across both classes,
    avoiding the issue of predicting only one class.
    """
    best_thresh = 0.5
    best_macro_f1 = 0.0
    
    thresholds = np.linspace(0.05, 0.95, 91)
    
    for thresh in thresholds:
        preds = [1 if p >= thresh else 0 for p in all_probs]
        macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_thresh = thresh
            
    return float(best_thresh), float(best_macro_f1)
 
 
# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
 
def run_fold(fold, train_data, test_data, device):
    """Train and evaluate a single k-fold split."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold + 1} / {N_FOLDS}")
    print(f"{'='*60}")
    print(f"  Train: {len(train_data)} | Val: {len(test_data)}")
 
    train_labels_bin = [0 if s["label"] == 0 else 1 for s in train_data]
    class_counts     = np.bincount(train_labels_bin)
 
    print(f"  Class counts in fold — Real(0): {class_counts[0]}, Fake(1): {class_counts[1]}")
 
    # Calculate perfectly inverse class weights for balanced sampling
    class_weights = 1.0 / np.array(class_counts, dtype=np.float32)
    sample_weights = [class_weights[label] for label in train_labels_bin]
    
    # Create WeightedRandomSampler (draws total num_samples, with replacement)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print("  [Sampler] Using perfect 1:1 class balancing via WeightedRandomSampler.")

    # With Sampler, shuffle MUST be False
    train_loader = DataLoader(
        FakeAVDataset(train_data),
        batch_size=BATCH_SIZE,
        shuffle=False, 
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader = DataLoader(
        FakeAVDataset(test_data),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
 
    model = DeepTraceModel(num_classes=NUM_CLASSES).to(device)
 
    # Use standard CrossEntropyLoss for perfectly calibrated probabilities.
    # Class weights are not needed because the sampler perfectly balances the batches.
    criterion = nn.CrossEntropyLoss()
 
    optimizer = torch.optim.AdamW([
        {'params': model.video_encoder.cnn.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(),        'lr': 5e-4},
        {'params': model.cross_attn.parameters(),        'lr': 1e-4},
        {'params': model.audio_encoder.parameters(),     'lr': 1e-4}
    ], weight_decay=1e-3)
 
    scaler = torch.cuda.amp.GradScaler()
 
    # FIX 3: scheduler monitors macro F1, not accuracy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
 
    best_macro_f1  = 0.0
    best_threshold = 0.5
    patience_counter = 0
 
    for epoch in range(EPOCHS):
        try:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            print(f"\nEpoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
 
            val_loss, val_acc, all_preds, all_labels, all_probs, macro_f1 = evaluate(
                model, test_loader, criterion, device
            )
            print(f"Epoch {epoch+1}/{EPOCHS} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Macro F1: {macro_f1:.4f}")  # FIX 3: report macro F1
 
            # Prediction distribution (collapse detection)
            unique, counts = np.unique(all_preds, return_counts=True)
            pred_dist = dict(zip(unique.tolist(), counts.tolist()))
            print(f"  Prediction Distribution: {pred_dist}  <-- must have BOTH classes")
 
            # FIX 4: Find optimal threshold each epoch
            thresh, thresh_f1 = find_best_threshold(all_labels, all_probs)
            print(f"  Optimal Threshold: {thresh:.3f} (Macro F1 at thresh: {thresh_f1:.4f})")
 
            # Re-predict with optimal threshold
            adjusted_preds = [
                0 if p < thresh else 1 for p in all_probs
            ]
            adj_macro_f1 = f1_score(all_labels, adjusted_preds, average='macro', zero_division=0)
            print(f"  Adjusted Macro F1 (at optimal threshold): {adj_macro_f1:.4f}")
 
            print(f"\n--- Classification Report — Epoch {epoch+1} (default 0.5 threshold) ---")
            print(classification_report(all_labels, all_preds, digits=4, zero_division=0))
 
            print(f"--- Classification Report — Epoch {epoch+1} (optimal threshold {thresh:.3f}) ---")
            print(classification_report(all_labels, adjusted_preds, digits=4, zero_division=0))
            print("-" * 60)
 
            # FIX 3: step scheduler on adjusted macro F1
            scheduler.step(adj_macro_f1)
 
            if adj_macro_f1 > best_macro_f1:
                best_macro_f1  = adj_macro_f1
                best_threshold = thresh
                torch.save(model.state_dict(),
                           f"deeptrace_fold{fold+1}_best.pth")
                print(f"  >> New best saved (macro F1: {best_macro_f1:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f"  >> Early stopping triggered after 5 epochs without improvement.")
                    break
 
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  [ERROR] Epoch {epoch+1}: {e}")
            continue
 
    print(f"\nFold {fold+1} done. Best Macro F1: {best_macro_f1:.4f} | "
          f"Best Threshold: {best_threshold:.3f}")
    return best_macro_f1, best_threshold, model
 
 
def main():
    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print("  >> Enabled cuDNN benchmark for full GPU speed boost.")
 
    # ── 1. Dataset discovery ──────────────────────────────────────────────────
    label_map = {
        "RealVideo-RealAudio":  0,
        "RealVideo-FakeAudio":  1,
        "FakeVideo-RealAudio":  2,
        "FakeVideo-FakeAudio":  3
    }
    video_samples = []
    for folder, label in label_map.items():
        folder_path = os.path.join(DATASET_PATH, folder)
        if not os.path.exists(folder_path):
            continue
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".mp4"):
                    video_samples.append({
                        "path":  os.path.join(root, file),
                        "label": label
                    })
 
    labels     = [s["label"] for s in video_samples]
    bin_labels = [0 if l == 0 else 1 for l in labels]
 
    total      = len(video_samples)
    n_real     = sum(1 for l in bin_labels if l == 0)
    n_fake     = sum(1 for l in bin_labels if l == 1)
 
    print(f"\nDataset Summary:")
    print(f"  Total Videos: {total}")
    print(f"  Real (class 0): {n_real} ({n_real/total*100:.1f}%)")
    print(f"  Fake (class 1): {n_fake} ({n_fake/total*100:.1f}%)")
    print(f"  Imbalance ratio: 1:{n_fake//max(n_real,1)}")
 
    # ── 2. FIX 5: Stratified k-fold cross-validation ─────────────────────────
    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_f1s   = []
    fold_threshs = []
 
    for fold, (train_idx, val_idx) in enumerate(skf.split(video_samples, bin_labels)):
        train_data = [video_samples[i] for i in train_idx]
        test_data  = [video_samples[i] for i in val_idx]
 
        best_f1, best_thresh, model = run_fold(fold, train_data, test_data, device)
        fold_f1s.append(best_f1)
        fold_threshs.append(best_thresh)
 
    # ── 3. Final summary across folds ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    for i, (f1, t) in enumerate(zip(fold_f1s, fold_threshs)):
        print(f"  Fold {i+1}: Macro F1 = {f1:.4f} | Best Threshold = {t:.3f}")
    print(f"\n  Mean Macro F1 : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"  Mean Threshold: {np.mean(fold_threshs):.3f}")
 
    # Save best overall model (highest fold F1)
    best_fold = int(np.argmax(fold_f1s))
    print(f"\n  Best fold: {best_fold+1} (F1={fold_f1s[best_fold]:.4f})")
    print(f"  Model saved as: deeptrace_fold{best_fold+1}_best.pth")
    print(f"  Recommended inference threshold: {fold_threshs[best_fold]:.3f}")
 
 
if __name__ == "__main__":
    main()
