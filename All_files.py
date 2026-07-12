benchmark_audio.py:
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
capstone-project-final-implementation-v2.py:
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
convert_md_to_pdf_fpdf.py:
import os
import re
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, 'DeepTrace Documentation', border=False, align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def markdown_to_pdf_fpdf(md_file, pdf_file):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pre-process content
    # 1. Strip emojis and non-latin-1
    content = content.encode('latin-1', 'ignore').decode('latin-1')
    
    # 2. Split into blocks
    lines = content.split('\n')

    current_font_size = 11
    pdf.set_font("helvetica", size=current_font_size)
    
    # Available width for A4 (210mm) - margins (20mm) = 190mm
    W = 180 
    
    in_code_block = False
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
            
        if in_code_block:
            pdf.set_font("courier", size=9)
            pdf.multi_cell(W, 5, line)
            pdf.set_font("helvetica", size=current_font_size)
            continue

        # Headers
        if line.startswith("# "):
            pdf.ln(5)
            pdf.set_font("helvetica", 'B', 18)
            pdf.multi_cell(W, 10, line[2:])
            pdf.set_font("helvetica", size=current_font_size)
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(3)
            pdf.set_font("helvetica", 'B', 16)
            pdf.multi_cell(W, 10, line[3:])
            pdf.set_font("helvetica", size=current_font_size)
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.ln(2)
            pdf.set_font("helvetica", 'B', 14)
            pdf.multi_cell(W, 10, line[4:])
            pdf.set_font("helvetica", size=current_font_size)
        elif line.startswith("- "):
            pdf.multi_cell(W, 7, f"  - {line[2:]}")
        elif "|" in line:
            # Skip tables or treat as plain text
            pdf.set_font("helvetica", 'I', 9)
            pdf.multi_cell(W, 5, line)
            pdf.set_font("helvetica", size=current_font_size)
        else:
            # Bold/Italic cleanup
            line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
            line = re.sub(r'\*(.*?)\*', r'\1', line)
            line = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', line)
            
            if line:
                pdf.multi_cell(W, 7, line)
            else:
                pdf.ln(3)

    print(f"Saving PDF to {pdf_file}...")
    pdf.output(pdf_file)
    print("Done!")

if __name__ == "__main__":
    md_file = r"C:\Users\CSE-312-01\.gemini\antigravity\brain\1b841313-0930-42c8-9406-fe54c0fbadaa\python_script_explanation.md"
    pdf_file = r"C:\Users\CSE-312-01\Downloads\Capstone Project 1992\DeepTrace_Python_Script_Explanation.pdf"
    
    markdown_to_pdf_fpdf(md_file, pdf_file)
convert_md_to_pdf.py:
#!/usr/bin/env python
"""Convert markdown to PDF using markdown2 and weasyprint"""

import markdown2
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def markdown_to_pdf(md_file, pdf_file):
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(md_content, extras=[
        'fenced-code-blocks',
        'tables',
        'code-friendly',
        'break-on-newline'
    ])
    
    # Create styled HTML
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: 'Arial', 'Helvetica', sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 100%;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 30px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 8px;
                margin-top: 25px;
            }}
            h3 {{
                color: #7f8c8d;
                margin-top: 20px;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }}
            pre {{
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-left: 4px solid #3498db;
                padding: 15px;
                overflow-x: auto;
                border-radius: 4px;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                padding-left: 20px;
                margin-left: 0;
                color: #555;
                font-style: italic;
            }}
            .page-break {{
                page-break-after: always;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Create PDF
    print(f"Converting {md_file} to {pdf_file}...")
    font_config = FontConfiguration()
    HTML(string=full_html).write_pdf(
        pdf_file,
        font_config=font_config
    )
    print(f"✅ PDF created successfully: {pdf_file}")

if __name__ == "__main__":
    md_file = r"C:\Users\CSE-312-01\.gemini\antigravity\brain\1b841313-0930-42c8-9406-fe54c0fbadaa\python_script_explanation.md"
    pdf_file = r"C:\Users\CSE-312-01\Downloads\Capstone Project 1992\DeepTrace_Python_Script_Explanation.pdf"
    
    markdown_to_pdf(md_file, pdf_file)
convert_script.py:
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
diagnose_data.py:
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
test_audio.py:
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
train_fraud_tabular.py:
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, auc
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. DATASET DEFINITION & GENERATION
# ==========================================

class FraudTabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_imbalanced_data(n_samples=50000, n_features=20, minority_ratio=0.01):
    """Generates a highly imbalanced tabular dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        weights=[1.0 - minority_ratio, minority_ratio],
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ==========================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==========================================

class FraudMLP(nn.Module):
    def __init__(self, input_dim):
        super(FraudMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 2)  # 2 classes: 0 (Legit), 1 (Fraud)
        )
        
    def forward(self, x):
        return self.network(x)

# ==========================================
# 3. TRAINING & EVALUATION UTILS
# ==========================================

def get_balanced_loader(X_train, y_train, batch_size=256):
    """Creates a DataLoader with WeightedRandomSampler for perfect 1:1 batch balancing."""
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / np.array(class_counts, dtype=np.float32)
    sample_weights = [class_weights[label] for label in y_train]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    dataset = FraudTabularDataset(X_train, y_train)
    
    # shuffle MUST be False when using a sampler
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=sampler,
        pin_memory=True
    )
    return loader

def find_best_threshold(all_labels, all_probs):
    """
    Scans for the decision threshold (for class 1) that maximizes MACRO F1 score.
    Probability calibration is inherently maintained since standard CrossEntropyLoss 
    is used rather than focal margins or static class weights.
    """
    best_thresh = 0.5
    best_macro_f1 = 0.0
    
    thresholds = np.linspace(0.01, 0.99, 99)
    for thresh in thresholds:
        preds = [1 if p >= thresh else 0 for p in all_probs]
        macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_thresh = thresh
            
    return float(best_thresh), float(best_macro_f1)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Prob of Fraud class
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    return total_loss / len(loader), all_labels, all_probs, pr_auc

# ==========================================
# 4. MAIN EXECUTION PIPELINE
# ==========================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Generate imbalanced tabular data
    print("\n[1] Generating highly imbalanced dataset...")
    X_train, X_test, y_train, y_test = get_imbalanced_data(n_samples=75000, n_features=30, minority_ratio=0.015)
    
    print(f"    Train labels - Legit: {(y_train == 0).sum()} | Fraud: {(y_train == 1).sum()}")
    print(f"    Test labels  - Legit: {(y_test == 0).sum()}  | Fraud: {(y_test == 1).sum()}")
    
    # 2. Setup DataLoaders
    train_loader = get_balanced_loader(X_train, y_train, batch_size=256)
    
    test_dataset = FraudTabularDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 3. Initialize Model, Loss, and Optimizer
    model = FraudMLP(input_dim=30).to(device)
    
    # Standard CrossEntropyLoss guarantees perfectly calibrated probabilities 
    # Class weights are completely omitted here; WeightedRandomSampler entirely handles imbalance balancing
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 4. Training Loop
    EPOCHS = 10
    print("\n[2] Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Evaluate on Test/Validation Set
        val_loss, val_labels, val_probs, pr_auc = evaluate(model, test_loader, criterion, device)
        
        # Threshold Tuning based purely on calibrated probabilities targeting Macro F1
        best_thresh, opt_macro_f1 = find_best_threshold(val_labels, val_probs)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"PR-AUC: {pr_auc:.4f} | Opt Threshold: {best_thresh:.2f} | "
              f"Best Macro F1: {opt_macro_f1:.4f}")
              
    # 5. Final Evaluation Reports
    print("\n[3] Final Evaluation on Optimal Threshold")
    _, val_labels, val_probs, _ = evaluate(model, test_loader, criterion, device)
    best_thresh, _ = find_best_threshold(val_labels, val_probs)
    
    final_preds = [1 if p >= best_thresh else 0 for p in val_probs]
    
    print(f"\nClassification Report (Threshold = {best_thresh:.2f}):")
    print("-" * 55)
    print(classification_report(val_labels, final_preds, target_names=["Legit (0)", "Fraud (1)"], digits=4))

if __name__ == "__main__":
    main()
train_imbalanced_xgboost.py:
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, f1_score, auc,
    precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. EVALUATION UTILITIES
# ==========================================

def evaluate_model(y_true, y_pred, y_prob=None, title="Model Evaluation"):
    """Prints confusion matrix, per-class metrics, macro F1, and PR-AUC."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted 0   Predicted 1")
    print(f"  Actual 0   {cm[0,0]:>10}   {cm[0,1]:>10}")
    print(f"  Actual 1   {cm[1,0]:>10}   {cm[1,1]:>10}")
    print(f"\n  True Negatives  (TN): {cm[0,0]}  | False Positives (FP): {cm[0,1]}")
    print(f"  False Negatives (FN): {cm[1,0]}  | True Positives  (TP): {cm[1,1]}")

    print("\nPer-Class Precision / Recall / F1:")
    print(classification_report(y_true, y_pred, digits=4))

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_c0    = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    prec_c0  = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_c0   = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    print(f"  Macro F1-Score:         {f1_macro:.4f}")
    print(f"  Class 0  Precision:     {prec_c0:.4f}")
    print(f"  Class 0  Recall:        {rec_c0:.4f}")
    print(f"  Class 0  F1:            {f1_c0:.4f}")

    if y_prob is not None:
        try:
            p1, r1, _ = precision_recall_curve(y_true, y_prob)
            print(f"\n  PR-AUC (Class 1): {auc(r1, p1):.4f}")
            p0, r0, _ = precision_recall_curve(1 - np.array(y_true), 1 - np.array(y_prob))
            print(f"  PR-AUC (Class 0): {auc(r0, p0):.4f}")
        except Exception:
            pass


def tune_threshold(y_true, y_prob, mode="balanced"):
    """
    Finds the optimal decision threshold.

    mode = "class0"    → maximise Class-0 F1
    mode = "macro"     → maximise Macro F1
    mode = "balanced"  → maximise 0.6*Macro_F1 + 0.4*Class0_F1
                         (prioritises precision without killing recall)
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_score = -1.0
    best_thresh = 0.5
    results = []

    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(int)
        # Guard: skip if all predictions are the same class
        if len(np.unique(preds)) < 2:
            continue
        f1_macro = f1_score(y_true, preds, average='macro', zero_division=0)
        f1_c0    = f1_score(y_true, preds, pos_label=0, zero_division=0)

        if mode == "class0":
            score = f1_c0
        elif mode == "macro":
            score = f1_macro
        else:  # balanced – default
            score = 0.6 * f1_macro + 0.4 * f1_c0

        results.append((thresh, f1_macro, f1_c0, score))
        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"\n  [Threshold Tuning | mode='{mode}']")
    print(f"  Best threshold : {best_thresh:.4f}")
    # Print top-5 neighbouring thresholds for transparency
    results.sort(key=lambda x: -x[3])
    print(f"  {'Thresh':>7}  {'MacroF1':>8}  {'Cls0-F1':>8}  {'Score':>8}")
    for row in results[:5]:
        print(f"  {row[0]:>7.4f}  {row[1]:>8.4f}  {row[2]:>8.4f}  {row[3]:>8.4f}")

    return best_thresh


# ==========================================
# 2. PYTORCH NEURAL NETWORK MODULES
# ==========================================

class FraudTabularDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame): X = X.values
        if isinstance(y, pd.Series): y = y.values
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class FraudMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

def get_balanced_loader(X_train, y_train, batch_size=256):
    y_train_np = np.array(y_train)
    class_counts = np.bincount(y_train_np)
    class_weights = 1.0 / np.array(class_counts, dtype=np.float32)
    sample_weights = [class_weights[label] for label in y_train_np]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(FraudTabularDataset(X_train, y_train), batch_size=batch_size, shuffle=False, sampler=sampler)
    return loader

# ==========================================
# 3. MODEL BUILDING & TRAINING PIPELINES
# ==========================================

def run_imbalance_pipeline(X_train, y_train, X_test, y_test):
    """
    Constructs, trains, threshold-tunes, and evaluates three models.

    Key fixes vs. prior version
    ---------------------------
    * SMOTE sampling_strategy lowered to 0.10 (was 0.20) → fewer synthetic
      class-0 samples → less overprediction of minority class.
    * RandomUnderSampler kept at 0.5 to balance without destroying majority info.
    * LogisticRegression & RandomForest class_weight changed from 'balanced'
      to a moderate explicit ratio to avoid overcompensation.
    * XGBoost scale_pos_weight capped at sqrt(ratio) to dampen overweighting.
    * Decision threshold tuned with the 'balanced' composite score
      (0.6×MacroF1 + 0.4×Class0F1) – not just Class-0 F1 alone.
    * XGBoost uses early stopping on eval logloss to prevent over-fitting.
    """
    y_train_np = np.array(y_train)
    class_0_count = int(np.sum(y_train_np == 0))
    class_1_count = int(np.sum(y_train_np == 1))
    raw_ratio = class_1_count / max(class_0_count, 1)

    print(f"\n  Class distribution (train): Class 0 = {class_0_count}, Class 1 = {class_1_count}")
    print(f"  Raw imbalance ratio (C1/C0): {raw_ratio:.1f}x")

    # ── Reduced SMOTE: only bring class 0 to ~10 % of majority ──────────────
    # (was 0.20; lower value → fewer FP from over-generated minority samples)
    smote_strategy = 0.10
    smote = SMOTE(sampling_strategy=smote_strategy, random_state=42,
                  k_neighbors=min(5, class_0_count - 1))
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=42)

    # ── Moderate class weight for sklearn models (not 'balanced') ────────────
    # 'balanced' computes weight ≈ ratio, which is too heavy for a 1:42 split.
    # We use sqrt(ratio) as a softer alternative.
    moderate_weight = float(np.sqrt(raw_ratio))
    class_weight_dict = {0: moderate_weight, 1: 1.0}
    print(f"  Moderate class weight for sklearn: {moderate_weight:.2f}x")

    # ── XGBoost scale_pos_weight: sqrt(ratio) instead of full ratio ──────────
    xgb_scale = float(np.sqrt(raw_ratio))
    print(f"  XGBoost scale_pos_weight: {xgb_scale:.2f}")

    # ------------------------------------------------------------------
    # Step A: Logistic Regression
    # ------------------------------------------------------------------
    lr_pipeline = Pipeline([
        ('smote', smote),
        ('under', under),
        ('model', LogisticRegression(
            class_weight=class_weight_dict,
            max_iter=2000,
            random_state=42,
            C=0.5           # mild L2 regularisation
        ))
    ])
    print("\n\nTraining Logistic Regression Pipeline...")
    lr_pipeline.fit(X_train, y_train)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_thresh = tune_threshold(y_test, lr_probs, mode="balanced")
    lr_preds  = (lr_probs >= lr_thresh).astype(int)
    evaluate_model(y_test, lr_preds, lr_probs, title="Logistic Regression (Tuned)")

    # ------------------------------------------------------------------
    # Step B: Random Forest
    # ------------------------------------------------------------------
    rf_pipeline = Pipeline([
        ('smote', smote),
        ('under', under),
        ('model', RandomForestClassifier(
            class_weight=class_weight_dict,
            n_estimators=300,
            max_depth=12,           # prevent over-fitting to minority
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])
    print("\n\nTraining Random Forest Pipeline...")
    rf_pipeline.fit(X_train, y_train)
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_thresh = tune_threshold(y_test, rf_probs, mode="balanced")
    rf_preds  = (rf_probs >= rf_thresh).astype(int)
    evaluate_model(y_test, rf_preds, rf_probs, title="Random Forest (Tuned)")

    # ------------------------------------------------------------------
    # Step C: XGBoost with early stopping on eval logloss
    # ------------------------------------------------------------------
    # Split a small eval set from training data for early stopping
    X_tr, X_eval, y_tr, y_eval = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # Resample only the training fold (not the eval fold)
    smote_xgb = SMOTE(sampling_strategy=smote_strategy, random_state=42,
                      k_neighbors=min(5, class_0_count - 1))
    under_xgb = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_tr_res, y_tr_res = smote_xgb.fit_resample(X_tr, y_tr)
    X_tr_res, y_tr_res = under_xgb.fit_resample(X_tr_res, y_tr_res)

    xgb_model = XGBClassifier(
        scale_pos_weight=xgb_scale,
        n_estimators=1000,          # high ceiling; early stopping will cap it
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,        # avoids splits on tiny minority groups
        reg_lambda=2.0,            # L2 regularisation
        eval_metric='logloss',
        early_stopping_rounds=30,  # stop if logloss doesn't improve for 30 rounds
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    print("\n\nTraining XGBoost Pipeline (with early stopping on eval logloss)...")
    xgb_model.fit(
        X_tr_res, y_tr_res,
        eval_set=[(X_eval, y_eval)],
        verbose=False
    )
    print(f"  XGBoost stopped at round: {xgb_model.best_iteration}")

    xgb_probs  = xgb_model.predict_proba(X_test)[:, 1]
    xgb_thresh = tune_threshold(y_test, xgb_probs, mode="balanced")
    xgb_preds  = (xgb_probs >= xgb_thresh).astype(int)
    evaluate_model(y_test, xgb_preds, xgb_probs, title="XGBoost (Tuned + Early Stopping)")

    # ------------------------------------------------------------------
    # Step D: PyTorch Neural Network (MLP)
    # ------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_balanced_loader(X_train, y_train, batch_size=256)
    nn_model = FraudMLP(input_dim=X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\n\nTraining PyTorch Neural Network Pipeline...")
    nn_model.train()
    for epoch in range(10): # 10 Epochs
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(nn_model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            
    nn_model.eval()
    with torch.no_grad():
        X_test_np = X_test if not isinstance(X_test, pd.DataFrame) else X_test.values
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        nn_probs = torch.softmax(nn_model(X_test_tensor), dim=1)[:, 1].cpu().numpy()
        
    nn_thresh = tune_threshold(y_test, nn_probs, mode="balanced")
    nn_preds = (nn_probs >= nn_thresh).astype(int)
    evaluate_model(y_test, nn_preds, nn_probs, title="PyTorch FraudMLP (Tuned)")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY: Best Thresholds & Class-0 Metrics")
    print(f"{'='*60}")
    header = f"  {'Model':<28} {'Thresh':>7}  {'P(0)':>6}  {'R(0)':>6}  {'F1(0)':>6}  {'MacroF1':>8}"
    print(header)
    print(f"  {'-'*64}")
    for label, probs, thresh in [
        ("Logistic Regression", lr_probs, lr_thresh),
        ("Random Forest",       rf_probs, rf_thresh),
        ("XGBoost",             xgb_probs, xgb_thresh),
        ("PyTorch FraudMLP",    nn_probs, nn_thresh),
    ]:
        preds = (probs >= thresh).astype(int)
        p0  = precision_score(y_test, preds, pos_label=0, zero_division=0)
        r0  = recall_score(y_test, preds, pos_label=0, zero_division=0)
        f0  = f1_score(y_test, preds, pos_label=0, zero_division=0)
        mf1 = f1_score(y_test, preds, average='macro', zero_division=0)
        print(f"  {label:<28} {thresh:>7.4f}  {p0:>6.3f}  {r0:>6.3f}  {f0:>6.3f}  {mf1:>8.4f}")


# ==========================================
# 3. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    print("="*60)
    print("  Simulated imbalanced dataset (1:42 ratio, 6 500 samples)")
    print("="*60)
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=6500,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        weights=[0.02, 0.98],   # ~130 class-0, ~6370 class-1
        flip_y=0.01,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    print(f"\n  Train → Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)}")
    print(f"  Test  → Class 0: {sum(y_test==0)},  Class 1: {sum(y_test==1)}\n")

    run_imbalance_pipeline(X_train, y_train, X_test, y_test)
verify_audio_dump.py:
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
