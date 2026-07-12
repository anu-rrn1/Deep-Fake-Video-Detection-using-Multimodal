#train_imbalanced_xgboost.py:
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
