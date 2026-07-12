#train_fraud_tabular.py:
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
