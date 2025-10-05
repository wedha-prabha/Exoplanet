# src/train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.data_loader import LightCurveDataset
from src.model_hybrid import HybridModel
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# ---------------------------
# Dataset and DataLoader
# ---------------------------
def build_loaders(samples=1000, batch=32, length=512):
    ds = LightCurveDataset(n_samples=samples, length=length)
    X, y, _ = ds.get_numpy()
    # Train/val split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train).float().unsqueeze(1),
                             torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float().unsqueeze(1),
                           torch.from_numpy(y_val).float())

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch)
    return train_loader, val_loader, ds

# ---------------------------
# Training Function
# ---------------------------
def train_model(samples=2000, batch=32, epochs=10, length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, ds = build_loaders(samples, batch, length)
    model = HybridModel(input_len=length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    best_auc = 0.0

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            cls_logits, reg = model(xb)
            cls_loss = bce(cls_logits, yb)
            # regression dummy target: zeros for simplicity
            reg_target = torch.zeros_like(reg)
            loss = cls_loss + 0.1 * mse(reg, reg_target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validation
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits, _ = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                ys.extend(yb.numpy().tolist())
                ps.extend(probs.tolist())
        try:
            auc = roc_auc_score(ys, ps)
        except:
            auc = 0.5

        print(f"Epoch {epoch+1} | Train Loss: {np.mean(losses):.4f} | Val AUC: {auc:.4f}")

        # Save best model
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join("checkpoints", "best_model.pth"))
            print("✅ Saved best model.")

    print("🎉 Training complete. Best AUC:", best_auc)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    train_model(samples=2000, batch=32, epochs=10, length=512)
