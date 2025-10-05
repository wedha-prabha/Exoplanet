# src/evaluate.py
import torch
from src.model_hybrid import HybridModel
from src.data_loader import LightCurveDataset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate(model_path="checkpoints/best_model.pth", samples=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LightCurveDataset(n_samples=samples)
    X,y, _ = ds.get_numpy()
    model = HybridModel(input_len=X.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).float().unsqueeze(1).to(device)
        logits, _ = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
    auc = roc_auc_score(y, probs)
    preds = (probs > 0.5).astype(int)
    p,r,f,_ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    print("AUC:", auc, "Precision:", p, "Recall:", r, "F1:", f)
    # plot some examples
    os.makedirs("figs", exist_ok=True)
    for i in range(3):
        idx = i
        plt.figure(figsize=(6,2))
        plt.plot(X[idx])
        plt.title(f"Label={y[idx]} Prob={probs[idx]:.3f}")
        plt.savefig(f"figs/sample_{i}.png")
        plt.close()

if __name__ == "__main__":
    evaluate()
