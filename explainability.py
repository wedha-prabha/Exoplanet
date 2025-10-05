# src/explainability.py
import shap
import torch
import numpy as np
from src.model_hybrid import HybridModel

def shap_explain(model_path="checkpoints/best_model.pth", background_samples=50, target_sample=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # use synthetic data as background
    from src.data_loader import LightCurveDataset
    ds = LightCurveDataset(n_samples=200)
    X,y,_ = ds.get_numpy()
    model = HybridModel(input_len=X.shape[1]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # wrap model prediction for shap
    def f(X_np):
        # X_np: (N, L)
        with torch.no_grad():
            xb = torch.from_numpy(X_np).float().unsqueeze(1).to(device)
            logits, _ = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.reshape(-1,1)
    background = X[:background_samples]
    explainer = shap.KernelExplainer(f, background)
    test = X[target_sample:target_sample+1]
    shap_values = explainer.shap_values(test, nsamples=100)
    # shap returns list; plot
    shap.summary_plot(shap_values, test)
    return shap_values

if __name__ == "__main__":
    shap_explain()
