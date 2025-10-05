# src/data_loader.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import batman
    HAS_BATMAN = True
except Exception:
    HAS_BATMAN = False

def simulate_transit(time, period=10., rp=0.05, duration=0.2, t0=1.0, noise_std=0.0005):
    """
    Very simple transit generator. If batman is present, use it.
    Otherwise, produce a box-shaped dip convolved with a small gaussian.
    """
    if HAS_BATMAN:
        params = batman.TransitParams()
        params.t0 = t0
        params.per = period
        params.rp = rp
        params.a = 15.0
        params.inc = 89.0
        params.ecc = 0.
        params.w = 90.
        params.u = [0.1, 0.3]
        params.limb_dark = "quadratic"
        m = batman.TransitModel(params, time)
        flux = m.light_curve(params)
        # normalize around 1.
        flux = flux / np.median(flux)
        flux += np.random.normal(0, noise_std, size=flux.shape)
        return flux
    else:
        flux = np.ones_like(time)
        # place a box dip centered at t0 with width=duration fraction of period
        dip_mask = np.abs(((time - t0 + 0.5*period) % period) - 0.5*period) < duration/2
        flux[dip_mask] -= rp
        flux += np.random.normal(0, noise_std, size=flux.shape)
        return flux

def simulate_non_transit(time, noise_std=0.0007):
    # Stellar variability + noise
    flux = 1 + 0.002*np.sin(2*np.pi*time/30.0) + 0.001*np.sin(2*np.pi*time/7.0)
    flux += np.random.normal(0, noise_std, size=flux.shape)
    return flux

class LightCurveDataset:
    """
    Simple in-memory dataset of windows of light curves.
    Each sample: (flux_array, label, meta)
    """
    def __init__(self, n_samples=1000, length=512, out_dir="data/synthetic"):
        self.n_samples = n_samples
        self.length = length
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.time = np.linspace(0, 30, length)
        self.samples = []
        self._build(n_samples)

    def _build_one(self, label):
        # randomize parameters
        period = np.random.uniform(3, 20)
        rp = np.random.uniform(0.005, 0.08) if label==1 else 0.0
        duration = np.random.uniform(0.05, 0.6)
        t0 = np.random.uniform(0, period)
        if label == 1:
            flux = simulate_transit(self.time, period=period, rp=rp, duration=duration, t0=t0)
        else:
            flux = simulate_non_transit(self.time)
        # normalize
        flux = (flux - np.median(flux)) / np.std(flux)
        meta = {"period": period if label==1 else None, "rp": rp if label==1 else None, "duration": duration if label==1 else None}
        return flux.astype(np.float32), label, meta

    def _build(self, n_samples):
        self.samples = []
        for i in tqdm(range(n_samples), desc="Generating synthetic data"):
            label = 1 if np.random.rand() < 0.5 else 0
            flux, label, meta = self._build_one(label)
            self.samples.append((flux, label, meta))
        # persist small csv summary
        df = pd.DataFrame([{"idx":i, "label":s[1], **(s[2] or {})} for i,s in enumerate(self.samples)])
        df.to_csv(os.path.join(self.out_dir, "manifest.csv"), index=False)

    def get_numpy(self):
        X = np.stack([s[0] for s in self.samples])
        y = np.array([s[1] for s in self.samples])
        metas = [s[2] for s in self.samples]
        return X, y, metas

if __name__ == "__main__":
    ds = LightCurveDataset(n_samples=500, length=512)
    X,y,_ = ds.get_numpy()
    print("X shape:", X.shape, "y dist:", np.bincount(y))
