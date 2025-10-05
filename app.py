# dashboard/app.py
import streamlit as st
import numpy as np
import torch
import io
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler

# Import your model and dataset
from src.model_hybrid import HybridModel
from src.data_loader import LightCurveDataset

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="🌌 A World Away — Integrated Demo", layout="wide")
st.title("🌌 A World Away — Integrated Exoplanet Demo (Animated Simulation)")

# ----------------------------
# Model loader (cached)
# ----------------------------
# Prefer the new 'checkpoints' folder but fall back to the existing 'checkpoint' folder
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "best_model.pth")
ALT_CHECKPOINT = os.path.join("checkpoint", "best_model.pth")
# Resolve to whichever file actually exists, defaulting to DEFAULT_CHECKPOINT when neither exists
if os.path.exists(DEFAULT_CHECKPOINT):
    CHECKPOINT_PATH = DEFAULT_CHECKPOINT
elif os.path.exists(ALT_CHECKPOINT):
    CHECKPOINT_PATH = ALT_CHECKPOINT
else:
    CHECKPOINT_PATH = DEFAULT_CHECKPOINT

@st.cache_resource
def load_model():
    model = HybridModel(input_len=512)
    if os.path.exists(CHECKPOINT_PATH):
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
            st.sidebar.success("✅ Loaded trained checkpoint.")
        except Exception as e:
 #           st.sidebar.warning("⚠️ Found checkpoint but failed to load. Using random init.")
            st.sidebar.write(str(e))
#    else:
 #       st.sidebar.warning("⚠️ Checkpoint not found — model initialized randomly.")
    model.eval()
    return model

model = load_model()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("⚙️ Controls")
use_uploaded = st.sidebar.file_uploader("Upload a light curve CSV/TXT (one or multiple numeric cols)", type=["csv", "txt"])
sample_index = st.sidebar.slider("Synthetic sample index (if no upload)", 0, 199, 0)
star_radius_km = st.sidebar.number_input("Star radius (km) — used to estimate planet size", value=696340.0)
animate = st.sidebar.checkbox("Enable animated light simulation", value=True)
frame_rate = st.sidebar.slider("Animation frame delay (ms)", 10, 200, 40)

# ----------------------------
# Load light curve (upload or synthetic)
# ----------------------------
def load_flux_from_upload(uploaded):
    try:
        df = pd.read_csv(uploaded)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            # fallback: read single-column CSV ignoring headers
            df2 = pd.read_csv(uploaded, header=None)
            series = pd.to_numeric(df2[0], errors='coerce').dropna().values
        else:
            # if there are multiple numeric columns, let’s choose the first or flatten
            # prefer columns named like 'flux' or similar
            if any('flux' in c.lower() for c in numeric.columns):
                col = [c for c in numeric.columns if 'flux' in c.lower()][0]
                series = numeric[col].values
            else:
                # flatten numeric values into a single 1D series
                series = numeric.values.flatten()
        return series.astype(np.float32)
    except Exception:
        # fallback robust method
        uploaded.seek(0)
        df = pd.read_csv(uploaded, header=None)
        series = pd.to_numeric(df[0], errors='coerce').dropna().values
        return series.astype(np.float32)

if use_uploaded is not None:
    flux_raw = load_flux_from_upload(use_uploaded)
    if flux_raw.size == 0:
        st.error("Uploaded file contains no numeric flux data.")
        st.stop()
    # resize/interpolate to length 512
    if flux_raw.shape[0] != 512:
        flux = np.interp(np.linspace(0, flux_raw.shape[0]-1, 512), np.arange(flux_raw.shape[0]), flux_raw)
    else:
        flux = flux_raw.copy()
else:
    ds = LightCurveDataset(n_samples=200, length=512)
    X, y, _ = ds.get_numpy()
    flux = X[sample_index]

# keep original flux copy (for depth calculation)
flux_original = flux.copy()

# ----------------------------
# Plot light curve
# ----------------------------
st.subheader("🌠 Light Curve")
fig, ax = plt.subplots(figsize=(10, 2))
ax.plot(flux, color="deepskyblue")
ax.set_xlabel("Time step")
ax.set_ylabel("Flux")
ax.set_title("Selected Light Curve")
st.pyplot(fig)

# ----------------------------
# Preprocess & model predict
# ----------------------------
# Simple scaler (we fit on this sample to normalize for visualization & model)
scaler = StandardScaler()
flux_scaled = scaler.fit_transform(flux.reshape(-1, 1)).flatten()

# Model inference (expecting input shape [B, 1, L])
with st.spinner("Running model inference..."):
    xb = torch.from_numpy(flux_scaled).float().unsqueeze(0).unsqueeze(0)
    try:
        logits, reg = model(xb)
        prob = float(torch.sigmoid(logits).item())
        reg_out = reg.detach().cpu().numpy().tolist()
    except Exception:
        # If model forward fails for any reason, fallback to random prob
        prob = float(np.random.rand())
        reg_out = [0.0, 0.0, 0.0]

st.metric("🪐 Transit Probability", f"{prob:.3f}")
st.write("**Regression Output (sample):**", reg_out)
st.write("---")

# ----------------------------
# Transit depth -> planet radius
# ----------------------------
st.subheader("📏 Transit Depth & Planet Size Estimation")

# For depth calculation we should use flux relative to baseline.
# Use median baseline estimate (robust)
baseline = np.median(flux_original)
min_flux = np.min(flux_original)
# Avoid negative or weird numbers: compute fractional depth (positive)
transit_depth_fraction = max(0.0, (baseline - min_flux) / baseline) if baseline != 0 else 0.0

R_star = float(star_radius_km)
R_planet_km = R_star * np.sqrt(transit_depth_fraction) if transit_depth_fraction > 0 else 0.0

st.write(f"**Transit depth (fraction):** {transit_depth_fraction:.6f}")
st.write(f"**Estimated planet radius:** {R_planet_km:.2f} km")

# Show approximate comparisons
JUPITER_RADIUS_KM = 69911.0
if R_planet_km > 0:
    st.write(f"≈ {R_planet_km / JUPITER_RADIUS_KM:.2f} × Jupiter radius")

# ----------------------------
# Animated light simulation
# ----------------------------
st.subheader("💡 Light Interaction Simulation")
st.caption("Animation shows vertical light rays and an object sized proportionally to the estimated planet radius. Use Start / Stop to control animation.")

# UI controls for animation
col1, col2 = st.columns([1, 3])
with col1:
    start_btn = st.button("Start Animation")
    stop_btn = st.button("Stop Animation")

with col2:
    st.write("Object width is proportional to estimated planet radius (clamped).")

# Determine object size in pixels (map planet km -> px)
# heuristic mapping: scale planet radius to pixels (tune as needed)
# clamp width to [20, 220] px for visibility
px_width = int(np.clip((R_planet_km / 1000.0), 20, 220)) if R_planet_km > 0 else 60
px_height = int(px_width * 1.2)  # slight aspect ratio

# Simulation params
W, H = 700, 320
LIGHT_GAP = 60
speed = 4  # pixels per frame
bg_color = (10, 10, 20)
ray_color = (255, 255, 200)
blocked_color = (255, 80, 80)
obj_color = (0, 200, 0)

# Prepare a placeholder for st.image
placeholder = st.empty()

# Animation loop guard variables
running = False
# If start pressed, start animation; if stop pressed, break
# Use session state to keep running across reruns
if "anim_running" not in st.session_state:
    st.session_state["anim_running"] = False

if start_btn:
    st.session_state["anim_running"] = True
if stop_btn:
    st.session_state["anim_running"] = False

# Animation frame generation
def gen_frame(x_pos):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = bg_color
    # draw rays
    for x in range(0, W, LIGHT_GAP):
        cv2.line(frame, (x, 0), (x, H), ray_color, 2)
    # draw object
    obj_x = int(x_pos)
    obj_y = (H - px_height) // 2
    cv2.rectangle(frame, (obj_x, obj_y), (obj_x + px_width, obj_y + px_height), obj_color, -1)
    # show blocked ray segments in red where rays intersect object
    for x in range(0, W, LIGHT_GAP):
        if obj_x < x < obj_x + px_width:
            cv2.line(frame, (x, 0), (x, obj_y), blocked_color, 2)
            cv2.line(frame, (x, obj_y + px_height), (x, H), blocked_color, 2)
    # overlay text (show object pixel dims and mapped radius)
    cv2.putText(frame, f"Obj WxH: {px_width}px x {px_height}px", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
    cv2.putText(frame, f"Est Rp: {R_planet_km:.0f} km", (12, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
    return frame

# Run the animation if session_state says running, otherwise show single frame
if st.session_state["anim_running"] and animate:
    x = 0
    direction = 1
    try:
        while st.session_state["anim_running"]:
            frame = gen_frame(x)
            # convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)
            x += speed * direction
            if x + px_width >= W or x <= 0:
                direction *= -1
            # sleep per frame rate
            time.sleep(frame_rate / 1000.0)
            # allow Streamlit to rerun / pick up stop button
            if not st.session_state["anim_running"]:
                break
    except st.script_runner.StopException:
        # Streamlit internal stop; just pass
        pass
    except Exception as e:
        st.error(f"Animation error: {e}")
else:
    # show a single static frame
    frame = gen_frame(W // 2 - px_width // 2)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, use_container_width=True)

st.write("---")
st.info("Tip: Train and save a model to 'checkpoints/best_model.pth' for real predictions; otherwise model uses random init.")
