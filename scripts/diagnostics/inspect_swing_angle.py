"""
Estimate stride length from gyroscope swing angle integration.
Model: stride_length ≈ 2 * L_foot * sin(theta_swing / 2)
where theta_swing is the foot rotation angle during swing phase.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

H5_KEY = "p_RHRHUG004-1/6MWT/start_2025-11-28T12-46-09Z/Left"
H5_PATH = "data/raw/gait_study_data.h5"
FS = 100.0
L_FOOT = 0.26  # metres, typical adult foot length

# Load pipeline results to get peaks and toe_offs
import sys
sys.path.insert(0, "src")
from gait_analysis.utils.config_loader import load_project_config
from gait_analysis.processor import GaitDataProcessor, ProcessConfig

config = load_project_config("config/config.yaml")
proc_cfg = ProcessConfig(**{k: v for k, v in config["processing"].items()
                            if k in ProcessConfig.model_fields})
processor = GaitDataProcessor(proc_cfg)

df_raw = pd.read_hdf(H5_PATH, key=H5_KEY)
df_proc, metrics, peaks, toe_offs, _ = processor.process_signals(
    df_raw, test_type="6MWT",
    clinical_tests_cfg=config.get("clinical_tests", {}),
    gps_estimation_cfg=config.get("gps_estimation", {}),
)

# Resample time axis
df_proc["_time"] = pd.to_datetime(df_proc["_time"])
t_sec = (df_proc["_time"] - df_proc["_time"].iloc[0]).dt.total_seconds().to_numpy()

# Filter gyroscope
nyq = 0.5 * FS
b, a = butter(4, 5.0 / nyq, btype="low")
gz_f = filtfilt(b, a, df_proc["Gz_filt"].to_numpy())

# For each stride: integrate Gz during swing phase (TO → next HS)
n_swings = min(len(toe_offs), len(peaks) - 1)
stride_lengths = []
swing_angles = []

for i in range(n_swings):
    to_idx = toe_offs[i]
    next_hs_idx = peaks[i + 1]
    if next_hs_idx <= to_idx:
        continue
    # Integrate Gz (deg/s) over swing duration → angle in degrees
    swing_gz = gz_f[to_idx:next_hs_idx]
    dt = 1.0 / FS
    theta_deg = np.abs(np.trapezoid(swing_gz, dx=dt))
    theta_rad = np.deg2rad(theta_deg)
    # Pendulum model: stride_length ≈ 2 * L * sin(theta/2)
    sl = 2.0 * L_FOOT * np.sin(theta_rad / 2.0)
    stride_lengths.append(sl)
    swing_angles.append(theta_deg)

stride_lengths = np.array(stride_lengths)
swing_angles = np.array(swing_angles)

print(f"Computed {len(stride_lengths)} stride lengths from gyroscope swing angle.")
print(f"\nSwing angle (deg):")
print(f"  mean = {swing_angles.mean():.1f}")
print(f"  std  = {swing_angles.std():.1f}")
print(f"  min  = {swing_angles.min():.1f}")
print(f"  max  = {swing_angles.max():.1f}")

print(f"\nStride length from gyroscope model (m):")
print(f"  mean = {stride_lengths.mean():.3f}")
print(f"  std  = {stride_lengths.std():.3f}")
print(f"  min  = {stride_lengths.min():.3f}")
print(f"  max  = {stride_lengths.max():.3f}")

# Compare with GPS-based stride length for the outdoor trial
print(f"\nReference (GPS outdoor trial RHRHUG004-1/Dec-02):")
print(f"  walking_speed = 1.106 m/s")
print(f"  stride_time   = 1.139 s")
print(f"  stride_length_gps = {1.106 * 1.139:.3f} m")
print(f"  stride_length_gyro = {stride_lengths.mean():.3f} m")

# Plot distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(swing_angles, bins=30, color="darkorange", edgecolor="white")
axes[0].set_xlabel("Swing angle (deg)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of swing angles")

axes[1].hist(stride_lengths, bins=30, color="darkgreen", edgecolor="white")
axes[1].set_xlabel("Stride length (m)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Stride length from gyro model (L_foot={L_FOOT} m)")
axes[1].axvline(stride_lengths.mean(), color="red", lw=1.5, label=f"mean={stride_lengths.mean():.3f} m")
axes[1].legend()

plt.tight_layout()
out = "reports/plots/gyro_stride_length.png"
plt.savefig(out, dpi=150)
print(f"\nPlot saved to {out}")