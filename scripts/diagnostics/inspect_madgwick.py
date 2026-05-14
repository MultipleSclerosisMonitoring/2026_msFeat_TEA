"""
Diagnose Madgwick filter output on a known 6MWT trial.
Estimates foot orientation and plots pitch angle vs time
to verify that the gait cycle is visible.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick

H5_KEY = "p_RHRHUG004-1/6MWT/start_2025-11-28T12-46-09Z/Left"
H5_PATH = "data/raw/gait_study_data.h5"
FS = 100.0
BETA = 0.1

# Load and resample
df = pd.read_hdf(H5_PATH, key=H5_KEY)
df["_time"] = pd.to_datetime(df["_time"])
df = df.sort_values("_time").reset_index(drop=True)
t_sec = (df["_time"] - df["_time"].iloc[0]).dt.total_seconds().to_numpy()
t_uniform = np.arange(0.0, t_sec[-1], 1.0 / FS)

def resample(col):
    return np.interp(t_uniform, t_sec, df[col].astype(float).to_numpy())

ax = resample("Ax"); ay = resample("Ay"); az = resample("Az")
gx = resample("Gx"); gy = resample("Gy"); gz = resample("Gz")
mx = resample("Mx"); my = resample("My"); mz = resample("Mz")

# Low-pass filter accelerometer to reduce impact spikes
nyq = 0.5 * FS
b, a = butter(4, 10.0 / nyq, btype="low")
ax_f = filtfilt(b, a, ax)
ay_f = filtfilt(b, a, ay)
az_f = filtfilt(b, a, az)

# Gyroscope: convert deg/s to rad/s
gx_r = np.deg2rad(gx)
gy_r = np.deg2rad(gy)
gz_r = np.deg2rad(gz)

# Stack for ahrs
acc = np.column_stack([ax_f, ay_f, az_f])
gyr = np.column_stack([gx_r, gy_r, gz_r])
mag = np.column_stack([mx, my, mz])

print(f"Running Madgwick on {len(t_uniform)} samples (beta={BETA})...")
madgwick = Madgwick(gyr=gyr, acc=acc, frequency=FS, beta=BETA)
Q = madgwick.Q  # shape (N, 4): [w, x, y, z]

# Extract pitch angle from quaternion
# Pitch = arcsin(2*(w*y - z*x))
w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
pitch_rad = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
pitch_deg = np.rad2deg(pitch_rad)

# Plot first 30 seconds
mask = t_uniform <= 30.0
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(t_uniform[mask], ay_f[mask], color="steelblue", lw=0.8)
axes[0].set_ylabel("Ay_filtered (g)")
axes[0].set_title("Vertical acceleration (filtered)")
axes[0].axhline(0, color="gray", lw=0.5)

axes[1].plot(t_uniform[mask], gz[mask], color="darkorange", lw=0.8)
axes[1].set_ylabel("Gz (deg/s)")
axes[1].set_title("Gyroscope Z — rotation in sagittal plane")
axes[1].axhline(0, color="gray", lw=0.5)

axes[2].plot(t_uniform[mask], pitch_deg[mask], color="darkgreen", lw=1.0)
axes[2].set_ylabel("Pitch (deg)")
axes[2].set_xlabel("Time (s)")
axes[2].set_title(f"Estimated foot pitch angle (Madgwick beta={BETA})")
axes[2].axhline(0, color="gray", lw=0.5)

plt.tight_layout()
out = "reports/plots/madgwick_diagnosis.png"
plt.savefig(out, dpi=150)
print(f"Plot saved to {out}")

# Stats
print(f"\nPitch stats (full trial):")
print(f"  mean = {pitch_deg.mean():.1f} deg")
print(f"  std  = {pitch_deg.std():.1f} deg")
print(f"  min  = {pitch_deg.min():.1f} deg")
print(f"  max  = {pitch_deg.max():.1f} deg")

# Check oscillation: count zero crossings in pitch (proxy for stride count)
zero_crossings = np.sum(np.diff(np.sign(pitch_deg - pitch_deg.mean())) != 0)
estimated_strides = zero_crossings // 2
print(f"\nZero crossings in pitch: {zero_crossings}")
print(f"Estimated strides from pitch oscillation: {estimated_strides}")
print(f"Expected strides (from pipeline): ~371")