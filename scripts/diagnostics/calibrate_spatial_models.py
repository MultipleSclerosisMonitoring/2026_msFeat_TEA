"""
Calibrate K constants for gyro-norm and biometric spatial models.
Uses the 8 outdoor 6MWT trials where GPS gives a reliable reference.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
import sys
sys.path.insert(0, "src")
from gait_analysis.utils.config_loader import load_project_config
from gait_analysis.processor import GaitDataProcessor, ProcessConfig

H5_PATH = "data/raw/gait_study_data.h5"
config = load_project_config("config/config.yaml")
proc_cfg = ProcessConfig(**{k: v for k, v in config["processing"].items()
                            if k in ProcessConfig.model_fields})
processor = GaitDataProcessor(proc_cfg)

# GPS-valid trials from previous audit
GPS_VALID_KEYS = [
    # p_01912299X-118/Right excluded: only 2 heel strikes detected,
    # stride_time=4.01s artefact contaminates gyro_int
    "p_01912299X-118/6MWT/start_2025-11-17T18-24-36Z/Left",
    "p_EPGHUG006-25/6MWT/start_2025-12-12T18-33-18Z/Left",
    "p_EPGHUG006-25/6MWT/start_2025-12-12T18-33-18Z/Right",
    # p_RHRHUG004-1/Dec-01 excluded: GPS outlier (multipath)
    "p_RHRHUG004-1/6MWT/start_2025-12-02T13-05-47Z/Left",
    "p_RHRHUG004-1/6MWT/start_2025-12-02T13-05-47Z/Right",
]

FS = 100.0
nyq = 0.5 * FS
b, a = butter(4, 5.0 / nyq, btype="low")

results = []
print(f"{'Key':<65} {'SL_gps':>8} {'gyro_int':>10} {'cadence':>9}")
print("=" * 95)

for h5_key in GPS_VALID_KEYS:
    # Load CSV metrics (already computed with GPS)
    safe_key = h5_key.replace("/", "_")
    csv_path = Path(f"reports/data/metrics_summary_{safe_key}.csv")
    if not csv_path.exists():
        print(f"SKIP {h5_key}: CSV not found")
        continue

    row = pd.read_csv(csv_path).iloc[0]
    if row["spatial_method"] != "gps":
        print(f"SKIP {h5_key}: spatial_method={row['spatial_method']}")
        continue

    sl_gps = float(row["walking_speed_mean_m_s"]) * float(row["stride_time_mean_s"])
    cadence = float(row["stride_cadence_spm"])

    # Compute gyro integral during swing phases
    df_raw = pd.read_hdf(H5_PATH, key=h5_key)
    df_proc, metrics, peaks, toe_offs, _ = processor.process_signals(
        df_raw, test_type="6MWT",
        clinical_tests_cfg=config.get("clinical_tests", {}),
        gps_estimation_cfg=config.get("gps_estimation", {}),
    )

    # Filter gyro axes
    gx_f = filtfilt(b, a, df_proc["Gx_filt"].to_numpy())
    gy_f = filtfilt(b, a, df_proc["Gy_filt"].to_numpy())
    gz_f = filtfilt(b, a, df_proc["Gz_filt"].to_numpy())
    gyro_norm = np.sqrt(gx_f**2 + gy_f**2 + gz_f**2)

    n_swings = min(len(toe_offs), len(peaks) - 1)
    swing_integrals = []
    for i in range(n_swings):
        to_idx = toe_offs[i]
        next_hs_idx = peaks[i + 1]
        if next_hs_idx <= to_idx:
            continue
        integral = np.trapezoid(gyro_norm[to_idx:next_hs_idx], dx=1.0/FS)
        swing_integrals.append(integral)

    if not swing_integrals:
        continue

    gyro_int_mean = float(np.mean(swing_integrals))

    print(f"{h5_key:<65} {sl_gps:>8.3f} {gyro_int_mean:>10.3f} {cadence:>9.2f}")
    results.append({
        "key": h5_key,
        "sl_gps": sl_gps,
        "gyro_int_mean": gyro_int_mean,
        "cadence_spm": cadence,
    })

df = pd.DataFrame(results)
print(f"\n{len(df)} trials used for calibration.\n")

if len(df) >= 2:
    # Calibrate K_gyro: sl_gps = K_gyro * gyro_int_mean
    K_gyro = float(np.dot(df["sl_gps"], df["gyro_int_mean"]) /
                   np.dot(df["gyro_int_mean"], df["gyro_int_mean"]))
    pred_gyro = K_gyro * df["gyro_int_mean"]
    err_gyro = df["sl_gps"] - pred_gyro
    rmse_gyro = float(np.sqrt((err_gyro**2).mean()))

    # Calibrate K_bio: sl_gps = K_bio * sqrt(cadence)
    sqrt_cad = np.sqrt(df["cadence_spm"])
    K_bio = float(np.dot(df["sl_gps"], sqrt_cad) /
                  np.dot(sqrt_cad, sqrt_cad))
    pred_bio = K_bio * sqrt_cad
    err_bio = df["sl_gps"] - pred_bio
    rmse_bio = float(np.sqrt((err_bio**2).mean()))

    print("=" * 50)
    print("CALIBRATION RESULTS")
    print("=" * 50)
    print(f"\nGyro-norm model:  K_gyro = {K_gyro:.6f}")
    print(f"  RMSE = {rmse_gyro:.4f} m  ({100*rmse_gyro/df['sl_gps'].mean():.1f}% of mean SL)")

    print(f"\nBiometric model:  K_bio = {K_bio:.6f}")
    print(f"  RMSE = {rmse_bio:.4f} m  ({100*rmse_bio/df['sl_gps'].mean():.1f}% of mean SL)")

    print(f"\nMean stride length GPS (reference): {df['sl_gps'].mean():.3f} m")