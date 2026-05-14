"""Run spatial analysis on all TUG and T25FW trials. Print summary table."""

import subprocess
import pandas as pd
from pathlib import Path

H5_PATH = "data/raw/gait_study_data.h5"

# Listar todas las keys de TUG y T25FW
with pd.HDFStore(H5_PATH, mode="r") as store:
    all_keys = sorted(store.keys())

target_keys = [
    k.lstrip("/") for k in all_keys
    if "/TUG/" in k or "/T25FW/" in k
]

print(f"Found {len(target_keys)} TUG/T25FW keys.")
print(f"Will process them all and gather the spatial metrics.\n")

# Ejecutar el pipeline para cada key
results = []
for i, h5_key in enumerate(target_keys, 1):
    print(f"[{i}/{len(target_keys)}] {h5_key}")
    result = subprocess.run(
        [
            "python", "-m", "gait_analysis.cli.analyze_gait",
            "--config", "config/config.yaml",
            "--h5-key", h5_key,
            "--no-plots",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", **__import__("os").environ},
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.splitlines()[-3:]}")
        continue

    # Cargar el CSV recién generado
    safe_key = h5_key.replace("/", "_")
    csv_path = Path(f"reports/data/metrics_summary_{safe_key}.csv")
    if not csv_path.exists():
        print(f"  CSV not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    row = df.iloc[0]
    parts = h5_key.split("/")
    results.append({
        "patient": parts[0],
        "test": parts[1],
        "foot": parts[3],
        "duration_s": row["walking_duration_s"],
        "n_strides": row.get("stride_cadence_spm", 0) * row["walking_duration_s"] / 60,
        "stride_time_s": row["stride_time_mean_s"],
        "cadence_spm": row["stride_cadence_spm"],
        "stance_pct": (
            100 * row["stance_time_mean_s"] / row["stride_time_mean_s"]
            if row["stride_time_mean_s"] > 0 else 0
        ),
        "speed_m_s": row["walking_speed_mean_m_s"],
        "stride_len_m": row["stride_length_mean_m"],
        "method": row["spatial_method"],
    })

# Tabla resumen
df_summary = pd.DataFrame(results)
print("\n" + "=" * 130)
print("RESULTS PER TRIAL")
print("=" * 130)
print(df_summary.to_string(index=False))

# Estadísticas por test
print("\n" + "=" * 130)
print("AGGREGATE STATS BY TEST TYPE (only spatial method = known_distance)")
print("=" * 130)
df_valid = df_summary[df_summary["method"] == "known_distance"]

for test_type in ["TUG", "T25FW"]:
    subset = df_valid[df_valid["test"] == test_type]
    if subset.empty:
        continue
    print(f"\n{test_type} ({len(subset)} valid trials):")
    print(f"  walking_speed_m_s:  mean={subset['speed_m_s'].mean():.3f}  "
          f"min={subset['speed_m_s'].min():.3f}  max={subset['speed_m_s'].max():.3f}")
    print(f"  stride_length_m:    mean={subset['stride_len_m'].mean():.3f}  "
          f"min={subset['stride_len_m'].min():.3f}  max={subset['stride_len_m'].max():.3f}")
    print(f"  cadence_spm:        mean={subset['cadence_spm'].mean():.1f}  "
          f"min={subset['cadence_spm'].min():.1f}  max={subset['cadence_spm'].max():.1f}")

# Trials descartados
df_invalid = df_summary[df_summary["method"] == "none"]
if not df_invalid.empty:
    print(f"\n{len(df_invalid)} trials con spatial_method='none' (duracion < min):")
    for _, row in df_invalid.iterrows():
        print(f"  {row['patient']:<25} {row['test']:>6} {row['foot']:>5}  "
              f"duration={row['duration_s']:.2f}s")