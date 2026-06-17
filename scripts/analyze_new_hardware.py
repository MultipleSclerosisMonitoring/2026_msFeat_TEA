"""
Análisis del dataset ampliado con hardware +/-8g y mapa GPS de trayectorias.

Ejecutar desde la raiz del proyecto:
    $env:PYTHONPATH="src"
    python scripts/analyze_new_hardware.py

Salida:
    - reports/plots/chapter5/fig_hardware_comparison.png
    - reports/plots/chapter5/mapa_gps_trayectorias.html
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / 'reports' / 'plots' / 'chapter5'
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_BLUE  = '#185FA5'
C_GREEN = '#1D9E75'
C_RED   = '#D85A30'
C_AMBER = '#E8A838'
C_GRAY  = '#5F5E5A'
C_LGRAY = '#F5F5F5'
DPI     = 180

# Cargar datos
df = pd.read_csv(PROJECT_ROOT / 'reports' / 'data' / 'batch_metrics.csv')

g6 = df[
    (df['test_type'] == '6MWT') &
    (df['foot'] == 'Left') &
    (~df['patient_id'].isin(['AMIR-48', 'SENSORIA-57']))
].copy()

g6['fecha'] = g6['analysis_h5_key'].str.extract(r'start_(\d{4}-\d{2}-\d{2})')
g6['hardware'] = g6['fecha'].apply(lambda x: 'pm8g' if x >= '2026-06-11' else 'pm2g')

outliers_pid = ['p_EDLROHGU044-29', 'p_CJMGHUG063-75']
gps_clean = g6[
    (g6['spatial_method'] == 'gps') &
    (g6['walking_speed_mean_m_s'] < 3.0) &
    (g6['walking_speed_mean_m_s'] > 0.1) &
    (~g6['patient_id'].isin(outliers_pid))
].copy()

g2 = g6[g6['hardware'] == 'pm2g']
g8 = g6[g6['hardware'] == 'pm8g']
gps2 = gps_clean[gps_clean['hardware'] == 'pm2g']
gps8 = gps_clean[gps_clean['hardware'] == 'pm8g']

print(f'Total trials 6MWT: {len(g6)}')
print(f'Hardware 2g: {len(g2)} | Hardware 8g: {len(g8)}')
print(f'GPS validos (outliers excluidos): {len(gps_clean)}')

print('\n=== ESTADISTICAS ===')
for label, sub, sub_gps in [('2g', g2, gps2), ('8g', g8, gps8)]:
    stance_pct = (sub['stance_time_mean_s'] / sub['stride_time_mean_s'] * 100)
    if len(sub_gps) > 0:
        err = ((sub_gps['gyro_norm_walking_speed_m_s'] - sub_gps['walking_speed_mean_m_s']).abs() /
               sub_gps['walking_speed_mean_m_s'] * 100).mean()
    else:
        err = float('nan')
    print(f'Hardware {label} (n={len(sub)}): cad={sub["stride_cadence_spm"].mean():.1f}spm '
          f'stance={stance_pct.mean():.1f}% slope={sub["stride_time_minute_slope"].mean()*1000:.1f}ms/bl '
          f'v_GPS={sub_gps["walking_speed_mean_m_s"].mean():.2f}m/s (n={len(sub_gps)}) error_gyro={err:.0f}%')

print('\n=== PACIENTES 8g ===')
for p in sorted(g8['patient_id'].unique()):
    sub = g8[g8['patient_id'] == p]
    sub_gps = gps8[gps8['patient_id'] == p]
    v = sub_gps['walking_speed_mean_m_s'].mean() if len(sub_gps) > 0 else float('nan')
    print(f'  {p}: n={len(sub)}, cad={sub["stride_cadence_spm"].mean():.0f}spm, '
          f'slope={sub["stride_time_minute_slope"].mean()*1000:.1f}ms/bl, GPS={v:.2f}m/s')

# =====================================================================
# FIGURA comparativa hardware
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Comparativa hardware 2g vs 8g - 6MWT (dataset completo)',
             fontsize=12, fontweight='bold')

# Panel 1 - Cadencia
bp1 = axes[0].boxplot(
    [g2['stride_cadence_spm'].dropna(), g8['stride_cadence_spm'].dropna()],
    labels=['2g (n=203)', '8g (n=13)'], patch_artist=True,
    medianprops=dict(color='white', lw=2))
for patch, color in zip(bp1['boxes'], [C_BLUE, C_GREEN]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[0].axhline(60, color=C_GRAY, ls='--', lw=1.2, label='Ref. sanos ~60 spm')
axes[0].set_title('Cadencia (spm)', fontweight='bold')
axes[0].set_ylabel('spm')
axes[0].legend(fontsize=8)
axes[0].set_facecolor(C_LGRAY)
axes[0].grid(axis='y', alpha=0.3)

# Panel 2 - Pendiente fatiga
bp2 = axes[1].boxplot(
    [g2['stride_time_minute_slope'].dropna() * 1000,
     g8['stride_time_minute_slope'].dropna() * 1000],
    labels=['2g (n=203)', '8g (n=13)'], patch_artist=True,
    medianprops=dict(color='white', lw=2))
for patch, color in zip(bp2['boxes'], [C_BLUE, C_GREEN]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1].axhline(4.2, color=C_GRAY, ls='--', lw=1.2, label='Ref. Muller +4.2ms/bl')
axes[1].axhline(0, color='#333', lw=1)
axes[1].set_title('Pendiente fatiga (ms/bloque)', fontweight='bold')
axes[1].set_ylabel('ms / bloque 60s')
axes[1].set_ylim(-200, 300)
axes[1].legend(fontsize=8)
axes[1].set_facecolor(C_LGRAY)
axes[1].grid(axis='y', alpha=0.3)

# Panel 3 - Velocidad GPS vs gyro-norm
x = np.array([0.8, 1.2, 1.8, 2.2])
means = [gps2['walking_speed_mean_m_s'].mean(),
         gps2['gyro_norm_walking_speed_m_s'].mean(),
         gps8['walking_speed_mean_m_s'].mean(),
         gps8['gyro_norm_walking_speed_m_s'].mean()]
stds = [gps2['walking_speed_mean_m_s'].std(),
        gps2['gyro_norm_walking_speed_m_s'].std(),
        gps8['walking_speed_mean_m_s'].std(),
        gps8['gyro_norm_walking_speed_m_s'].std()]
cols = [C_BLUE, C_BLUE, C_GREEN, C_GREEN]
alps = [0.9, 0.5, 0.9, 0.5]

for xi, mv, sv, cv, av in zip(x, means, stds, cols, alps):
    if not np.isnan(mv):
        axes[2].bar(xi, mv, 0.3, color=cv, alpha=av, edgecolor='white')
        axes[2].errorbar(xi, mv, yerr=sv, fmt='none', color='#333', capsize=4, lw=1.5)
        axes[2].text(xi, mv + 0.04, f'{mv:.2f}', ha='center', fontsize=8, fontweight='bold')

axes[2].set_xticks([1.0, 2.0])
axes[2].set_xticklabels(['2g', '8g'])
axes[2].set_title('Velocidad GPS vs gyro-norm', fontweight='bold')
axes[2].set_ylabel('m/s')
axes[2].set_facecolor(C_LGRAY)
axes[2].grid(axis='y', alpha=0.3)
legend_v = [
    mpatches.Patch(color=C_BLUE,  alpha=0.9, label='GPS 2g'),
    mpatches.Patch(color=C_BLUE,  alpha=0.5, label='Gyro-norm 2g'),
    mpatches.Patch(color=C_GREEN, alpha=0.9, label='GPS 8g'),
    mpatches.Patch(color=C_GREEN, alpha=0.5, label='Gyro-norm 8g'),
]
axes[2].legend(handles=legend_v, fontsize=7.5)

fig.patch.set_facecolor('white')
plt.tight_layout()
out_fig = OUT_DIR / 'fig_hardware_comparison.png'
plt.savefig(out_fig, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\nFigura guardada: {out_fig}')

# =====================================================================
# MAPA GPS con folium (basado en gps_map_generator.py del proyecto)
# =====================================================================
try:
    import folium
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'folium', '-q'])
    import folium

HDF5_PATH = PROJECT_ROOT / 'data' / 'raw' / 'gait_study_data.h5'

# Colores por paciente
colors_map = {
    'p_RSLHUG073-4':   '#E8A838',
    'p_EMGHUG072-7':   '#1D9E75',
    'p_RHRHUG004-1':   '#185FA5',
    'p_EPGHUG006-25':  '#D85A30',
    'p_AMIR2026-54':   '#8B5CF6',
    'p_AEMDHUG060-70': '#EC4899',
}
default_color = '#64748B'

m = folium.Map(location=[40.33, -3.77], zoom_start=15, tiles='OpenStreetMap')

n_traj = 0
lats_all, lngs_all = [], []

for _, row in gps_clean.iterrows():
    pid = row['patient_id']
    key = row['analysis_h5_key']
    hw_tag = '8g' if row['hardware'] == 'pm8g' else '2g'
    v_gps = row['walking_speed_mean_m_s']
    color = colors_map.get(pid, default_color)
    pid_clean = pid.replace('p_', '')

    for foot in ['Left', 'Right']:
        # analysis_h5_key ya incluye /Left o /Right al final
        # Construir clave base sin foot y añadir foot
        key_base = key.rsplit('/', 1)[0]  # quita /Left o /Right del final
        hdf_key = f"/{key_base}/{foot}"
        try:
            df_t = pd.read_hdf(str(HDF5_PATH), key=hdf_key)
            if 'lat' not in df_t.columns:
                continue

            # Aplicar misma logica que gps_map_generator: ordenar y eliminar duplicados
            df_t = df_t.sort_values('_time') if '_time' in df_t.columns else df_t
            lats = df_t['lat'].astype(float)
            lngs = df_t['lng'].astype(float)

            # Filtrar coordenadas validas cerca del hospital
            mask = (lats > 40.0) & (lats < 40.6) & (lngs > -4.1) & (lngs < -3.5)
            lats = lats[mask].dropna()
            lngs = lngs[mask].dropna()

            # Submuestreo cada 50 muestras para reducir densidad
            # (el GPS actualiza lento ~1Hz pero la señal es a 100Hz)
            step = max(1, len(lats) // 300)
            lats_s = lats.values[::step]
            lngs_s = lngs.values[::step]

            # Eliminar duplicados consecutivos tras submuestreo
            coords_df = pd.DataFrame({'lat': lats_s, 'lng': lngs_s})
            coords_df = coords_df.loc[
                (coords_df['lat'].shift() != coords_df['lat']) |
                (coords_df['lng'].shift() != coords_df['lng'])
            ].reset_index(drop=True)

            if len(coords_df) < 3:
                continue

            coords = coords_df[['lat', 'lng']].values.tolist()

            # Polyline de la trayectoria
            folium.PolyLine(
                coords,
                color=color,
                weight=4,
                opacity=0.85,
                tooltip=f"{pid_clean} | {foot} | {hw_tag} | {v_gps:.2f} m/s"
            ).add_to(m)

            # Marcador inicio (verde) y fin (rojo) como en gps_map_generator
            folium.Marker(
                location=coords[0],
                popup=f"Inicio: {pid_clean}",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
            folium.Marker(
                location=coords[-1],
                popup=f"Fin: {pid_clean} | {v_gps:.2f} m/s",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)

            lats_all.extend([c[0] for c in coords])
            lngs_all.extend([c[1] for c in coords])
            n_traj += 1
            break  # Un pie por trial

        except Exception:
            continue

# Ajustar zoom automaticamente
if lats_all:
    m.fit_bounds([
        [min(lats_all), min(lngs_all)],
        [max(lats_all), max(lngs_all)]
    ])

# Leyenda
legend_html = '''
<div style="position:fixed;bottom:30px;left:30px;z-index:1000;
     background:white;padding:14px;border-radius:10px;
     border:2px solid #ccc;font-size:13px;font-family:Arial;
     box-shadow:2px 2px 6px rgba(0,0,0,0.2);">
<b>MS-Feat &mdash; Trayectorias GPS 6MWT</b><br><br>
<span style="color:#E8A838;font-size:16px">&#9644;</span> RSLHUG073-4 (8g)<br>
<span style="color:#1D9E75;font-size:16px">&#9644;</span> EMGHUG072-7 (8g)<br>
<span style="color:#8B5CF6;font-size:16px">&#9644;</span> AMIR2026-54 (8g)<br>
<span style="color:#EC4899;font-size:16px">&#9644;</span> AEMDHUG060-70<br>
<span style="color:#185FA5;font-size:16px">&#9644;</span> RHRHUG004-1 (2g)<br>
<span style="color:#D85A30;font-size:16px">&#9644;</span> EPGHUG006-25 (2g)<br>
<span style="color:#64748B;font-size:16px">&#9644;</span> Otros pacientes<br><br>
<img src="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png" 
     style="height:16px"> Inicio &nbsp;
<img src="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png" 
     style="height:16px"> Fin
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

out_map = OUT_DIR / 'mapa_gps_trayectorias.html'
m.save(str(out_map))
print(f'Mapa GPS guardado: {out_map} ({n_traj} trayectorias)')
print(f'\nTodo guardado en: {OUT_DIR}')