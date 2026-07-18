# Informe pseudo-clínico actualizado con cruce EDSS

## Premisas de análisis

- Se ha agregado cada ensayo por ambos pies para evitar doble conteo del mismo test.
- Se ha usado como corte de fiabilidad espacial el `2026-06-03` para evaluar si la nueva cobertura confirma o no el problema de saturación de IMU.
- El cruce con EDSS se ha hecho mediante el código base del paciente: en el CSV aparece con sufijo final (`-n`) y en el Excel `Bs_Gait_Sen_Algorit.xlsx` sin ese sufijo.
- El índice de severidad de marcha es exploratorio: media de z-scores robustos por tipo de prueba, orientados para que valores mayores indiquen peor desempeño.

## Resumen del dataset actualizado

- Filas del CSV: 839
- Pacientes únicos del CSV: 92
- Ensayos únicos agregados: 440
- Ensayos bilaterales completos: 399
- Rango temporal del CSV: 2025-11-17 a 2026-06-18
- Filas posteriores al 2026-06-03: 50
- Pacientes con referencia EDSS emparejable: 18

## ¿Se constata el problema espacial de la IMU?

- Sí. Hay 112 ensayos `imu_zupt`, de los cuales 25 son posteriores al 2026-06-03.
- Antes del corte, la velocidad IMU/ZUPT tiene mediana 0.290 [0.072-0.718] m/s y la razón IMU/gyro es 0.122 [0.055-0.264].
- Después del corte, la velocidad IMU/ZUPT tiene mediana 0.046 [0.021-0.169] m/s y la razón IMU/gyro es 0.033 [0.013-0.150].
- La nueva cobertura de junio permite constatar el problema: incluso tras el 3 de junio la salida espacial `imu_zupt` sigue siendo mayoritariamente implausible frente a las estimaciones cinemáticas alternativas y conserva outliers extremos. Por tanto, el componente espacial IMU no debe usarse todavía como referencia clínica principal.

## Lectura pseudo-clínica de la marcha

### 6MWT
- Ensayos: 228; pacientes: 90; bilaterales completos: 221.
- Cadencia mediana: 52.5 spm; tiempo de zancada mediano: 1.142 s; CV de zancada mediano: 0.321.
- Asimetría bilateral mediana: 4.0%; doble apoyo mediano: 24.9%.
- Pendiente temporal mediana: 12.0 ms/min, compatible a nivel grupal con fatigabilidad leve-moderada.

### T25FW
- Ensayos: 127; pacientes: 60; bilaterales completos: 105.
- Cadencia mediana: 53.7 spm; tiempo de zancada mediano: 0.865 s; CV de zancada mediano: 0.056.
- Asimetría bilateral mediana: 2.5%; doble apoyo mediano: 26.3%.

### TUG
- Ensayos: 82; pacientes: 60; bilaterales completos: 70.
- Cadencia mediana: 44.0 spm; tiempo de zancada mediano: 1.096 s; CV de zancada mediano: 0.225.
- Asimetría bilateral mediana: 13.1%; doble apoyo mediano: 25.6%.

### 2MWT
- Ensayos: 3; pacientes: 2; bilaterales completos: 3.
- Cadencia mediana: 32.0 spm; tiempo de zancada mediano: 1.809 s; CV de zancada mediano: 0.385.
- Asimetría bilateral mediana: 23.1%; doble apoyo mediano: 16.1%.

## Alineamiento con EDSS

- Solape final: 18 pacientes con EDSS y al menos una familia de prueba usable.
- Correlación global entre EDSS y severidad de marcha normalizada: rho de Spearman = 0.749, p = 0.0003, n = 18.
- Asociaciones más claras con EDSS en el subconjunto disponible:
  - T25FW / bio_speed: rho=-0.889, p=0.0000, n=17.
  - T25FW / cadence: rho=-0.829, p=0.0000, n=17.
  - T25FW / gait_severity_index: rho=0.817, p=0.0001, n=17.
  - T25FW / stride_time: rho=0.788, p=0.0002, n=17.
  - 6MWT / stride_time: rho=0.640, p=0.0043, n=18.
  - TUG / stride_time: rho=0.670, p=0.0045, n=16.
  - 6MWT / gyro_speed: rho=-0.572, p=0.0131, n=18.
  - 6MWT / bio_speed: rho=-0.560, p=0.0156, n=18.

## Casos con desalineamiento relevante

- Marcha peor de lo esperable para su EDSS:
  - JASAHUG010: EDSS 1.5, severidad 0.30, pruebas 6MWT,T25FW,TUG, gap 0.33.
  - EMMHUG008: EDSS 3.0, severidad 0.64, pruebas 6MWT,T25FW,TUG, gap 0.25.
  - MJCRHUG013: EDSS 4.0, severidad 2.48, pruebas 6MWT,T25FW,TUG, gap 0.25.
  - RSBHUG005: EDSS 1.0, severidad -0.01, pruebas 6MWT,T25FW,TUG, gap 0.19.
  - RHRHUG004: EDSS 0.0, severidad -0.22, pruebas 6MWT,T25FW,TUG, gap 0.17.
  - CDOHUG015: EDSS 2.0, severidad 0.09, pruebas 6MWT, gap 0.14.
- EDSS mayor con marcha relativamente mejor de lo esperable:
  - JMAHUG009: EDSS 5.5, severidad -0.15, pruebas 6MWT,T25FW,TUG, gap -0.44.
  - JMGHUG016: EDSS 3.0, severidad -0.33, pruebas 6MWT,T25FW,TUG, gap -0.42.
  - RBGHUG018: EDSS 6.0, severidad 0.46, pruebas 6MWT,T25FW, gap -0.14.
  - AJGFHUG011: EDSS 6.5, severidad 0.75, pruebas 6MWT,T25FW,TUG, gap -0.11.
  - FMGHUG012: EDSS 2.0, severidad -0.22, pruebas 6MWT,T25FW,TUG, gap -0.08.
  - JCGMHUG007: EDSS 3.0, severidad 0.03, pruebas 6MWT,T25FW,TUG, gap -0.08.

## Conclusión breve

- Los parámetros temporales y bilaterales de marcha sí muestran un alineamiento clínico moderado con EDSS en este subconjunto de referencias.
- Las señales que más se alinean con EDSS aquí son, sobre todo, el enlentecimiento temporal: `stride_time` en T25FW y 6MWT, más que la asimetría o la fatigabilidad.
- El bloque espacial `imu_zupt` sigue siendo no confiable incluso tras junio de 2026, así que la comparación con EDSS debe descansar principalmente en atributos temporales, de variabilidad y de bilateralidad.

## Ficheros generados

- Informe: `/home/jordieres/soft/sclerosis/2026_msFeat_TEA/reports/pseudo_clinical_report_GUILLE_v2.md`
- Resumen por ensayo: `/home/jordieres/soft/sclerosis/2026_msFeat_TEA/reports/pseudo_clinical_trial_summary_GUILLE_v2.csv`
- Tabla de alineamiento con EDSS: `/home/jordieres/soft/sclerosis/2026_msFeat_TEA/reports/edss_alignment_GUILLE.csv`