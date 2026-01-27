import pandas as pd
import numpy as np
import matplotlib
#Configuración para evitar errores en servidores sin entorno gráfico 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from scipy.signal import butter, filtfilt, find_peaks
from pydantic import BaseModel, Field
from typing import Dict, Any, Tuple, Optional 
from pathlib import Path    

# Configuración del sistema de registros (Logging)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ProcessConfig(BaseModel):
   """Esquema de configuración para el procesamiento de señales de marcha.

    Utiliza Pydantic v2 para la validación estricta de tipos y gestión de valores por defecto.
    
    Attributes:
        fs (float): Frecuencia de muestreo de los sensores en Hz.
        cutoff_pressure (float): Frecuencia de corte para el filtro de los sensores de presión.
        cutoff_gyro (float): Frecuencia de corte para las señales del giroscopio.
        gyro_threshold (float): Umbral de velocidad angular (°/s) para detectar giros.
        save_plots (bool): Indica si se deben exportar gráficos a disco.
    """
   fs: float = Field(default=100.0, ge=0.1)
   cutoff_pressure: float = Field(default=5.0, ge=0.1)
   cutoff_gyro: float = Field(default=2.0, ge=0.1)
   gyro_threshold: float = Field(default=50.0)
   save_plots: bool = Field(default=True)
   min_peak_distance: int = Field(default=50)
   min_peak_height: float = Field(default=200.0)

class GaitDataProcessor:
   """Lógica avanzada para el análisis y procesamiento digital de señales de marcha.

    Esta clase encapsula un pipeline de procesamiento digital de señales (DSP) diseñado
    para ser hardware-agnóstico. Implementa algoritmos de autocalibración para deducir
    la orientación del sensor mediante física, eliminando la necesidad de configurar 
    manualmente los ejes espaciales.

    El pipeline incluye:
        1. Autodetección de la vertical mediante análisis de componentes gravitatorias.
        2. Filtrado digital zero-phase para la eliminación de artefactos.
        3. Segmentación espacial para identificar y excluir fases de giro.
        4. Detección heurística de eventos de impacto (Heel Strikes).

    Arguments:
        config (Optional[ProcessConfig]): Objeto de configuración que define los 
            umbrales de filtrado, detección y parámetros de salida. Si es None, 
            se instancian los valores por defecto validados por Pydantic.

    Attributes:
        config (ProcessConfig): Configuración activa del procesador.
        logger (logging.Logger): Instancia para el registro profesional de eventos 
            y advertencias biomecánicas.
    """
   def __init__(self, config: Optional[ProcessConfig] = None):
      """Inicializa el procesador con una configuración específica."""
      self.config = config or ProcessConfig()
      self.logger = logging.getLogger(__name__)

   def _autodetect_vertical_axis(self, df: pd.DataFrame) -> str:
      """Deduce automáticamente el eje vertical analizando la gravedad.

        Busca el componente de aceleración con mayor magnitud constante para 
        establecer el eje de referencia de impactos, garantizando la robustez 
        del análisis ante cambios en la colocación del sensor.

        Arguments:
            df (pd.DataFrame): Datos crudos con ejes de aceleración (Ax, Ay, Az).

        Returns:
            str: Identificador de la columna detectada como vertical.
        """
      axes = ['Ax', 'Ay', 'Az']
      available_axes = [ax for ax in axes if ax in df.columns]
      if not available_axes:
        logger.warning("No se hallaron ejes Ax, Ay, Az. Se asume 'Az' por defecto.")
        return 'Az'
      # El eje con mayor aceleración media constante es la vertical (Gravedad)
      means = df[available_axes].abs().mean()
      detected_axis = means.idxmax()
      
      logger.info(f" Autocalibración exitosa. Eje vertical detectado: {detected_axis}")
      return detected_axis
    
   def _butter_lowpass_filter(self, data: np.ndarray, cutoff: float) -> np.ndarray:
    """Aplica un filtro Butterworth paso bajo de cuarto orden y fase cero.

        Arguments:
            data (np.ndarray): Array 1D que contiene la señal cruda del sensor.
            cutoff (float): Frecuencia de corte deseada en Hz.

        Returns:
            np.ndarray: Señal suavizada con el ruido de alta frecuencia eliminado.
    """
    nyq = 0.5 * self.config.fs
    normal_cutoff = cutoff / nyq
    b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
   
   def process_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
    """Ejecuta el pipeline completo de análisis sobre un ensayo de marcha.

        Realiza la limpieza, autocalibración, filtrado y detección de eventos, 
        integrando metadatos de contexto como coordenadas GPS si están disponibles.

        Arguments:
            df (pd.DataFrame): DataFrame de entrada con señales de presión (S0), 
                acelerometría (Ax, Ay, Az), giroscopio (Gz) y marcas de tiempo.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
                - df_proc: DataFrame enriquecido con señales filtradas y máscaras.
                - metrics: Diccionario con resultados (conteo de pasos, stride, GPS).
                - peaks: Array de índices correspondientes a los Heel Strikes detectados.

        Raises:
            ValueError: Si el DataFrame no contiene las columnas mínimas requeridas.
        """
    if 'S0' not in df.columns or '_time' not in df.columns:
            raise ValueError("El DataFrame debe contener al menos las columnas 'S0' y '_time'.")
    
    # 1. Preparación y ordenación cronológica
    df = df.sort_values('_time').reset_index(drop=True)
    df['_time'] = pd.to_datetime(df['_time'])

    # 2. Autodetección de orientación
    v_axis = self._autodetect_vertical_axis(df)

    # 3. Filtrado de señales
    df['S0_filt'] = self._butter_lowpass_filter(df['S0'].values, self.config.cutoff_pressure)
    df['Gz_filt'] = self._butter_lowpass_filter(df['Gz'].values, self.config.cutoff_gyro)

    # 4. Segmentación de giros (Giroscopio > Umbral)
    df['is_turning'] = df['Gz_filt'].abs() > self.config.gyro_threshold

    # 5. Detección de pasos (Heel Strikes) en zonas de marcha estable
    s0_clean = df['S0_filt'].copy()
    s0_clean[df['is_turning']] = 0 # Ignorar picos durante rotaciones
    peaks, _ = find_peaks(
            s0_clean, 
            distance=self.config.min_peak_distance, 
            height=self.config.min_peak_height
    )
    
    # 6. Extracción de métricas y contexto geográfico
    metrics = {
        'pasos_detectados': len(peaks),
        'stride_medio_s': 0.0,
        'eje_vertical_utilizado': v_axis,            
        'posicion_gps': "No disponible" # Valor por defecto
    }
    if 'lat' in df.columns and 'lng' in df.columns:
        lat = df['lat'].iloc[0]
        lng = df['lng'].iloc[0]
        metrics['posicion_gps'] = f"{lat}, {lng}"


    if len(peaks) > 1:
            tiempos_pasos = df['_time'].iloc[peaks].values
            metrics['stride_medio_s'] = np.mean(np.diff(tiempos_pasos) / np.timedelta64(1, 's'))

    return df, metrics, peaks