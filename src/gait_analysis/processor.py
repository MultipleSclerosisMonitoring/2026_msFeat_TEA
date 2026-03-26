"""
Digital Signal Processing (DSP) Module for Human Gait Analysis.

This module provides high-level abstractions for processing inertial and 
pressure sensor data. It focuses on zero-phase digital filtering, 
automated spatial calibration, and clinical metric extraction such as 
stride time variability.

The architecture ensures hardware-agnostic processing by implementing 
gravity-based orientation detection and pydantic-validated configuration.

"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from scipy.signal import butter, filtfilt, find_peaks
from pydantic import BaseModel, Field

# Local module logger
logger = logging.getLogger(__name__)

class ProcessConfig(BaseModel):
    """
    Configuration schema for gait signal processing parameters.

    Validated via Pydantic v2 to ensure strict typing and range constraints 
    on DSP hyperparameters.

    Attributes
    ----------
    fs : float
        Sampling frequency in Hz. Must be positive.
    cutoff_pressure : float
        Low-pass cutoff frequency for pressure sensors (S0) in Hz.
    cutoff_gyro : float
        Low-pass cutoff frequency for gyroscope signals (Gz) in Hz.
    gyro_threshold : float
        Angular velocity threshold (°/s) to trigger turn detection.
    min_peak_distance : int
        Minimum number of samples between consecutive Heel Strikes.
    min_peak_height : float
        Normalized amplitude threshold for peak detection on S0.
    """
    fs: float = Field(default=100.0, ge=0.1)
    cutoff_pressure: float = Field(default=5.0, ge=0.1)
    cutoff_gyro: float = Field(default=2.0, ge=0.1)
    gyro_threshold: float = Field(default=50.0)
    min_peak_distance: int = Field(default=50)
    min_peak_height: float = Field(default=0.2)


class GaitDataProcessor:
    """
    Main engine for biomechanical signal decomposition and event detection.

    Provides a robust pipeline for transforming raw sensor telemetry into 
    validated gait cycles. Includes automated vertical axis detection and 
    steady-state walking segmentation.

    Parameters
    ----------
    config : Optional[ProcessConfig], default=None
        Injected configuration object. If None, default values are used.

    Attributes
    ----------
    config : ProcessConfig
        The active configuration schema for the processor instance.
    logger : logging.Logger
        Scoped logger for tracking DSP events and warnings.
    """

    def __init__(self, config: Optional[ProcessConfig] = None):
        """Initialize the processor with validated configuration."""
        self.config = config or ProcessConfig()
        self.logger = logging.getLogger(__name__)

    def _autodetect_vertical_axis(self, df: pd.DataFrame) -> str:
        """
        Identify the vertical axis based on gravitational component analysis.

        Assumes that the axis with the highest absolute mean acceleration 
        represents the primary vertical vector (gravity) during steady-state 
        or static periods.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing at least 'Ax', 'Ay', 'Az' columns.

        Returns
        -------
        str
            The label of the detected vertical axis (e.g., 'Az').
        """
        axes = ['Ax', 'Ay', 'Az']
        available_axes = [ax for ax in axes if ax in df.columns]
        
        if not available_axes:
            self.logger.warning("Inertial axes missing. Defaulting to 'Az'.")
            return 'Az'

        # Eje con mayor magnitud constante (Gravedad)
        means = df[available_axes].abs().mean()
        detected_axis = means.idxmax()
        
        self.logger.debug(f"Gravity analysis results: {means.to_dict()}")
        self.logger.info(f"Spatial autocalibration successful. Vertical axis: {detected_axis}")
        
        return detected_axis

    def _butter_lowpass_filter(self, data: np.ndarray, cutoff: float) -> np.ndarray:
        """
        Apply a 4th-order zero-phase Butterworth low-pass filter.

        Uses filtfilt to ensure zero phase distortion, which is critical for 
        maintaining the temporal alignment of biomechanical events.

        Parameters
        ----------
        data : np.ndarray
            1D array of raw sensor telemetry.
        cutoff : float
            Cutoff frequency in Hz.

        Returns
-------
        np.ndarray
            Smoothed signal with high-frequency artifacts removed.
        """
        nyq = 0.5 * self.config.fs
        normal_cutoff = cutoff / nyq
        b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def process_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
        """
        Execute the full analysis pipeline on a gait trial.

        The pipeline follows a deterministic sequence:
        1. Temporal sorting and datetime conversion.
        2. Orientation autodetection via gravity vectors.
        3. Zero-phase digital filtering of pressure and rotational signals.
        4. Spatial segmentation (exclusion of turning phases).
        5. Heuristic event detection (Heel Strikes) and metric calculation.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset containing '_time', 'S0', 'Gz', and IMU axes.

        Returns
        -------
        df_proc : pd.DataFrame
            Enriched dataframe with filtered signals and masks.
        metrics : Dict[str, Any]
            Dictionary containing 'pasos_detectados', 'stride_medio_s', 
            'stride_std_s', and 'posicion_gps'.
        peaks : np.ndarray
            Array of indices identifying the detected Heel Strike events.

        Raises
        ------
        ValueError
            If 'S0' or '_time' columns are missing from the input dataframe.
        """
        self.logger.debug("Starting gait signal processing pipeline.")

        if 'S0' not in df.columns or '_time' not in df.columns:
            msg = "Data Integrity Error: 'S0' and '_time' are mandatory."
            self.logger.error(msg)
            raise ValueError(msg)

        # 1. Chronological sorting
        df = df.sort_values('_time').reset_index(drop=True)
        df['_time'] = pd.to_datetime(df['_time'])

        # 2. Calibration and Filtering
        v_axis = self._autodetect_vertical_axis(df)
        df['S0_filt'] = self._butter_lowpass_filter(df['S0'].values, self.config.cutoff_pressure)
        
        # 3. Turn Segmentation (Gz-based)
        if 'Gz' in df.columns:
            df['Gz_filt'] = self._butter_lowpass_filter(df['Gz'].values, self.config.cutoff_gyro)
            df['is_turning'] = df['Gz_filt'].abs() > self.config.gyro_threshold
        else:
            self.logger.warning("Gz signal missing. Turn segmentation disabled.")
            df['is_turning'] = pd.Series(False, index=df.index)

        # Logging granular for DSP verification
        self.logger.debug(f"Filtered S0 head: {df['S0_filt'].head().values}")

        # 4. Heel Strike Detection
        # Isolate steady-state walking by masking turns
        s0_clean = df['S0_filt'].copy()
        s0_clean[df['is_turning']] = 0 
        
        peaks, _ = find_peaks(
            s0_clean, 
            distance=self.config.min_peak_distance, 
            height=self.config.min_peak_height
        )

        # 5. Scientific Metric Consolidation
        metrics = {
            'pasos_detectados': len(peaks),
            'stride_medio_s': 0.0,
            'stride_std_s': 0.0,
            'eje_vertical_utilizado': v_axis,
            'posicion_gps': "N/A"
        }

        # Contextual GPS data extraction
        if {'lat', 'lng'}.issubset(df.columns):
            metrics['posicion_gps'] = f"{df['lat'].iloc[0]}, {df['lng'].iloc[0]}"

        # Statistical analysis of gait cycles
        if len(peaks) > 1:
            step_times = df['_time'].iloc[peaks].values
            intervals = np.diff(step_times) / np.timedelta64(1, 's')
            metrics['stride_medio_s'] = float(np.mean(intervals))
            metrics['stride_std_s'] = float(np.std(intervals))
            
            self.logger.info(
                f"Processing complete: {len(peaks)} steps. Mean Stride: {metrics['stride_medio_s']:.2f}s"
            )

        return df, metrics, peaks
def main():
    """
    Punto de entrada para el comando CLI 'analyze-gait'.
    
    Inicializa el procesador con la configuración por defecto y 
    sirve como demostración de la integridad del paquete.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Iniciando motor biomecánico gait-analysis-tfg...")
    
    try:
        processor = GaitDataProcessor()
        logger.info("Procesador instanciado correctamente. Listo para análisis masivo.")
        # Aquí podrías añadir una pequeña prueba de carga de un CSV si quisieras
    except Exception as e:
        logger.error(f"Error al iniciar el motor: {e}")

if __name__ == "__main__":
    main()
      