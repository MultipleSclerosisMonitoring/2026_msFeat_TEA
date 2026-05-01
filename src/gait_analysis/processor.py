"""
Digital Signal Processing (DSP) Module for Human Gait Analysis.

This module implements a deterministic pipeline for extracting
spatiotemporal gait features from wearable sensor data.

The design follows a research-oriented approach:
- Zero-phase filtering (Butterworth)
- Automatic spatial calibration (gravity-based)
- Event detection (Heel Strikes)
- Temporal and variability feature extraction
- Temporal trend estimation for fatigue-sensitive metrics

The output aligns with clinically relevant gait metrics and
provides a foundation for fatigue analysis.

"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from scipy.signal import butter, filtfilt, find_peaks
from pydantic import BaseModel, ConfigDict,Field

logger = logging.getLogger(__name__)


class ProcessConfig(BaseModel):
    """
    Configuration schema for gait signal processing parameters.

    Attributes
    ----------
    fs : float
        Sampling frequency in Hz.
    cutoff_pressure : float
        Low-pass cutoff frequency for plantar pressure signal.
    cutoff_gyro : float
        Low-pass cutoff frequency for gyroscope signal.
    gyro_threshold : float
        Threshold to detect turning phases.
    min_peak_distance_s : float
        Minimum distance between detected peaks (seconds).
    min_peak_height : float
        Minimum normalized amplitude for peak detection.
    """

    model_config = ConfigDict(extra="forbid")

    fs: float = Field(default=100.0, ge=0.1)
    cutoff_pressure: float = Field(default=5.0, ge=0.1)
    cutoff_gyro: float = Field(default=2.0, ge=0.1)
    gyro_threshold: float = Field(default=50.0)
    min_peak_distance_s: float = Field(default=0.5, ge=0.05)
    min_peak_height: float = Field(default=0.2)


class GaitDataProcessor:
    """
    Core processing engine for gait signal analysis.

    This class transforms raw sensor data into clinically relevant
    spatiotemporal features and variability metrics.

    Pipeline stages
    ---------------
    1. Temporal alignment
    2. Signal filtering
    3. Turn segmentation
    4. Event detection (Heel Strikes)
    5. Temporal, variability, and trend feature extraction

    Parameters
    ----------
    config : ProcessConfig, optional
        Processing configuration. If omitted, default values are used.
    """

    def __init__(self, config: Optional[ProcessConfig] = None):
        self.config = config or ProcessConfig()
        self.logger = logging.getLogger(__name__)

    def _autodetect_vertical_axis(self, df: pd.DataFrame) -> str:
        """
        Detect the vertical inertial axis using gravity magnitude.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing inertial acceleration axes.

        Returns
        -------
        str
            Detected vertical axis label ("Ax", "Ay", or "Az").
        """
        axes = ["Ax", "Ay", "Az"]
        available_axes = [ax for ax in axes if ax in df.columns]

        if not available_axes:
            self.logger.warning("Inertial axes missing. Defaulting to 'Az'.")
            return "Az"

        means = df[available_axes].abs().mean()
        axis = means.idxmax()

        self.logger.debug(f"Gravity mean magnitudes: {means.to_dict()}")
        self.logger.info(f"Spatial autocalibration successful. Vertical axis: {axis}")
        return axis

    def _butter_lowpass_filter(self, data: np.ndarray, cutoff: float) -> np.ndarray:
        """
        Apply zero-phase Butterworth low-pass filtering.

        Parameters
        ----------
        data : np.ndarray
            Raw 1D signal.
        cutoff : float
            Cutoff frequency in Hz.

        Returns
        -------
        np.ndarray
            Filtered signal.
        """
        nyq = 0.5 * self.config.fs
        normal_cutoff = cutoff / nyq
        b, a = butter(4, normal_cutoff, btype="low")
        return filtfilt(b, a, data)
    
    def _estimate_sampling_frequency(self, df: pd.DataFrame) -> float:
        """
        Estimate the effective sampling frequency from timestamps.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing the '_time' column.

        Returns
        -------
        float
            Effective sampling frequency in Hz, computed as the total number
            of samples divided by the total duration of the recording. This
            estimate is robust against irregular timestamps (bursts and gaps)
            common with BLE-streamed sensors, which bias both mean and median
            of inter-sample intervals.

        Raises
        ------
        ValueError
            If the dataframe does not contain enough valid timestamps.
        """
        if "_time" not in df.columns or len(df) < 2:
            raise ValueError("Cannot estimate sampling frequency without at least two timestamps.")
        time_seconds = (
            (pd.to_datetime(df["_time"]) - pd.to_datetime(df["_time"]).iloc[0])
            / np.timedelta64(1, "s")
        ).astype(float).to_numpy()

        # Effective sampling frequency = total samples / total duration.
        # This is robust against irregular timestamps (BLE-streamed sensors
        # often emit data in bursts of dt < 1 ms followed by gaps), which
        # bias both mean and median of inter-sample intervals.
        duration = time_seconds[-1] - time_seconds[0]
        if duration <= 0:
            raise ValueError("Invalid timestamps: non-positive total duration.")
        fs_est = (len(time_seconds) - 1) / duration
        return float(fs_est)

    def _resample_to_uniform_timebase(self, df: pd.DataFrame, target_fs: float) -> pd.DataFrame:
        """
        Resample the dataframe to a uniform time base using linear interpolation.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with irregular timestamps.
        target_fs : float
            Target resampling frequency in Hz.

        Returns
        -------
        pd.DataFrame
            Resampled dataframe with uniform timestamps.
        """
        t0 = pd.to_datetime(df["_time"]).iloc[0]
        time_original = (
            (pd.to_datetime(df["_time"]) - t0) / np.timedelta64(1, "s")
        ).astype(float).to_numpy()

        duration = time_original[-1]
        dt_uniform = 1.0 / target_fs
        time_uniform = np.arange(0.0, duration, dt_uniform)

        df_resampled = pd.DataFrame({
            "_time": t0 + pd.to_timedelta(time_uniform, unit="s")
        })

        candidate_columns = [col for col in df.columns if col != "_time"]

        for col in candidate_columns:
            series = pd.to_numeric(df[col], errors="coerce")
            valid = series.notna().to_numpy()

            if np.sum(valid) < 2:
                continue

            df_resampled[col] = np.interp(
                time_uniform,
                time_original[valid],
                series.to_numpy()[valid]
            )

        return df_resampled

    def _safe_linear_slope(self, values: np.ndarray) -> float:
        """
        Compute the linear slope of a sequence using first-order polynomial fit.

        Parameters
        ----------
        values : np.ndarray
            One-dimensional numeric array.

        Returns
        -------
        float
            Estimated slope. Returns 0.0 if fewer than 2 samples are available.
        """
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values), dtype=float)
        slope = np.polyfit(x, values.astype(float), 1)[0]
        return float(slope)

    def _safe_cv(self, values: np.ndarray) -> float:
        """
        Compute coefficient of variation safely.

        Parameters
        ----------
        values : np.ndarray
            Numeric values.

        Returns
        -------
        float
            Standard deviation divided by mean, or 0.0 if mean is zero
            or not enough values are available.
        """
        if len(values) < 2:
            return 0.0

        mean_val = float(np.mean(values))
        if np.isclose(mean_val, 0.0):
            return 0.0

        return float(np.std(values) / mean_val)

    def process_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray]:
        """
        Execute the full gait processing pipeline.

        The method performs:
        - chronological sorting
        - zero-phase filtering
        - turn masking
        - heel strike detection
        - extraction of temporal, variability, and trend metrics

        Parameters
        ----------
        df : pd.DataFrame
            Raw gait dataframe. Must contain at least '_time' and 'S0'.

        Returns
        -------
        df_proc : pd.DataFrame
            Processed dataframe with filtered signals and turn mask.
        metrics : Dict[str, Any]
            Extracted gait metrics and metadata.
        peaks : np.ndarray
            Detected heel strike indices.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        self.logger.debug("Starting gait signal processing pipeline.")

        if "S2" not in df.columns or "_time" not in df.columns:
            raise ValueError("Missing required columns: 'S2' (heel pressure) and '_time'")

        # 1. Chronological ordering
        df = df.sort_values("_time").reset_index(drop=True)
        df["_time"] = pd.to_datetime(df["_time"])

        # Estimate real sampling frequency from timestamps
        fs_est = self._estimate_sampling_frequency(df)
        self.logger.info(f"Estimated raw sampling frequency: {fs_est:.2f} Hz")

        # Resample to a uniform time base using the configured target frequency
        target_fs = self.config.fs
        self.logger.info(f"Resampling to uniform time base at {target_fs:.2f} Hz")

        df = self._resample_to_uniform_timebase(df, target_fs=target_fs)

        self.logger.debug(f"Resampled signal length: {len(df)} samples")
        self.logger.debug(
            f"Resampled duration: "
            f"{(df['_time'].iloc[-1] - df['_time'].iloc[0]).total_seconds():.3f} s"
        )

        # 2. Filtering
        v_axis = self._autodetect_vertical_axis(df)
        df["S2_filt"] = self._butter_lowpass_filter(
            df["S2"].values,
            self.config.cutoff_pressure,
)

        # 3. Turn detection
        if "Gz" in df.columns:
            df["Gz_filt"] = self._butter_lowpass_filter(
                df["Gz"].values,
                self.config.cutoff_gyro,
            )
            df["is_turning"] = df["Gz_filt"].abs() > self.config.gyro_threshold
        else:
            df["is_turning"] = False
            self.logger.warning("Gz not found. Turn segmentation disabled.")

        self.logger.debug(f"Filtered S2 preview: {df['S2_filt'].head().to_list()}")

        # 4. Heel strike detection outside turning phases
        # The plantar pressure sensor produces HIGH values when the foot is in the air
        # and LOW values during contact. Heel strikes are therefore minima (valleys)
        # of S2_filt. We invert and normalize the signal to [0, 1] so that find_peaks()
        # can use a meaningful, scale-independent height threshold.
        s2_inv_raw = -df["S2_filt"].to_numpy(dtype=float)

        # Robust normalization to [0, 1] using min/max of the entire signal.
        s2_min = float(np.min(s2_inv_raw))
        s2_max = float(np.max(s2_inv_raw))
        if s2_max - s2_min > 1e-9:
            s2_norm = (s2_inv_raw - s2_min) / (s2_max - s2_min)
        else:
            s2_norm = np.zeros_like(s2_inv_raw)

        # Suppress detection during turning phases by forcing the normalized
        # inverted signal to 0 there (so it never reaches min_peak_height).
        s2_norm[df["is_turning"].to_numpy()] = 0.0

        min_peak_distance_samples = max(1, int(self.config.min_peak_distance_s * self.config.fs))
        self.logger.debug(
            f"Peak detection minimum distance: {self.config.min_peak_distance_s:.3f} s "
            f"({min_peak_distance_samples} samples at {self.config.fs:.2f} Hz)"
        )
        self.logger.debug(
            f"Normalized inverted signal range used for peak detection: "
            f"min=0.000, max=1.000 (original min={s2_min:.1f}, max={s2_max:.1f})"
        )

        peaks, properties = find_peaks(
            s2_norm,
            distance=min_peak_distance_samples,
            height=self.config.min_peak_height,
        )

        self.logger.debug(f"Signal length: {len(df)} samples")
        self.logger.debug(f"Turning samples: {df['is_turning'].sum()}")
        
        self.logger.info(f"Detected {len(peaks)} steps")
        self.logger.debug(f"Peak heights preview: {properties.get('peak_heights', [])[:5]}")
        self.logger.debug(f"Turning ratio: {df['is_turning'].sum() / len(df):.3f}")

        # 5. Feature extraction
        metrics: Dict[str, Any] = {
            "pasos_detectados": int(len(peaks)),
            "eje_vertical_utilizado": v_axis,
            "posicion_gps": "N/A",
            "walking_duration_s": 0.0,
            "step_time_mean_s": 0.0,
            "step_time_std_s": 0.0,
            "step_time_cv": 0.0,
            "cadence_spm": 0.0,
            "stride_time_mean_s": 0.0,
            "stride_time_std_s": 0.0,
            "stride_time_cv": 0.0,
            "step_time_slope": 0.0,
            "stride_time_slope": 0.0,
            "cadence_first_half_spm": 0.0,
            "cadence_second_half_spm": 0.0,
            "cadence_change_spm": 0.0,
        }

        # GPS metadata
        if {"lat", "lng"}.issubset(df.columns):
            gps_valid = df[["lat", "lng"]].dropna()
            if not gps_valid.empty:
                first = gps_valid.iloc[0]
                metrics["posicion_gps"] = {
                    "lat": float(first["lat"]),
                    "lng": float(first["lng"]),
                }
                self.logger.debug(f"GPS fix found: {metrics['posicion_gps']}")
            else:
                self.logger.warning("GPS columns found, but no valid lat/lng values are available.")
        else:
            self.logger.warning("GPS columns are missing from the extracted dataset.")

        # Temporal metrics
        if len(peaks) > 1:
            step_times = df["_time"].iloc[peaks].values
            step_intervals = np.diff(step_times) / np.timedelta64(1, "s")
            step_intervals = step_intervals.astype(float)

            duration = float((df["_time"].iloc[-1] - df["_time"].iloc[0]).total_seconds())
            metrics["walking_duration_s"] = duration
            metrics["step_time_mean_s"] = float(np.mean(step_intervals))
            metrics["step_time_std_s"] = float(np.std(step_intervals))
            metrics["step_time_cv"] = self._safe_cv(step_intervals)

            if duration > 0:
                metrics["cadence_spm"] = float(len(peaks) / duration * 60.0)

            # Fatigue-sensitive temporal trend
            metrics["step_time_slope"] = self._safe_linear_slope(step_intervals)

            # Stride intervals approximated as every second step interval
            if len(step_intervals) >= 3:
                stride_intervals = step_intervals[::2]
                metrics["stride_time_mean_s"] = float(np.mean(stride_intervals))
                metrics["stride_time_std_s"] = float(np.std(stride_intervals))
                metrics["stride_time_cv"] = self._safe_cv(stride_intervals)
                metrics["stride_time_slope"] = self._safe_linear_slope(stride_intervals)

            # First-half vs second-half cadence
            total_duration = metrics["walking_duration_s"]
            if total_duration > 0:
                t0 = df["_time"].iloc[0]
                relative_step_times = (
                    (pd.to_datetime(step_times) - t0) / np.timedelta64(1, "s")
                ).astype(float)

                half_time = total_duration / 2.0
                first_half_steps = int(np.sum(relative_step_times < half_time))
                second_half_steps = int(np.sum(relative_step_times >= half_time))

                first_half_duration = half_time
                second_half_duration = total_duration - half_time

                if first_half_duration > 0:
                    metrics["cadence_first_half_spm"] = float(first_half_steps / first_half_duration * 60.0)

                if second_half_duration > 0:
                    metrics["cadence_second_half_spm"] = float(second_half_steps / second_half_duration * 60.0)

                metrics["cadence_change_spm"] = float(
                    metrics["cadence_second_half_spm"] - metrics["cadence_first_half_spm"]
                )

            self.logger.info(
                "Temporal features extracted successfully: "
                f"step_mean={metrics['step_time_mean_s']:.3f}s, "
                f"cadence={metrics['cadence_spm']:.1f} spm, "
                f"step_slope={metrics['step_time_slope']:.6f}"
            )

        else:
            self.logger.warning("Not enough detected steps to compute temporal gait features.")

        return df, metrics, peaks