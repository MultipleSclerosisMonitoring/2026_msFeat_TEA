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
    minute_block_duration_s : float
        Duration of each block in the per-minute fatigue analysis (seconds).
        Default is 60 s, matching the standard segmentation used in 6MWT
        fatigue studies. Set higher for shorter trials, lower for finer-grained
        temporal resolution.
    """

    model_config = ConfigDict(extra="forbid")

    fs: float = Field(default=100.0, ge=0.1)
    cutoff_pressure: float = Field(default=5.0, ge=0.1)
    cutoff_gyro: float = Field(default=2.0, ge=0.1)
    gyro_threshold: float = Field(default=50.0)
    min_peak_distance_s: float = Field(default=0.5, ge=0.05)
    min_peak_height: float = Field(default=0.2)
    minute_block_duration_s: float = Field(default=60.0, ge=5.0)


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
    
    def _compute_minute_metrics(
        self,
        df: pd.DataFrame,
        peaks: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute stride metrics within fixed-duration blocks of the trial.

        This is the core of the fatigue analysis: by tracking how stride
        time and stride cadence evolve over consecutive time blocks of
        the trial (typically 60 s each in a 6MWT), we can detect gradual
        deterioration that is not visible in trial-wide averages.

        The trial is divided into blocks of `minute_block_duration_s`.
        Blocks shorter than 50 % of that duration are discarded to avoid
        artefacts at the trial's tail.

        Parameters
        ----------
        df : pd.DataFrame
            Resampled, filtered signal dataframe with a '_time' column.
        peaks : np.ndarray
            Indices of detected heel-strike peaks.

        Returns
        -------
        pd.DataFrame
            One row per accepted block, with columns:
            - block_index (int): 0-based block number
            - block_start_s (float): seconds since trial start
            - block_end_s (float): seconds since trial start
            - n_strides (int): strides starting in this block
            - stride_time_mean_s (float): mean stride time in this block
            - stride_time_std_s (float): standard deviation of stride times
            - stride_cadence_spm (float): strides per minute in this block
        """
        if len(peaks) < 2:
            self.logger.warning(
                "Not enough peaks to compute per-minute metrics; need at least 2."
            )
            return pd.DataFrame()

        block_dur = float(self.config.minute_block_duration_s)
        if block_dur <= 0:
            self.logger.error("minute_block_duration_s must be positive.")
            return pd.DataFrame()

        # Time of each peak (seconds since trial start) and stride intervals.
        t0 = df["_time"].iloc[0]
        peak_times_s = (
            (pd.to_datetime(df["_time"].iloc[peaks]) - t0)
            / np.timedelta64(1, "s")
        ).astype(float).to_numpy()

        # Each stride is the interval between consecutive peaks; we attribute
        # each stride to the block where it STARTS (i.e. where the first peak
        # of the pair falls). This is the standard convention.
        stride_intervals = np.diff(peak_times_s)
        stride_start_times = peak_times_s[:-1]

        total_duration = float(
            (df["_time"].iloc[-1] - df["_time"].iloc[0]).total_seconds()
        )
        n_blocks_full = int(np.floor(total_duration / block_dur))
        last_partial = total_duration - n_blocks_full * block_dur
        # Accept the last partial block only if it covers >= 50 % of block_dur.
        n_blocks_total = (
            n_blocks_full + 1 if last_partial >= 0.5 * block_dur else n_blocks_full
        )

        if n_blocks_total < 1:
            self.logger.warning(
                f"Trial duration {total_duration:.1f}s shorter than half a block "
                f"({0.5 * block_dur:.1f}s); no per-minute metrics computed."
            )
            return pd.DataFrame()

        rows = []
        for i in range(n_blocks_total):
            start_s = i * block_dur
            end_s = min((i + 1) * block_dur, total_duration)
            mask = (stride_start_times >= start_s) & (stride_start_times < end_s)
            block_strides = stride_intervals[mask]

            if len(block_strides) == 0:
                row = {
                    "block_index": i,
                    "block_start_s": start_s,
                    "block_end_s": end_s,
                    "n_strides": 0,
                    "stride_time_mean_s": np.nan,
                    "stride_time_std_s": np.nan,
                    "stride_cadence_spm": 0.0,
                }
            else:
                row = {
                    "block_index": i,
                    "block_start_s": start_s,
                    "block_end_s": end_s,
                    "n_strides": int(len(block_strides)),
                    "stride_time_mean_s": float(np.mean(block_strides)),
                    "stride_time_std_s": float(np.std(block_strides)),
                    "stride_cadence_spm": float(
                        len(block_strides) / (end_s - start_s) * 60.0
                    ),
                }
            rows.append(row)

        return pd.DataFrame(rows)


    def process_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray, pd.DataFrame]:
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
        
        # 'peaks' are heel strikes from a single foot, so each one represents
        # the start of a stride for that foot.
        self.logger.info(f"Detected {len(peaks)} strides")
        self.logger.debug(f"Peak heights preview: {properties.get('peak_heights', [])[:5]}")
        self.logger.debug(f"Turning ratio: {df['is_turning'].sum() / len(df):.3f}")

        # 5. Feature extraction
        metrics: Dict[str, Any] = {
            "posicion_gps": "N/A",
            "walking_duration_s": 0.0,
            "stride_time_mean_s": 0.0,
            "stride_time_std_s": 0.0,
            "stride_time_cv": 0.0,
            "stride_time_slope": 0.0,
            "stride_cadence_spm": 0.0,
            "stride_cadence_first_half_spm": 0.0,
            "stride_cadence_second_half_spm": 0.0,
            "stride_cadence_change_spm": 0.0,
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
        # Heel strikes are detected on a single foot's S2 sensor, so each
        # consecutive pair of strikes spans one full gait cycle of that foot.
        # The interval between them is therefore the STRIDE TIME of that foot,
        # not the step time (step time would require alternating strikes from
        # both feet and is calculated by combining Left and Right HDF5 keys,
        # which is left as future work).
        if len(peaks) > 1:
            stride_times = df["_time"].iloc[peaks].values
            stride_intervals = np.diff(stride_times) / np.timedelta64(1, "s")
            stride_intervals = stride_intervals.astype(float)

            duration = float((df["_time"].iloc[-1] - df["_time"].iloc[0]).total_seconds())
            metrics["walking_duration_s"] = duration
            metrics["stride_time_mean_s"] = float(np.mean(stride_intervals))
            metrics["stride_time_std_s"] = float(np.std(stride_intervals))
            metrics["stride_time_cv"] = self._safe_cv(stride_intervals)

            # Fatigue-sensitive temporal trend (slope of stride duration over time).
            metrics["stride_time_slope"] = self._safe_linear_slope(stride_intervals)

            # Per-foot cadence (strides of this foot per minute). Clinical cadence
            # would be the sum of strides of BOTH feet per minute, computed by
            # combining Left and Right keys. This is single-foot cadence only.
            if duration > 0:
                metrics["stride_cadence_spm"] = float(len(peaks) / duration * 60.0)

            # First-half vs second-half stride cadence (basic fatigue indicator).
            if duration > 0:
                t0 = df["_time"].iloc[0]
                relative_stride_times = (
                    (pd.to_datetime(stride_times) - t0) / np.timedelta64(1, "s")
                ).astype(float)

                half_time = duration / 2.0
                first_half_strides = int(np.sum(relative_stride_times < half_time))
                second_half_strides = int(np.sum(relative_stride_times >= half_time))

                first_half_duration = half_time
                second_half_duration = duration - half_time

                if first_half_duration > 0:
                    metrics["stride_cadence_first_half_spm"] = float(
                        first_half_strides / first_half_duration * 60.0
                    )
                if second_half_duration > 0:
                    metrics["stride_cadence_second_half_spm"] = float(
                        second_half_strides / second_half_duration * 60.0
                    )

                metrics["stride_cadence_change_spm"] = float(
                    metrics["stride_cadence_second_half_spm"]
                    - metrics["stride_cadence_first_half_spm"]
                )

            self.logger.info(
                "Temporal features extracted successfully: "
                f"stride_mean={metrics['stride_time_mean_s']:.3f}s, "
                f"stride_cadence={metrics['stride_cadence_spm']:.1f} spm, "
                f"stride_slope={metrics['stride_time_slope']:.6f}"
            )
        else:
            self.logger.warning("Not enough detected strides to compute temporal gait features.")

        # ────────────────────────────────────────────────────────────────────
        # Per-minute fatigue analysis
        # ────────────────────────────────────────────────────────────────────
        # Compute stride metrics in fixed-duration blocks (typically 60 s)
        # and derive the linear trend across blocks. The slope across blocks
        # captures progressive fatigue more sensitively than the trial-wide
        # slope (stride_time_slope), which averages variability within blocks
        # and may miss progressive deterioration in pace.
        per_minute_df = self._compute_minute_metrics(df, peaks)
        if not per_minute_df.empty:
            metrics["n_minute_blocks"] = int(len(per_minute_df))
            valid_blocks = per_minute_df.dropna(subset=["stride_time_mean_s"])
            if len(valid_blocks) >= 2:
                metrics["stride_time_minute_slope"] = self._safe_linear_slope(
                    valid_blocks["stride_time_mean_s"].to_numpy()
                )
                metrics["stride_cadence_minute_slope"] = self._safe_linear_slope(
                    valid_blocks["stride_cadence_spm"].to_numpy()
                )
                self.logger.info(
                    f"Per-minute fatigue analysis: {len(per_minute_df)} blocks, "
                    f"stride_time_slope={metrics['stride_time_minute_slope']:.6f} s/block, "
                    f"cadence_slope={metrics['stride_cadence_minute_slope']:.3f} spm/block"
                )
            else:
                self.logger.warning(
                    "Less than 2 valid blocks; per-minute slopes left at 0."
                )

        return df, metrics, peaks, per_minute_df