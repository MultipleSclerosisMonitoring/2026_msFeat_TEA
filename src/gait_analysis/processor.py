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
import math 
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from scipy.signal import butter, filtfilt, find_peaks
from pydantic import BaseModel, ConfigDict,Field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Module-level GPS helpers.
# ──────────────────────────────────────────────────────────────────────
def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Great-circle distance in metres between two GPS points.

    Uses the haversine formula on a spherical Earth model with
    R = 6_371_000 m. Accuracy is sufficient for path-length estimation
    over walking distances (sub-metre error for separations under 1 km).
    """
    R = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    )
    return 2.0 * R * math.asin(math.sqrt(a))


def _latlng_to_local_xy_m(
    lat_array: np.ndarray, lng_array: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project latitude/longitude arrays to a local Cartesian plane in metres."""
    lat0 = float(lat_array[0])
    lng0 = float(lng_array[0])
    mean_lat_rad = math.radians(float(np.nanmean(lat_array)))
    x_m = (lng_array - lng0) * 111_320.0 * math.cos(mean_lat_rad)
    y_m = (lat_array - lat0) * 110_540.0
    return x_m.astype(float), y_m.astype(float)



def smooth_gps_trajectory(
    lat_array: np.ndarray,
    lng_array: np.ndarray,
    outlier_distance_m: float = 35.0,
) -> dict[str, np.ndarray | int]:
    """
    Replace isolated GPS spikes with a local median estimate.

    A sample is smoothed only when both adjacent jumps are large but the
    direct bridge between its neighbours is short, which is the typical
    fingerprint of a single bad fix.
    """
    lat = np.asarray(lat_array, dtype=float).copy()
    lng = np.asarray(lng_array, dtype=float).copy()
    n = len(lat)
    if n < 3:
        return {"lat": lat, "lng": lng, "n_replaced": 0}

    replaced = 0
    for i in range(1, n - 1):
        d_prev = haversine_m(lat[i - 1], lng[i - 1], lat[i], lng[i])
        d_next = haversine_m(lat[i], lng[i], lat[i + 1], lng[i + 1])
        d_bridge = haversine_m(lat[i - 1], lng[i - 1], lat[i + 1], lng[i + 1])
        if (
            d_prev > outlier_distance_m
            and d_next > outlier_distance_m
            and d_bridge < outlier_distance_m
        ):
            lat[i] = float(np.median(lat[i - 1 : i + 2]))
            lng[i] = float(np.median(lng[i - 1 : i + 2]))
            replaced += 1

    return {"lat": lat, "lng": lng, "n_replaced": replaced}



def estimate_indoor_corridor_path_m(
    lat_array: np.ndarray,
    lng_array: np.ndarray,
) -> float:
    """
    Estimate indoor back-and-forth distance along the dominant corridor axis.

    GPS fixes are projected to a local plane, collapsed onto their first
    principal axis and integrated as absolute displacement. This lets an
    ida-vuelta corridor accumulate distance even when the net displacement is
    small.
    """
    if len(lat_array) < 2:
        return 0.0

    x_m, y_m = _latlng_to_local_xy_m(lat_array, lng_array)
    coords = np.column_stack([x_m, y_m])
    centered = coords - coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    proj = centered @ axis
    return float(np.abs(np.diff(proj)).sum())



def _count_turn_episodes(turn_flags: np.ndarray) -> int:
    """Count contiguous turning episodes from a boolean turning mask."""
    turns = np.asarray(turn_flags, dtype=bool)
    if turns.size == 0:
        return 0
    starts = np.flatnonzero(turns & ~np.r_[False, turns[:-1]])
    return int(len(starts))


def compute_gps_path_metrics(
    lat_array: np.ndarray, lng_array: np.ndarray
) -> Dict[str, float]:
    """
    Compute path-based GPS metrics from latitude / longitude arrays.

    The raw GPS trajectory is first smoothed to suppress isolated outlier fixes.
    Then we report both the direct consecutive-path estimate and an indoor
    corridor estimate based on absolute motion along the dominant axis.
    """
    n = len(lat_array)
    if n < 2:
        return {
            "n_unique_points": int(n),
            "span_m": 0.0,
            "total_path_m": 0.0,
            "corridor_path_m": 0.0,
            "n_smoothed_outliers": 0,
        }

    smoothed = smooth_gps_trajectory(lat_array, lng_array)
    lat_s = smoothed["lat"]
    lng_s = smoothed["lng"]

    rounded = np.column_stack([np.round(lat_s, 6), np.round(lng_s, 6)])
    n_unique = int(len(np.unique(rounded, axis=0)))

    distances_from_start = np.array([
        haversine_m(lat_s[0], lng_s[0], lat_s[i], lng_s[i])
        for i in range(n)
    ])
    span_m = float(distances_from_start.max())

    consecutive = np.array([
        haversine_m(lat_s[i - 1], lng_s[i - 1], lat_s[i], lng_s[i])
        for i in range(1, n)
    ])
    total_path_m = float(consecutive.sum())
    corridor_path_m = float(estimate_indoor_corridor_path_m(lat_s, lng_s))

    return {
        "n_unique_points": n_unique,
        "span_m": span_m,
        "total_path_m": total_path_m,
        "corridor_path_m": corridor_path_m,
        "n_smoothed_outliers": int(smoothed["n_replaced"]),
    }


# NumPy >= 2.0 renamed trapz to trapezoid; support both versions.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def compute_mean_swing_gyro_integral(
    df: pd.DataFrame,
    peaks: np.ndarray,
    toe_offs: np.ndarray,
    fs: float,
) -> float:
    """Mean integral of gyroscope L2 norm during swing phases (deg).
    Used by gyro-norm spatial model: stride_length = K * result.
    Returns 0.0 if no valid swings."""
    if len(peaks) < 2 or len(toe_offs) < 1:
        return 0.0
    
    gx = df["Gx_filt"].to_numpy()
    gy = df["Gy_filt"].to_numpy()
    gz = df["Gz_filt"].to_numpy()
    gyro_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    dt = 1.0 / fs
    n_swings = min(len(toe_offs), len(peaks) - 1)
    integrals = []
    for i in range(n_swings):
        to_idx = int(toe_offs[i])
        next_hs = int(peaks[i + 1])
        if next_hs <= to_idx:
            continue
        integrals.append(float(_trapz(gyro_norm[to_idx:next_hs], dx=dt)))
    
    return float(np.mean(integrals)) if integrals else 0.0


def compute_bilateral_metrics(
    metrics_left: Dict[str, Any],
    metrics_right: Dict[str, Any],
    peaks_left: np.ndarray,
    peaks_right: np.ndarray,
    toe_offs_left: np.ndarray,
    toe_offs_right: np.ndarray,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    compute_step_time: bool = True,
) -> Dict[str, Any]:
    """
    Compute bilateral gait metrics from paired left and right foot data.

    Calculates stride asymmetry and double support time, which require
    simultaneous knowledge of events from both feet. Asymmetry is
    expressed as the absolute percentage difference between feet,
    following the convention of Plotnik et al. (2020). Double support
    time is the interval during which both feet are simultaneously in
    contact with the ground, computed as the overlap between the stance
    phase of one foot and the stance phase of the contralateral foot.

    Parameters
    ----------
    metrics_left, metrics_right : dict
        Trial-wide metrics already computed by process_signals for each foot.
    peaks_left, peaks_right : np.ndarray
        Heel-strike indices for each foot.
    toe_offs_left, toe_offs_right : np.ndarray
        Toe-off indices for each foot.
    df_left, df_right : pd.DataFrame
        Processed dataframes with '_time' column for each foot.

    Returns
    -------
    dict
        Bilateral metrics, all prefixed with ``bilateral_``.
    """
    result: Dict[str, Any] = {
        "bilateral_stride_time_asymmetry_pct": np.nan,
        "bilateral_cadence_asymmetry_pct": np.nan,
        "bilateral_stance_time_asymmetry_pct": np.nan,
        "bilateral_double_support_mean_s": np.nan,
        "bilateral_double_support_pct": np.nan,
        "bilateral_available": False,
    }

    # ── Asymmetry metrics ────────────────────────────────────────────
    def _asymmetry_pct(val_l: float, val_r: float) -> float:
        """Absolute % difference relative to the mean of both values."""
        mean = (val_l + val_r) / 2.0
        if mean <= 0:
            return 0.0
        return float(abs(val_l - val_r) / mean * 100.0)

    st_l = float(metrics_left.get("stride_time_mean_s", 0.0))
    st_r = float(metrics_right.get("stride_time_mean_s", 0.0))
    cad_l = float(metrics_left.get("stride_cadence_spm", 0.0))
    cad_r = float(metrics_right.get("stride_cadence_spm", 0.0))
    stance_l = float(metrics_left.get("stance_time_mean_s", 0.0))
    stance_r = float(metrics_right.get("stance_time_mean_s", 0.0))

    if st_l > 0 and st_r > 0:
        result["bilateral_stride_time_asymmetry_pct"] = _asymmetry_pct(st_l, st_r)
        result["bilateral_cadence_asymmetry_pct"] = _asymmetry_pct(cad_l, cad_r)
    if stance_l > 0 and stance_r > 0:
        result["bilateral_stance_time_asymmetry_pct"] = _asymmetry_pct(stance_l, stance_r)

    # Step time and step time asymmetry are computed later in the try block
    # once absolute timestamps are available (requires both feet aligned).
    result["bilateral_step_time_LR_mean_s"] = np.nan  # HS_left  → HS_right
    result["bilateral_step_time_RL_mean_s"] = np.nan  # HS_right → HS_left
    result["bilateral_step_time_asymmetry_pct"] = np.nan

    # ── Double support time ──────────────────────────────────────────
    # Double support occurs when HS of one foot falls before the TO of
    # the contralateral foot. We need a shared time axis, so we use
    # absolute timestamps from the dataframes.
    if (
        len(peaks_left) < 2 or len(toe_offs_left) < 1
        or len(peaks_right) < 2 or len(toe_offs_right) < 1
    ):
        # Asymmetry computed but double support requires timestamp alignment.
        # Leave double support as NaN if timestamp processing fails later.
        result["bilateral_available"] = st_l > 0 and st_r > 0
        return result

    try:
        # Use the earliest timestamp across both feet as the common
        # time origin, so that absolute event times are comparable.
        t0 = min(
            pd.to_datetime(df_left["_time"].iloc[0]),
            pd.to_datetime(df_right["_time"].iloc[0]),
        )

        def to_sec(df: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
            return (
                (pd.to_datetime(df["_time"].iloc[idx]) - t0)
                / np.timedelta64(1, "s")
            ).astype(float).to_numpy()

        hs_l = to_sec(df_left, peaks_left)
        to_l = to_sec(df_left, toe_offs_left)
        hs_r = to_sec(df_right, peaks_right)
        to_r = to_sec(df_right, toe_offs_right)

        # ── Step time ────────────────────────────────────────────────────
        # Step time is defined as the interval between consecutive HS of
        # alternating feet. Following Plotnik et al. (2020), we merge all
        # HS events from both feet into a single chronological sequence,
        # then compute the interval between each pair of consecutive
        # events from different feet.
        #
        # step_LR: HS_right(n) → HS_left(n)  — the step taken by left foot
        # step_RL: HS_left(n)  → HS_right(n+1) — the step taken by right foot
        #
        # This approach is robust to which foot leads, unlike the
        # "find next contralateral" method which produces artifacts
        # when one foot consistently precedes the other.
        # Step time: merge all HS from both feet chronologically and
        # compute intervals between consecutive alternating-foot events.
        all_hs = sorted(
            [(t, 'L') for t in hs_l] + [(t, 'R') for t in hs_r],
            key=lambda x: x[0]
        )
        step_times_lr = []  # R→L steps
        step_times_rl = []  # L→R steps
        for i in range(len(all_hs) - 1):
            t1, foot1 = all_hs[i]
            t2, foot2 = all_hs[i + 1]
            if foot1 != foot2:
                dt = float(t2 - t1)
                if 0.2 < dt < 2.0:
                    if foot1 == 'R' and foot2 == 'L':
                        step_times_lr.append(dt)
                    elif foot1 == 'L' and foot2 == 'R':
                        step_times_rl.append(dt)
        if step_times_lr and step_times_rl:
            st_lr = float(np.mean(step_times_lr))
            st_rl = float(np.mean(step_times_rl))
            result["bilateral_step_time_LR_mean_s"] = st_lr
            result["bilateral_step_time_RL_mean_s"] = st_rl
            result["bilateral_step_time_asymmetry_pct"] = _asymmetry_pct(st_lr, st_rl)

        # For each left stance [hs_l[i], to_l[i]], find overlap with
        # right stance phases [hs_r[j], to_r[j]].
        n_l = min(len(hs_l), len(to_l))
        n_r = min(len(hs_r), len(to_r))
        ds_intervals = []
        for i in range(n_l):
            for j in range(n_r):
                overlap_start = max(hs_l[i], hs_r[j])
                overlap_end = min(to_l[i], to_r[j])
                if overlap_end > overlap_start:
                    ds_intervals.append(overlap_end - overlap_start)

        if ds_intervals:
            ds_mean = float(np.mean(ds_intervals))
            stride_mean = (st_l + st_r) / 2.0
            result["bilateral_double_support_mean_s"] = ds_mean
            result["bilateral_double_support_pct"] = (
                float(ds_mean / stride_mean * 100.0) if stride_mean > 0 else 0.0
            )

        result["bilateral_available"] = True

    except Exception:
        pass

    return result

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
    edge_threshold : float
        Normalized amplitude threshold (0-1) used to identify Heel Strike
        and Toe-Off as rising/falling edge crossings of the S2 signal.
        Default is 0.5 (= 50 % of the trial's dynamic range).
    """

    model_config = ConfigDict(extra="forbid")

    fs: float = Field(default=100.0, ge=0.1)
    cutoff_pressure: float = Field(default=20.0, ge=0.1)
    cutoff_gyro: float = Field(default=5.0, ge=0.1)
    gyro_threshold: float = Field(default=150.0)
    min_peak_distance_s: float = Field(default=0.6, ge=0.05)
    min_peak_height: float = Field(default=0.4)
    step_time_enabled: bool = Field(default=True)
    minute_block_duration_s: float = Field(default=60.0, ge=5.0)
    edge_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    toe_off_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    toe_off_method: str = Field(default="derivative")
    # 'threshold': first falling-edge crossing below toe_off_threshold.
    # 'derivative': point of maximum negative slope in the S2 signal
    #               after heel strike (more robust, independent of
    #               signal baseline).


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
    
    def _detect_gait_events(
        self,
        s2_filt: np.ndarray,
        is_turning: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect heel-strike and toe-off events from the filtered S2 signal.

        Implementation of the two-step "valley + threshold-crossing" method
        commonly used in plantar-pressure-based gait analysis:

        1. Find mid-swing valleys on the inverted normalized signal (these
           mark the deepest part of the swing phase, when the heel sensor
           reads its lowest value).
        2. For each valley, the next time the upright normalized signal
           rises above ``edge_threshold`` is a Heel Strike (foot lands).
        3. For each Heel Strike, the next time the signal drops below
           ``edge_threshold`` is a Toe-Off (foot lifts off).

        Stance phase = HS -> TO. Swing phase = TO -> next HS.

        Parameters
        ----------
        s2_filt : np.ndarray
            Filtered S2 signal (heel pressure), upright (high = stance).
        is_turning : np.ndarray
            Boolean array of the same length, True where the gyro indicates
            high-magnitude rotation. Valleys inside turning regions are
            ignored.

        Returns
        -------
        heel_strikes : np.ndarray
            Indices of heel-strike events.
        toe_offs : np.ndarray
            Indices of toe-off events. ``len(toe_offs) == len(heel_strikes)``;
            each toe-off is the one that closes the stance started by the
            corresponding heel strike.
        valleys : np.ndarray
            Indices of mid-swing valleys (kept for diagnostics / plotting).
        """
        # 1. Mid-swing valleys: peaks of the inverted normalized signal.
        s2_inv = -s2_filt
        s2_inv_min = float(np.min(s2_inv))
        s2_inv_max = float(np.max(s2_inv))
        if s2_inv_max - s2_inv_min > 1e-9:
            s2_inv_norm = (s2_inv - s2_inv_min) / (s2_inv_max - s2_inv_min)
        else:
            s2_inv_norm = np.zeros_like(s2_inv)
        # Suppress detection during turning regions.
        s2_inv_norm[is_turning] = 0.0

        min_peak_distance_samples = max(
            1, int(self.config.min_peak_distance_s * self.config.fs)
        )
        
        valleys, _ = find_peaks(
            s2_inv_norm,
            distance=min_peak_distance_samples,
            prominence=self.config.min_peak_height,
        )

        # 2. Upright normalized signal for threshold-crossing detection.
        s2_min = float(np.min(s2_filt))
        s2_max = float(np.max(s2_filt))
        if s2_max - s2_min > 1e-9:
            s2_norm = (s2_filt - s2_min) / (s2_max - s2_min)
        else:
            s2_norm = np.zeros_like(s2_filt)

        hs_threshold = float(self.config.edge_threshold)
        to_threshold = float(self.config.toe_off_threshold)
        n = len(s2_norm)
        heel_strikes: list = []
        toe_offs: list = []
        for i, v_idx in enumerate(valleys):
            next_v = valleys[i + 1] if i + 1 < len(valleys) else n
            # Heel Strike: first rising-edge crossing AFTER the valley.
            segment_after_valley = s2_norm[v_idx:next_v]
            above = np.where(segment_after_valley > hs_threshold)[0]
            if len(above) == 0:
                continue
            hs_idx = v_idx + above[0]
            # Toe-Off detection: two methods available via toe_off_method.
            segment_after_hs = s2_norm[hs_idx:next_v]
            if len(segment_after_hs) < 2:
                continue
            if self.config.toe_off_method == "derivative":
                # Point of maximum negative slope: the sample where the
                # pressure signal falls fastest, which corresponds to the
                # actual foot lift-off. More robust than threshold crossing
                # because it is independent of the signal baseline level.
                deriv = np.diff(segment_after_hs)
                neg_mask = deriv < 0
                if not np.any(neg_mask):
                    # No falling segment: fall back to threshold method
                    below = np.where(segment_after_hs < to_threshold)[0]
                    if len(below) == 0:
                        continue
                    to_idx = hs_idx + below[0]
                else:
                    to_idx = hs_idx + int(np.argmin(deriv)) + 1
            else:
                # threshold method: first falling-edge crossing
                below = np.where(segment_after_hs < to_threshold)[0]
                if len(below) == 0:
                    continue
                to_idx = hs_idx + below[0]

            heel_strikes.append(hs_idx)
            toe_offs.append(to_idx)

        return (
            np.array(heel_strikes, dtype=int),
            np.array(toe_offs, dtype=int),
            valleys,
        )

    def _compute_minute_metrics(
        self,
        df: pd.DataFrame,
        peaks: np.ndarray,
        toe_offs: np.ndarray,
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

        # Stance and swing per stride.
        # Stance_i = TO_i - HS_i (toe-off i closes the stance started by HS_i).
        # Swing_i  = HS_{i+1} - TO_i (the air phase between TO_i and next HS).
        # Both arrays are aligned to stride index (so stride i has stance[i]
        # and swing[i]). Non-physiological values (<= 0 or >= 3 s) are masked
        # as NaN so they do not distort block-wise means.
        if len(toe_offs) >= 1 and len(peaks) >= 1:
            to_times_s = (
                (pd.to_datetime(df["_time"].iloc[toe_offs]) - t0)
                / np.timedelta64(1, "s")
            ).astype(float).to_numpy()
            stance_per_stride = to_times_s - peak_times_s[: len(to_times_s)]
            stance_per_stride = np.where(
                (stance_per_stride > 0) & (stance_per_stride < 3.0),
                stance_per_stride,
                np.nan,
            )
        else:
            stance_per_stride = np.array([], dtype=float)

        if len(toe_offs) >= 2 and len(peaks) >= 2:
            # Swing_i = HS_{i+1} - TO_i. The last toe-off has no following
            # heel strike within this trial, so we compute swing only for
            # the first (N-1) strides where N = min(len(peaks), len(toe_offs)).
            n_swings = min(len(peak_times_s) - 1, len(to_times_s) - 1)
            swing_per_stride = peak_times_s[1 : n_swings + 1] - to_times_s[:n_swings]
            swing_per_stride = np.where(
                (swing_per_stride > 0) & (swing_per_stride < 3.0),
                swing_per_stride,
                np.nan,
            )
        else:
            swing_per_stride = np.array([], dtype=float)

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
                    "stance_time_mean_s": np.nan,
                    "stance_time_std_s": np.nan,
                    "swing_time_mean_s": np.nan,
                    "swing_time_std_s": np.nan,
                }
            else:
                # Indices of strides attributed to this block.
                block_stride_idx = np.where(mask)[0]

                # Stance values for strides in this block.
                if len(stance_per_stride) > 0:
                    valid_stance_idx = block_stride_idx[
                        block_stride_idx < len(stance_per_stride)
                    ]
                    block_stance = stance_per_stride[valid_stance_idx]
                    block_stance = block_stance[~np.isnan(block_stance)]
                else:
                    block_stance = np.array([])

                # Swing values for strides in this block.
                if len(swing_per_stride) > 0:
                    valid_swing_idx = block_stride_idx[
                        block_stride_idx < len(swing_per_stride)
                    ]
                    block_swing = swing_per_stride[valid_swing_idx]
                    block_swing = block_swing[~np.isnan(block_swing)]
                else:
                    block_swing = np.array([])

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
                    "stance_time_mean_s": (
                        float(np.mean(block_stance))
                        if len(block_stance) > 0
                        else np.nan
                    ),
                    "stance_time_std_s": (
                        float(np.std(block_stance))
                        if len(block_stance) > 0
                        else np.nan
                    ),
                    "swing_time_mean_s": (
                        float(np.mean(block_swing))
                        if len(block_swing) > 0
                        else np.nan
                    ),
                    "swing_time_std_s": (
                        float(np.std(block_swing))
                        if len(block_swing) > 0
                        else np.nan
                    ),
                }
            rows.append(row)
        return pd.DataFrame(rows)


    def process_signals(
        self,
        df: pd.DataFrame,
        test_type: str | None = None,
        clinical_tests_cfg: Dict[str, Any] | None = None,
        gps_estimation_cfg: Dict[str, Any] | None = None,
        spatial_models_cfg: Dict[str, Any] | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
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

        # 3. Turn detection.
        #
        # We compute the L2 norm of the angular velocity vector
        # (||(Gx, Gy, Gz)||) instead of relying on |Gz| alone. The norm
        # is invariant to sensor orientation: independently of how the
        # IMU is positioned inside the sock, the magnitude of rotation
        # is the same. This avoids missing rotational activity when the
        # vertical world axis aligns with Gx or Gy rather than Gz.
        gyro_axes = ["Gx", "Gy", "Gz"]
        if all(axis in df.columns for axis in gyro_axes):
            for axis in gyro_axes:
                df[f"{axis}_filt"] = self._butter_lowpass_filter(
                    df[axis].values,
                    self.config.cutoff_gyro,
                )
            gyro_magnitude = np.sqrt(
                df["Gx_filt"] ** 2
                + df["Gy_filt"] ** 2
                + df["Gz_filt"] ** 2
            )
            
            df["gyro_magnitude"] = gyro_magnitude

            # ── Turn detection: peak-MAD adaptive method ─────────────────
            # The swing phase of each step produces a gyro spike of ~500
            # deg/s, making the raw signal unsuitable for z-score detection.
            # A 0.5 Hz envelope approach was also evaluated but failed
            # because normal gait already produces ~170 deg/s mean, making
            # turns indistinguishable from gait in the low-frequency domain.
            #
            # Solution (two-pass approach):
            # Pass 1 — bootstrap with fixed threshold to get initial peaks.
            # Pass 2 — compute the per-stride gyro peak series, apply a
            #           sliding window of median + N * MAD (Median Absolute
            #           Deviation). MAD is robust to outliers (turns inside
            #           the window do not corrupt the baseline estimate).
            #           Strides whose peak exceeds the adaptive threshold
            #           are flagged as turns and their samples marked.
            # Pass 3 — re-run gait event detection with the refined mask.
            #
            # This method is patient-adaptive: the threshold scales with
            # each patient's own gait dynamics, making it robust to the
            # high inter-subject variability typical of MS.

            gyro_mag_np = gyro_magnitude.to_numpy() if hasattr(gyro_magnitude, "to_numpy") else gyro_magnitude

            # Pass 1: fixed-threshold bootstrap
            is_turning_bootstrap = gyro_mag_np > self.config.gyro_threshold
            df["is_turning"] = is_turning_bootstrap

            try:
                # Get bootstrap S2 signal for preliminary peak detection
                s2_filt_bootstrap = df["S2_filt"].to_numpy()
                peaks_bootstrap, _, _ = self._detect_gait_events(
                    s2_filt_bootstrap,
                    is_turning_bootstrap,
                )

                if len(peaks_bootstrap) >= 4:
                    # Pass 2: compute gyro peak per stride
                    stride_peaks_gyro = []
                    stride_peak_indices = []  # index of each stride in peaks array
                    for i in range(len(peaks_bootstrap) - 1):
                        start = peaks_bootstrap[i]
                        end   = peaks_bootstrap[i + 1]
                        stride_gyro = gyro_mag_np[start:end]
                        if len(stride_gyro) > 0:
                            stride_peaks_gyro.append(float(np.max(stride_gyro)))
                            stride_peak_indices.append((start, end))

                    stride_peaks_gyro = np.array(stride_peaks_gyro)
                    n_strides = len(stride_peaks_gyro)

                    # Sliding window (default: 10 strides ≈ 10 s at 60 spm)
                    window = max(5, min(10, n_strides // 3))
                    is_turn_stride = np.zeros(n_strides, dtype=bool)

                    for i in range(n_strides):
                        lo = max(0, i - window // 2)
                        hi = min(n_strides, i + window // 2 + 1)
                        local_peaks = stride_peaks_gyro[lo:hi]
                        local_median = float(np.median(local_peaks))
                        local_mad    = float(np.median(np.abs(local_peaks - local_median)))
                        # N=3: conservative threshold; increase to reduce
                        # sensitivity for patients with very variable gait
                        adaptive_threshold = local_median + 3.0 * local_mad
                        if stride_peaks_gyro[i] > adaptive_threshold:
                            is_turn_stride[i] = True

                    # Expand stride-level flags to sample-level
                    is_turning_adaptive = np.zeros(len(df), dtype=bool)
                    for i, (start, end) in enumerate(stride_peak_indices):
                        if is_turn_stride[i]:
                            is_turning_adaptive[start:end] = True

                    # Fallback: if adaptive detects nothing but fixed would,
                    # keep fixed threshold result
                    if not np.any(is_turning_adaptive) and np.any(is_turning_bootstrap):
                        df["is_turning"] = is_turning_bootstrap
                        self.logger.debug("Peak-MAD turn detection found no turns; keeping fixed threshold.")
                    else:
                        df["is_turning"] = is_turning_adaptive
                        n_turns = int(is_turn_stride.sum())
                        self.logger.info(
                            f"Peak-MAD turn detection: {n_turns} turning strides detected "
                            f"({is_turning_adaptive.mean()*100:.1f}% of samples)."
                        )
                else:
                    # Not enough strides for adaptive detection
                    self.logger.debug("Not enough strides for peak-MAD detection; using fixed threshold.")
                    df["is_turning"] = is_turning_bootstrap

            except Exception as e:
                self.logger.warning(f"Peak-MAD turn detection failed ({e}); falling back to fixed threshold.")
                df["is_turning"] = is_turning_bootstrap

        else:
            df["is_turning"] = False
            missing = [a for a in gyro_axes if a not in df.columns]
            self.logger.warning(
                f"Gyroscope axes missing ({missing}). Turn segmentation disabled."
            )

        self.logger.debug(f"Filtered S2 preview: {df['S2_filt'].head().to_list()}")

        # 4. Heel-Strike and Toe-Off detection outside turning phases.
        #
        # Sensor convention (validated empirically by cross-referencing the
        # S2 peaks with positive Gz peaks that mark Toe-Off in three different
        # patients): S2 reads HIGH while the foot is bearing weight on the
        # ground (stance) and LOW while the foot is in the air (swing).
        #
        # We detect events using the two-step "valley + edge crossing" method
        # encapsulated in `_detect_gait_events`:
        #   - Heel Strike (HS) = first rising-edge crossing of `edge_threshold`
        #     after a mid-swing valley.
        #   - Toe-Off    (TO) = first falling-edge crossing after the HS.
        # Stance phase = HS -> TO. Swing phase = TO -> next HS.
        s2_signal = df["S2_filt"].to_numpy(dtype=float)
        is_turning = df["is_turning"].to_numpy()

        peaks, toe_offs, _valleys = self._detect_gait_events(
            s2_filt=s2_signal,
            is_turning=is_turning,
        )

        self.logger.debug(
            f"Edge threshold: HS={self.config.edge_threshold:.2f}, "
            f"TO={self.config.toe_off_threshold:.2f} "
            f"(applied on signal normalized to [0, 1])"
        )
        self.logger.debug(
            f"Original S2 range: min={s2_signal.min():.1f}, max={s2_signal.max():.1f}"
        )
        self.logger.debug(f"Signal length: {len(df)} samples")
        self.logger.debug(f"Turning samples: {df['is_turning'].sum()}")
        self.logger.debug(f"Turning ratio: {df['is_turning'].sum() / len(df):.3f}")

        # Each peak is a Heel Strike of the analysed foot; each gait cycle
        # begins there. Toe-offs come paired 1:1 with heel-strikes.
        self.logger.info(
            f"Detected {len(peaks)} heel strikes and {len(toe_offs)} toe-offs"
        )

        # 5. Feature extraction
        metrics: Dict[str, Any] = {
            "posicion_gps": "N/A",
            "walking_duration_s": 0.0,
            # Stride-level metrics (HS to next HS of the same foot).
            "stride_time_mean_s": 0.0,
            "stride_time_std_s": 0.0,
            "stride_time_cv": 0.0,
            "stride_time_slope": 0.0,
            "stride_cadence_spm": 0.0,
            "stride_cadence_first_half_spm": 0.0,
            "stride_cadence_second_half_spm": 0.0,
            "stride_cadence_change_spm": 0.0,
            # Sub-cycle phases (HS to TO = stance, TO to next HS = swing).
            "stance_time_mean_s": 0.0,
            "stance_time_std_s": 0.0,
            "stance_time_cv": 0.0,
            "swing_time_mean_s": 0.0,
            "swing_time_std_s": 0.0,
            "swing_time_cv": 0.0,
            "stance_swing_ratio": 0.0,
            # Per-minute fatigue slopes (linear regression over 60-s blocks).
            "n_minute_blocks": 0,
            "stride_time_minute_slope": 0.0,
            "stride_cadence_minute_slope": 0.0,
            "stance_time_minute_slope": 0.0,
            "swing_time_minute_slope": 0.0,
            # Spatial metrics (only computed when distance is known).
            # spatial_method documents the source: 'none' (not computed),
            # 'known_distance' (clinical test with fixed distance from
            # config), 'gps' or 'imu_zupt' (future work for 6MWT).
            "spatial_method": "none",
            "spatial_distance_m": 0.0,
            "walking_speed_mean_m_s": 0.0,
            "stride_length_mean_m": 0.0,
            "gps_n_unique_points": 0,
            "gps_span_m": 0.0,
            "gps_total_path_m": 0.0,
            "gps_corridor_path_m": 0.0,
            "gps_smoothed_outliers": 0,
            "gyro_norm_stride_length_m": 0.0,
            "gyro_norm_walking_speed_m_s": 0.0,
            "biometric_stride_length_m": 0.0,
            "biometric_walking_speed_m_s": 0.0,
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
        # both feet). Bilateral metrics (asymmetry, double support) are computed
        # separately in compute_bilateral_metrics() after processing both feet.
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

            # Stance / swing decomposition.
            # Stance = HS_i -> TO_i (foot on the ground).
            # Swing  = TO_i -> HS_{i+1} (foot in the air).
            # We filter out non-physiological values (<= 0 or >= 3 s) that
            # could appear at trial edges or when an event is missing.
            if len(toe_offs) >= 1 and len(peaks) >= 1:
                stance_times = (
                    (df["_time"].iloc[toe_offs].values - df["_time"].iloc[peaks].values)
                    / np.timedelta64(1, "s")
                ).astype(float)
                stance_times = stance_times[(stance_times > 0) & (stance_times < 3.0)]
                if len(stance_times) >= 1:
                    metrics["stance_time_mean_s"] = float(np.mean(stance_times))
                    metrics["stance_time_std_s"] = float(np.std(stance_times))
                    metrics["stance_time_cv"] = self._safe_cv(stance_times)

            if len(toe_offs) >= 1 and len(peaks) >= 2:
                swing_times = (
                    (df["_time"].iloc[peaks[1:]].values - df["_time"].iloc[toe_offs[:-1]].values)
                    / np.timedelta64(1, "s")
                ).astype(float)
                swing_times = swing_times[(swing_times > 0) & (swing_times < 3.0)]
                if len(swing_times) >= 1:
                    metrics["swing_time_mean_s"] = float(np.mean(swing_times))
                    metrics["swing_time_std_s"] = float(np.std(swing_times))
                    metrics["swing_time_cv"] = self._safe_cv(swing_times)

            if metrics["swing_time_mean_s"] > 0:
                metrics["stance_swing_ratio"] = float(
                    metrics["stance_time_mean_s"] / metrics["swing_time_mean_s"]
                )

            self.logger.info(
                "Temporal features extracted successfully: "
                f"stride_mean={metrics['stride_time_mean_s']:.3f}s, "
                f"stride_cadence={metrics['stride_cadence_spm']:.1f} spm, "
                f"stride_slope={metrics['stride_time_slope']:.6f}, "
                f"stance%={100*metrics['stance_time_mean_s']/metrics['stride_time_mean_s']:.1f}% "
                f"of stride"
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
        per_minute_df = self._compute_minute_metrics(df, peaks, toe_offs)
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
                # Sub-cycle phase slopes: only use blocks where stance/swing
                # are not NaN (a block may have strides but missing stance/swing
                # if the corresponding TO event was filtered out as non-physiological).
                stance_blocks = per_minute_df.dropna(subset=["stance_time_mean_s"])
                if len(stance_blocks) >= 2:
                    metrics["stance_time_minute_slope"] = self._safe_linear_slope(
                        stance_blocks["stance_time_mean_s"].to_numpy()
                    )
                    swing_blocks = per_minute_df.dropna(subset=["swing_time_mean_s"])
                    if len(swing_blocks) >= 2:
                        metrics["swing_time_minute_slope"] = self._safe_linear_slope(
                            swing_blocks["swing_time_mean_s"].to_numpy()
                        )
                    self.logger.info(
                        f"Per-minute fatigue analysis: {len(per_minute_df)} blocks, "
                        f"stride_time_slope={metrics['stride_time_minute_slope']:.6f} s/block, "
                        f"cadence_slope={metrics['stride_cadence_minute_slope']:.3f} spm/block, "
                        f"stance_slope={metrics['stance_time_minute_slope']:.6f} s/block, "
                        f"swing_slope={metrics['swing_time_minute_slope']:.6f} s/block"
                    )
            else:
                self.logger.warning(
                    "Less than 2 valid blocks; per-minute slopes left at 0."
                )

        # ──────────────────────────────────────────────────────────────────
        # Spatial metrics: walking speed and stride length.
        # ──────────────────────────────────────────────────────────────────
        # Strategy 1 (implemented here): tests with KNOWN, fixed distance
        # such as TUG (6 m) or T25FW (7.62 m). The distance is configured
        # in `clinical_tests.<test_type>.distance_m` and the velocity is
        # simply walking_speed = distance / walking_duration. Stride
        # length follows from velocity * mean stride time.
        #
        # Strategies 2 (GPS) and 3 (IMU + ZUPT) for tests without known
        # distance, like 6MWT, are out of scope of this method and will
        # be implemented as separate spatial estimators in future work.
        if (
            test_type is not None
            and clinical_tests_cfg
            and test_type in clinical_tests_cfg
        ):
            test_cfg = clinical_tests_cfg[test_type]
            distance_m = float(test_cfg.get("distance_m", 0.0))
            min_duration_s = float(test_cfg.get("min_duration_s", 0.0))
            min_strides = int(test_cfg.get("min_strides", 0))
            walking_duration_s = float(metrics.get("walking_duration_s", 0.0))
            stride_time_mean_s = float(metrics.get("stride_time_mean_s", 0.0))
            # Each pair of consecutive heel strikes defines one stride, so
            # n_strides = len(peaks) - 1 (matches what we use for stride_time).
            n_strides = max(0, len(peaks) - 1)

            metrics["spatial_distance_m"] = distance_m

            if distance_m <= 0:
                self.logger.warning(
                    f"Configured distance for {test_type} is non-positive "
                    f"({distance_m}). Spatial metrics not computed."
                )
                metrics["spatial_method"] = "none"
            elif walking_duration_s < min_duration_s:
                self.logger.warning(
                    f"{test_type} trial duration ({walking_duration_s:.2f}s) "
                    f"is below the plausibility threshold "
                    f"({min_duration_s:.2f}s). Spatial metrics not computed."
                )
                metrics["spatial_method"] = "none"
            elif n_strides < min_strides:
                self.logger.warning(
                    f"{test_type} trial has only {n_strides} strides "
                    f"(below threshold {min_strides}). Stride-time mean "
                    f"would be unreliable, so spatial metrics are not "
                    f"computed."
                )
                metrics["spatial_method"] = "none"
            else:
                walking_speed = distance_m / walking_duration_s
                metrics["spatial_method"] = "known_distance"
                metrics["walking_speed_mean_m_s"] = float(walking_speed)
                if stride_time_mean_s > 0:
                    metrics["stride_length_mean_m"] = float(
                        walking_speed * stride_time_mean_s
                    )
                self.logger.info(
                    f"Spatial metrics ({test_type}, known distance "
                    f"{distance_m:.2f} m, {n_strides} strides): "
                    f"walking_speed="
                    f"{metrics['walking_speed_mean_m_s']:.3f} m/s, "
                    f"stride_length="
                    f"{metrics['stride_length_mean_m']:.3f} m"
                )
        elif test_type is not None:
            # ──────────────────────────────────────────────────────
            # Strategy 2 — GPS-based spatial estimation.
            # ──────────────────────────────────────────────────────
            # For tests without a fixed known distance (typically
            # 6MWT), we attempt to derive walking speed from GPS
            # fixes. This is only reliable when the patient walked
            # outdoor and far enough that GPS jitter is dwarfed by
            # actual displacement. We enforce two quality gates:
            #
            #   - span_m >= min_span_m: the patient must reach a
            #     point at least min_span_m metres from the start.
            #     This rejects indoor sessions where the GPS drifts
            #     in place but the patient does not actually travel
            #     a long path.
            #   - n_unique_points >= min_unique_points: enough fixes
            #     to integrate a path. With a 0.06 Hz GPS, a 6-min
            #     trial yields ~22 unique points; below 10 the path
            #     reconstruction is too coarse to be meaningful.
            #
            # Trials that pass both filters get spatial_method='gps'
            # and a walking_speed estimated as total_path_m divided
            # by walking_duration_s. Trials that fail are tagged
            # spatial_method='none' but the underlying GPS quality
            # metrics (gps_span_m, gps_n_unique_points,
            # gps_total_path_m) are still recorded for traceability.
            gps_cfg = (
                gps_estimation_cfg if gps_estimation_cfg is not None else {}
            )
            min_span_m = float(gps_cfg.get("min_span_m", 0.0))
            min_unique = int(gps_cfg.get("min_unique_points", 0))
            max_path_span_ratio = float(gps_cfg.get("max_path_span_ratio", 3.0))
            min_turn_episodes = int(gps_cfg.get("min_turn_episodes", 2))

            if {"lat", "lng"}.issubset(df.columns):
                gps_df = (
                    df[["lat", "lng"]]
                    .apply(pd.to_numeric, errors="coerce")
                    .dropna()
                )
                if len(gps_df) >= 2:
                    lat_arr = gps_df["lat"].to_numpy()
                    lng_arr = gps_df["lng"].to_numpy()
                    gps_metrics = compute_gps_path_metrics(lat_arr, lng_arr)

                    metrics["gps_n_unique_points"] = gps_metrics["n_unique_points"]
                    metrics["gps_span_m"] = gps_metrics["span_m"]
                    metrics["gps_total_path_m"] = gps_metrics["total_path_m"]
                    metrics["gps_corridor_path_m"] = gps_metrics["corridor_path_m"]
                    metrics["gps_smoothed_outliers"] = gps_metrics["n_smoothed_outliers"]

                    walking_duration_s = float(
                        metrics.get("walking_duration_s", 0.0)
                    )
                    stride_time_mean_s = float(
                        metrics.get("stride_time_mean_s", 0.0)
                    )
                    turn_episodes = 0
                    if "is_turning" in df.columns:
                        turn_episodes = _count_turn_episodes(
                            df["is_turning"].to_numpy(dtype=bool)
                        )

                    direct_distance_m = float(gps_metrics["total_path_m"])
                    corridor_distance_m = float(gps_metrics["corridor_path_m"])
                    span_m = float(gps_metrics["span_m"])
                    path_span_ratio = (
                        direct_distance_m / span_m if span_m > 0 else float("inf")
                    )
                    using_indoor_rescue = span_m < min_span_m
                    if using_indoor_rescue:
                        candidate_distance_m = corridor_distance_m
                    else:
                        candidate_distance_m = direct_distance_m

                    if gps_metrics["n_unique_points"] < min_unique:
                        self.logger.warning(
                            f"GPS unique points "
                            f"({gps_metrics['n_unique_points']}) below "
                            f"threshold ({min_unique}). Insufficient fixes "
                            f"for path reconstruction in {test_type}."
                        )
                    elif walking_duration_s <= 0:
                        self.logger.warning(
                            f"walking_duration_s is non-positive; cannot "
                            f"derive GPS speed for {test_type}."
                        )
                    elif path_span_ratio > max_path_span_ratio:
                        self.logger.warning(
                            f"GPS path/span ratio ({path_span_ratio:.2f}) above "
                            f"threshold ({max_path_span_ratio:.2f}). Likely GPS jitter; "
                            f"spatial metrics not computed for {test_type}."
                        )
                    elif using_indoor_rescue and turn_episodes < min_turn_episodes:
                        self.logger.warning(
                            f"Indoor corridor candidate detected but only {turn_episodes} "
                            f"turn episodes found (threshold {min_turn_episodes}). "
                            f"Spatial metrics not computed for {test_type}."
                        )
                    elif candidate_distance_m < min_span_m:
                        self.logger.warning(
                            f"Reconstructed GPS distance ({candidate_distance_m:.1f} m) below "
                            f"threshold ({min_span_m:.1f} m). GPS-based spatial metrics not "
                            f"computed for {test_type}."
                        )
                    else:
                        walking_speed = candidate_distance_m / walking_duration_s
                        metrics["spatial_method"] = "gps"
                        metrics["spatial_distance_m"] = candidate_distance_m
                        metrics["walking_speed_mean_m_s"] = float(walking_speed)
                        if stride_time_mean_s > 0:
                            metrics["stride_length_mean_m"] = float(
                                walking_speed * stride_time_mean_s
                            )
                        self.logger.info(
                            f"Spatial metrics ({test_type}, GPS-based): "
                            f"span={span_m:.1f} m, "
                            f"path={direct_distance_m:.1f} m, "
                            f"corridor_path={corridor_distance_m:.1f} m, "
                            f"turn_episodes={turn_episodes}, "
                            f"path_span_ratio={path_span_ratio:.2f}, "
                            f"smoothed_outliers={gps_metrics['n_smoothed_outliers']}, "
                            f"selected_distance={candidate_distance_m:.1f} m, "
                            f"walking_speed="
                            f"{metrics['walking_speed_mean_m_s']:.3f} m/s, "
                            f"stride_length="
                            f"{metrics['stride_length_mean_m']:.3f} m"
                        )
                else:
                    self.logger.info(
                        f"Test type '{test_type}' has no configured distance "
                        f"and not enough GPS fixes; spatial metrics not "
                        f"computed."
                    )
            else:
                self.logger.info(
                    f"Test type '{test_type}' has no configured distance "
                    f"and no GPS columns; spatial metrics not computed."
                )

        # ── Nivel 3a: Gyro-norm model ─────────────────────────────────────
        # stride_length = K_gyro * mean(integral ||G|| during swing)
        # Orientation-invariant. Active for all test types when enabled.
        spatial_models_cfg = (
            spatial_models_cfg if spatial_models_cfg is not None else {}
        )
        gyro_cfg = spatial_models_cfg.get("gyro_norm", {})
        bio_cfg = spatial_models_cfg.get("biometric", {})
        stride_time_mean_s = float(metrics.get("stride_time_mean_s", 0.0))
        cadence_spm = float(metrics.get("stride_cadence_spm", 0.0))

        if gyro_cfg.get("enabled", False):
            K_gyro = float(gyro_cfg.get("K", 0.0))
            gyro_int = compute_mean_swing_gyro_integral(
                df, peaks, toe_offs, self.config.fs
            )
            if K_gyro > 0 and gyro_int > 0:
                sl_gyro = K_gyro * gyro_int
                metrics["gyro_norm_stride_length_m"] = float(sl_gyro)
                if stride_time_mean_s > 0:
                    metrics["gyro_norm_walking_speed_m_s"] = float(
                        sl_gyro / stride_time_mean_s
                    )
                self.logger.info(
                    f"Gyro-norm model: gyro_int={gyro_int:.3f} deg, "
                    f"stride_length={sl_gyro:.3f} m, "
                    f"walking_speed={metrics['gyro_norm_walking_speed_m_s']:.3f} m/s"
                )

        # ── Nivel 3b: Biometric model ──────────────────────────────────────
        # stride_length = K_bio * sqrt(cadence_spm)
        # Weinbach-type regression. Requires no IMU integration.
        if bio_cfg.get("enabled", False):
            K_bio = float(bio_cfg.get("K", 0.0))
            if K_bio > 0 and cadence_spm > 0:
                sl_bio = K_bio * math.sqrt(cadence_spm)
                metrics["biometric_stride_length_m"] = float(sl_bio)
                if stride_time_mean_s > 0:
                    metrics["biometric_walking_speed_m_s"] = float(
                        sl_bio / stride_time_mean_s
                    )
                self.logger.info(
                    f"Biometric model: cadence={cadence_spm:.1f} spm, "
                    f"stride_length={sl_bio:.3f} m, "
                    f"walking_speed={metrics['biometric_walking_speed_m_s']:.3f} m/s"
                )

        # ── Nivel 3c: IMU + ZUPT via Madgwick ─────────────────────────────
        # Orientation estimation + world-frame acceleration integration with
        # Zero-velocity UPdaTes (ZUPT) during stance phases.
        #
        # STATUS: Prepared for hardware recalibration to ±8g.
        # DISABLED by default: the current Sensoria sensor saturates at
        # ±2g during heel-strike impacts, causing Madgwick to accumulate
        # orientation drift that makes double integration unreliable.
        # When the accelerometer range is extended to ±8g (planned), set
        # imu_zupt.enabled=true in config. The magnetometer should also
        # be enabled at that point to stabilise the heading reference.
        imu_cfg = spatial_models_cfg.get("imu_zupt", {})
        if imu_cfg.get("enabled", False):
            try:
                from ahrs.filters import Madgwick as _Madgwick
                beta = float(imu_cfg.get("madgwick_beta", 0.1))
                use_mag = bool(imu_cfg.get("use_magnetometer", True))

                # Build input arrays (already resampled and filtered)
                ax = df["Ax"].to_numpy(dtype=float)
                ay = df["Ay"].to_numpy(dtype=float)
                az = df["Az"].to_numpy(dtype=float)
                gx = np.deg2rad(df["Gx_filt"].to_numpy(dtype=float))
                gy = np.deg2rad(df["Gy_filt"].to_numpy(dtype=float))
                gz = np.deg2rad(df["Gz_filt"].to_numpy(dtype=float))
                acc = np.column_stack([ax, ay, az])
                gyr = np.column_stack([gx, gy, gz])

                if use_mag and {"Mx", "My", "Mz"}.issubset(df.columns):
                    mx = pd.to_numeric(df["Mx"], errors="coerce").to_numpy(dtype=float)
                    my = pd.to_numeric(df["My"], errors="coerce").to_numpy(dtype=float)
                    mz = pd.to_numeric(df["Mz"], errors="coerce").to_numpy(dtype=float)
                    mag = np.column_stack([mx, my, mz])
                    madgwick = _Madgwick(
                        gyr=gyr, acc=acc, mag=mag,
                        frequency=self.config.fs, beta=beta,
                    )
                else:
                    madgwick = _Madgwick(
                        gyr=gyr, acc=acc,
                        frequency=self.config.fs, beta=beta,
                    )

                Q = madgwick.Q  # (N, 4): [w, x, y, z]

                # Rotate acceleration to world frame and remove gravity
                w, qx, qy, qz = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
                # Rotation matrix rows for each sample (vectorised)
                ax_w = (
                    (1 - 2*(qy**2 + qz**2)) * ax
                    + 2*(qx*qy - w*qz) * ay
                    + 2*(qx*qz + w*qy) * az
                )
                ay_w = (
                    2*(qx*qy + w*qz) * ax
                    + (1 - 2*(qx**2 + qz**2)) * ay
                    + 2*(qy*qz - w*qx) * az
                )
                az_w = (
                    2*(qx*qz - w*qy) * ax
                    + 2*(qy*qz + w*qx) * ay
                    + (1 - 2*(qx**2 + qy**2)) * az
                )
                # Remove gravity (world Z is vertical, ~1g upward)
                az_w_net = az_w - 1.0  # g units

                # ZUPT: zero velocity during stance phases
                dt = 1.0 / self.config.fs
                vel = np.zeros(len(ax_w))
                is_stance = ~df["is_turning"].to_numpy(dtype=bool)
                # Simple integration with ZUPT reset at each stance sample
                for i in range(1, len(vel)):
                    vel[i] = vel[i - 1] + az_w_net[i] * dt * 9.81
                    if is_stance[i]:
                        vel[i] = 0.0

                # Integrate velocity to position
                pos = np.cumsum(vel) * dt
                total_displacement_m = float(np.abs(pos[-1] - pos[0]))
                walking_dur = float(metrics.get("walking_duration_s", 0.0))

                if walking_dur > 0 and total_displacement_m > 0:
                    speed_zupt = total_displacement_m / walking_dur
                    metrics["spatial_method"] = "imu_zupt"
                    metrics["spatial_distance_m"] = total_displacement_m
                    metrics["walking_speed_mean_m_s"] = float(speed_zupt)
                    st = float(metrics.get("stride_time_mean_s", 0.0))
                    if st > 0:
                        metrics["stride_length_mean_m"] = float(speed_zupt * st)
                    self.logger.info(
                        f"IMU+ZUPT: displacement={total_displacement_m:.1f} m, "
                        f"walking_speed={speed_zupt:.3f} m/s"
                    )
                else:
                    self.logger.warning(
                        "IMU+ZUPT: displacement or duration is zero; "
                        "spatial metrics not updated."
                    )
            except ImportError:
                self.logger.warning(
                    "ahrs library not found; IMU+ZUPT disabled. "
                    "Install with: pip install ahrs"
                )
            except Exception as exc:
                self.logger.warning(f"IMU+ZUPT failed: {exc}")

        # Fallback for trials without a validated primary spatial estimate.
        if metrics.get("spatial_method") == "none":
            fallback_priority = spatial_models_cfg.get(
                "fallback_priority", ["gyro_norm", "biometric"]
            )
            fallback_values = {
                "gyro_norm": (
                    float(metrics.get("gyro_norm_walking_speed_m_s", 0.0) or 0.0),
                    float(metrics.get("gyro_norm_stride_length_m", 0.0) or 0.0),
                ),
                "biometric": (
                    float(metrics.get("biometric_walking_speed_m_s", 0.0) or 0.0),
                    float(metrics.get("biometric_stride_length_m", 0.0) or 0.0),
                ),
            }

            for method_name in fallback_priority:
                if method_name not in fallback_values:
                    continue
                speed_val, stride_val = fallback_values[method_name]
                if speed_val > 0 and stride_val > 0:
                    metrics["spatial_method"] = method_name
                    metrics["walking_speed_mean_m_s"] = speed_val
                    metrics["stride_length_mean_m"] = stride_val
                    self.logger.info(
                        "Spatial fallback activated: %s model promoted to main spatial metrics.",
                        method_name,
                    )
                    break

        return df, metrics, peaks, toe_offs, per_minute_df