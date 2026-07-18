"""
Unit tests for gait_analysis.processor — MS-Feat.

Covers the five most critical functions:
    1. haversine_m              — GPS distance calculation
    2. _autodetect_vertical_axis — IMU axis calibration
    3. _detect_gait_events      — Heel Strike / Toe-Off detection
    4. compute_bilateral_metrics — Asymmetry and double support
    5. ProcessConfig            — Schema validation and defaults
"""

import numpy as np
import pandas as pd
import pytest

from gait_analysis.processor import (
    GaitDataProcessor,
    ProcessConfig,
    compute_bilateral_metrics,
    compute_gps_path_metrics,
    estimate_indoor_corridor_path_m,
    haversine_m,
    smooth_gps_trajectory,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic_s2(
    fs: float = 100.0,
    n_strides: int = 5,
    stride_time_s: float = 1.0,
    stance_pct: float = 0.65,
) -> pd.DataFrame:
    """
    Build a minimal synthetic dataframe with a clean S2 pressure signal
    and flat gyroscope channels (no turning).

    The S2 signal is a square wave: HIGH during stance, LOW during swing.
    Gx, Gy, Gz are all zero so is_turning is never activated.
    """
    total_samples = int(n_strides * stride_time_s * fs)
    t = np.arange(total_samples) / fs

    # Square-wave pressure: HIGH = stance, LOW = swing
    stance_samples = int(stance_pct * stride_time_s * fs)
    stride_samples = int(stride_time_s * fs)
    s2 = np.zeros(total_samples)
    for i in range(n_strides):
        start = i * stride_samples
        s2[start : start + stance_samples] = 1.0

    # Add small Gaussian noise to avoid degenerate edge cases
    rng = np.random.default_rng(42)
    s2 += rng.normal(0, 0.02, total_samples)
    s2 = np.clip(s2, 0, None)

    t0 = pd.Timestamp("2025-01-01 00:00:00")
    timestamps = pd.date_range(t0, periods=total_samples, freq=pd.to_timedelta(1 / fs, unit="s"))

    df = pd.DataFrame({
        "_time": timestamps,
        "S2": s2,
        "Ax": rng.normal(0, 0.05, total_samples),
        "Ay": 1.0 + rng.normal(0, 0.05, total_samples),  # vertical axis
        "Az": rng.normal(0, 0.05, total_samples),
        "Gx": np.zeros(total_samples),
        "Gy": np.zeros(total_samples),
        "Gz": np.zeros(total_samples),
    })
    return df


def _make_metrics(stride_time_s: float, cadence_spm: float, stance_time_s: float) -> dict:
    return {
        "stride_time_mean_s": stride_time_s,
        "stride_cadence_spm": cadence_spm,
        "stance_time_mean_s": stance_time_s,
    }


# ── 1. haversine_m ────────────────────────────────────────────────────────────

class TestGpsPathMetrics:

    def test_smooth_gps_trajectory_removes_isolated_spike(self):
        lat = np.array([40.0, 40.00001, 40.01, 40.00002, 40.00003])
        lng = np.array([-3.0, -3.0, -3.01, -3.0, -3.0])
        result = smooth_gps_trajectory(lat, lng, outlier_distance_m=50.0)
        assert result["n_replaced"] == 1
        assert abs(result["lat"][2] - 40.00002) < 1e-4

    def test_indoor_corridor_path_accumulates_back_and_forth_distance(self):
        lat = np.array([40.0, 40.0, 40.0, 40.0, 40.0])
        lng = np.array([-3.0, -2.9998, -2.9996, -2.9998, -3.0])
        corridor_path = estimate_indoor_corridor_path_m(lat, lng)
        direct_span = haversine_m(lat[0], lng[0], lat[2], lng[2])
        assert corridor_path > direct_span
        assert corridor_path > 60.0

    def test_compute_gps_path_metrics_reports_corridor_and_outliers(self):
        lat = np.array([40.0, 40.00001, 40.01, 40.00002, 40.00003])
        lng = np.array([-3.0, -2.9999, -3.01, -2.9998, -2.9997])
        metrics = compute_gps_path_metrics(lat, lng)
        assert "corridor_path_m" in metrics
        assert "n_smoothed_outliers" in metrics
        assert metrics["n_smoothed_outliers"] >= 1
        assert metrics["corridor_path_m"] >= 0.0


class TestHaversine:

    def test_same_point_is_zero(self):
        assert haversine_m(40.4, -3.7, 40.4, -3.7) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_madrid_barcelona(self):
        # Approx straight-line distance ~504 km
        d = haversine_m(40.4168, -3.7038, 41.3851, 2.1734)
        assert 500_000 < d < 510_000

    def test_symmetry(self):
        d1 = haversine_m(40.0, -3.0, 41.0, -4.0)
        d2 = haversine_m(41.0, -4.0, 40.0, -3.0)
        assert d1 == pytest.approx(d2, rel=1e-9)

    def test_short_distance_precision(self):
        # Two points ~111 m apart (1 arc-second latitude)
        d = haversine_m(40.0, -3.0, 40.001, -3.0)
        assert 100 < d < 120


# ── 2. _autodetect_vertical_axis ─────────────────────────────────────────────

class TestAutodetectVerticalAxis:

    def setup_method(self):
        self.processor = GaitDataProcessor(ProcessConfig())

    def test_detects_ay_as_vertical(self):
        """Ay has mean ~1g; Ax and Az are near zero."""
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame({
            "Ax": rng.normal(0, 0.05, n),
            "Ay": 1.0 + rng.normal(0, 0.05, n),
            "Az": rng.normal(0, 0.05, n),
        })
        assert self.processor._autodetect_vertical_axis(df) == "Ay"

    def test_detects_az_as_vertical(self):
        rng = np.random.default_rng(1)
        n = 500
        df = pd.DataFrame({
            "Ax": rng.normal(0, 0.05, n),
            "Ay": rng.normal(0, 0.05, n),
            "Az": 1.0 + rng.normal(0, 0.05, n),
        })
        assert self.processor._autodetect_vertical_axis(df) == "Az"

    def test_missing_axes_returns_default(self):
        df = pd.DataFrame({"S2": [1, 2, 3]})
        result = self.processor._autodetect_vertical_axis(df)
        assert result == "Az"


# ── 3. _detect_gait_events ────────────────────────────────────────────────────

class TestDetectGaitEvents:

    def setup_method(self):
        self.config = ProcessConfig(
            fs=100.0,
            cutoff_pressure=20.0,
            min_peak_distance_s=0.5,
            min_peak_height=0.3,
            edge_threshold=0.5,
            toe_off_threshold=0.5,
            toe_off_method="threshold",
        )
        self.processor = GaitDataProcessor(self.config)

    def _filtered_s2(self, df: pd.DataFrame) -> np.ndarray:
        return self.processor._butter_lowpass_filter(
            df["S2"].values, self.config.cutoff_pressure
        )

    def test_detects_correct_number_of_strides(self):
        n_strides = 6
        df = _make_synthetic_s2(n_strides=n_strides)
        s2_filt = self._filtered_s2(df)
        is_turning = np.zeros(len(s2_filt), dtype=bool)
        peaks, toe_offs, _ = self.processor._detect_gait_events(s2_filt, is_turning)
        # Expect n_strides HS (one per stride), with some tolerance for edge effects
        assert len(peaks) >= n_strides - 1
        assert len(peaks) == len(toe_offs)

    def test_peaks_before_toe_offs(self):
        """Each HS must precede its corresponding TO."""
        df = _make_synthetic_s2(n_strides=5)
        s2_filt = self._filtered_s2(df)
        is_turning = np.zeros(len(s2_filt), dtype=bool)
        peaks, toe_offs, _ = self.processor._detect_gait_events(s2_filt, is_turning)
        for hs, to in zip(peaks, toe_offs):
            assert hs < to, f"HS {hs} is not before TO {to}"

    def test_turning_suppresses_detection(self):
        """When is_turning is True everywhere, no events should be detected."""
        df = _make_synthetic_s2(n_strides=5)
        s2_filt = self._filtered_s2(df)
        is_turning = np.ones(len(s2_filt), dtype=bool)
        peaks, toe_offs, _ = self.processor._detect_gait_events(s2_filt, is_turning)
        assert len(peaks) == 0
        assert len(toe_offs) == 0

    def test_derivative_method_returns_same_count(self):
        """Derivative TO detection should find the same number of events."""
        config_deriv = ProcessConfig(
            fs=100.0,
            cutoff_pressure=20.0,
            min_peak_distance_s=0.5,
            min_peak_height=0.3,
            edge_threshold=0.5,
            toe_off_method="derivative",
        )
        proc_deriv = GaitDataProcessor(config_deriv)
        df = _make_synthetic_s2(n_strides=5)
        s2_filt = self.processor._butter_lowpass_filter(df["S2"].values, 20.0)
        is_turning = np.zeros(len(s2_filt), dtype=bool)
        peaks_t, to_t, _ = self.processor._detect_gait_events(s2_filt, is_turning)
        peaks_d, to_d, _ = proc_deriv._detect_gait_events(s2_filt, is_turning)
        # Both methods should find the same HS (only TO may differ slightly)
        assert len(peaks_t) == len(peaks_d)


# ── 4. compute_bilateral_metrics ─────────────────────────────────────────────

class TestComputeBilateralMetrics:

    def _make_df_with_events(self, n_events: int, fs: float = 100.0) -> tuple:
        """Return (df, peaks, toe_offs) for a synthetic symmetric trial."""
        stride_s = 1.0
        total = int(n_events * stride_s * fs)
        t0 = pd.Timestamp("2025-01-01")
        times = pd.date_range(t0, periods=total, freq=pd.to_timedelta(1 / fs, unit="s"))
        df = pd.DataFrame({"_time": times, "S2": np.ones(total)})
        peaks = np.array([int(i * stride_s * fs) for i in range(n_events)], dtype=int)
        toe_offs = np.array(
            [int((i + 0.65) * stride_s * fs) for i in range(n_events)], dtype=int
        )
        return df, peaks, toe_offs

    def test_symmetric_gait_low_asymmetry(self):
        """Identical left and right metrics should give ~0% asymmetry."""
        m = _make_metrics(stride_time_s=1.0, cadence_spm=60.0, stance_time_s=0.65)
        df_l, peaks_l, to_l = self._make_df_with_events(8)
        df_r, peaks_r, to_r = self._make_df_with_events(8)
        result = compute_bilateral_metrics(
            m, m, peaks_l, peaks_r, to_l, to_r, df_l, df_r
        )
        assert result["bilateral_stride_time_asymmetry_pct"] == pytest.approx(0.0, abs=1e-6)
        assert result["bilateral_available"] is True

    def test_asymmetric_gait_detects_difference(self):
        """Different stride times should produce non-zero asymmetry."""
        m_l = _make_metrics(1.0, 60.0, 0.65)
        m_r = _make_metrics(1.2, 50.0, 0.75)
        df_l, peaks_l, to_l = self._make_df_with_events(8)
        df_r, peaks_r, to_r = self._make_df_with_events(8)
        result = compute_bilateral_metrics(
            m_l, m_r, peaks_l, peaks_r, to_l, to_r, df_l, df_r
        )
        assert result["bilateral_stride_time_asymmetry_pct"] > 5.0

    def test_double_support_is_positive(self):
        """Double support time should be > 0 when stances overlap."""
        m = _make_metrics(1.0, 60.0, 0.65)
        df_l, peaks_l, to_l = self._make_df_with_events(8)
        # Shift right foot by half a stride so stances overlap
        offset = 50  # samples = 0.5 s
        t0 = pd.Timestamp("2025-01-01")
        total = len(df_l)
        times_r = pd.date_range(
            t0 + pd.Timedelta(milliseconds=500), periods=total, freq=pd.to_timedelta(0.01, unit="s")
        )
        df_r = pd.DataFrame({"_time": times_r, "S2": np.ones(total)})
        peaks_r = np.clip(peaks_l + offset, 0, total - 1)
        to_r = np.clip(to_l + offset, 0, total - 1)
        result = compute_bilateral_metrics(
            m, m, peaks_l, peaks_r, to_l, to_r, df_l, df_r
        )
        assert result["bilateral_double_support_mean_s"] > 0.0

    def test_missing_events_returns_unavailable(self):
        """If one foot has < 2 peaks, bilateral metrics should be NaN."""
        m = _make_metrics(1.0, 60.0, 0.65)
        df_l, peaks_l, to_l = self._make_df_with_events(8)
        df_r = df_l.copy()
        result = compute_bilateral_metrics(
            m, m,
            peaks_l, np.array([], dtype=int),
            to_l, np.array([], dtype=int),
            df_l, df_r,
        )
        # Double support requires peak timestamps — NaN when peaks are missing
        assert np.isnan(result["bilateral_double_support_mean_s"])
        assert np.isnan(result["bilateral_double_support_pct"])
        # Asymmetry is computed from metrics dicts (valid here) so it calculates
        assert result["bilateral_stride_time_asymmetry_pct"] == pytest.approx(0.0)


# ── 5. ProcessConfig ──────────────────────────────────────────────────────────

class TestProcessConfig:

    def test_default_values_match_config_yaml(self):
        """Default values must match the calibrated config.yaml values."""
        cfg = ProcessConfig()
        assert cfg.cutoff_pressure == pytest.approx(20.0)
        assert cfg.cutoff_gyro == pytest.approx(5.0)
        assert cfg.gyro_threshold == pytest.approx(150.0)
        assert cfg.min_peak_height == pytest.approx(0.4)
        assert cfg.fs == pytest.approx(100.0)

    def test_custom_values_accepted(self):
        cfg = ProcessConfig(fs=50.0, cutoff_pressure=10.0, gyro_threshold=200.0)
        assert cfg.fs == 50.0
        assert cfg.cutoff_pressure == 10.0
        assert cfg.gyro_threshold == 200.0

    def test_invalid_fs_raises(self):
        with pytest.raises(Exception):
            ProcessConfig(fs=-1.0)

    def test_invalid_edge_threshold_raises(self):
        with pytest.raises(Exception):
            ProcessConfig(edge_threshold=1.5)  # must be <= 0.95

    def test_toe_off_method_default(self):
        cfg = ProcessConfig()
        assert cfg.toe_off_method == "derivative"

class TestGpsAcceptanceLogic:

    def _make_gps_trial(self, corridor_lngs, turn_segments=None):
        df = _make_synthetic_s2(fs=100.0, n_strides=max(8, len(corridor_lngs)), stride_time_s=1.0)
        n = len(df)
        anchor_idx = np.linspace(0, n - 1, len(corridor_lngs)).astype(int)
        lat = np.interp(np.arange(n), anchor_idx, np.full(len(corridor_lngs), 40.0))
        lng = np.interp(np.arange(n), anchor_idx, np.asarray(corridor_lngs, dtype=float))
        df['lat'] = lat
        df['lng'] = lng
        df['Gx'] = 0.0
        df['Gy'] = 0.0
        df['Gz'] = 0.0
        if turn_segments:
            for start, end in turn_segments:
                start_idx = anchor_idx[start]
                end_idx = anchor_idx[min(end, len(anchor_idx) - 1)] + 1
                df.loc[start_idx:end_idx, 'Gz'] = 400.0
        return df

    def test_gps_rejects_implausible_path_span_ratio(self):
        proc = GaitDataProcessor(ProcessConfig(fs=100.0, min_peak_distance_s=0.5))
        df = self._make_gps_trial([-3.0, -2.9995, -2.9999, -2.9994, -2.9998, -2.9993])
        _df_out, metrics, *_ = proc.process_signals(
            df,
            test_type='6MWT',
            clinical_tests_cfg={},
            gps_estimation_cfg={
                'min_span_m': 100.0,
                'min_unique_points': 4,
                'max_path_span_ratio': 1.5,
                'min_turn_episodes': 1,
            },
            spatial_models_cfg={
                'gyro_norm': {'enabled': False},
                'biometric': {'enabled': False},
                'imu_zupt': {'enabled': False},
            },
        )
        assert metrics['spatial_method'] == 'none'

    def test_gps_indoor_corridor_requires_turn_evidence(self):
        proc = GaitDataProcessor(ProcessConfig(fs=100.0, min_peak_distance_s=0.5))
        lngs = [-3.0, -2.99985, -2.9997, -2.99985, -3.0, -2.99985, -2.9997, -2.99985, -3.0]
        df_no_turns = self._make_gps_trial(lngs, turn_segments=None)
        _df_out, metrics_no_turns, *_ = proc.process_signals(
            df_no_turns,
            test_type='6MWT',
            clinical_tests_cfg={},
            gps_estimation_cfg={
                'min_span_m': 100.0,
                'min_unique_points': 4,
                'max_path_span_ratio': 5.0,
                'min_turn_episodes': 1,
            },
            spatial_models_cfg={
                'gyro_norm': {'enabled': False},
                'biometric': {'enabled': False},
                'imu_zupt': {'enabled': False},
            },
        )
        assert metrics_no_turns['spatial_method'] == 'none'

        df_with_turns = self._make_gps_trial(lngs, turn_segments=[(2, 3), (6, 7)])
        _df_out, metrics_with_turns, *_ = proc.process_signals(
            df_with_turns,
            test_type='6MWT',
            clinical_tests_cfg={},
            gps_estimation_cfg={
                'min_span_m': 100.0,
                'min_unique_points': 4,
                'max_path_span_ratio': 5.0,
                'min_turn_episodes': 1,
            },
            spatial_models_cfg={
                'gyro_norm': {'enabled': False},
                'biometric': {'enabled': False},
                'imu_zupt': {'enabled': False},
            },
        )
        assert metrics_with_turns['spatial_method'] == 'gps'
        assert metrics_with_turns['spatial_distance_m'] >= 100.0


class TestSpatialFallback:

    def test_gyro_fallback_has_priority_over_biometric(self):
        proc = GaitDataProcessor(ProcessConfig())
        df = _make_synthetic_s2(n_strides=5)
        # Ensure the gyro-norm model has a non-zero swing integral.
        phase = np.linspace(0.0, 8.0 * np.pi, len(df))
        df["Gx"] = 120.0 * np.sin(phase)
        df["Gy"] = 40.0 * np.cos(phase)
        df["Gz"] = 15.0 * np.sin(phase / 2.0)
        _df_out, metrics, *_ = proc.process_signals(
            df,
            test_type="6MWT",
            clinical_tests_cfg={},
            gps_estimation_cfg={"min_span_m": 1000.0, "min_unique_points": 999},
            spatial_models_cfg={
                "gyro_norm": {"enabled": True, "K": 0.05},
                "biometric": {"enabled": True, "K": 0.2},
                "imu_zupt": {"enabled": False},
            },
        )
        assert metrics["spatial_method"] == "gyro_norm"
        assert metrics["walking_speed_mean_m_s"] == pytest.approx(metrics["gyro_norm_walking_speed_m_s"])
        assert metrics["stride_length_mean_m"] == pytest.approx(metrics["gyro_norm_stride_length_m"])
