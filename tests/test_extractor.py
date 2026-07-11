import pandas as pd

from gait_analysis.extractor import GaitDataExtractor


def test_coerce_query_result_accepts_dataframe():
    df = pd.DataFrame({"_time": [1], "S2": [0.5], "Foot": ["Left"]})
    result = GaitDataExtractor._coerce_query_result_to_dataframe(df)
    assert result.equals(df)


def test_coerce_query_result_concatenates_dataframe_list():
    left = pd.DataFrame({"_time": [1], "S2": [0.5], "Foot": ["Left"]})
    right = pd.DataFrame({"_time": [2], "S2": [0.7], "Foot": ["Right"]})
    result = GaitDataExtractor._coerce_query_result_to_dataframe([left, right])
    assert len(result) == 2
    assert set(result["Foot"]) == {"Left", "Right"}


def test_build_event_query_uses_curated_window_and_codeid():
    extractor = GaitDataExtractor.__new__(GaitDataExtractor)
    extractor._bucket = "test-bucket"
    row = pd.Series(
        {
            "id": 66,
            "codeid": "EPSHUG067-10",
            "t_code": "6MWT",
            "d_from": "2026-01-22T11:00:00Z",
            "d_until": "2026-01-22T11:06:00Z",
        }
    )
    query, p_id, event_id, test_type, start_dt = extractor._build_event_query(row)
    assert 'from(bucket: "test-bucket")' in query
    assert 'r["CodeID"] == "EPSHUG067-10"' in query
    assert p_id == "EPSHUG067-10"
    assert event_id == 66
    assert test_type == "6MWT"
    assert str(start_dt) == "2026-01-22 11:00:00+00:00"
