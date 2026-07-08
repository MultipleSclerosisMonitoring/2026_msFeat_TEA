import pandas as pd

from gait_analysis.postgresql import EventSelector, build_event_query, extract_result_payload, normalize_event_rows


def test_build_event_query_with_all_filters():
    selector = EventSelector(
        ids=[10, 20],
        codeids=["P001"],
        test_types=["6MWT"],
        date_from=pd.Timestamp("2025-11-01", tz="UTC").to_pydatetime(),
        date_to=pd.Timestamp("2025-11-30", tz="UTC").to_pydatetime(),
    )
    query, params = build_event_query(selector, '"public"."healthywear_event"')

    assert 'FROM "public"."healthywear_event"' in query
    assert 'id = ANY(%s)' in query
    assert 'codeid = ANY(%s)' in query
    assert 't_code = ANY(%s)' in query
    assert 'd_until >= %s' in query
    assert 'd_from <= %s' in query
    assert params[0] == [10, 20]
    assert params[1] == ["P001"]
    assert params[2] == ["6MWT"]


def test_normalize_event_rows_parses_expected_columns():
    df = pd.DataFrame([
        {
            "id": 1,
            "codeid": "P001",
            "t_code": "TUG",
            "d_from": "2025-11-28T12:43:14Z",
            "d_until": "2025-11-28T12:45:14Z",
        }
    ])
    result = normalize_event_rows(df)
    assert list(result.columns) == ["id", "codeid", "t_code", "d_from", "d_until"]
    assert str(result["d_from"].dtype).startswith("datetime64")


def test_extract_result_payload_normalizes_json_and_missing_values():
    payload = extract_result_payload(
        {
            "healthywear_event_id": 99,
            "analysis_h5_key": "p_P001/6MWT/start_2025-11-28T12-46-09Z/Left",
            "foot": "Left",
            "pipeline_version": "0.2.0",
            "posicion_gps": {"lat": 40.4, "lng": -3.7},
            "stride_time_mean_s": float("nan"),
            "bilateral_available": False,
        }
    )
    assert payload["healthywear_event_id"] == 99
    assert payload["foot"] == "Left"
    assert payload["pipeline_version"] == "0.2.0"
    assert "bilateral_step_time_lr_mean_s" not in payload
    assert payload["posicion_gps"] == '{"lat": 40.4, "lng": -3.7}'
    assert payload["stride_time_mean_s"] is None
    assert payload["bilateral_available"] is False
