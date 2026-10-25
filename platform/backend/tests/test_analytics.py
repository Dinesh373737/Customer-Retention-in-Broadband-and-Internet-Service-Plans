import json
import pytest
from app.services.analytics import overview, eda, segment_summary, apply_filters
from app.services.data_engine import validate_configuration
from test_data_engine import fixture_frame, fixture_config


def test_overview_and_dynamic_eda_are_actual_computed_data():
    frame = fixture_frame()
    config = validate_configuration(frame, fixture_config())
    result = overview(frame, config)
    assert result["churned"] == frame.Exited.eq("Left").sum()
    assert result["numeric_summaries"][0]["mean"] == pytest.approx(frame.fee.mean())
    plots = eda(frame, config, numeric="fee", categorical="plan", second="months")
    assert sum(row["count"] for row in plots["histogram"]) == len(frame)
    assert len(plots["scatter"]) == len(frame)
    assert sum(row["count"] for row in plots["categories"]) == len(frame)
    json.dumps(plots, allow_nan=False)


def test_filtered_single_class_and_empty_views_are_supported():
    frame = fixture_frame()
    config = validate_configuration(frame, fixture_config())
    result = overview(frame, config, {"Exited": ["Left"]})
    assert result["churn_rate"] == 1
    empty = eda(frame, config, filters={"plan": ["does not exist"]})
    assert empty["histogram"] == []
    assert empty["findings"]


def test_exploratory_without_target_has_no_churn_claim():
    frame, config = fixture_frame().drop(columns="Exited"), fixture_config()
    config.update(target_column=None, positive_label=None)
    config = validate_configuration(frame, config)
    result = overview(frame, config)
    assert result["churned"] is None and result["churn_rate"] is None
    assert eda(frame, config)["associations"] == []


def test_filters_and_unknown_selectors_validate_schema():
    frame = fixture_frame()
    config = validate_configuration(frame, fixture_config())
    with pytest.raises(ValueError, match="Filter column"):
        apply_filters(frame, {"bad": ["x"]})
    with pytest.raises(ValueError, match="numeric"):
        eda(frame, config, numeric="plan")
