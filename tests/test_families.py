# Description: Tests for the family registry, the declared seasonality, and
#   the family classes' declared structure, save and load (P11.04.02).
#   Covers spec_P11.04.02 section 6, including the P11.04.V-8 gate test.
# Usage: python -m pytest tests/test_families.py -v --tb=short
# Dependencies: pytest, numpy, pandas, its2s

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import _has_neuralprophet, declared

from its2s.frequency import SeriesFrequency
from its2s.models.families import FAMILIES, get_family, resolve_space
from its2s.seasonality import cycle_period, legacy_seasonality, make_seasonality


def _freq(alias):
    return SeriesFrequency.from_alias(alias)


def _series(freq, n, seed):
    """Seeded level + annual cycle + noise on a regular grid of n rows."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-05", periods=n, freq=freq)
    days = (dates - dates[0]).days.to_numpy()
    y = 100 + 10 * np.sin(2 * np.pi * days / 365.25) + rng.normal(0, 2, n)
    return pd.DataFrame({"ds": dates, "y": y})


# ===================================================================
# Registry (spec 4.1)
# ===================================================================
class TestRegistry:

    def test_exactly_three_families(self):
        assert sorted(FAMILIES) == ["arima", "neuralprophet", "prophet_xgb"]

    def test_unknown_family_raises_listing_the_three(self):
        with pytest.raises(ValueError, match="prophet_xgb.*arima.*neuralprophet"):
            get_family("lstm")

    def test_registry_imports_no_model_library(self):
        # A fresh interpreter, so libraries imported by other tests do not count.
        code = ("import sys, its2s.models.families; "
                "print(sorted(m for m in ('prophet', 'pmdarima', 'neuralprophet') "
                "if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, check=True).stdout.strip()
        assert out == "[]"

    def test_neuralprophet_is_withheld_naming_issue_53(self):
        spec = get_family("neuralprophet")
        assert spec.status == "withheld"
        assert spec.withheld_issue == 53
        assert spec.withheld_reason
        assert {f.status for n, f in FAMILIES.items() if n != "neuralprophet"} == {"available"}


# ===================================================================
# fixed= and bounds= (spec 4.3, contract 5.7)
# ===================================================================
class TestResolveSpace:

    @pytest.fixture
    def weekly_daily(self):
        return declared("D", "weekly")

    def test_fixed_removes_a_dimension(self, weekly_daily):
        rs = resolve_space(get_family("arima"), weekly_daily, fixed={"max_p": 3})
        assert "max_p" not in rs.sample
        assert rs.fixed == {"max_p": 3}
        assert any("max_p fixed" in f for f in rs.flags)

    def test_bounds_replace_a_range_and_may_widen_it(self, weekly_daily):
        rs = resolve_space(get_family("arima"), weekly_daily,
                           bounds={"max_p": (2, 8)})
        assert rs.sample["max_p"] == (2, 8, "int", "linear")

    @pytest.mark.parametrize("kwargs, match", [
        ({"fixed": {"n_layers": 2}}, "no dimension"),
        ({"fixed": {"max_p": 3}, "bounds": {"max_p": (2, 4)}}, "both fixed= and bounds="),
        ({"bounds": {"max_p": (4, 4)}}, "must be below"),
        ({"bounds": {"max_p": (2.5, 4)}}, "not an integer"),
        ({"fixed": {"max_p": 2.5}}, "not an integer"),
    ])
    def test_arima_raises(self, weekly_daily, kwargs, match):
        with pytest.raises(ValueError, match=match):
            resolve_space(get_family("arima"), weekly_daily, **kwargs)

    def test_nonpositive_bound_on_log_dimension_raises(self, weekly_daily):
        with pytest.raises(ValueError, match="log-scaled"):
            resolve_space(get_family("prophet_xgb"), weekly_daily,
                          bounds={"xgb__learning_rate": (0.0, 0.3)})

    def test_seasonal_only_dimensions_vanish_without_a_cycle(self):
        rs = resolve_space(get_family("arima"), declared("D"))
        assert not {"max_P", "max_D", "max_Q"} & set(rs.sample)
        rs = resolve_space(get_family("prophet_xgb"), declared("D"))
        assert "prophet__seasonality_prior_scale" not in rs.sample

    def test_naming_a_seasonal_only_dimension_without_a_cycle_raises(self):
        with pytest.raises(ValueError, match="seasonal component"):
            resolve_space(get_family("arima"), declared("D"), fixed={"max_P": 1})


# ===================================================================
# The declared seasonality (spec 4.2)
# ===================================================================
class TestSeasonality:

    @pytest.mark.parametrize("cycle, alias, m", [
        ("annual", "D", 365.25),
        ("annual", "W-SUN", 365.25 / 7),
        ("annual", "MS", 12.0),
        ("annual", "QS-JAN", 4.0),
        ("weekly", "D", 7.0),
    ])
    def test_period_arithmetic(self, cycle, alias, m):
        assert cycle_period(cycle, _freq(alias)) == pytest.approx(m)

    def test_weekly_on_weekly_raises(self):
        with pytest.raises(ValueError, match="more than two"):
            make_seasonality(["weekly"], _freq("W-SUN"))

    def test_unknown_cycle_raises(self):
        with pytest.raises(ValueError, match="Unknown cycles"):
            make_seasonality(["daily"], _freq("D"))

    def test_cycles_are_ordered_and_round_trip(self):
        s = make_seasonality(["annual", "weekly"], _freq("D"))
        assert s.cycles == ("weekly", "annual")
        assert type(s).from_dict(s.to_dict()) == s


# ===================================================================
# The old path's declaration: explicit ARIMA m (spec OQ-9)
# ===================================================================
class TestLegacyExplicitM:

    @pytest.mark.parametrize("alias, m, cycles", [
        ("D", 7, ("weekly",)),
        ("W-SUN", 52, ("annual",)),
        ("MS", 12, ("annual",)),
        ("QS-JAN", 4, ("annual",)),
        ("W-SUN", 1, ()),
    ])
    def test_explicit_m_maps_to_its_cycle(self, alias, m, cycles):
        df = _series(alias, 200, seed=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = legacy_seasonality("arima", {"m": m}, df, _freq(alias))
        assert s.cycles == cycles

    def test_non_calendar_m_raises(self):
        df = _series("W-SUN", 200, seed=1)
        with pytest.raises(ValueError, match="not the period of a declarable cycle"):
            legacy_seasonality("arima", {"m": 26}, df, _freq("W-SUN"))

    def test_daily_prophet_component_raises(self):
        df = _series("D", 200, seed=1)
        params = {"prophet": {"daily_seasonality": True}}
        with pytest.raises(ValueError, match="daily cycle is not declarable"):
            legacy_seasonality("prophet_xgb", params, df, _freq("D"))


# ===================================================================
# ARIMA carries only what it can honour (spec OQ-3)
# ===================================================================
class TestARIMADeclaration:

    def test_annual_on_daily_is_refused(self):
        from its2s.models.arima import ARIMAModel
        with pytest.raises(ValueError, match="prophet_xgb"):
            ARIMAModel(params={}, seasonality=declared("D", "annual"))

    def test_two_cycles_are_refused(self):
        from its2s.models.arima import ARIMAModel
        with pytest.raises(ValueError, match="one seasonal period"):
            ARIMAModel(params={}, seasonality=declared("D", "weekly", "annual"))


# ===================================================================
# P11.04.V-8: every fit of one declaration carries the declared components
# ===================================================================
# Short and long windows straddle the old per-fit rules: 400 days is under
# the 730-day yearly rule. ARIMA's short window is 600 days (85 weekly rows),
# under its old 2m = 104 guard but above the ~70 rows a seasonal m=52 fit
# needs at all; 400 days (57 rows) cannot be fit as declared (spec OQ-11).
_LONG_DAYS = 1100
_SHORT_DAYS = {"prophet_xgb": 400, "arima": 600}

_V8_CASES = [
    # (family, frequency, cycles, expected seasonal_components subset)
    ("prophet_xgb", "D", ("weekly", "annual"), {"prophet": ["weekly", "yearly"]}),
    ("prophet_xgb", "D", ("annual",), {"prophet": ["yearly"]}),
    ("prophet_xgb", "D", (), {"prophet": []}),
    ("arima", "W-SUN", ("annual",), {"m": 52, "seasonal_search": True}),
    ("arima", "W-SUN", (), {"m": 1, "seasonal_search": False}),
]


def _fit_window(family, alias, cycles, days, seed):
    from its2s.models.families import load_family_class
    rows = days if alias == "D" else days // 7
    df = _series(alias, rows, seed)
    params = {"max_p": 2, "max_q": 2, "max_d": 1} if family == "arima" else {}
    model = load_family_class(family)(params=params,
                                      seasonality=declared(alias, *cycles))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df)
    return model.seasonal_components()


@pytest.mark.parametrize("family, alias, cycles, expected", _V8_CASES,
                         ids=[f"{c[0]}-{'+'.join(c[2]) or 'none'}" for c in _V8_CASES])
def test_v8_declared_components_on_every_window(family, alias, cycles, expected):
    short = _fit_window(family, alias, cycles, _SHORT_DAYS[family], seed=8)
    long = _fit_window(family, alias, cycles, _LONG_DAYS, seed=9)
    for key, value in expected.items():
        assert short[key] == value
        assert long[key] == value
    # The fitted seasonal order is an estimate (OQ-5): recorded, not held equal.
    short.pop("fitted_seasonal_order", None)
    long.pop("fitted_seasonal_order", None)
    assert short == long


# ===================================================================
# Save and load (spec 4.5)
# ===================================================================
class TestSaveLoad:

    @pytest.mark.parametrize("family, alias, cycles", [
        ("prophet_xgb", "D", ("weekly",)),
        ("arima", "D", ("weekly",)),
    ])
    def test_round_trip_predicts_bit_identically(self, tmp_path, family, alias, cycles):
        from its2s.models.families import load_family_class
        cls = load_family_class(family)
        df = _series(alias, 200, seed=3)
        params = {"max_p": 2, "max_q": 2, "max_d": 1} if family == "arima" else {}
        model = cls(params=params, seasonality=declared(alias, *cycles))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df.iloc[:170])
            before = model.predict(df.iloc[170:]).predicted
            model.save(tmp_path / family)
            loaded = cls.load(tmp_path / family)
            after = loaded.predict(df.iloc[170:]).predicted
        np.testing.assert_array_equal(before, after)
        assert loaded.seasonality == model.seasonality
        assert loaded.params == model.params

    def test_loading_another_familys_save_raises(self, tmp_path):
        from its2s.models.arima import ARIMAModel
        from its2s.models.prophet_xgb import ProphetXGBHybridModel
        df = _series("D", 120, seed=4)
        model = ARIMAModel(params={"max_p": 2, "max_q": 2, "max_d": 1},
                           seasonality=declared("D"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        model.save(tmp_path)
        with pytest.raises(ValueError, match="holds a 'arima' model"):
            ProphetXGBHybridModel.load(tmp_path)

    def test_saving_an_unfitted_model_raises(self, tmp_path):
        from its2s.models.arima import ARIMAModel
        with pytest.raises(RuntimeError, match="fit the model"):
            ARIMAModel(params={}, seasonality=declared("D")).save(tmp_path)

    @pytest.mark.skipif(not _has_neuralprophet(), reason="neuralprophet not installed")
    def test_neuralprophet_save_raises_naming_issue_53(self, tmp_path):
        from its2s.models.neuralprophet import NeuralProphetModel
        model = NeuralProphetModel(params={}, seasonality=declared("D"))
        with pytest.raises(NotImplementedError, match="issue 53"):
            model.save(tmp_path)


# ===================================================================
# The YAML channel is gone (spec 4.6)
# ===================================================================
class TestYamlRetired:

    def test_no_params_yaml_ships(self):
        import its2s
        assert not (Path(its2s.__file__).parent / "params.yaml").exists()

    def test_load_config_rejects_config_path(self):
        from its2s.settings import load_config
        with pytest.raises(TypeError):
            load_config(config_path="params.yaml")

    @pytest.mark.parametrize("name", ["run_single_its", "tune_model",
                                      "time_series_cv", "run_batch"])
    def test_public_calls_reject_config_path(self, name):
        import its2s
        with pytest.raises(TypeError, match="config_path"):
            getattr(its2s, name)(None, config_path="params.yaml")
