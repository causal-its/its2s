# Description: Tests for the two conveniences around step 0: Design.starter,
#   the one-call first-run design, and explore(), the optional look at the
#   series before declaring, with its boundary. Synthetic data only. Nothing
#   here fits a forecasting model.
# Usage: python -m pytest tests/test_starter_explore.py -v --tb=short
# Dependencies: pytest, numpy, pandas, matplotlib, statsmodels, scipy, its2s

import json
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

import its2s
from its2s import Design, DesignWarning, declare_design, explore
from its2s.design import resolve
from its2s.design_record import RECORD_DIR, RECORD_FILE
from its2s.explore import CAUTIONS, PLOT_FILE, SUMMARY_FILE, Exploration

from test_design import EVENT, _contract_example, _design, _frame

BOUNDARY = "2025-06-16"


def _explore(df=None, **kw):
    df = _frame() if df is None else df
    return explore(df, "week", "deaths", before=kw.pop("before", BOUNDARY), **kw)


def _seasonal_frame(periods=260, amplitude=0.3, seed=0):
    dates = pd.date_range("2020-01-06", periods=periods, freq="W-MON")
    t = np.arange(periods)
    lam = 100 * (1 + amplitude * np.sin(2 * np.pi * t / (365.25 / 7)))
    y = np.random.default_rng(seed).poisson(lam).astype(float)
    return pd.DataFrame({"week": dates, "deaths": y})


def _starter(tmp, label="first_run"):
    return Design.starter(data=_frame(), date_col="week", outcome_col="deaths",
                          event_date=EVENT, horizon=3, seed=42, output_dir=str(tmp),
                          series_label=label)


def _plain(tmp, label="first_run"):
    d = Design(_frame(), "week", "deaths", None, EVENT, 3, 42, str(tmp), label)
    d.add_split("sp1", train_frac=0.80, primary=True)
    d.add_cv("cv1", n_folds=3, fold_eval_obs=3)
    d.add_family("prophet_xgb", n_trials=100)
    return d


def _seal(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DesignWarning)
        return declare_design(d)


# ---------------------------------------------------------------------------
# Design.starter
# ---------------------------------------------------------------------------

class TestStarter:

    def test_record_equals_the_plain_design_bar_starter_used(self, tmp_path, capsys):
        # Same output path for both, so every field but the stamps can match:
        # the plain one is sealed, removed, and the starter sealed in its place.
        plain = _seal(_plain(tmp_path))
        path = tmp_path / "first_run" / RECORD_DIR / RECORD_FILE
        a = json.loads(path.read_text())
        shutil.rmtree(tmp_path / "first_run")
        _seal(_starter(tmp_path))
        b = json.loads(path.read_text())
        assert plain.starter_used is False
        assert (a.pop("starter_used"), b.pop("starter_used")) == (False, True)
        for r in (a, b):
            r.pop("declared_at")
            r.pop("seal")
        assert a == b

    def test_prints_the_three_calls(self, tmp_path, capsys):
        _starter(tmp_path)
        out = capsys.readouterr().out.splitlines()
        assert out[1:] == [
            "  design.add_split('sp1', train_frac=0.8, primary=True)",
            "  design.add_cv('cv1', n_folds=3, fold_eval_obs=3)",
            "  design.add_family('prophet_xgb', n_trials=100)",
        ]

    def test_fold_eval_obs_is_H(self, tmp_path):
        d = Design.starter(data=_frame(), date_col="week", outcome_col="deaths",
                           event_date=EVENT, horizon=5, seed=42,
                           output_dir=str(tmp_path), series_label="x")
        assert d._cvs["cv1"]["fold_eval_obs"] == 5

    def test_report_says_starter_used(self, tmp_path, capsys):
        d = _starter(tmp_path)
        capsys.readouterr()
        d.preview()
        assert "  starter      used" in capsys.readouterr().out

    def test_is_validated_and_editable_like_any_design(self, tmp_path):
        d = _starter(tmp_path)
        d.remove_family("prophet_xgb")
        d.add_family("arima", n_trials=20)
        resolved, findings = resolve(d)
        assert list(resolved["families"]) == ["arima"]
        assert resolved["starter_used"] is True
        assert [f for f in findings if f.level == "error"] == []

    def test_covariates_pass_through(self, tmp_path):
        df = _frame()
        df["temp"] = np.linspace(0, 1, len(df))
        d = Design.starter(data=df, date_col="week", outcome_col="deaths",
                           event_date=EVENT, horizon=3, seed=42,
                           output_dir=str(tmp_path), series_label="x",
                           covariate_cols=["temp"])
        assert d.covariate_cols == ["temp"]


# ---------------------------------------------------------------------------
# explore: what it reads
# ---------------------------------------------------------------------------

class TestExploreBoundary:

    def test_reads_only_rows_strictly_before(self):
        x = _explore()
        df = _frame()
        before = df[df["week"] < pd.Timestamp(BOUNDARY)]
        assert x.rows_read == len(before) == 128
        assert x.last_date == before["week"].iloc[-1] < x.boundary
        assert x.first_date == df["week"].iloc[0]

    def test_values_after_the_boundary_change_nothing(self):
        df = _frame()
        poisoned = df.copy()
        poisoned.loc[poisoned["week"] >= pd.Timestamp(BOUNDARY), "deaths"] = np.nan
        a, b = _explore(df), _explore(poisoned)
        assert a.to_dict() == b.to_dict()

    def test_rows_after_the_boundary_may_be_irregular(self):
        df = _frame()
        df = df.drop(df.index[df["week"] >= pd.Timestamp(BOUNDARY)][5:9])
        assert _explore(df).rows_read == 128

    def test_unsorted_frame_is_read_in_date_order(self):
        df = _frame()
        assert _explore(df.sample(frac=1, random_state=0)).to_dict() == _explore(df).to_dict()

    def test_missing_value_before_the_boundary_is_an_error(self):
        df = _frame()
        df.loc[10, "deaths"] = np.nan
        with pytest.raises(ValueError, match="1 of the 128 rows"):
            _explore(df)

    def test_boundary_must_be_a_date_of_the_series(self):
        # A Wednesday between two Monday rows: the Monday row before it may
        # hold days after it, so it is refused, naming the rows either side.
        with pytest.raises(ValueError, match="between the grid dates 2025-06-16 and 2025-06-23"):
            _explore(before="2025-06-18")
        with pytest.raises(ValueError, match="before the first date"):
            _explore(before="2020-01-06")
        with pytest.raises(ValueError, match="after the last date"):
            _explore(before="2030-01-07")

    def test_too_few_rows_before_the_boundary(self):
        with pytest.raises(ValueError, match="2 rows"):
            _explore(before="2023-01-16")

    def test_bad_arguments(self):
        with pytest.raises(ValueError, match="not a column"):
            explore(_frame(), "week", "nope", before=BOUNDARY)
        with pytest.raises(ValueError, match="transform"):
            _explore(transform="boxcox")
        with pytest.raises(ValueError, match="Unknown cycle"):
            _explore(cycles=["monthly"])
        with pytest.raises(TypeError, match="string"):
            _explore(cycles="annual")
        df = _frame()
        df.loc[3, "deaths"] = 0.0
        with pytest.raises(ValueError, match="positive"):
            _explore(df, transform="log")


# ---------------------------------------------------------------------------
# explore: what it computes and writes
# ---------------------------------------------------------------------------

class TestExploreOutputs:

    def test_numbers_for_the_annual_cycle(self):
        x = _explore()
        a = x.numbers["cycles"]["annual"]
        assert a["m"] == pytest.approx(365.25 / 7, abs=1e-4)
        assert a["acf_lag"] == 52
        assert len(a["periodogram_bins"]) == 3
        assert x.numbers["stl"]["period"] == 52

    def test_finds_a_real_annual_cycle(self):
        x = _explore(_seasonal_frame(), before="2024-12-23")
        a = x.numbers["cycles"]["annual"]
        assert a["power_rank"] == 1
        assert a["acf"] > x.numbers["acf_band"]

    def test_shuffled_reference_beside_the_strength(self):
        # On a real cycle the strength clears its shuffled copy; on noise of
        # the same length it does not, which is what the reference is for.
        s = _explore(_seasonal_frame(), before="2024-12-23").numbers["stl"]
        assert s["seasonal_strength"] > s["shuffled_strength"] + 0.2
        noise = _seasonal_frame(amplitude=0.0)
        n = _explore(noise, before="2024-12-23").numbers["stl"]
        assert abs(n["seasonal_strength"] - n["shuffled_strength"]) < 0.2
        assert n["shuffle_seed"] == s["shuffle_seed"] == 0

    def test_deterministic(self):
        assert _explore().to_dict() == _explore().to_dict()

    def test_stl_skipped_under_two_periods(self):
        x = _explore(before="2024-10-07")
        assert x.rows_read < 104
        assert "fewer than two periods" in x.numbers["stl"]["skipped"]

    def test_no_cycles(self):
        x = _explore(cycles=None)
        assert x.cycles == () and x.numbers["cycles"] == {} and x.numbers["stl"] is None

    def test_prints_numbers_and_cautions(self, capsys):
        _explore()
        out = capsys.readouterr().out
        assert "rows read    128" in out
        for c in CAUTIONS:
            assert c in out

    def test_writes_nothing_by_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _explore()
        assert list(tmp_path.iterdir()) == []

    def test_save_to(self, tmp_path):
        x = _explore(save_to=tmp_path / "look")
        saved = json.loads((tmp_path / "look" / SUMMARY_FILE).read_text())
        assert saved == json.loads(json.dumps(x.to_dict()))
        assert (tmp_path / "look" / PLOT_FILE).stat().st_size > 0

    def test_figure_has_four_panels(self):
        assert len(_explore().figure.axes) == 4

    def test_public_name(self):
        assert its2s.explore is explore and "explore" in its2s.__all__


# ---------------------------------------------------------------------------
# explore and the design
# ---------------------------------------------------------------------------

class TestExplorationInTheDesign:

    def test_boundary_and_rows_stamped_in_the_record(self, tmp_path):
        d = _contract_example(_design(tmp=str(tmp_path)))
        x = _explore()
        d.set_exploration(x)
        _seal(d)
        path = tmp_path / "euromomo_heat2026" / RECORD_DIR / RECORD_FILE
        stamped = json.loads(path.read_text())["exploration"]
        assert stamped == {"boundary": BOUNDARY, "rows_read": 128,
                           "first_date": "2023-01-02", "last_date": "2025-06-09",
                           "transform": "sqrt", "cycles": ["annual"],
                           "fingerprint": json.loads(json.dumps(x.fingerprint))}

    def test_split_ending_before_the_boundary_warns(self):
        # Both splits' training windows end before 2025-12-01, so a look up to
        # that date read rows of both test windows: a warning each, no error.
        d = _contract_example(_design())
        d.set_exploration(_explore(before="2025-12-01"))
        _, findings = resolve(d)
        hits = [f for f in findings if f.title == "exploration"]
        assert all(f.level == "warning" for f in hits)
        assert sorted(f.message.split("'")[0] for f in hits) == ["sp1", "sp2"]

    def test_boundary_at_the_first_test_date_does_not_warn(self):
        d = _contract_example(_design())
        resolved, _ = resolve(d)
        first_test = resolved["splits"]["sp1"]["test"]["start"]
        d.set_exploration(_explore(before=first_test))
        _, findings = resolve(d)
        assert not [f for f in findings if f.title == "exploration"
                    and f.message.startswith("sp1")]

    def test_same_rows_do_not_warn(self):
        d = _contract_example(_design())
        d.set_exploration(_explore())
        _, findings = resolve(d)
        assert [f for f in findings if f.title == "exploration"] == []

    def test_other_rows_warn(self):
        # One outcome value before the boundary differs between the frame
        # explored and the frame declared.
        other = _frame()
        other.loc[20, "deaths"] += 1
        d = _contract_example(_design())
        d.set_exploration(_explore(other))
        _, findings = resolve(d)
        hits = [f for f in findings if f.title == "exploration"]
        assert len(hits) == 1 and hits[0].level == "warning"
        assert "not this design's 128 rows" in hits[0].message

    def test_rows_after_the_boundary_may_differ(self):
        other = _frame()
        other.loc[190, "deaths"] += 1
        d = _contract_example(_design())
        d.set_exploration(_explore(other))
        _, findings = resolve(d)
        assert [f for f in findings if f.title == "exploration"] == []

    def test_set_exploration_takes_an_exploration(self):
        with pytest.raises(TypeError):
            _design().set_exploration("2025-06-16")
        assert isinstance(_explore(), Exploration)
