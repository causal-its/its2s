# Description: Step 4's computation (its2s.estimate.estimate) and its public
#   call (its2s.estimate_excess): every argument is checked and refused before
#   any bootstrap, an existing run_label is refused before the data is read,
#   the saved model must reproduce the fitted values step 3 sealed, failed
#   replications are removed and counted, period totals and percent excess are
#   formed within each path, and the sealed record round-trips and names the
#   four records it descends from; the report, both CSVs and the figure are
#   written beside the record and agree with it. Synthetic data; stand-in
#   families except where a real family's in-sample fit is the point.
# Usage: pytest tests/test_estimate_excess.py
# Dependencies: pytest, numpy, pandas

import contextlib
import io
import json
import os
import stat
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import its2s.check as check_module
import its2s.estimate as estimate_module
import its2s.final_fit as final_fit
import its2s.fit_record as fit_record
import its2s.screening as screening
from its2s import (CandidatesRecord, CheckRecord, Design, DesignRecord, DesignWarning,
                   EstimateError, EstimateRecord, EstimateWarning, FitRecord, SealError,
                   declare_design, estimate_excess)
from its2s.bootstrap.mbb import Replication, replicate, replication_seeds
from its2s.estimate import estimate
from its2s.estimate_record import RECORD_DIR, RECORD_FILE, REPORT_FILE
from its2s.estimate_tables import (COUNTERFACTUAL_FILE, EXCESS_TABLE_FILE, bound_columns,
                                   level_percent, to_csv_text)
from its2s.outputs.estimate_plot import DEFAULT_STYLE, PLOT_FILE
from its2s.metrics.excess import (PCT_FORECAST_NOT_POSITIVE, PCT_PATH_NOT_POSITIVE,
                                  excess_from_paths)
from its2s.records import seal_record

from test_check_selected_model import SP1, _check
from test_design import EVENT, _frame
from test_fit_final_model import _fit, _saving_family
from test_screen_models import _quiet

# Six post-event weeks, split into two declared sub-periods.
H = 6
SUB_PERIODS = (("early", "2026-06-22", "2026-07-06"), ("late", "2026-07-13", "2026-07-27"))
FIT_ROWS = 181
N = 30


def _step4_family(**kw):
    """A stand-in family with everything step 4 calls: a fresh clone for each
    refit, no start-up rows, and in-sample fitted values on a fitted or
    loaded model."""
    base = _saving_family(**kw)

    class Step4(base):
        def clone_fresh(self):
            return type(self)(params=self.params, seasonality=self.seasonality)

        def startup_rows(self):
            return 0

        def in_sample(self, fit_df, target_col="y", date_col="ds", covariate_cols=None):
            return np.full(len(fit_df), self.level)
    return Step4


@pytest.fixture
def family(monkeypatch):
    cls = _step4_family()
    for module in (screening, check_module, final_fit, fit_record):
        monkeypatch.setattr(module, "load_family_class", lambda name: cls)
    return cls


def _declare(tmp, label="heat2026", output_dir=None, sub_periods=SUB_PERIODS,
             split=(("train_obs", 129),)):
    d = Design(_frame(), "week", "deaths", None, EVENT, H, 42,
               str(tmp) if output_dir is None else output_dir, label)
    d.set_seasonality(cycles=["annual"])
    d.add_split("sp1", test_obs=52, primary=True, **dict(split))
    d.add_cv("cv1", n_folds=3, fold_eval_obs=6)
    d.add_family("prophet_xgb", n_trials=3)
    for name, start, end in sub_periods:
        d.add_sub_period(name, start, end)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", DesignWarning)
        return declare_design(d)


def _chain(tmp, **kw):
    """declare -> screen -> check -> fit, with the stand-in family."""
    return _fit(_check(_quiet(_declare(tmp, **kw), data=_frame()), SP1, data=_frame()),
                data=_frame())


@pytest.fixture
def fitted(tmp_path, family):
    return _chain(tmp_path)


def _est(fit, **kw):
    kw.setdefault("n_replications", N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EstimateWarning)
        return estimate_excess(fit, data=kw.pop("data", _frame()), **kw)


def _estimate_dir(fit):
    return os.path.join(os.path.dirname(os.path.dirname(fit.path)), RECORD_DIR)


def _files(directory):
    out = {}
    for root, _dirs, names in os.walk(directory):
        for n in names:
            p = os.path.join(root, n)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, directory)] = (fh.read(), os.stat(p).st_mtime_ns)
    return out


def _strip(record, *keys):
    """The record without its timestamps, seal, run_label and the keys given."""
    drop = {"estimated_at", "sealed_at", "seal", "run_label", *keys}
    return {k: v for k, v in record.to_dict().items() if k not in drop}


@pytest.fixture
def no_bootstrap(monkeypatch):
    """Any replication raises, so a test can prove none is reached."""
    def refuse(*args, **kw):
        raise AssertionError("a bootstrap replication was reached")
    monkeypatch.setattr(estimate_module, "replicate", refuse)


class Sentinel:
    """Data that raises on any access, to prove it was never read."""

    def __getattr__(self, name):
        raise AssertionError(f"the data was read ({name})")

    def __getitem__(self, key):
        raise AssertionError("the data was read")


# ---------------------------------------------------------------------------
# Refused before any bootstrap; nothing written
# ---------------------------------------------------------------------------

BAD_ARGUMENTS = [
    ({"sub_periods": ["middle"]}, "not declared at step 0"),
    ({"sub_periods": "early"}, "sub_periods must be"),
    ({"interval_type": "credible"}, "interval_type must be"),
    ({"level": 0.0}, "level must be"),
    ({"level": 1.0}, "level must be"),
    ({"level": "0.9"}, "level must be"),
    ({"level": True}, "level must be"),
    ({"n_replications": 0}, "n_replications must be"),
    ({"n_replications": 2.5}, "n_replications must be"),
    ({"n_replications": True}, "n_replications must be"),
    ({"block_length": 0}, "block_length must be"),
    ({"block_length": "14"}, "block_length must be"),
    ({"block_length": FIT_ROWS}, "shorter than the residual pool"),
    ({"style": {"colour": {}}}, "unknown key 'colour'"),
    ({"style": {"colors": {"observed": 3}}}, "must be a colour string"),
    ({"style": {"fonts": {"title": -1}}}, "must be a positive size"),
    ({"style": {"figsize": (8,)}}, "figsize must be two positive numbers"),
    ({"style": {"dpi": 1.5}}, "dpi must be a positive integer"),
    ({"run_label": ""}, "run_label must be"),
    ({"run_label": "a/b"}, "run_label must be"),
    ({"run_label": ".."}, "run_label must be"),
]


class TestRefusals:

    @pytest.mark.parametrize("kw,message", BAD_ARGUMENTS)
    def test_each_bad_argument_is_refused_before_any_bootstrap(self, fitted, no_bootstrap,
                                                                kw, message):
        with pytest.raises(EstimateError, match=message):
            _est(fitted, **kw)
        assert not os.path.exists(_estimate_dir(fitted))

    def test_a_dict_is_a_type_error_naming_the_loader(self, fitted, no_bootstrap):
        with pytest.raises(TypeError, match="FitRecord.load"):
            estimate_excess(fitted.to_dict(), data=_frame())

    def test_positional_arguments_after_data_are_a_type_error(self, fitted, no_bootstrap):
        with pytest.raises(TypeError):
            estimate_excess(fitted, _frame(), None)

    def test_other_data_is_refused_naming_step_4(self, fitted, no_bootstrap):
        df = _frame()
        df.loc[184, "deaths"] += 1  # a post-event week, inside the fingerprint
        with pytest.raises(EstimateError, match="fingerprint.*Nothing has been estimated"):
            _est(fitted, data=df)
        assert not os.path.exists(_estimate_dir(fitted))

    def test_a_withheld_family_is_refused(self, fitted, no_bootstrap):
        record = fitted.to_dict()
        record["family"] = "neuralprophet"
        record["handoff"]["candidate"]["family"] = "neuralprophet"
        forged = FitRecord.from_dict(seal_record(record), path=fitted.path)
        with pytest.raises(EstimateError, match="neuralprophet is not supported"):
            _est(forged)
        assert not os.path.exists(_estimate_dir(fitted))

    def test_a_model_off_by_one_ulp_is_refused_naming_library_versions(
            self, fitted, family, no_bootstrap, monkeypatch):
        def moved(self, fit_df, **kw):
            v = np.full(len(fit_df), self.level)
            v[7] = np.nextafter(v[7], np.inf)
            return v
        monkeypatch.setattr(family, "in_sample", moved)
        with pytest.raises(EstimateError, match="does not reproduce.*Fit with:.*Installed:"):
            _est(fitted)
        assert not os.path.exists(_estimate_dir(fitted))

    def test_a_changed_saved_model_is_refused_on_load(self, fitted, no_bootstrap):
        loaded = FitRecord.load(fitted.path)
        level = os.path.join(loaded.model_dir(), "level.json")
        os.chmod(level, os.stat(level).st_mode | stat.S_IWUSR)
        with open(level, "w") as fh:
            json.dump({"level": 1.0}, fh)
        with pytest.raises(SealError, match="does not match the fit record"):
            _est(loaded)

    def test_level_with_few_replications_beyond_a_bound_warns(self, fitted):
        with pytest.warns(EstimateWarning, match="fewer than 5"):
            estimate_excess(fitted, data=_frame(), n_replications=N)


# ---------------------------------------------------------------------------
# run_label
# ---------------------------------------------------------------------------

class TestRunLabel:

    def test_an_existing_run_label_is_refused_before_the_data_is_read(self, fitted):
        _est(fitted)
        with pytest.raises(EstimateError) as e:
            _est(fitted, data=Sentinel())
        today = f"{date.today():%Y-%m-%d}"
        assert f"estimated on {today}" in str(e.value)
        assert f"'main_{date.today():%Y%m%d}'" in str(e.value)

    def test_a_second_run_label_leaves_the_first_untouched(self, fitted):
        _est(fitted)
        first = _files(os.path.join(_estimate_dir(fitted), "main"))
        second = _est(fitted, run_label="ci90", interval_type="confidence", level=0.9)
        assert second.run_label == "ci90"
        assert _files(os.path.join(_estimate_dir(fitted), "main")) == first

    def test_relative_output_path_and_another_working_directory(self, tmp_path, family,
                                                                monkeypatch):
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        fit = _chain(tmp_path, output_dir="results")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        est = _est(FitRecord.load(str(work / "results" / "heat2026")))
        assert est.path == str(work / "results" / "heat2026" / RECORD_DIR / "main"
                               / RECORD_FILE)
        assert not os.listdir(elsewhere)


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------

def _settings(**kw):
    out = {"interval_type": "prediction", "level": 0.95, "n_replications": N,
           "block_length": 14, "block_rule": "fixed-14-observations",
           "block_argument": "auto", "periods": ["full_window", "early", "late"]}
    out.update(kw)
    return out


def _inputs(fit):
    model = fit.model()
    return model, np.full(FIT_ROWS, model.level)


class TestComputation:

    def test_the_handoff_alone_gives_the_full_body(self, fitted):
        est = _est(fitted)
        model, values = _inputs(fitted)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EstimateWarning)
            body = estimate(fitted.to_dict()["handoff"], _frame(), model, values,
                            _settings())
        record = est.to_dict()
        for key, value in body.items():
            assert record[key] == json.loads(json.dumps(value)), key

    def test_two_identical_runs_are_identical_but_for_label_and_time(self, fitted):
        a = _est(fitted, run_label="a")
        b = _est(fitted, run_label="b")
        assert _strip(a) == _strip(b)

    def test_a_replication_depends_only_on_its_own_seed(self, fitted):
        model, values = _inputs(fitted)
        h = fitted.to_dict()["handoff"]
        fit_df = estimate_module.fit_rows(h, _frame())
        target = estimate_module.target_rows(h, _frame())
        cols = estimate_module.ReplicateColumns("deaths", "week", None)
        short, _ = estimate_module._run_replications(
            model, fit_df, target, values, cols, _settings(n_replications=5),
            h["bootstrap_seed"], 0)
        long_, _ = estimate_module._run_replications(
            model, fit_df, target, values, cols, _settings(n_replications=12),
            h["bootstrap_seed"], 0)
        assert replication_seeds(7, 12)[:5] == replication_seeds(7, 5)
        for s, lo in zip(short, long_):
            np.testing.assert_array_equal(s.path, lo.path)

    def test_noise_is_recorded_for_a_prediction_run_only(self, fitted):
        p = _est(fitted, run_label="p").bootstrap
        c = _est(fitted, run_label="c", interval_type="confidence").bootstrap
        assert dict(p["noise"]) == {"pool": estimate_module.NOISE_POOL, "n_pool": FIT_ROWS,
                                    "block_length": 14, "propagation": "none"}
        assert c["noise"] is None

    def test_a_poisoned_fit_result_changes_nothing(self, tmp_path, family):
        clean = _est(_chain(tmp_path / "a"))
        fit = _chain(tmp_path / "b")
        fit.model()._fit_result = "garbage"
        assert _strip(_est(fit), "provenance") == _strip(clean, "provenance")


# ---------------------------------------------------------------------------
# Failed replications
# ---------------------------------------------------------------------------

class _Broken:
    """A model whose refit fails in a chosen way."""

    def __init__(self, model, how):
        self.model, self.how = model, how

    def clone_fresh(self):
        if self.how == "exception":
            raise ZeroDivisionError("injected")
        fresh = self.model.clone_fresh()
        how = self.how

        class Wrapped:
            def fit(self, *a, **kw):
                return fresh.fit(*a, **kw)

            def predict(self, *a, **kw):
                pred = fresh.predict(*a, **kw)
                if how == "wrong_length":
                    pred.predicted = pred.predicted[:-1]
                else:
                    pred.predicted = np.full(len(pred.predicted), np.nan)
                return pred
        return Wrapped()


def _failing_at(monkeypatch, failing, captured):
    """Replications in failing fail as named; every outcome is captured."""
    def wrapped(r, model, *args, **kw):
        if r in failing:
            model = _Broken(model, failing[r])
        outcome = replicate(r, model, *args, **kw)
        captured.append(outcome)
        return outcome
    monkeypatch.setattr(estimate_module, "replicate", wrapped)


class TestFailures:

    FAILING = {2: "exception", 5: "wrong_length", 11: "non_finite", 17: "exception"}

    def test_failures_are_removed_counted_and_intervals_use_survivors(
            self, fitted, monkeypatch):
        captured = []
        _failing_at(monkeypatch, self.FAILING, captured)
        with pytest.warns(EstimateWarning, match="4 of 30 bootstrap replications failed"):
            est = estimate_excess(fitted, data=_frame(), n_replications=N)
        b = est.bootstrap
        assert b["n_succeeded"] == N - 4
        assert b["failures"]["exception"] == {"ZeroDivisionError": 2}
        assert b["failures"]["wrong_length"] == 1 and b["failures"]["non_finite"] == 1

        survivors = np.column_stack([o.path for o in captured if o.ok])
        h = fitted.to_dict()["handoff"]
        target = estimate_module.target_rows(h, _frame())
        dates = target["week"].reset_index(drop=True)
        masks = estimate_module._period_masks(h, dates, ["full_window", "early", "late"])
        steps, periods = excess_from_paths(
            target["deaths"].to_numpy(dtype=float), np.full(H, fitted.model().level),
            survivors, 0.95, masks, percent="within_path")
        for row, s in zip(est.per_step, steps):
            assert (row["excess_lo"], row["excess_hi"]) == (s["excess_lo"], s["excess_hi"])
        for row, p in zip(est.periods, periods):
            assert (row["excess"], row["excess_lo"], row["excess_hi"]) == (
                p["total_excess"], p["excess_lo"], p["excess_hi"])

    def test_the_failure_warning_is_given_once(self, fitted, monkeypatch):
        _failing_at(monkeypatch, self.FAILING, [])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimate_excess(fitted, data=_frame(), n_replications=N)
        assert sum("bootstrap replications failed" in str(w.message) for w in caught) == 1

    def test_every_replication_failing_refuses_and_writes_nothing(self, fitted,
                                                                   monkeypatch):
        _failing_at(monkeypatch, {r: "exception" for r in range(N)}, [])
        with pytest.raises(EstimateError, match=f"All {N} bootstrap replications failed"):
            _est(fitted)
        assert not os.path.exists(_estimate_dir(fitted))

    def test_refit_warnings_are_counted_not_shown(self, fitted, family, monkeypatch):
        fit_ = family.fit

        def noisy(self, *a, **kw):
            warnings.warn("refit", RuntimeWarning)
            return fit_(self, *a, **kw)
        monkeypatch.setattr(family, "fit", noisy)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est = estimate_excess(fitted, data=_frame(), n_replications=N)
        assert dict(est.bootstrap["refit_warnings"]) == {"RuntimeWarning": N}
        assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


# ---------------------------------------------------------------------------
# Totals and percent excess within each path
# ---------------------------------------------------------------------------

def _known_paths(monkeypatch, paths):
    """Replication r returns column r of paths."""
    def known(r, *args, **kw):
        return Replication(r, mean_path=paths[:, r], path=paths[:, r])
    monkeypatch.setattr(estimate_module, "replicate", known)


class TestWithinPaths:

    def test_period_and_percent_bounds_are_within_path_quantiles(self, fitted,
                                                                  monkeypatch):
        rng = np.random.default_rng(3)
        paths = rng.normal(100, 8, size=(H, N))
        _known_paths(monkeypatch, paths)
        est = _est(fitted)
        observed = _frame()["deaths"].to_numpy()[FIT_ROWS:FIT_ROWS + H]
        # Equal to rounding: numpy may add a path's values in another order.
        same = lambda x: pytest.approx(x, rel=1e-12, abs=0)  # noqa: E731
        full = est.periods[0]
        totals = paths.sum(axis=0)
        assert full["excess_lo"] == same(np.percentile(observed.sum() - totals, 2.5))
        assert full["excess_hi"] == same(np.percentile(observed.sum() - totals, 97.5))
        pct = (observed.sum() - totals) / totals * 100
        assert full["excess_pct_lo"] == same(np.percentile(pct, 2.5))
        assert full["excess_pct_hi"] == same(np.percentile(pct, 97.5))
        late = est.periods[2]
        late_totals = paths[3:].sum(axis=0)
        assert late["median_path_minus_point"] == same(
            np.median(late_totals) - late["forecast"])
        step = est.per_step[4]
        assert step["counterfactual_lo"] == same(np.percentile(paths[4], 2.5))
        step_pct = (observed[4] - paths[4]) / paths[4] * 100
        assert step["excess_pct_hi"] == same(np.percentile(step_pct, 97.5))

    def test_a_shifted_point_is_flagged_outside_its_interval(self, fitted, monkeypatch):
        _known_paths(monkeypatch, np.full((H, N), 1000.0)
                     + np.arange(N, dtype=float))
        full = _est(fitted).periods[0]
        assert full["point_outside_interval"] is True
        assert full["median_path_minus_point"] > 0

    def test_a_path_not_positive_empties_the_percent_bounds_only(self, fitted,
                                                                 monkeypatch):
        paths = np.full((H, N), 100.0) + np.arange(N, dtype=float)
        paths[1, 4] = -2.0
        _known_paths(monkeypatch, paths)
        est = _est(fitted)
        step = est.per_step[1]
        assert step["excess_pct"] is not None
        assert step["excess_pct_lo"] is None and step["excess_pct_hi"] is None
        assert step["excess_pct_null"] == PCT_PATH_NOT_POSITIVE
        assert step["excess_lo"] is not None
        assert dict(est.percent_null["per_step"]) == {PCT_PATH_NOT_POSITIVE: 1}
        # The period totals over that week still have positive denominators.
        assert dict(est.percent_null["periods"]) == {}

    def test_a_point_forecast_not_positive_empties_the_percent(self):
        steps, periods = excess_from_paths(
            np.array([5.0, 6.0]), np.array([0.0, 3.0]), np.full((2, 4), 2.0), 0.9,
            [np.array([True, True])], percent="within_path")
        assert steps[0]["excess_pct"] is None and steps[0]["excess_pct_lo"] is None
        assert steps[0]["excess_pct_null"] == PCT_FORECAST_NOT_POSITIVE
        assert steps[1]["excess_pct_null"] is None
        assert periods[0]["excess_pct_null"] is None

    def test_the_old_path_percent_is_unchanged(self):
        observed, point = np.array([5.0, 6.0]), np.array([4.0, 3.0])
        paths = np.array([[3.0, 4.0, 5.0], [2.0, 3.0, 4.0]])
        steps, periods = excess_from_paths(observed, point, paths, 0.9,
                                           [np.array([True, True])])
        # Divided by the point forecast, not by each path; equal to rounding,
        # since the level arithmetic puts the quantile at 5.000000000000002.
        assert steps[0]["excess_pct_lo"] == pytest.approx(
            np.percentile((5.0 - paths[0]) / 4.0 * 100, 5), rel=1e-12)
        assert "excess_pct_lo" not in periods[0]


# ---------------------------------------------------------------------------
# Sub-periods and block length
# ---------------------------------------------------------------------------

class TestSettings:

    @pytest.mark.parametrize("chosen,names", [
        (None, ["full_window", "early", "late"]),
        ([], ["full_window"]),
        (["late"], ["full_window", "late"]),
        (["late", "early"], ["full_window", "early", "late"]),
        (["full_window"], ["full_window"]),
    ])
    def test_sub_periods(self, fitted, chosen, names):
        est = _est(fitted, sub_periods=chosen)
        assert [p["name"] for p in est.periods] == names

    def test_period_dates_are_inclusive(self, fitted):
        early = _est(fitted).periods[1]
        assert (early["first"], early["last"], early["n"]) == ("2026-06-22", "2026-07-06", 3)

    def test_auto_block_length_records_its_rule(self, fitted):
        b = _est(fitted).bootstrap["block_length"]
        assert dict(b) == {"argument": "auto", "rule": "fixed-14-observations", "value": 14}

    def test_a_given_block_length_is_recorded_as_given(self, fitted):
        b = _est(fitted, block_length=7).bootstrap["block_length"]
        assert dict(b) == {"argument": 7, "rule": "as given", "value": 7}

    def test_fewer_than_ten_blocks_warns(self, fitted):
        with pytest.warns(EstimateWarning, match="9 whole blocks of 20"):
            estimate_excess(fitted, data=_frame(), n_replications=N, block_length=20)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

class TestRecord:

    def test_the_record_on_disk_is_the_record_returned(self, fitted):
        est = _est(fitted)
        run_dir = os.path.join(_estimate_dir(fitted), "main")
        assert EstimateRecord.load(run_dir) == est
        assert EstimateRecord.load(est.path).to_dict() == est.to_dict()
        assert not os.stat(est.path).st_mode & stat.S_IWUSR

    def test_a_one_character_edit_breaks_the_seal(self, fitted):
        est = _est(fitted)
        os.chmod(est.path, os.stat(est.path).st_mode | stat.S_IWUSR)
        with open(est.path, encoding="utf-8") as fh:
            text = fh.read()
        with open(est.path, "w", encoding="utf-8") as fh:
            fh.write(text.replace('"type": "prediction"', '"type": "predictiom"'))
        with pytest.raises(SealError, match="estimate record"):
            EstimateRecord.load(est.path)

    def test_provenance_names_the_four_records_on_disk(self, fitted):
        prov = _est(fitted).provenance
        assert list(prov) == ["design", "candidates", "check", "fit"]
        for name, cls in (("design", DesignRecord), ("candidates", CandidatesRecord),
                          ("check", CheckRecord), ("fit", FitRecord)):
            assert cls.load(prov[name]["path"]).seal["digest"] == prov[name]["seal"]

    def test_every_interval_is_labelled_not_validated(self, fitted):
        est = _est(fitted)
        assert est.interval["validated"] is False
        assert est.interval["method"] == "quantile"
        assert est.caveats[0].startswith("The coverage of this interval has not been "
                                         "validated")

    def test_the_caveats_follow_the_interval_type_and_family(self):
        from its2s.estimate_record import caveats
        noise = "The added noise is not propagated through the model's dynamics."
        order = ("The model's order is held fixed, so uncertainty in choosing it is not "
                 "included.")
        assert noise in caveats("prediction", "prophet_xgb")
        assert noise not in caveats("confidence", "prophet_xgb")
        assert order in caveats("confidence", "arima")
        assert order not in caveats("prediction", "prophet_xgb")

    def test_the_handoff_is_empty_and_stamps_are_carried(self, fitted):
        est = _est(fitted)
        assert dict(est.handoff) == {}
        assert est.to_dict()["stamps"] == fitted.to_dict()["stamps"]


# ---------------------------------------------------------------------------
# The report, the CSVs and the figure
# ---------------------------------------------------------------------------

OUTPUT_FILES = sorted([RECORD_FILE, REPORT_FILE, COUNTERFACTUAL_FILE, EXCESS_TABLE_FILE,
                       PLOT_FILE])


def _read(est, name):
    with open(os.path.join(os.path.dirname(est.path), name), encoding="utf-8") as fh:
        return fh.read()


def _csv(est, name):
    return pd.read_csv(os.path.join(os.path.dirname(est.path), name),
                       keep_default_na=False, na_values=[""])


class TestOutputs:

    def test_the_five_files_are_written_read_only(self, fitted):
        run_dir = os.path.dirname(_est(fitted).path)
        assert sorted(os.listdir(run_dir)) == OUTPUT_FILES
        for name in OUTPUT_FILES:
            assert not os.stat(os.path.join(run_dir, name)).st_mode & stat.S_IWUSR

    def test_the_report_is_rendered_from_the_record_byte_for_byte(self, fitted):
        est = _est(fitted)
        run_dir = os.path.dirname(est.path)
        assert EstimateRecord.load(run_dir).report() == _read(est, REPORT_FILE)

    def test_the_report_carries_every_caveat_that_applies(self, fitted):
        for kind in ("prediction", "confidence"):
            est = _est(fitted, interval_type=kind, run_label=kind)
            flat = " ".join(_read(est, REPORT_FILE).split())
            assert est.caveats and all(c in flat for c in est.caveats)
            assert f"95% {kind} interval, NOT VALIDATED" in flat

    def test_excess_table_equals_the_csv(self, fitted):
        est = _est(fitted)
        loaded = EstimateRecord.load(est.path)
        assert to_csv_text(loaded.excess_table()) == _read(est, EXCESS_TABLE_FILE)
        table = est.excess_table()
        assert list(table["period"]) == ["full_window", "early", "late"]
        assert list(table["last"]) == ["2026-07-27", "2026-07-06", "2026-07-27"]

    @pytest.mark.parametrize("level,tag", [(0.95, "95"), (0.975, "97_5"), (0.57, "57"),
                                           (0.5, "50")])
    def test_bound_columns_name_type_and_level_in_full(self, level, tag):
        interval = {"type": "confidence", "level": level}
        assert bound_columns("excess", interval) == (f"excess_confidence{tag}_lo",
                                                     f"excess_confidence{tag}_hi")

    def test_level_percent_is_from_the_decimal_string(self):
        assert 0.57 * 100 != 57
        assert level_percent(0.57) == "57"
        assert level_percent(0.975) == "97.5"

    def test_csv_headers_carry_the_interval(self, fitted):
        est = _est(fitted, interval_type="confidence", level=0.975)
        cf = _csv(est, COUNTERFACTUAL_FILE)
        assert list(cf.columns) == [
            "date", "window", "observed", "forecast",
            "counterfactual_confidence97_5_lo", "counterfactual_confidence97_5_hi",
            "excess", "excess_confidence97_5_lo", "excess_confidence97_5_hi",
            "excess_pct", "excess_pct_confidence97_5_lo", "excess_pct_confidence97_5_hi"]
        assert list(_csv(est, EXCESS_TABLE_FILE).columns) == [
            "period", "first", "last", "n", "observed", "forecast",
            "excess", "excess_confidence97_5_lo", "excess_confidence97_5_hi",
            "excess_pct", "excess_pct_confidence97_5_lo", "excess_pct_confidence97_5_hi",
            "point_outside_interval", "median_path_minus_point"]

    def test_counterfactual_rows_windows_and_empty_pre_event_columns(self, fitted):
        est = _est(fitted)
        cf = _csv(est, COUNTERFACTUAL_FILE)
        df = _frame()
        df = df[df["week"] < "2026-08-03"]  # the data runs on past the post-event window
        assert len(cf) == len(df) == FIT_ROWS + H
        assert cf["date"].iloc[0] == "2023-01-02" and cf["date"].iloc[-1] == "2026-07-27"
        assert list(cf["window"].value_counts().sort_index().items()) == [
            ("post_event", H), ("test", 52), ("training", 129)]
        np.testing.assert_array_equal(cf["observed"], df["deaths"].astype(float))
        before = cf[cf["window"] != "post_event"]
        assert before.drop(columns=["date", "window", "observed"]).isna().all().all()
        post = cf[cf["window"] == "post_event"]
        assert not post.isna().any().any()
        np.testing.assert_allclose(post["excess"], [s["excess"] for s in est.per_step])

    def test_rows_before_train_start_are_labelled_before_training(self, tmp_path, family):
        fit = _chain(tmp_path, split=(("train_start", "2023-03-06"), ("train_obs", 120)))
        cf = _csv(_est(fit), COUNTERFACTUAL_FILE)
        assert (cf["window"].iloc[:9] == "before_training").all()
        assert cf["window"].iloc[9] == "training"

    def test_style_reaches_the_figure_and_is_recorded(self, fitted, monkeypatch):
        drawn = []
        real = estimate_module.counterfactual_figure

        def keep(*args):
            drawn.append(real(*args))
            return drawn[-1]
        monkeypatch.setattr(estimate_module, "counterfactual_figure", keep)
        style = {"colors": {"observed": "#123456"}, "figsize": (8, 4), "dpi": 72}
        est = _est(fitted, style=style)
        fig = drawn[0]
        assert fig.dpi == 72
        assert tuple(fig.get_size_inches()) == (8, 4)
        assert to_hex(fig.axes[0].lines[0].get_color()) == "#123456"
        resolved = est.to_dict()["presentation"]["resolved"]
        assert est.presentation["style"]["dpi"] == 72
        assert resolved["colors"]["observed"] == "#123456"
        assert resolved["colors"]["event"] == DEFAULT_STYLE["colors"]["event"]
        assert resolved["figsize"] == [8, 4]

    def test_the_figure_names_the_interval_and_draws_no_fit_line(self, fitted,
                                                                  monkeypatch):
        drawn = []
        real = estimate_module.counterfactual_figure
        monkeypatch.setattr(estimate_module, "counterfactual_figure",
                            lambda *a: drawn.append(real(*a)) or drawn[-1])
        _est(fitted, interval_type="confidence", level=0.9)
        ax = drawn[0].axes[0]
        labels = ax.get_legend_handles_labels()[1]
        assert "90% confidence interval, not validated" in labels
        assert "test (used in the final fit)" in labels
        # observed and the counterfactual forecast; the event line has two points
        forecast = [ln for ln in ax.lines if ln.get_label() == "counterfactual forecast"]
        assert len(forecast[0].get_xdata()) == H

    def test_a_failure_while_writing_leaves_nothing(self, fitted, monkeypatch):
        def broken(*args):
            raise RuntimeError("drawing failed")
        monkeypatch.setattr(estimate_module, "counterfactual_figure", broken)
        with pytest.raises(RuntimeError, match="drawing failed"):
            _est(fitted)
        root = _estimate_dir(fitted)
        assert not os.path.exists(root) or os.listdir(root) == []


# ---------------------------------------------------------------------------
# in_sample on the real families
# ---------------------------------------------------------------------------

def _series(n=120, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    t = np.arange(n)
    y = 100 + 0.2 * t + 5 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 2, n)
    return pd.DataFrame({"ds": dates, "y": y})


def _real(name, params=None):
    from its2s.frequency import SeriesFrequency
    from its2s.models.families import load_family_class
    from its2s.seasonality import make_seasonality
    seasonality = make_seasonality(["annual"], SeriesFrequency.from_alias("W-MON"))
    return load_family_class(name)(params=params, seasonality=seasonality)


class TestInSample:

    @pytest.mark.parametrize("name,params", [
        ("prophet_xgb", {"xgb": {"n_estimators": 20}}),
        ("arima", {"max_p": 1, "max_q": 1, "max_P": 0, "max_Q": 0, "max_D": 0}),
    ])
    def test_in_sample_equals_the_fit_in_memory_and_after_load(self, tmp_path, name,
                                                               params):
        df = _series()
        model = _real(name, params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(df)
            in_memory = model.in_sample(df)
            model.save(tmp_path / "m")
            loaded = type(model).load(tmp_path / "m").in_sample(df)
        np.testing.assert_array_equal(in_memory, result.fitted_values)
        np.testing.assert_array_equal(loaded, result.fitted_values)

    def test_arima_refuses_rows_other_than_its_fit(self):
        df = _series()
        model = _real("arima", {"max_p": 1, "max_q": 1, "max_P": 0, "max_Q": 0, "max_D": 0})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        with pytest.raises(ValueError, match="fit on 120 observations"):
            model.in_sample(df.iloc[:-1])


class TestPublicNames:

    def test_exported(self):
        import its2s
        for name in ("estimate_excess", "EstimateRecord", "EstimateError",
                     "EstimateWarning"):
            assert name in its2s.__all__ and hasattr(its2s, name)
        assert issubclass(EstimateError, ValueError)
        assert issubclass(EstimateWarning, UserWarning)
