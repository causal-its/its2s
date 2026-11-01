# Description: Tests for declare_design, step 0's sealing call: the label
#   refusal, the grouped DesignError, the atomic read-only write, the sealed
#   record and its handoff, the data fingerprint, DesignRecord, and the
#   figure. Synthetic data only. Nothing here fits a model, and one test
#   proves no fit is reachable.
# Usage: python -m pytest tests/test_declare.py -v --tb=short
# Dependencies: pytest, numpy, pandas, matplotlib, its2s

import json
import os
import re
import stat
import subprocess
import sys
import warnings
from datetime import datetime, timezone

import pandas as pd
import pytest

import its2s
from its2s import (Design, DesignError, DesignRecord, DesignWarning,
                   FrozenDesignError, SealError, declare_design)
from its2s.design import CLOSING_LINE, resolve, sealing_footer
from its2s.design_record import (PLOT_FILE, RECORD_DIR, RECORD_FILE, REPORT_FILE,
                                 fingerprint)
from its2s.models.families import FAMILIES, load_family_class
from its2s.outputs.design_plot import fold_counts, plotted_combinations
from its2s.seeds import derive_seed

from test_design import EVENT, _contract_example, _design, _frame, _minimal

LABEL = "euromomo_heat2026"


def _declare(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DesignWarning)
        return declare_design(d)


def _failing(tmp):
    """The worked failed declaration: a duplicate split and a thin cv design."""
    d = _contract_example(_design(tmp=tmp))
    d.add_split("sp3", train_frac=0.80)
    d.add_cv("cv3", n_folds=4, fold_eval_obs=13)
    return d


def _files(root):
    return sorted(os.path.relpath(os.path.join(r, f), root)
                  for r, _, fs in os.walk(root) for f in fs) + sorted(
        os.path.relpath(os.path.join(r, x), root)
        for r, ds, _ in os.walk(root) for x in ds)


# ---------------------------------------------------------------------------
# What is written
# ---------------------------------------------------------------------------

class TestWrite:

    def test_three_read_only_files(self, tmp_path, capsys):
        _declare(_contract_example(_design(tmp=tmp_path)))
        sub = tmp_path / LABEL / RECORD_DIR
        assert sorted(os.listdir(sub)) == sorted([RECORD_FILE, REPORT_FILE, PLOT_FILE])
        for name in os.listdir(sub):
            mode = os.stat(sub / name).st_mode
            assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        assert (sub / PLOT_FILE).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        # Nothing else beside the label: no partial directory left behind.
        assert os.listdir(tmp_path) == [LABEL]

    def test_prints_the_report(self, tmp_path, capsys):
        _declare(_contract_example(_design(tmp=tmp_path)))
        out = capsys.readouterr().out
        assert out == (tmp_path / LABEL / RECORD_DIR / REPORT_FILE).read_text()

    def test_preview_prints_the_report_above_its_footer(self, tmp_path, capsys):
        d = _contract_example(_design(tmp=tmp_path))
        d.preview()
        previewed = capsys.readouterr().out
        record = _declare(d)
        capsys.readouterr()
        written = (tmp_path / LABEL / RECORD_DIR / REPORT_FILE).read_text()
        assert written == previewed + sealing_footer(record.to_dict())

    def test_footer_states_time_and_fingerprint(self, tmp_path):
        record = _declare(_contract_example(_design(tmp=tmp_path)))
        written = (tmp_path / LABEL / RECORD_DIR / REPORT_FILE).read_text()
        footer = written[written.rindex("SEALED"):]
        assert f"SEALED  {record.declared_at}" in footer
        assert record.data["fingerprint"]["digest"] in footer
        assert "184 rows, 2023-01-02 to 2026-07-06; columns week, deaths" in footer

    def test_failure_while_writing_leaves_nothing(self, tmp_path, monkeypatch):
        import its2s.design as design_module

        def boom(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(design_module, "record_json", boom)
        d = _contract_example(_design(tmp=tmp_path))
        with pytest.raises(OSError, match="disk full"):
            _declare(d)
        assert os.listdir(tmp_path) == []
        # The design was not sealed, so it can be declared once the fault is gone.
        monkeypatch.undo()
        _declare(d)
        assert os.listdir(tmp_path) == [LABEL]

    def test_failure_in_the_figure_leaves_nothing(self, tmp_path, monkeypatch):
        from matplotlib.figure import Figure

        def boom(*a, **k):
            raise RuntimeError("renderer failed")
        monkeypatch.setattr(Figure, "savefig", boom)
        with pytest.raises(RuntimeError):
            _declare(_contract_example(_design(tmp=tmp_path)))
        assert os.listdir(tmp_path) == []

    def test_creates_output_dir(self, tmp_path):
        out = tmp_path / "a" / "b"
        _declare(_contract_example(_design(tmp=out)))
        assert (out / LABEL / RECORD_DIR / RECORD_FILE).exists()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

class TestRefusal:

    def test_existing_label_refused_with_date_and_suggestion(self, tmp_path):
        record = _declare(_contract_example(_design(tmp=tmp_path)))
        before = _files(tmp_path)
        with pytest.raises(DesignError) as info:
            _declare(_contract_example(_design(tmp=tmp_path)))
        msg = str(info.value)
        assert repr(LABEL) in msg
        assert f"sealed on {record.declared_at[:10]}" in msg
        today = datetime.now().strftime("%Y%m%d")
        assert f"{LABEL}_{today}" in msg
        assert msg.endswith(CLOSING_LINE)
        assert _files(tmp_path) == before

    def test_existing_label_without_record_uses_directory_date(self, tmp_path):
        (tmp_path / LABEL).mkdir()
        os.utime(tmp_path / LABEL, (0, 86400 * 365))      # 1971-01-01
        with pytest.raises(DesignError, match="sealed on 1971-01-01"):
            _declare(_contract_example(_design(tmp=tmp_path)))

    def test_label_checked_before_validation(self, tmp_path):
        (tmp_path / LABEL).mkdir()
        with pytest.raises(DesignError, match="already exists"):
            _declare(_failing(tmp_path))

    def test_failed_design_is_one_grouped_error(self, tmp_path):
        with pytest.raises(DesignError) as info:
            _declare(_failing(tmp_path))
        e = info.value
        msg = str(e)
        assert msg.startswith("this design cannot be sealed. 2 errors, 4 warnings.")
        assert msg.endswith(CLOSING_LINE)
        assert "duplicate split" in msg and "thin folds" in msg
        assert len(e.errors) == 2 and len(e.warnings) == 4
        assert isinstance(e, ValueError)
        assert os.listdir(tmp_path) == []

    def test_unresolved_groups_are_named_in_the_error(self, tmp_path):
        d = _minimal(_design(tmp=tmp_path, event="2026-06-24"))   # off the grid
        with pytest.raises(DesignError, match="NOT CHECKED"):
            _declare(d)

    def test_declared_design_is_frozen(self, tmp_path):
        d = _contract_example(_design(tmp=tmp_path))
        _declare(d)
        mutators = [
            lambda: d.set_seasonality(None),
            lambda: d.add_split("sp9", train_obs=100, test_obs=81),
            lambda: d.remove_split("sp1"),
            lambda: d.add_cv("cv9", n_folds=2, fold_eval_obs=3),
            lambda: d.remove_cv("cv1"),
            lambda: d.add_family("arima", n_trials=5),
            lambda: d.remove_family("arima"),
            lambda: d.set_criteria(S2=True),
            lambda: d.set_check(n_origins=3),
            lambda: d.add_sub_period("late", "2026-07-06", "2026-07-06"),
            lambda: d.set_plot(all_combinations=True),
            lambda: declare_design(d),
        ]
        for call in mutators:
            with pytest.raises(FrozenDesignError):
                call()

    def test_redeclaring_names_label_date_and_a_new_label(self, tmp_path):
        d = _contract_example(_design(tmp=tmp_path))
        record = _declare(d)
        with pytest.raises(FrozenDesignError) as info:
            declare_design(d)
        msg = str(info.value)
        today = datetime.now().strftime("%Y%m%d")
        assert f"already declared as {LABEL!r} on {record.declared_at[:10]}" in msg
        assert "build a new Design with a new series_label" in msg
        assert f"{LABEL}_{today}" in msg

    def test_not_a_design(self):
        with pytest.raises(TypeError):
            declare_design({"series_label": "x"})


def test_no_fit_is_reachable(tmp_path, monkeypatch):
    """A failed declaration costs no fits, and a sealed one runs none."""
    patched = 0
    for name, spec in FAMILIES.items():
        if spec.status == "withheld":
            continue
        try:
            cls = load_family_class(name)
        except ImportError:
            continue

        def refuse(*a, **k):
            raise AssertionError("step 0 reached a model fit")
        monkeypatch.setattr(cls, "fit", refuse)
        patched += 1
    assert patched >= 1
    with pytest.raises(DesignError):
        _declare(_failing(tmp_path / "failed"))
    assert not (tmp_path / "failed").exists()
    _declare(_contract_example(_design(tmp=tmp_path / "sealed")))


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

TOP_LEVEL = ["schema_version", "package_version", "declared_at", "seal", "label",
             "data", "question", "seasonality", "splits", "cvs", "combinations",
             "families", "criteria", "check", "sub_periods", "seeds", "starter_used",
             "exploration", "warnings", "handoff", "report_fields"]


class TestRecord:

    @pytest.fixture
    def sealed(self, tmp_path):
        record = _declare(_contract_example(_design(tmp=tmp_path)))
        return tmp_path / LABEL / RECORD_DIR / RECORD_FILE, record

    def test_top_level_keys(self, sealed):
        path, _ = sealed
        raw = json.loads(path.read_text())
        assert list(raw) == TOP_LEVEL
        assert set(raw["report_fields"]) == {"output_path", "seed", "candidates",
                                             "sweep_fits", "plot_all_combinations",
                                             "not_checked"}
        assert raw["schema_version"] == 1
        assert raw["package_version"] == its2s.__version__
        assert raw["seal"]["algorithm"] == "sha256"

    def test_declared_at_is_utc_with_microseconds(self, sealed, tmp_path):
        _, record = sealed
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00",
                            record.declared_at)
        stamped = datetime.fromisoformat(record.declared_at)
        written = datetime.fromtimestamp(
            os.stat(tmp_path / LABEL / RECORD_DIR / RECORD_FILE).st_mtime, tz=timezone.utc)
        assert stamped <= written

    def test_returned_record_equals_loaded(self, sealed):
        path, record = sealed
        loaded = DesignRecord.load(path)
        assert loaded.to_dict() == record.to_dict() == json.loads(path.read_text())
        assert DesignRecord.load(path.parent.parent).to_dict() == record.to_dict()
        assert DesignRecord.load(path.parent).to_dict() == record.to_dict()

    def test_round_trip_renders_the_written_report(self, sealed):
        path, _ = sealed
        written = (path.parent / REPORT_FILE).read_text()
        assert DesignRecord.load(path).report() == written

    def test_public_names(self):
        assert its2s.DesignRecord is DesignRecord
        assert issubclass(SealError, DesignError)

    def test_record_is_read_only(self, sealed):
        _, record = sealed
        with pytest.raises(AttributeError):
            record.label = "other"
        with pytest.raises(TypeError):
            record.handoff["question"]["H"] = 4
        copy = record.to_dict()
        copy["label"] = "other"
        assert record.label == LABEL

    def test_seal_detects_a_one_character_edit(self, sealed):
        path, _ = sealed
        text = path.read_text()
        os.chmod(path, 0o644)
        path.write_text(text.replace('"H": 3', '"H": 4', 1))
        with pytest.raises(SealError, match="does not match its seal"):
            DesignRecord.load(path)

    def test_seal_survives_reformatting(self, sealed):
        path, record = sealed
        os.chmod(path, 0o644)
        path.write_text(json.dumps(json.loads(path.read_text()), indent=4))
        assert DesignRecord.load(path).to_dict() == record.to_dict()

    def test_invalid_json_and_missing_record(self, sealed, tmp_path):
        path, _ = sealed
        os.chmod(path, 0o644)
        path.write_text(path.read_text()[:-5])
        with pytest.raises(SealError, match="not valid JSON"):
            DesignRecord.load(path)
        with pytest.raises(SealError, match="No design record"):
            DesignRecord.load(tmp_path / "nowhere")

    def test_warnings_are_recorded_and_warned_once(self, tmp_path):
        d = _contract_example(_design(tmp=tmp_path))
        d.set_criteria(S2=True)              # cv1 has two folds: S2 not computable
        with pytest.warns(DesignWarning) as caught:
            record = declare_design(d)
        assert len([w for w in caught if w.category is DesignWarning]) == 1
        assert "fold stability" in str(caught[0].message)
        assert [w["title"] for w in record.warnings] == ["fold stability"]
        report = record.report()
        assert "ISSUES  0 errors, 1 warning" in report
        assert CLOSING_LINE not in report

    def test_clean_design_warns_nothing(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DesignWarning)
            record = declare_design(_contract_example(_design(tmp=tmp_path)))
        assert record.warnings == ()


# ---------------------------------------------------------------------------
# The handoff
# ---------------------------------------------------------------------------

class TestHandoff:

    @pytest.fixture
    def record(self, tmp_path):
        return _declare(_contract_example(_design(tmp=tmp_path))).to_dict()

    def test_sections(self, record):
        assert list(record["handoff"]) == [
            "data", "question", "seasonality", "splits", "combinations",
            "candidates", "criteria", "origins", "sub_periods", "bootstrap_seed",
            "output_path"]

    def test_data_and_question(self, record):
        h = record["handoff"]
        assert h["data"]["fingerprint"] == record["data"]["fingerprint"]["digest"]
        assert h["data"]["rows"] == {"first": "2023-01-02", "last": "2026-07-06", "n": 184}
        assert h["data"]["freq"] == "W-MON"
        assert h["data"]["outcome_col"] == "deaths"
        assert h["question"] == {"event_date": EVENT, "H": 3}

    def test_windows_are_half_open_bounds(self, record):
        sp1 = record["handoff"]["splits"]["sp1"]
        assert sp1["primary"] is True
        assert sp1["train"] == {"start": "2023-01-02", "end": "2025-06-23", "n": 129}
        assert sp1["test"] == {"start": "2025-06-23", "end": EVENT, "n": 52}
        assert sp1["post_event"] == {"start": EVENT, "end": "2026-07-13", "n": 3}
        folds = record["handoff"]["combinations"]["sp1/cv1"]["folds"]
        assert folds[-1]["evaluate"]["end"] == sp1["train"]["end"]

    def test_candidates_carry_space_and_seed(self, record):
        cands = record["handoff"]["candidates"]
        assert len(cands) == 8
        c = cands["arima/sp2/cv1"]
        assert (c["family"], c["split"], c["cv"], c["n_trials"]) == ("arima", "sp2", "cv1", 40)
        assert c["seed"] == derive_seed(42, "arima/sp2/cv1")
        assert set(c["space"]) == {"sample", "fixed"}
        assert c["m"] == 52
        assert cands["prophet_xgb/sp1/cv1"]["m"] is None
        assert record["handoff"]["bootstrap_seed"] == derive_seed(42, "bootstrap")

    def test_criteria_origins_sub_periods(self, record):
        h = record["handoff"]
        assert h["criteria"]["S2"] == {"selected": False, "op": "<=", "threshold": 1.5}
        assert h["criteria"]["S4"] == {"selected": True, "op": ">=", "threshold": 2.0}
        assert h["criteria"]["S5"] == {"selected": False, "op": "band",
                                       "threshold": [0.5, 2.0]}
        assert len(h["origins"]["sp1"]["dates"]) == 4
        assert set(h["sub_periods"]) == {"full_window", "acute"}
        assert h["sub_periods"]["full_window"]["n"] == 3

    def test_output_path(self, record, tmp_path):
        assert record["handoff"]["output_path"] == os.path.join(str(tmp_path), LABEL)


# ---------------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------------

def _fp(df, end="2026-07-13", cols=("deaths",)):
    return fingerprint(df, "week", list(cols), end)["digest"]


class TestFingerprint:

    def test_stable_across_runs_and_copies(self):
        assert _fp(_frame()) == _fp(_frame()) == _fp(_frame().copy(deep=True))

    def test_changes_on_one_used_value(self):
        df = _frame()
        df.loc[0, "deaths"] += 1          # the first row, before any window
        assert _fp(df) != _fp(_frame())
        df = _frame()
        df.loc[183, "deaths"] += 1        # the last post-event row
        assert _fp(df) != _fp(_frame())

    def test_unchanged_outside_the_used_rows_and_columns(self):
        df = _frame()
        df["temperature"] = 1.0
        assert _fp(df) == _fp(_frame())
        df = _frame()
        df.loc[184, "deaths"] += 1        # first row after the post-event window
        assert _fp(df) == _fp(_frame())
        assert _fp(_frame(periods=250)) == _fp(_frame())

    def test_covers_covariates_and_names(self):
        df = _frame()
        df["temperature"] = 1.0
        base = _fp(df, cols=("deaths", "temperature"))
        assert base != _fp(df)
        df.loc[5, "temperature"] = 2.0
        assert _fp(df, cols=("deaths", "temperature")) != base
        renamed = _frame().rename(columns={"deaths": "count"})
        assert fingerprint(renamed, "week", ["count"], "2026-07-13")["digest"] != _fp(_frame())

    def test_integer_and_float_outcomes_agree(self):
        df = _frame()
        df["deaths"] = df["deaths"].astype(int)
        assert _fp(df) == _fp(_frame())

    def test_record_holds_range_and_columns(self, tmp_path):
        record = _declare(_contract_example(_design(tmp=tmp_path)))
        fp = record.data["fingerprint"]
        assert fp["columns"] == ("week", "deaths")
        assert fp["rows"]["n"] == 184
        assert fp["digest"] == _fp(_frame())


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

class TestFigure:

    def test_primary_split_only_past_six_combinations(self):
        d = _failing("unused")
        d.remove_split("sp3")
        d.add_split("sp3", train_obs=120, test_obs=61)
        d.remove_family("arima")
        resolved, _ = resolve(d)
        assert len(resolved["combinations"]) == 9
        assert plotted_combinations(resolved) == ["sp1/cv1", "sp1/cv2", "sp1/cv3"]
        d.set_plot(all_combinations=True)
        assert len(plotted_combinations(resolve(d)[0])) == 9

    def test_trimmed_counts_are_labelled(self):
        from its2s.outputs.design_plot import design_figure
        d = _failing("unused")
        d.remove_split("sp3")
        d.add_split("sp3", train_obs=120, test_obs=61)
        d.remove_family("arima")
        trimmed = design_figure(resolve(d)[0])
        assert "primary split" in trimmed.axes[1].get_ylabel()
        d.set_plot(all_combinations=True)
        assert design_figure(resolve(d)[0]).axes[1].get_ylabel() == "folds"

    def test_fold_counts(self):
        resolved, _ = resolve(_contract_example(_design()))
        keys = plotted_combinations(resolved)
        grid, train, evaluate = fold_counts(resolved, keys)
        folds = [f for k in keys for f in resolved["combinations"][k]["folds"]]
        assert train.sum() == sum(f["train"]["n"] for f in folds)
        assert evaluate.sum() == sum(f["evaluate"]["n"] for f in folds)
        # The first date is trained on by every fold; nothing after the
        # training windows is used by any fold.
        assert train[0] == len(folds)
        assert (train[grid >= pd.Timestamp(EVENT)] == 0).all()

    def test_redrawn_from_record(self, tmp_path):
        record = _declare(_contract_example(_design(tmp=tmp_path)))
        fig = record.figure()
        assert len(fig.axes) == 2


def test_declare_imports_no_heavy_library(tmp_path):
    code = (
        "import sys, warnings, numpy as np, pandas as pd\n"
        "from its2s.design import Design, declare_design\n"
        "df = pd.DataFrame({'week': pd.date_range('2023-01-02', periods=201, "
        "freq='W-MON'), 'deaths': np.ones(201)})\n"
        f"d = Design(df, 'week', 'deaths', None, '2026-06-22', 3, 42, {str(tmp_path)!r}, 'x')\n"
        "d.add_split('sp1', train_obs=129, test_obs=52)\n"
        "d.add_cv('cv1', n_folds=2, fold_eval_obs=3)\n"
        "d.add_family('arima', n_trials=10)\n"
        "d.add_family('prophet_xgb', n_trials=10)\n"
        "d.preview()\n"
        "warnings.simplefilter('ignore')\n"
        "declare_design(d)\n"
        "print([m for m in ('prophet', 'pmdarima', 'neuralprophet', 'xgboost', "
        "'torch') if m in sys.modules])\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"
    assert (tmp_path / "x" / RECORD_DIR / RECORD_FILE).exists()
