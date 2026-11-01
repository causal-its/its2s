# Description: The sealed design record that step 0 writes and every later
#   step reads. The JSON on disk is canonical; DesignRecord is a frozen,
#   typed view of it. This module builds the record from step 0's resolved
#   description, fingerprints the data the analysis uses, seals the record
#   with a hash of its own content, and selects the HANDOFF section: the only
#   fields the next step may read.
# Usage: from its2s.design_record import DesignRecord, build_record, fingerprint
# Dependencies: numpy, pandas

import hashlib
import json
import os
from dataclasses import dataclass, field
from types import MappingProxyType

import pandas as pd

SCHEMA_VERSION = 1
SEAL_ALGORITHM = "sha256"
RECORD_DIR = "00_design"
RECORD_FILE = "00-design.json"
REPORT_FILE = "00-design_report.txt"
PLOT_FILE = "00-design_plot.png"


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def _iso(ts):
    """A timestamp as an ISO date, or date and time when it has one."""
    ts = pd.Timestamp(ts)
    return ts.date().isoformat() if ts == ts.normalize() else ts.isoformat()


def _hash_column(h, name, array):
    """Feed one column into h: its name and byte count first, so no two
    different tables can produce the same byte stream."""
    name_bytes = name.encode("utf-8")
    data = array.tobytes()
    h.update(len(name_bytes).to_bytes(8, "little"))
    h.update(name_bytes)
    h.update(len(data).to_bytes(8, "little"))
    h.update(data)


def fingerprint(data, date_col, value_cols, end):
    """Fingerprint the rows and columns the analysis uses.

    Covers the date column and the value columns of every row dated before
    `end`, the half-open end of the post-event window: from the first row
    through the last post-event row. Rows appended after the post-event
    window and columns not declared do not change it; any change to a used
    value does. Dates are hashed as int64 nanoseconds and values as
    little-endian float64, so the result does not depend on how a pandas
    version formats them.

    Parameters
    ----------
    data : pd.DataFrame
        Sorted by date (step 0 refuses an unsorted series before this runs).
    date_col : str
    value_cols : list of str
        Outcome then covariates, in declared order.
    end : date-like
        Half-open end of the fingerprinted range.

    Returns
    -------
    dict
        algorithm, digest, the row range (first and last date, row count)
        and the columns covered.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(data[date_col])).as_unit("ns")
    keep = dates < pd.Timestamp(end)
    h = hashlib.sha256()
    _hash_column(h, date_col, dates[keep].asi8.astype("<i8"))
    for col in value_cols:
        _hash_column(h, col, data[col].to_numpy(dtype="<f8")[keep])
    used = dates[keep]
    return {
        "algorithm": SEAL_ALGORITHM,
        "digest": h.hexdigest(),
        "rows": {"first": _iso(used[0]), "last": _iso(used[-1]),
                 "n": int(keep.sum())},
        "columns": [date_col, *value_cols],
    }


# ---------------------------------------------------------------------------
# Seal
# ---------------------------------------------------------------------------

def _canonical(record):
    """The record's content as one fixed byte string, seal excluded.

    Sorted keys and no whitespace, so re-indenting the file does not break
    the seal while any change to a value does.
    """
    body = {k: v for k, v in record.items() if k != "seal"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def seal_digest(record):
    """The hash that seals a record."""
    return hashlib.sha256(_canonical(record)).hexdigest()


def seal_matches(record):
    """True when the record's seal is the hash of its own content."""
    seal = record.get("seal")
    return (isinstance(seal, dict) and seal.get("algorithm") == SEAL_ALGORITHM
            and seal.get("digest") == seal_digest(record))


# ---------------------------------------------------------------------------
# The handoff: what the next step reads, and nothing else
# ---------------------------------------------------------------------------

def _bounds(w):
    """A window as a later step needs it: half-open start and end, rows."""
    return {"start": w["start"], "end": w["end"], "n": w["n"]}


def build_handoff(resolved, fp):
    """Select the fields later steps need from the resolved description.

    Each step reads only its predecessor's handoff, and steps 2 to 4 need
    facts declared here, so this section carries everything any later step
    needs; each later record copies forward what its successors need.
    Everything outside it (typed forms, flags, coverage figures, the
    rendering) exists for readers and may change without touching a step.
    """
    r = resolved
    d = r["data"]
    rf = r["report_fields"]
    return {
        "data": {
            "fingerprint": fp["digest"],
            "rows": dict(fp["rows"]),
            "date_col": d["date_col"],
            "outcome_col": d["outcome_col"],
            "covariate_cols": list(d["covariate_cols"]),
            "freq": d["freq"],
        },
        "question": dict(r["question"]),
        "seasonality": {"cycles": list(r["seasonality"]["cycles"]),
                        "periods": dict(r["seasonality"]["periods"])},
        "splits": {
            sid: {"primary": s["primary"],
                  "train": _bounds(s["train"]),
                  "test": _bounds(s["test"]),
                  "post_event": _bounds(s["post_event"])}
            for sid, s in r["splits"].items()},
        "combinations": {
            key: {"split": c["split"], "cv": c["cv"],
                  "folds": [{"k": f["k"], "train": _bounds(f["train"]),
                             "evaluate": _bounds(f["evaluate"])} for f in c["folds"]]}
            for key, c in r["combinations"].items()},
        # m is the seasonal period the family fits at (arima only), so a
        # later step uses the period this record was sealed with rather than
        # one a later package version might derive differently.
        "candidates": {
            cid: {"family": c["family"], "split": c["split"], "cv": c["cv"],
                  "n_trials": c["n_trials"],
                  "space": r["families"][c["family"]]["space"],
                  "m": r["families"][c["family"]]["m"],
                  "seed": c["seed"]}
            for cid, c in rf["candidates"].items()},
        # op is recorded for the same reason: each criterion is judged by the
        # comparison it was sealed under.
        "criteria": {cid: {"selected": c["selected"], "op": c["op"],
                           "threshold": c["threshold"]}
                     for cid, c in r["criteria"].items()},
        "origins": {sid: {"dates": list(o["dates"]), "phase_origin": o["phase_origin"]}
                    for sid, o in r["check"]["splits"].items()},
        "sub_periods": {name: _bounds(w) for name, w in r["sub_periods"].items()},
        "bootstrap_seed": r["seeds"]["bootstrap"],
        "output_path": rf["output_path"],
    }


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def build_record(resolved, fp, declared_at, package_version):
    """The sealed design record, as a JSON-ready dict.

    It holds the declaration as resolved, with the validation findings
    under "warnings" (a sealed design has no errors); the schema and
    package versions, the declaration time, the data fingerprint, the
    handoff and the seal; and, nested under "report_fields", what the
    report and the figure need beyond the declaration, so both can be
    rendered from the record alone.
    """
    body = {k: v for k, v in resolved.items() if k not in ("issues", "report_fields")}
    body["data"] = {**resolved["data"], "fingerprint": fp}
    record = {
        "schema_version": SCHEMA_VERSION,
        "package_version": package_version,
        "declared_at": declared_at,
        "seal": None,
        **body,
        "warnings": [i for i in resolved["issues"] if i["level"] == "warning"],
        "handoff": build_handoff(resolved, fp),
        "report_fields": resolved["report_fields"],
    }
    record["seal"] = {"algorithm": SEAL_ALGORITHM, "digest": seal_digest(record)}
    return record


def record_json(record):
    """The record as written to disk."""
    return json.dumps(record, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def _freeze(value):
    """A read-only copy: dicts become mapping proxies, lists tuples."""
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _record_path(path):
    """Accept the record file, its 00_design directory, or the label directory."""
    path = os.fspath(path)
    if os.path.isdir(os.path.join(path, RECORD_DIR)):
        path = os.path.join(path, RECORD_DIR)
    if os.path.isdir(path):
        path = os.path.join(path, RECORD_FILE)
    return path


@dataclass(frozen=True)
class DesignRecord:
    """A sealed design record, read-only.

    Built from the JSON on disk (load) or from the dict declare_design
    wrote (from_dict); either way the seal is verified first. Nested
    sections are read-only mappings and tuples; to_dict() returns a fresh
    plain copy.
    """

    schema_version: int
    package_version: str
    declared_at: str
    seal: object
    label: str
    data: object
    question: object
    seasonality: object
    splits: object
    cvs: object
    combinations: object
    families: object
    criteria: object
    check: object
    sub_periods: object
    seeds: object
    starter_used: bool
    exploration: object
    warnings: tuple
    handoff: object
    report_fields: object
    path: str | None = None
    _json: str = field(default="", repr=False, compare=False)

    @classmethod
    def from_dict(cls, record, path=None):
        """A DesignRecord from a record dict; raises SealError if the seal
        does not match the content."""
        from .design import SealError
        if not seal_matches(record):
            where = f" at {path}" if path else ""
            raise SealError(
                f"The design record{where} does not match its seal: it has been "
                "edited since it was declared, so it is not the design that was "
                "sealed. Declare again under a new label.")
        names = [f for f in cls.__dataclass_fields__ if f not in ("path", "_json")]
        return cls(**{n: _freeze(record[n]) for n in names}, path=path,
                   _json=json.dumps(record))

    @classmethod
    def load(cls, path):
        """Read and verify a design record.

        path may be the record file, its 00_design directory, or the label
        directory that holds it.
        """
        from .design import SealError
        file = _record_path(path)
        try:
            with open(file, encoding="utf-8") as fh:
                record = json.load(fh)
        except FileNotFoundError:
            raise SealError(f"No design record at {file}.") from None
        except json.JSONDecodeError as e:
            raise SealError(f"The design record at {file} is not valid JSON ({e}); "
                              "it has been edited since it was declared.") from None
        return cls.from_dict(record, path=file)

    def to_dict(self):
        """A fresh, mutable copy of the record as JSON values."""
        return json.loads(self._json)

    def report(self):
        """The text report, exactly as written to 00-design_report.txt."""
        from .design import sealed_report
        return sealed_report(self.to_dict())

    def figure(self):
        """The design figure, redrawn from the record."""
        from .outputs.design_plot import design_figure
        return design_figure(self.to_dict())
