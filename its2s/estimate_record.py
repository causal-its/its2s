# Description: The sealed estimate record that step 4 writes, one per run.
#   The JSON on disk is canonical; EstimateRecord is a frozen, typed view of
#   it. Step 4 may run more than once under a label, so each run's record
#   sits in its own directory, 04_estimate/<run_label>/, and loading takes
#   that directory. Nothing reads after step 4, so the record's handoff
#   section is empty.
# Usage: from its2s.estimate_record import EstimateRecord, build_record
# Dependencies: none beyond the package

import os
from dataclasses import dataclass, field

from .records import SealedRecord, seal_record

SCHEMA_VERSION = 1
RECORD_DIR = "04_estimate"
RECORD_FILE = "04-estimate.json"
REPORT_FILE = "04-estimate_report.txt"

# The caveats every report carries until the interval method is rebuilt and
# its coverage measured. Each is (applies, text); applies takes the interval
# type and the family.
_CAVEATS = (
    (lambda kind, family: True,
     "The coverage of this interval has not been validated, and it is known to "
     "undercover; worst for differenced ARIMA at long horizons and for "
     "prophet_xgb, whose in-sample residuals understate its forecast error."),
    (lambda kind, family: kind == "prediction",
     "The added noise is not propagated through the model's dynamics."),
    (lambda kind, family: True,
     "The residual noise is pooled across seasons."),
    (lambda kind, family: family == "arima",
     "The model's order is held fixed, so uncertainty in choosing it is not "
     "included."),
)


def caveats(interval_type, family):
    """The caveats that apply to a run, as the record and report state them."""
    return [text for applies, text in _CAVEATS if applies(interval_type, family)]


def build_record(fields):
    """The sealed estimate record, as a JSON-ready dict.

    fields holds every top-level value but the schema version, the seal and
    the handoff, which is empty: no step reads after this one.
    """
    record = {"schema_version": SCHEMA_VERSION, **fields, "seal": None, "handoff": {}}
    return seal_record(record)


@dataclass(frozen=True)
class EstimateRecord(SealedRecord):
    """A sealed estimate record, read-only.

    Built from the JSON on disk (load, given the run directory or the record
    file) or from the dict estimate_excess wrote (from_dict); either way the
    seal is verified first.
    """

    KIND = "estimate"
    RECORD_DIR = RECORD_DIR
    RECORD_FILE = RECORD_FILE
    SEALED_WHEN = "estimated"
    REDO = "Estimate again under a new run_label."

    schema_version: int
    package_version: str
    estimated_at: str
    sealed_at: str
    seal: object
    label: str
    run_label: str
    candidate_id: str
    family: str
    interval: object
    bootstrap: object
    per_step: object
    periods: object
    percent_null: object
    caveats: object
    presentation: object
    stamps: object
    provenance: object
    data_fingerprint: object
    handoff: object
    path: str | None = None
    _json: str = field(default="", repr=False, compare=False)

    @classmethod
    def record_path(cls, path):
        """Accept the record file or the run directory that holds it.

        A label holds one directory per run under 04_estimate/, so the label
        directory alone does not name a record.
        """
        path = os.fspath(path)
        if os.path.isdir(path):
            path = os.path.join(path, cls.RECORD_FILE)
        return path

    def report(self):
        """The text report, exactly as written to 04-estimate_report.txt."""
        from .estimate_report import render_report
        return render_report(self.to_dict())

    def excess_table(self):
        """One row per period, the same columns and values as
        04-excess_table.csv, rendered from this record's fields.

        Returns
        -------
        pd.DataFrame
        """
        from .estimate_tables import excess_table
        return excess_table(self.to_dict())
