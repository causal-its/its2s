# Description: The sealed candidates record that step 1 writes and step 2
#   reads. The JSON on disk is canonical; CandidatesRecord is a frozen, typed
#   view of it. The record embeds the design record it screened, and its
#   HANDOFF section holds exactly what step 2 needs: the design facts carried
#   forward and, per candidate, the family, the exact parameters, the
#   tuning error and whether it passed every applicable criterion.
# Usage: from its2s.candidates_record import CandidatesRecord, build_record
# Dependencies: none beyond the package

from dataclasses import dataclass, field

from .records import SealedRecord, seal_record

SCHEMA_VERSION = 1
RECORD_DIR = "01_screening"
RECORD_FILE = "01-candidates.json"
REPORT_FILE = "01-candidates_report.txt"

# Step 2's criteria, carried forward from the design so step 2 reads the
# candidates handoff only.
CHECK_CRITERIA = ("C1", "C2", "C3")


def build_handoff(design_handoff, body):
    """Select what step 2 reads: the design facts it needs, carried forward,
    and each candidate's choice-relevant fields."""
    h = design_handoff
    return {
        "data": dict(h["data"]),
        "question": dict(h["question"]),
        "seasonality": dict(h["seasonality"]),
        "splits": dict(h["splits"]),
        "origins": dict(h["origins"]),
        "criteria": {cid: dict(h["criteria"][cid]) for cid in CHECK_CRITERIA},
        "sub_periods": dict(h["sub_periods"]),
        "bootstrap_seed": h["bootstrap_seed"],
        "output_path": h["output_path"],
        # cv_rmse is the tuning metric, the denominator of step 2's C2. A
        # failed candidate carries no params and cannot be chosen.
        "candidates": {
            cid: {"status": c["status"], "family": c["family"], "split": c["split"],
                  "primary": c["primary"],
                  "params": None if c["best"] is None else c["best"]["params"],
                  "m": c["m"], "seed": c["seed"],
                  "cv_rmse": None if c["best"] is None else c["best"]["mean_rmse"],
                  "passed_all": c["passed"]["passed_all"],
                  "n_applicable": c["passed"]["n_applicable"],
                  "n_passed": c["passed"]["n_passed"]}
            for cid, c in body["candidates"].items()},
        "multiplicity": {k: body["multiplicity"][k]
                         for k in ("n_candidates", "n_fits", "n_passed_all")},
    }


def build_record(design, data_fingerprint, body, screening_started_at, sealed_at,
                 package_version):
    """The sealed candidates record, as a JSON-ready dict.

    design is the complete design record, embedded verbatim; body is what
    screening.screen returned.
    """
    record = {
        "schema_version": SCHEMA_VERSION,
        "package_version": package_version,
        "screening_started_at": screening_started_at,
        "first_fit_at": body["first_fit_at"],
        "sealed_at": sealed_at,
        "seal": None,
        "design": design,
        "data_fingerprint": data_fingerprint,
        "combinations": body["combinations"],
        "budgets": body["budgets"],
        "candidates": body["candidates"],
        "multiplicity": body["multiplicity"],
        "facts": body["facts"],
        "handoff": build_handoff(design["handoff"], body),
    }
    return seal_record(record)


@dataclass(frozen=True)
class CandidatesRecord(SealedRecord):
    """A sealed candidates record, read-only.

    Built from the JSON on disk (load) or from the dict screen_models wrote
    (from_dict); either way the seal is verified first.
    """

    KIND = "candidates"
    RECORD_DIR = RECORD_DIR
    RECORD_FILE = RECORD_FILE
    SEALED_WHEN = "screened"
    REDO = "Declare the design again under a new label and screen that."

    schema_version: int
    package_version: str
    screening_started_at: str
    first_fit_at: str | None
    sealed_at: str
    seal: object
    design: object
    data_fingerprint: object
    combinations: object
    budgets: object
    candidates: object
    multiplicity: object
    facts: object
    handoff: object
    path: str | None = None
    _json: str = field(default="", repr=False, compare=False)

    def report(self):
        """The text report, exactly as written to 01-candidates_report.txt."""
        from .screening_report import render_report
        return render_report(self.to_dict())
