# Description: The sealed check record that step 2 writes and step 3 reads.
#   The JSON on disk is canonical; CheckRecord is a frozen, typed view of it.
#   The record embeds the judged candidate's step-1 entry and the seal
#   digests of the design and candidates records it descends from, and its
#   HANDOFF section holds exactly what step 3 needs and what step 3 must
#   carry forward for step 4.
# Usage: from its2s.check_record import CheckRecord, build_record
# Dependencies: none beyond the package

from dataclasses import dataclass, field

from .records import SealedRecord, seal_record

SCHEMA_VERSION = 1
RECORD_DIR = "02_check"
RECORD_FILE = "02-check.json"
REPORT_FILE = "02-check_report.txt"


def build_handoff(cand_handoff, label, cid, verdict, stamps):
    """Select what step 3 reads: the facts it fits with, and what it carries
    forward for step 4."""
    h = cand_handoff
    c = h["candidates"][cid]
    split = h["splits"][c["split"]]
    return {
        "label": label,
        "verdict": verdict,
        "data": dict(h["data"]),
        "question": dict(h["question"]),
        "seasonality": dict(h["seasonality"]),
        "candidate": {
            "id": cid, "family": c["family"], "params": c["params"], "m": c["m"],
            "seed": c["seed"], "split": c["split"],
            "train": dict(split["train"]), "test": dict(split["test"]),
            "post_event": dict(split["post_event"]),
        },
        "sub_periods": dict(h["sub_periods"]),
        "bootstrap_seed": h["bootstrap_seed"],
        "output_path": h["output_path"],
        "stamps": stamps,
    }


def build_record(fields, cand_handoff):
    """The sealed check record, as a JSON-ready dict.

    fields holds every top-level value but the seal and the handoff; the
    handoff is built from the candidates handoff and those fields.
    """
    stamps = {
        "split_chosen_after_evidence": fields["split_chosen_after_evidence"],
        "prior_checks": fields["prior_checks"]["count"],
        # [step 0 threshold, applied threshold]; applied is None when the
        # override deselected the criterion.
        "overrides": {cid: [c["step0_threshold"], c["override"]]
                      for cid, c in fields["criteria"].items() if c["overridden"]},
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        **fields,
        "seal": None,
        "handoff": build_handoff(cand_handoff, fields["label"], fields["candidate_id"],
                                 fields["verdict"], stamps),
    }
    return seal_record(record)


@dataclass(frozen=True)
class CheckRecord(SealedRecord):
    """A sealed check record, read-only.

    Built from the JSON on disk (load) or from the dict check_selected_model
    wrote (from_dict); either way the seal is verified first. A record
    carrying FAIL is a complete record: the analysis it belongs to stops.
    """

    KIND = "check"
    RECORD_DIR = RECORD_DIR
    RECORD_FILE = RECORD_FILE
    SEALED_WHEN = "checked"
    REDO = "Declare the design again under a new label."

    schema_version: int
    package_version: str
    checked_at: str
    first_fit_at: str | None
    sealed_at: str
    seal: object
    label: str
    candidate_id: str
    rationale: str | None
    verdict: str
    criteria: object
    origins: object
    aggregate: object
    max_seasonal_gap: float | None
    candidate: object
    screening_facts: object
    multiplicity: object
    split_chosen_after_evidence: bool
    prior_checks: object
    test_window_spent: bool
    provenance: object
    data_fingerprint: object
    pre_event_fingerprint: object
    handoff: object
    path: str | None = None
    _json: str = field(default="", repr=False, compare=False)

    def report(self):
        """The text report, exactly as written to 02-check_report.txt."""
        from .check_report import render_report
        return render_report(self.to_dict())
