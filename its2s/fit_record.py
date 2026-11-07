# Description: The sealed fit record that step 3 writes and step 4 reads,
#   and the saved model beside it. The JSON on disk is canonical; FitRecord
#   is a frozen, typed view of it that also hands back the fitted model:
#   the one held in memory after fit_final_model, or the saved one, loaded
#   from disk once every saved file matches the hash the record sealed. The
#   record holds no error figure of any kind: the final fit is never scored.
# Usage: from its2s.fit_record import FitRecord, build_record
# Dependencies: none beyond the package

import hashlib
import os
from dataclasses import dataclass, field

from .models.families import load_family_class
from .records import SealedRecord, seal_record

SCHEMA_VERSION = 1
RECORD_DIR = "03_fit"
RECORD_FILE = "03-fit.json"
MODEL_DIR = "03-model"
# The saved model's place under the label directory, as the record states it:
# with forward slashes on every system, so the record is the same everywhere.
MODEL_PATH = f"{RECORD_DIR}/{MODEL_DIR}"


def file_digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def model_files(directory):
    """{path relative to directory, with forward slashes: sha256} for every
    file the saved model holds."""
    out = {}
    for root, _dirs, files in os.walk(directory):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, directory).replace(os.sep, "/")
            out[rel] = file_digest(full)
    return dict(sorted(out.items()))


def build_handoff(check_handoff, body, model):
    """Select what step 4 reads: everything the check carried forward, the
    span the model was fit on, and where the model is."""
    h = check_handoff
    return {
        "label": h["label"],
        "data": dict(h["data"]),
        "question": dict(h["question"]),
        "seasonality": dict(h["seasonality"]),
        "candidate": dict(h["candidate"]),
        "fit_span": dict(body["span"]),
        "forecast_origin": body["forecast_origin"],
        "model": model,
        "sub_periods": dict(h["sub_periods"]),
        "bootstrap_seed": h["bootstrap_seed"],
        "output_path": h["output_path"],
        "stamps": h["stamps"],
    }


def build_record(fields, check_handoff):
    """The sealed fit record, as a JSON-ready dict.

    fields holds every top-level value but the seal and the handoff; the
    handoff is built from the check handoff and those fields.
    """
    record = {
        "schema_version": SCHEMA_VERSION,
        **fields,
        "seal": None,
        "handoff": build_handoff(check_handoff, fields, fields["saved_model"]),
    }
    return seal_record(record)


@dataclass(frozen=True)
class FitRecord(SealedRecord):
    """A sealed fit record, read-only, and the model it describes.

    Built from the JSON on disk (load) or from the dict fit_final_model
    wrote (from_dict); either way the seal is verified first. model()
    returns the fitted model.
    """

    KIND = "fit"
    RECORD_DIR = RECORD_DIR
    RECORD_FILE = RECORD_FILE
    SEALED_WHEN = "fitted"
    REDO = "Declare the design again under a new label."

    schema_version: int
    package_version: str
    fitted_at: str
    sealed_at: str
    seal: object
    label: str
    candidate_id: str
    family: str
    params: object
    seed: int | None
    span: object
    forecast_origin: str
    selected_order: object
    seasonal_components: object
    fitted_values: object
    library_versions: object
    warnings: object
    stamps: object
    provenance: object
    saved_model: object
    data_fingerprint: object
    handoff: object
    path: str | None = None
    _json: str = field(default="", repr=False, compare=False)
    # The model fit_final_model just fit; None for a record read from disk.
    _fitted: object = field(default=None, repr=False, compare=False)

    def model_dir(self):
        """Where the saved model is: under the label directory that holds
        this record's file."""
        if self.path is None:
            raise ValueError("This fit record was not read from disk, so there is no "
                             "saved model beside it; read it with FitRecord.load.")
        label_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.path)))
        return os.path.join(label_dir, *self.handoff["model"]["path"].split("/"))

    def verify_model(self):
        """Raise SealError unless the saved model's files are exactly the
        files the record sealed, byte for byte."""
        from .design import SealError
        directory = self.model_dir()
        sealed = dict(self.handoff["model"]["files"])
        found = model_files(directory) if os.path.isdir(directory) else {}
        changed = sorted(n for n in sealed if n in found and found[n] != sealed[n])
        missing = sorted(set(sealed) - set(found))
        extra = sorted(set(found) - set(sealed))
        if changed or missing or extra:
            what = "; ".join(f"{kind}: {', '.join(names)}" for kind, names in
                             (("changed", changed), ("missing", missing), ("added", extra))
                             if names)
            raise SealError(
                f"The saved model at {directory} does not match the fit record ({what}). "
                "It has been changed since it was fitted, so it is not the model that "
                f"was sealed. {self.REDO}")

    def model(self):
        """The fitted model.

        After fit_final_model, the model it fit. For a record read from
        disk, the saved model, loaded once every file matches its hash.
        """
        if self._fitted is not None:
            return self._fitted
        self.verify_model()
        family = self.handoff["candidate"]["family"]
        return load_family_class(family).load(self.model_dir())
