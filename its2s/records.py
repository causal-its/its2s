# Description: What every sealed step record shares: the canonical JSON form,
#   the seal (a hash of the record's own content), the on-disk JSON text, the
#   read-only view of a loaded record, and the all-or-nothing directory write.
#   The design record (step 0) and the candidates record (step 1) both use
#   these, so a record sealed by one step is checked by the same code in the
#   next. File names and record fields stay with each record's own module.
# Usage: from its2s.records import seal_record, seal_matches, record_json
# Dependencies: none beyond the standard library

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from types import MappingProxyType

SEAL_ALGORITHM = "sha256"


# ---------------------------------------------------------------------------
# Canonical form and seal
# ---------------------------------------------------------------------------

def canonical_json(value):
    """value as one fixed byte string: sorted keys, no whitespace, ASCII.

    Two values with the same content give the same bytes however they were
    built or indented, which is what the seal hashes and what "identical"
    means when two records are compared.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _canonical(record):
    """The record's content as canonical bytes, seal excluded.

    Re-indenting the file does not break the seal; any change to a value does.
    """
    return canonical_json({k: v for k, v in record.items() if k != "seal"})


def seal_digest(record):
    """The hash that seals a record."""
    return hashlib.sha256(_canonical(record)).hexdigest()


def seal_record(record):
    """Set the record's seal to the hash of its content; return the record."""
    record["seal"] = {"algorithm": SEAL_ALGORITHM, "digest": seal_digest(record)}
    return record


def seal_matches(record):
    """True when the record's seal is the hash of its own content."""
    seal = record.get("seal")
    return (isinstance(seal, dict) and seal.get("algorithm") == SEAL_ALGORITHM
            and seal.get("digest") == seal_digest(record))


# ---------------------------------------------------------------------------
# On disk and in memory
# ---------------------------------------------------------------------------

def record_json(record):
    """The record as written to disk."""
    return json.dumps(record, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def freeze(value):
    """A read-only copy: dicts become mapping proxies, lists tuples."""
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value


class SealedRecord:
    """Reading and verifying a sealed step record; shared by every step.

    A subclass is a frozen dataclass whose fields are the record's top-level
    keys plus path, _json and any other underscored field it keeps beside
    the record (none is record content), and which sets the class
    attributes below.
    Nested sections are read-only mappings and tuples; to_dict() returns a
    fresh plain copy.
    """

    KIND = ""            # "design", "candidates": names the record in messages
    RECORD_DIR = ""      # the step's directory under the label
    RECORD_FILE = ""     # the record file in it
    SEALED_WHEN = ""     # "declared": what made the record
    REDO = ""            # what to do instead of using an edited record

    @classmethod
    def record_path(cls, path):
        """Accept the record file, its step directory, or the label directory."""
        path = os.fspath(path)
        if os.path.isdir(os.path.join(path, cls.RECORD_DIR)):
            path = os.path.join(path, cls.RECORD_DIR)
        if os.path.isdir(path):
            path = os.path.join(path, cls.RECORD_FILE)
        return path

    @classmethod
    def from_dict(cls, record, path=None, **kept):
        """The record from a dict; raises SealError if the seal does not
        match the content. kept sets underscored fields held beside it."""
        from .design import SealError
        if not seal_matches(record):
            where = f" at {path}" if path else ""
            raise SealError(
                f"The {cls.KIND} record{where} does not match its seal: it has been "
                f"edited since it was {cls.SEALED_WHEN}, so it is not the {cls.KIND} "
                f"that was sealed. {cls.REDO}")
        # path and the underscored fields are not record content.
        names = [f for f in cls.__dataclass_fields__
                 if f != "path" and not f.startswith("_")]
        return cls(**{n: freeze(record[n]) for n in names}, path=path,
                   _json=json.dumps(record), **kept)

    @classmethod
    def load(cls, path):
        """Read and verify a record from the file, its step directory, or the
        label directory that holds it."""
        from .design import SealError
        file = cls.record_path(path)
        try:
            with open(file, encoding="utf-8") as fh:
                record = json.load(fh)
        except FileNotFoundError:
            raise SealError(f"No {cls.KIND} record at {file}.") from None
        except json.JSONDecodeError as e:
            raise SealError(f"The {cls.KIND} record at {file} is not valid JSON ({e}); "
                            f"it has been edited since it was {cls.SEALED_WHEN}.") from None
        return cls.from_dict(record, path=file)

    def to_dict(self):
        """A fresh, mutable copy of the record as JSON values."""
        return json.loads(self._json)


@contextmanager
def atomic_directory(target):
    """Build a new directory in full, then move it into place in one step.

    Yields a hidden directory beside target to write into. On a clean exit
    every file in it, at any depth, is made read-only and the directory is
    renamed to target; on any error it is removed, so a failure part-way
    leaves nothing behind to block a retry. target must not exist.
    """
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = os.path.join(parent, f".{os.path.basename(target)}.partial-{uuid.uuid4().hex}")
    os.mkdir(tmp)
    try:
        yield tmp
        make_read_only(tmp)
        os.rename(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def make_read_only(directory):
    """Mark every file under directory read-only; directories are left as they are."""
    for root, _dirs, files in os.walk(directory):
        for name in files:
            os.chmod(os.path.join(root, name), 0o444)
