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
