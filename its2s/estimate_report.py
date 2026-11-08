# Description: The step 4 report, 04-estimate_report.txt, rendered from the
#   estimate record alone. It computes nothing, so the report and the record
#   cannot disagree. The header carries the run, the candidate, the interval
#   in words and every stamp carried from the earlier steps; then the excess
#   by period with percent excess and the diagnostic flag; the bootstrap as
#   resolved; and the caveats, which every report prints until the interval
#   method is rebuilt and its coverage measured.
# Usage: from its2s.estimate_report import render_report
# Dependencies: none beyond the package

import textwrap

from .design import _count
from .estimate_tables import interval_words
from .metrics.excess import PCT_FORECAST_NOT_POSITIVE, PCT_PATH_NOT_POSITIVE

RULE = "=" * 80
THIN = "-" * 80
PROVENANCE = ("design", "candidates", "check", "fit")
PERCENT_NULL = {
    PCT_FORECAST_NOT_POSITIVE: "percent excess not defined: the forecast total is not "
                               "positive",
    PCT_PATH_NOT_POSITIVE: "percent bounds not defined: some bootstrap path's "
                           "counterfactual total is not positive",
}


def _num(value, places=1):
    return "n/a" if value is None else f"{value:,.{places}f}"


def _bounds(lo, hi, places=1):
    if lo is None or hi is None:
        return "n/a"
    return f"[{_num(lo, places)}, {_num(hi, places)}]"


def _wrap(text, indent):
    return textwrap.fill(text, width=80, initial_indent=indent,
                         subsequent_indent=" " * len(indent))


def _header(r, add):
    words = interval_words(r["interval"])
    add(f"STEP 4 -- ESTIMATE, SEALED          series: {r['label']}")
    add(f"its2s {r['package_version']}")
    add(f"  estimated          {r['estimated_at']}")
    add(f"  sealed             {r['sealed_at']}")
    fp = r["data_fingerprint"]
    add(f"  data fingerprint   {fp['algorithm']} {fp['digest']}")
    add("")
    add(f"  run_label          {r['run_label']}")
    add(f"  candidate          {r['candidate_id']}")
    add(f"  interval           {words}, NOT VALIDATED")
    s = r["stamps"]
    if s["split_chosen_after_evidence"]:
        add("  SPLIT CHOSEN AFTER EVIDENCE: the candidate's split is not the primary split")
    add(f"  prior checks on this data   {s['prior_checks']}")
    for cid, (step0, applied) in sorted(s["overrides"].items()):
        add(f"  OVERRIDDEN {cid}      step 0 {step0}  ->  applied "
            f"{'deselected' if applied is None else applied}")


def _periods(r, add):
    words = interval_words(r["interval"])
    periods = r["periods"]
    w = max(len("period"), *(len(p["name"]) for p in periods))
    add("")
    add(f"EXCESS BY PERIOD  ({words}, not validated)")
    add(f"  {'period':<{w}}  first .. last (inclusive)    n    observed    forecast"
        "      excess  interval")
    for p in periods:
        add(f"  {p['name']:<{w}}  {p['first']} .. {p['last']}  {p['n']:>5}  "
            f"{_num(p['observed']):>10}  {_num(p['forecast']):>10}  "
            f"{_num(p['excess']):>10}  {_bounds(p['excess_lo'], p['excess_hi'])}")
    add("")
    add(f"  {'period':<{w}}   percent  interval              median path - point  flag")
    for p in periods:
        flag = "POINT OUTSIDE INTERVAL" if p["point_outside_interval"] else "--"
        pct = _bounds(p["excess_pct_lo"], p["excess_pct_hi"])
        add(f"  {p['name']:<{w}}  {_num(p['excess_pct']):>7}%  {pct:<20}  "
            f"{_num(p['median_path_minus_point'], 2):>19}  {flag}")
    for p in periods:
        if p["excess_pct_null"] is not None:
            add(f"  {p['name']}: {PERCENT_NULL[p['excess_pct_null']]}")
    add("  Totals and percent excess are formed within each bootstrap path, then")
    add("  summarised across paths. The flag marks a point estimate outside its own")
    add("  interval; median path - point shows how far the paths sit from it.")


def _failures(b):
    f = b["failures"]
    parts = [f"{kind} {n}" for kind, n in f["exception"].items()]
    parts += [f"{kind} {f[kind]}" for kind in ("non_finite", "wrong_length") if f[kind]]
    return ", ".join(parts) if parts else "none"


def _bootstrap(r, add):
    b = r["bootstrap"]
    bl = b["block_length"]
    add("")
    add("BOOTSTRAP  (moving block, as resolved)")
    add(f"  replications       {b['n_requested']} requested, {b['n_succeeded']} succeeded")
    add(f"  failed             {_failures(b)}")
    warned = ", ".join(f"{k} {n}" for k, n in b["refit_warnings"].items())
    add(f"  refit warnings     {warned or 'none'}")
    add(f"  block length       {bl['value']} rows (argument {bl['argument']!r}, "
        f"rule {bl['rule']})")
    add(f"  start-up rows      {b['start_up_rows']}, left out of the residual pool")
    add(f"  residual pool      {b['n_pool']} rows, "
        f"{'centred' if b['pool_centred'] else 'not centred'}")
    n = b["noise"]
    if n is None:
        add("  noise              none: a confidence interval for the counterfactual mean")
    else:
        add(f"  noise              {n['pool']};")
        add(f"                     blocks of {n['block_length']} rows; propagation "
            f"{n['propagation']}")
    add(f"  seed               {b['seed']}")


def _caveats(r, add):
    add("")
    add("CAVEATS")
    for text in r["caveats"]:
        add(_wrap(text, "  - "))


def _provenance(r, add):
    add("")
    add("PROVENANCE")
    for name in PROVENANCE:
        p = r["provenance"][name]
        add(f"  {name:<11}{p['seal']}")
        add(f"  {'':<11}{p['path']}")


def render_report(record):
    """The step 4 report text, from the estimate record alone.

    Parameters
    ----------
    record : dict

    Returns
    -------
    str
    """
    lines = []
    add = lines.append
    add(RULE)
    _header(record, add)
    add(THIN)
    _periods(record, add)
    _bootstrap(record, add)
    _caveats(record, add)
    _provenance(record, add)
    add(RULE)
    return "\n".join(lines) + "\n"
