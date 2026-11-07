# Description: The step 2 report, rendered from the check record alone. It
#   computes nothing: every number is read from the record, so the report and
#   the record cannot disagree. The header carries every stamp a reader must
#   see before the verdict (a split chosen after evidence, earlier checks on
#   the same data, overridden thresholds); then the criteria, one line per
#   origin so the spread behind each mean is visible, the seasonal gap the
#   origins leave, the screening facts, and what the errors do and do not
#   describe.
# Usage: from its2s.check_report import render_report
# Dependencies: none beyond the package

from .design import _count, _fmt_threshold

RULE = "=" * 80
THIN = "-" * 80
CHECK = ("C1", "C2", "C3")
STATISTIC = {"C1": "mean MASE across origins",
             "C2": "mean origin RMSE / step 1 CV RMSE",
             "C3": "worst origin RMSE / median origin RMSE"}


def _num(value, places):
    return "n/a" if value is None else f"{value:.{places}f}"


def _verdict_word(c):
    return {"pass": "PASS", "fail": "FAIL", "n/a": "n/a", "not-selected": "--"}[c["verdict"]]


def _header(r, add):
    add(f"STEP 2 -- CHECK RECORD, SEALED          series: {r['label']}")
    add(f"its2s {r['package_version']}")
    add(f"  checked            {r['checked_at']}")
    add(f"  first fit          {r['first_fit_at']}")
    add(f"  sealed             {r['sealed_at']}")
    fp = r["data_fingerprint"]
    add(f"  data fingerprint   {fp['algorithm']} {fp['digest']}")
    add("")
    add(f"  candidate          {r['candidate_id']}")
    add(f"  VERDICT            {r['verdict']}")
    add(f"  rationale          {r['rationale'] if r['rationale'] else 'none given'}")
    if r["split_chosen_after_evidence"]:
        add(f"  SPLIT CHOSEN AFTER EVIDENCE: {r['candidate']['split']} is not the "
            "primary split")
    prior = r["prior_checks"]
    add(f"  prior checks on this data   {prior['count']}")
    for p in prior["records"]:
        when = (p["checked_at"] or "unknown date")[:10]
        broken = "" if p["seal_intact"] else "  (record no longer matches its seal)"
        add(f"    {p['label']}  {when}  {p['candidate_id']}  {p['verdict']}{broken}")
    for u in prior["unreadable"]:
        add(f"  UNREADABLE earlier check record, not counted: {u['folder']} ({u['why']})")
    for cid in CHECK:
        c = r["criteria"][cid]
        if c["overridden"]:
            applied = ("deselected" if c["override"] is None
                       else _fmt_threshold(c["op"], c["threshold"]))
            step0 = _fmt_threshold(c["op"], c["step0_threshold"])
            add(f"  OVERRIDDEN {cid}      step 0 {step0}"
                f"{'' if c['step0_selected'] else ' (not selected)'}  ->  applied {applied}")
    add("  The test window of this label is now SPENT.")


def _criteria(r, add):
    add("")
    add("CHECK CRITERIA                                   value   threshold    set by   verdict")
    for cid in CHECK:
        c = r["criteria"][cid]
        threshold = _fmt_threshold(c["op"], c["threshold"]) if c["selected"] else "--"
        set_by = ("override" if c["overridden"]
                  else "default" if c["defaulted"] else "user")
        add(f"  {cid}  {STATISTIC[cid]:<40} {_num(c['value'], 3):>7}   {threshold:<12} "
            f"{set_by:<8} {_verdict_word(c)}")
        if c["reason"] is not None:
            add(f"      {cid} n/a: {c['reason']}")
    a = r["aggregate"]
    add(f"  mean origin RMSE {_num(a['mean_rmse'], 2)}; step 1 CV RMSE {_num(a['cv_rmse'], 2)}; "
        f"worst {_num(a['worst_rmse'], 2)}, median {_num(a['median_rmse'], 2)}")
    add(f"  C1's seasonal naive repeats the value {a['mase_m']} rows earlier, as at S1.")
    add("  A criterion not selected still prints its value; only its threshold and")
    add("  verdict are withheld (shown --), and it does not enter the verdict.")
    for reason in a["verdict_reasons"]:
        add(f"  NOT PASSED: {reason}")


def _origins(r, add):
    add("")
    add(f"ORIGINS  ({_count(len(r['origins']), 'origin')}, each fit on every row before it "
        f"and forecast H = {r['handoff']['question']['H']} rows)")
    add("    origin       fit span                    n        RMSE        MAE     MASE")
    for o in r["origins"]:
        mark = "  event phase" if o["event_phase"] else ""
        f = o["fit"]
        if o["status"] == "failed":
            add(f"  {o['k']} {o['date']}   {f['first']} .. {f['last']}  {f['n']:>4}   "
                f"FAILED: {o['error']}{mark}")
            continue
        add(f"  {o['k']} {o['date']}   {f['first']} .. {f['last']}  {f['n']:>4}   "
            f"{_num(o['rmse'], 2):>9} {_num(o['mae'], 2):>10} {_num(o['mase'], 3):>8}{mark}")
    gap = r["max_seasonal_gap"]
    if gap is not None:
        add(f"  maximum seasonal gap {gap:.2f}: the largest share of the cycle no origin tests")


def _screening(r, add):
    s, m = r["screening_facts"], r["multiplicity"]
    add("")
    add("FROM STEP 1")
    add(f"  {r['candidate_id']} passed {s['n_passed']} of {s['n_applicable']} applicable "
        f"screening criteria" + (" (all)" if s["passed_all"] else ""))
    add(f"  {_count(s['n_candidates_passed_all'], 'candidate')} of "
        f"{m['n_candidates']} passed every applicable criterion")
    add(f"  {_count(m['n_fits'], 'fit')} at screening, {_count(m['n_check_fits'], 'fit')} "
        "here; report both counts in the methods section.")


def _caveat(r, add):
    add("")
    add("WHAT THESE ERRORS DESCRIBE")
    add("  The candidate as fit at each origin, on the rows before it: the procedure,")
    add("  not the final model. Step 3 fits once more on every pre-event row and is not")
    add("  scored; its forecast error is unknowable by design.")
    if r["verdict"] == "FAIL":
        add("")
        add(f"FAILED. No later step will run on {r['label']}.")
    add(RULE)


def render_report(record):
    """The report text for a check record (a plain dict)."""
    lines = []
    add = lines.append
    _header(record, add)
    _criteria(record, add)
    _origins(record, add)
    _screening(record, add)
    _caveat(record, add)
    return "\n".join(lines) + "\n"
