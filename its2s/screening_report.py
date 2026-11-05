# Description: The step 1 report, rendered from the candidates record alone.
#   It computes nothing: every number is read from the record, including the
#   embedded design record, so the report and the record cannot disagree.
#   Grouped so that invalid comparisons cannot share a table: an index that
#   is not a ranking, one table per combination with every mean followed by
#   the per-fold figures it averages, the search spread beside the default
#   configuration, which comparisons are valid, the failed candidates, and
#   what step 2 needs. It states facts and never makes a choice.
# Usage: from its2s.screening_report import render_report
# Dependencies: none beyond the package

from .design import _count, _fmt_threshold

RULE = "=" * 80
THIN = "-" * 80
SCREENING = ("S1", "S2", "S3", "S4", "S5")


def _num(value, places):
    return "n/a" if value is None else f"{value:.{places}f}"


def _verdict_word(c):
    return {"pass": "PASS", "fail": "FAIL", "n/a": "n/a", "not-selected": "--"}[c["verdict"]]


def _window(w):
    return f"{w['start']} .. {w['last']}"


def _order(s):
    if s is None:
        return "n/a"
    p, d, q = s["order"]
    P, D, Q, m = s["seasonal_order"]
    return f"({p},{d},{q})({P},{D},{Q},{m})" + (" with intercept" if s["with_intercept"]
                                               else " no intercept")


def _header(r, add):
    d = r["design"]
    fp = r["data_fingerprint"]
    add(f"STEP 1 -- CANDIDATES RECORD, SEALED          series: {d['label']}")
    add(f"its2s {r['package_version']}   seed {d['report_fields']['seed']}")
    add(f"  screening started  {r['screening_started_at']}")
    add(f"  first fit          {r['first_fit_at']}")
    add(f"  sealed             {r['sealed_at']}")
    add(f"  data fingerprint   {fp['algorithm']} {fp['digest']}")
    rows = fp["rows"]
    add(f"                     {_count(rows['n'], 'row')}, {rows['first']} to {rows['last']}; "
        f"columns {', '.join(fp['columns'])}")


def _declared(r, add):
    d = r["design"]
    budgets = r["budgets"]
    add("")
    add(f"DECLARED BEFORE THE SEARCH                    {d['declared_at']}")
    fams = ", ".join(f"{n} ({_count(t, 'trial')})" for n, t in budgets["n_trials"].items())
    add(f"  families      {fams}")
    if budgets["unequal"]:
        add("                BUDGETS ARE UNEQUAL between families -- reported with every score")
    for name, f in d["families"].items():
        for flag in f["flags"]:
            add(f"                NON-DEFAULT {name}: {flag}")
    splits = ", ".join(f"{sid} (primary)" if s["primary"] else sid
                       for sid, s in d["splits"].items())
    add(f"  splits        {splits}")
    add(f"  cv designs    {', '.join(d['cvs'])}")
    mult = r["multiplicity"]
    n_fam = len(d["families"])
    add(f"  sweep         {n_fam} famil{'y' if n_fam == 1 else 'ies'} x "
        f"{_count(len(d['splits']), 'split')} x {_count(len(d['cvs']), 'cv design')} = "
        f"{_count(mult['n_candidates'], 'candidate')}; {_count(mult['n_fits'], 'fit')} run "
        f"({d['report_fields']['sweep_fits']} declared at most)")
    q = d["question"]
    primary = next(s for s in d["splits"].values() if s["primary"])
    add(f"  event {q['event_date']}   H = {q['H']}   post-event "
        f"{_window(primary['post_event'])}")
    for sid, o in d["check"]["splits"].items():
        add(f"  check origins (sealed)   {sid}: {' '.join(o['dates'])}")
        extra = []
        if o["phase_origin"] is not None:
            extra.append(f"event-phase origin {o['phase_origin']}")
        if o["max_gap"] is not None:
            extra.append(f"max seasonal gap {o['max_gap']:.2f}")
        if extra:
            add(f"                           {'; '.join(extra)}")
    sw = d["seasonality"]
    how = "declared" if sw["declared"] else "default"
    cycles = ", ".join(f"{c} (m = {sw['periods'][c]:.2f})" for c in sw["cycles"]) or "none"
    add(f"  {d['data']['freq']} sampling; cycles {cycles}, {how}")


def _criteria(r, add):
    d = r["design"]
    crit = d["criteria"]
    add("")
    add("SCREENING CRITERIA                          threshold     set by   basis       selected")
    for cid in SCREENING:
        c = crit[cid]
        add(f"  {cid}  {c['statistic']:<38} {_fmt_threshold(c['op'], c['threshold']):<13} "
            f"{'default' if c['threshold_defaulted'] else 'user':<8} {c['basis']:<11} "
            f"{'yes' if c['selected'] else 'no'}")
    changed = [cid for cid in SCREENING if not crit[cid]["threshold_defaulted"]]
    if changed:
        add(f"  NON-DEFAULT THRESHOLD: {', '.join(changed)}")
    ms = {c["mase_m"] for c in r["candidates"].values()}
    add(f"  S1's seasonal naive repeats the value {', '.join(str(m) for m in sorted(ms))} "
        "rows earlier (one turn of the longest declared cycle; 1 when none).")
    add("  A criterion you did not select still prints its number; only its verdict is")
    add("  withheld (shown --). S2 is not computable below 3 folds. S4 takes its verdict")
    add("  on the SMALLEST fold; every fold's figure and the full window are shown.")


def _index(r, add):
    add("")
    add("CANDIDATE INDEX -- read an id from here and hand it to step 2. NOT A RANKING.")
    width = max(len(cid) for cid in r["candidates"])
    add(f"  {'candidate_id':<{width}}  " + "  ".join(f"{s:>4}" for s in SCREENING)
        + "   passed")
    for cid, c in r["candidates"].items():
        if c["status"] == "failed":
            add(f"  {cid:<{width}}  FAILED at {c['failure']['stage']}; cannot be chosen")
            continue
        verdicts = "  ".join(f"{_verdict_word(c['criteria'][s]):>4}" for s in SCREENING)
        p = c["passed"]
        add(f"  {cid:<{width}}  {verdicts}   {p['n_passed']}/{p['n_applicable']}")


def _family_rows(c, add):
    fam = c["family"]
    if c["status"] == "failed":
        last = c["failure"]["error"].strip().splitlines()[-1]
        add(f"  {fam:<14} FAILED at {c['failure']['stage']}: {last}")
        return
    b, crit = c["best"], c["criteria"]
    add(f"  {fam:<14} {_num(b['mean_rmse'], 2):>12}   "
        f"{_num(crit['S1']['value'], 3):>7} {_verdict_word(crit['S1']):<4}   "
        f"{_num(crit['S2']['value'], 3):>7} {_verdict_word(crit['S2']):<4}   "
        f"{_num(crit['S3']['value'], 3):>7} {_verdict_word(crit['S3']):<4}")
    for f in b["folds"]:
        add(f"      fold {f['k']}  RMSE {_num(f['rmse'], 2):>10}   MASE {_num(f['mase'], 3):>7}")
    add(f"      in-sample RMSE {_num(c['whole_window']['insample_rmse'], 2)} on the whole "
        f"training window ({c['whole_window']['n']} obs), S3's denominator")
    if fam == "arima":
        fs = c["fitted_structure"]
        distinct = {_order(s) for s in fs["folds"]}
        add(f"      selected order, whole window {_order(fs['whole_window'])}; the folds "
            f"selected {_count(len(distinct), 'distinct order')}")
    for cid in ("S1", "S2", "S3"):
        if crit[cid]["reason"] is not None:
            add(f"      {cid} n/a: {crit[cid]['reason']}")


def _spread_rows(c, add):
    fam = c["family"]
    if c["search"] is None:
        add(f"  {fam:<14} no search (n_trials = 0): the default configuration was scored")
        return
    s, ref = c["spread"], c["default_reference"]
    n_failed = sum(t["status"] == "failed" for t in c["search"])
    ref_text = (_num(ref["mean_rmse"], 2) if ref["status"] == "ok"
                else f"failed at fold {ref['fold']}")
    if s is None:
        add(f"  {fam:<14} {c['n_trials']:>6}   every setting failed   default {ref_text}")
        return
    add(f"  {fam:<14} {c['n_trials']:>6} {_num(s['best'], 2):>10} {_num(s['median'], 2):>10} "
        f"{_num(s['worst'], 2):>10}   {ref_text:>14}")
    if ref["status"] == "ok":
        folds = "  ".join(f"fold {f['k']} {_num(f['rmse'], 2)}" for f in ref["folds"])
        add(f"      default configuration per fold: {folds}")
    if n_failed:
        add(f"      {_count(n_failed, 'setting')} failed and scored nothing; see the record")


def _combination(r, key, combo, add):
    d = r["design"]
    dc = d["combinations"][key]
    split = d["splits"][combo["split"]]
    cv = d["cvs"][combo["cv"]]
    crit = d["criteria"]
    add("")
    add(f"COMBINATION  {combo['split']} / {combo['cv']}"
        + ("        PRIMARY SPLIT" if split["primary"] else ""))
    add(f"  training  {_window(split['train'])}   {split['train']['n']} obs")
    add(f"  test      {_window(split['test'])}   {split['test']['n']} obs, unspent")
    add(f"  cv        {_count(combo['n_folds'], 'fold')} (the evidence count), "
        f"{cv['fold_eval_obs']} obs scored in each")
    s4, s5 = combo["S4"], combo["S5"]
    add(f"  S4 seasonal coverage, smallest fold  {_num(s4['value'], 2):>6}  "
        f"{_fmt_threshold('>=', crit['S4']['threshold']):<12} {s4['verdict']}"
        "  (combination property)")
    add(f"  S5 fold horizon / H                  {_num(s5['value'], 2):>6}  "
        f"{_fmt_threshold('band', crit['S5']['threshold']):<12} {s5['verdict']}"
        "  (combination property)")
    add("  folds (identical across families in this combination)")
    for f in dc["folds"]:
        t, ev = f["train"], f["evaluate"]
        cyc = "" if t["cycles"] is None else f" ({t['cycles']:.2f} cycles)"
        add(f"    fold {f['k']}  train {t['n']:>4} obs{cyc}   eval {_window(ev)}")
    full = split["train"]["cycles"]
    if full is not None:
        add(f"    seasonal coverage  smallest fold {s4['value']:.2f} cycles   "
            f"full window {full:.2f} cycles")
    cands = [c for c in r["candidates"].values()
             if c["split"] == combo["split"] and c["cv"] == combo["cv"]]
    add("")
    add("  family         mean CV RMSE   mean CV MASE     worst/median     CV/in-sample")
    add("                                      S1                S2                S3")
    for c in cands:
        _family_rows(c, add)
    add("")
    add("  search spread   trials     lowest     median    highest   default config")
    for c in cands:
        _spread_rows(c, add)
    add(THIN)


def _facts(r, add):
    f, mult = r["facts"], r["multiplicity"]
    add("")
    add("WHY THERE ARE SEPARATE TABLES")
    add("  within a combination, across families : valid")
    add("  across cv designs                     : NOT valid on RMSE (different forecast "
        "distance); MASE indicative only")
    add("  across splits                         : NOT valid (different training data and "
        "fold windows)")
    add("")
    add("NO CHOICE IS MADE HERE")
    add("  Error metrics are not on one scale across splits or cv designs, so the")
    add("  candidates cannot be ranked. The analyst selects a candidate_id at step 2.")
    low = f["lowest_mean_cv_rmse"]
    if low is not None:
        add(f"  As a fact, not advice: the lowest mean CV RMSE anywhere is "
            f"{', '.join(low['ids'])} ({_num(low['mean_rmse'], 2)})")
    passing = f["passing_all_applicable"]
    add("  passes every applicable criterion: " + (", ".join(passing) if passing else "none"))
    add(f"  {_count(mult['n_candidates'], 'candidate')} over {_count(mult['n_fits'], 'fit')}. "
        "A choice made by eye is still a choice among")
    add("  these; report the count in the methods section.")


def _failures(r, add):
    failed = {cid: c for cid, c in r["candidates"].items() if c["status"] == "failed"}
    if not failed:
        return
    add("")
    add(f"FAILED CANDIDATES  ({len(failed)}; each is in the record with its traceback)")
    for cid, c in failed.items():
        last = c["failure"]["error"].strip().splitlines()[-1]
        add(f"  {cid}  at {c['failure']['stage']}: {last}")


def _next(add):
    add("")
    add("NEXT -- step 2")
    add("  candidate_id  one id from the index above")
    add("  rationale     one sentence, optional; stamped into the step-2 record if given")
    add("  choosing a candidate on a non-primary split stamps split_chosen_after_evidence=true")
    add("  into the step-2 record and every record after it")
    add("  Step 2 forecasts H steps from each sealed check origin, ONCE. It is the")
    add("  only step that can end the analysis; the test window is not reusable.")
    add(RULE)


def render_report(record):
    """The report text for a candidates record (a plain dict)."""
    lines = []
    add = lines.append
    _header(record, add)
    _declared(record, add)
    _criteria(record, add)
    _index(record, add)
    add("")
    add(RULE)
    add("DETAIL BY COMBINATION.  Every row inside one table is strictly comparable:")
    add("same training data, same folds, same forecast distance. Comparisons ACROSS")
    add("tables are not valid -- see the foot of this report.")
    add(RULE)
    for key, combo in record["combinations"].items():
        _combination(record, key, combo, add)
    _facts(record, add)
    _failures(record, add)
    _next(add)
    return "\n".join(lines) + "\n"
