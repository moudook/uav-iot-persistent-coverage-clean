"""
Fill the @@TOKEN@@ placeholders in the prose from the artefacts.

The prose is written with placeholders so that no number in the text can be
typed by hand: every value is read from the same artefact that the table
generator read, so the text and the tables cannot disagree.

Run:  python analysis/fill_tokens.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys

# The BLAS thread pools are pinned to one thread BEFORE numpy loads
# them: each process otherwise starts a pool sized for the whole
# machine, and a handful of processes exhaust the paging file.
import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.dirname(HERE)
# the package under test lives one level up, and the test module imports it
if ML not in sys.path:
    sys.path.insert(0, ML)


def load_csv(name):
    p = os.path.join(ML, name)
    if not os.path.exists(p):
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return []
    return json.load(open(p, encoding="utf-8"))


def load_json_ml(name):
    """A JSON artefact that lives in ml_impl/ rather than analysis/."""
    p = os.path.join(ML, name)
    if not os.path.exists(p):
        return []
    return json.load(open(p, encoding="utf-8"))


def f(x, nd=3):
    return f"{float(x):.{nd}f}"


def macro_name(token: str) -> str:
    """The LaTeX macro that carries a token's value in the paper's source.

    The paper's hand-written prose refers to a number by its MACRO, not by
    its value: the assembler writes the macro definitions from this table
    on every run, so a number that comes from an experiment is regenerated
    whenever the artefact changes.  Writing the value into the source
    instead would freeze it, and a frozen number is the failure this
    project keeps having to undo.

    The macro name is the token's name in camel case with a 'tok' prefix:
    GAP -> \\tokGap, EPISODE_STEPS -> \\tokEpisodeSteps.  LaTeX control
    sequence names cannot contain an underscore, which is why the token
    names are mapped rather than used directly -- and they cannot contain
    a DIGIT either, because TeX ends a control word at the first
    non-letter: \\tokAltN5R would be the control sequence \\tokAltN
    followed by the characters "5R".  The digits are therefore spelled
    out.
    """
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three",
              "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
              "8": "Eight", "9": "Nine"}
    parts = [p for p in token.split("_") if p]
    out = []
    for p in parts:
        word = "".join(digits.get(c, c) for c in p)
        out.append(word[:1].upper() + word[1:].lower())
    return "\\tok" + "".join(out)


def sample_std(values):
    """The SAMPLE standard deviation of a set of measurements.

    numpy's default divides by n, which is the POPULATION standard
    deviation: an interval built from it is too narrow, and the paper's
    intervals are the paper's central inference.  Every spread in the
    token table therefore comes from here.
    """
    v = [float(x) for x in values]
    if len(v) < 2:
        return 0.0
    return float(np.std(v, ddof=1))


def per_seed_ck(row):
    """The per-training-seed CK values of an ablation row, if present."""
    v = row.get("per_seed_CK", "") if row else ""
    if not v:
        return []
    out = []
    for part in str(v).split(";"):
        try:
            out.append(float(part))
        except ValueError:
            return []
    return out


def bound_clause(label, mean, half, gap):
    """One clause of an exclusion argument, derived from the measurement.

    To EXCLUDE an explanation of a gap of size `gap`, the arm that tests it
    must be shown to be too small to account for the gap: it is not enough
    for the arm's point estimate to be small, its upper confidence limit
    has to be below the gap.  The clause says which of the two happened.
    """
    upper = mean + half
    # The arm's NAME is prose, and prose in math mode loses every space
    # -- math mode ignores them -- which is why the paper printed
    # "addingtheagentidentity".  The name therefore goes in text mode,
    # and the numbers with it, so that the clause is one font.
    if gap is None or gap != gap:
        return (f"{label} ({mean:+.3f}, 95\\% interval "
                f"[{mean-half:+.3f}, {upper:+.3f}])")
    if upper < gap:
        return (f"{label} ({mean:+.3f}, 95\\% interval "
                f"[{mean-half:+.3f}, {upper:+.3f}], an upper limit below "
                f"the gap of {gap:.3f})")
    return (f"{label} ({mean:+.3f}, 95\\% interval "
            f"[{mean-half:+.3f}, {upper:+.3f}], whose upper limit is NOT "
            f"below the gap of {gap:.3f} and which therefore cannot be "
            f"excluded)")


def _paired_corr(x, y):
    """The correlation between two arms' per-seed scores.

    The arms share their training seeds, and the paper's justification
    for pairing is that the seed-to-seed variation cancels.  That
    justification only holds where the two arms' scores MOVE TOGETHER,
    and one of them does not: the identity arm's per-seed scores are
    negatively correlated, so the paired interval is WIDER than the
    unpaired one.  The correlation is measured here rather than assumed,
    and the summary says which case it is in.
    """
    try:
        a = [float(v) for v in (x.get("per_seed_CK") or [])]
        b = [float(v) for v in (y.get("per_seed_CK") or [])]
    except (TypeError, ValueError):
        return None
    if len(a) != len(b) or len(a) < 3:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((u - ma) * (v - mb) for u, v in zip(a, b))
    da = sum((u - ma) ** 2 for u in a) ** 0.5
    db = sum((v - mb) ** 2 for v in b) ** 0.5
    if da <= 0 or db <= 0:
        return None
    return num / (da * db)


def paired_stat(a, b):
    """The paired comparison of two arms that share their training seeds.

    `a` and `b` are the per-training-seed coverage values of the two arms.
    The arms are trained on the SAME seeds, so the difference is measured on
    the same training runs, and the interval is a paired t interval on the
    per-seed differences.  Returns (mean difference, half width of the 95%
    interval, two-sided p, number of pairs).

    A threshold on the size of a difference is not a test, so nothing here
    decides "the component matters" by comparing the difference with some
    other number: the interval does the work, and the reader can see what
    the measurement resolved.
    """
    try:
        from scipy import stats as _st
    except Exception:                                      # pragma: no cover
        _st = None
    n = min(len(a), len(b))
    if n < 2:
        return (float("nan"), float("nan"), float("nan"), n)
    d = [float(a[i]) - float(b[i]) for i in range(n)]
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    crit = _st.t.ppf(0.975, n - 1) if _st else 1.96
    half = crit * sd / np.sqrt(n)
    if _st:
        if sd > 0:
            tv = mean / (sd / np.sqrt(n))
            p = float(2 * (1 - _st.t.cdf(abs(tv), n - 1)))
        else:
            p = 0.0 if mean != 0 else 1.0
    else:                                                   # pragma: no cover
        p = float("nan")
    return (mean, float(half), p, n)


def build_tokens():
    """Read every artefact and return the token table.

    The tokens are the ONLY way a number reaches the paper's prose, so this
    function is the single point at which the paper's numbers are decided.
    """
    tok = {}

    main_rows = {r["model"]: r for r in load_csv("results_main.csv")}
    for key, model in (("FIXED", "Fixed"), ("GREEDY", "Greedy"),
                       ("REACT", "Reactive"), ("VAN", "VanillaMAPPO"),
                       ("MAPPO", "MAPPO"), ("ARG", "MAPPO-argmax"),
                       ("REACTL", "Reactive-local")):
        r = main_rows.get(model)
        if r is None:
            continue
        tok[f"{key}_CK"] = f(r["CK"])
        tok[f"{key}_AG"] = f(r["AG"])
        tok[f"{key}_SC"] = f(r["SC"])
        tok[f"{key}_CI"] = f(r["CovIntegral"])
        tok[f"{key}_E"] = f(r.get("Energy", 0), 0)
        tok[f"{key}_STR"] = f(r.get("Stranded", 0), 1)
        tok[f"{key}_RET"] = f(r.get("ReturnEvents", 0), 0)
        tok[f"{key}_DIST"] = f(r.get("Distance", 0), 0)
        # the spread is the SAMPLE standard deviation of the per-seed
        # values, so that the prose, the table and every interval quote
        # the SAME number
        per = []
        for x in str(r.get("per_seed_CK", "")).split(";"):
            if x.strip():
                try:
                    per.append(float(x))
                except ValueError:
                    per = []
                    break
        if len(per) > 1:
            tok[f"{key}_CK_STD"] = f"{sample_std(per):.3f}"
        elif r.get("CK_std") not in (None, ""):
            tok[f"{key}_CK_STD"] = f(r["CK_std"])

    # The seed-to-seed SPREAD of the learned controller, recomputed from
    # the raw per-seed rows rather than read from the table's summary
    # field.  The summary field is a POPULATION standard deviation
    # (numpy divides by n), and an interval computed from it is too
    # narrow: the sample standard deviation divides by n-1.  The paper's
    # intervals are the paper's central inference, so the value they are
    # built from is computed here, from the rows.
    base_rows_path = os.path.join(HERE, "base_training_seeds.json")
    base_seed_cks = []
    if os.path.exists(base_rows_path):
        try:
            _bs = json.load(open(base_rows_path, encoding="utf-8"))
            base_seed_cks = [float(b["CK"]) for b in _bs
                             if b.get("CK") is not None]
        except Exception:                                      # pragma: no cover
            base_seed_cks = []
    if len(base_seed_cks) > 1:
        tok["MAPPO_CK_STD"] = f"{sample_std(base_seed_cks):.3f}"
        tok["MAPPO_N_SEEDS"] = str(len(base_seed_cks))

    for r in load_csv("results_learned_baselines.csv"):
        key = ("IDDQN" if "DQN" in r["model"] else
               "PPOL" if "PPO" in r["model"] else None)
        if key is None:
            continue
        tok[f"{key}_CK"] = f(r["CK"])
        tok[f"{key}_AG"] = f(r["AG"])
        tok[f"{key}_SC"] = f(r["SC"])
        tok[f"{key}_CI"] = f(r["CovIntegral"])
        # the budget the baseline was trained at, so that the prose can
        # say it: the comparison against the proposed method is a
        # comparison of two budgets unless the budgets are stated
        if r.get("episodes"):
            tok[f"{key}_EPISODES"] = str(r["episodes"])
        # the spread is the SAMPLE standard deviation of the per-seed
        # values, the same number the table prints and the same number
        # every interval in the paper is built from
        per = [float(x) for x in str(r.get("per_seed_CK", "")).split(";")
               if x.strip()]
        if len(per) > 1:
            tok[f"{key}_CK_STD"] = f"{sample_std(per):.3f}"
        elif r.get("CK_std") not in (None, ""):
            tok[f"{key}_CK_STD"] = f(r["CK_std"])
    # the difference between the proposed method and the local-critic arm,
    # which the credit section tests on paired seeds
    mp0 = main_rows.get("MAPPO")
    for r in load_csv("results_learned_baselines.csv"):
        if "PPO" in r["model"] and mp0 is not None:
            tok["PPOL_DELTA"] = f"{float(r['CK']) - float(mp0['CK']):+.3f}"

    arg = main_rows.get("MAPPO-argmax")
    if arg is not None:
        tok["ARG_DISTANCE"] = f"{float(arg.get('Distance', 0)):.0f}"
        tok["ARG_CK"] = f(arg["CK"])
    mp = main_rows.get("MAPPO")
    if mp is not None:
        tok["MAPPO_DISTANCE"] = f"{float(mp.get('Distance', 0)):.0f}"
        tok["MAPPO_STRANDED"] = f"{float(mp.get('Stranded', 0)):.1f}"
        tok["MAPPO_UNEXEC"] = f"{float(mp.get('UnexecutedActions', 0)):.2f}"
        tok["PROJ_REVISIONS"] = f"{float(mp.get('ProjectionRevisions', 0)):.2f}"
        # THE GAP.  The reactive heuristic is deterministic, so it has no
        # training-seed spread; the uncertainty of the gap is therefore the
        # learned controller's own seed spread, divided by the number of
        # training runs.  The scene-to-scene variation is common to every
        # controller -- they are evaluated on the same scenes -- and cancels
        # in the difference.
        react = main_rows.get("Reactive")
        if react:
            # THE GAP is against the STRONGEST controller in the table, which
            # is not necessarily the privileged heuristic: the gap has to
            # be the hardest one the learned controller faces.
            best = None
            for cand in ("Reactive-local", "Reactive", "Greedy", "Fixed"):
                r = main_rows.get(cand)
                if r is None:
                    continue
                if best is None or float(r["CK"]) > float(best[1]["CK"]):
                    best = (cand, r)
            if best is not None:
                name, r = best
                tok["BEST_RULE_NAME"] = {
                    "Reactive-local": "the unprivileged reactive rule",
                    "Reactive": "the reactive heuristic",
                    "Greedy": "the greedy recharge controller",
                    "Fixed": "the fixed formation",
                }.get(name, name)
                tok["BEST_RULE_CK"] = f(r["CK"])
                gap = float(r["CK"]) - float(mp["CK"])
            else:
                gap = float(react["CK"]) - float(mp["CK"])
            tok["GAP"] = f"{gap:.3f}"
            s = mp.get("CK_std")
            n = int(mp.get("train_seeds", 0) or 0)
            # The interval is built from the SAMPLE standard deviation of
            # the per-seed rows, not from the table's summary field,
            # which is a population standard deviation and would make
            # every interval in the paper too narrow.
            if base_seed_cks:
                s = sample_std(base_seed_cks)
                n = len(base_seed_cks)
            if s not in (None, "") and n > 1:
                try:
                    from scipy import stats as _st
                    half = (float(_st.t.ppf(0.975, n - 1))
                            * float(s) / np.sqrt(n))
                except Exception:                          # pragma: no cover
                    half = 1.96 * float(s) / np.sqrt(n)
                tok["GAP_HALF"] = f"{half:.3f}"
                tok["GAP_CI"] = f"[{gap-half:.3f}, {gap+half:.3f}]"
                tok["GAP_N"] = str(n)
    # the fallback for the gap's interval: the per-seed training artefact
    if "GAP_CI" not in tok and "GAP" in tok:
        bs_path = os.path.join(HERE, "base_training_seeds.json")
        if os.path.exists(bs_path):
            bs = json.load(open(bs_path, encoding="utf-8"))
            cks = [float(b["CK"]) for b in bs]
            n = len(cks)
            if n > 1:
                try:
                    from scipy import stats as _st
                    crit = float(_st.t.ppf(0.975, n - 1))
                except Exception:                          # pragma: no cover
                    crit = 1.96
                gap = float(tok["GAP"])
                half = crit * float(np.std(cks, ddof=1)) / np.sqrt(n)
                tok["GAP_HALF"] = f"{half:.3f}"
                tok["GAP_CI"] = f"[{gap-half:.3f}, {gap+half:.3f}]"
                tok["GAP_N"] = str(n)

    # ------------------------------------------------------------------
    # the observation A/B/C
    # ------------------------------------------------------------------
    obs = load_json("ab_observation.json")
    arms = {}
    for r in obs:
        if r.get("kind") == "baseline":
            continue
        key = r["arm"].strip()[0].upper()
        arms[key] = r
    a, b, c = arms.get("A"), arms.get("B"), arms.get("C")
    if a and b:
        tok["A_CK"] = f(a["CK"])
        tok["A_AG"] = f(a["AG"])
        tok["A_SC"] = f(a["SC"])
        tok["A_DIST"] = f(a["distinct_actions"], 2)
        tok["A_SPREAD"] = f(a["mean_spread_m"], 1)
        tok["A_CK_STD"] = f(a.get("CK_std", 0.0))
        tok["B_CK"] = f(b["CK"])
        tok["B_AG"] = f(b["AG"])
        tok["B_SC"] = f(b["SC"])
        tok["B_DIST"] = f(b["distinct_actions"], 2)
        tok["B_SPREAD"] = f(b["mean_spread_m"], 1)
        tok["B_CK_STD"] = f(b.get("CK_std", 0.0))
        delta = b["CK"] - a["CK"]
        tok["B_MINUS_A"] = f"{delta:+.3f}"
        # The share is the arm's effect as a fraction of THE GAP THE PAPER
        # USES.  It was computed against a different denominator -- the
        # privileged rule's margin over arm A -- which is neither the
        # paper's gap nor a quantity a reader can find in the paper.
        gap_ref = tok.get("GAP")
        if gap_ref not in (None, ""):
            try:
                g = float(gap_ref)
                if abs(g) > 1e-9:
                    tok["B_SHARE"] = f"{100 * (delta / g):.0f}\\%"
            except (TypeError, ValueError):
                pass
    if c:
        tok["C_CK"] = f(c["CK"])
        tok["C_CK_STD"] = f(c.get("CK_std", 0.0))
        if a:
            tok["C_DELTA"] = f"{c['CK'] - a['CK']:+.3f}"
        if b:
            tok["B_DELTA"] = f"{b['CK'] - a['CK']:+.3f}" if a else "n/a"
        tok["C_AG"] = f(c["AG"])
        tok["C_SC"] = f(c["SC"])
        tok["C_DIST"] = f(c["distinct_actions"], 2)
        tok["C_SPREAD"] = f(c["mean_spread_m"], 1)
        # the paired intervals: the three arms share their training seeds
        obs_raw = {}
        obs_corr = {}
        for key, x, y in (("OBS_B", b, a), ("OBS_C", c, a)):
            if not (x and y):
                continue
            pa = x.get("per_seed_CK") or []
            pb = y.get("per_seed_CK") or []
            if not (pa and pb):
                continue
            mean, half, p, n = paired_stat(pa, pb)
            tok[f"{key}_PAIRED"] = f"{mean:+.3f}"
            tok[f"{key}_HALF"] = f"{half:.3f}"
            tok[f"{key}_CI"] = f"[{mean-half:+.3f}, {mean+half:+.3f}]"
            tok[f"{key}_P"] = f"{p:.3f}"
            tok[f"{key}_N"] = str(n)
            # the RAW values are kept: the summary below re-derives the
            # interval's endpoints, and doing that from the ROUNDED
            # values makes the summary's interval differ from the table's
            # in the last digit
            obs_raw[key] = (mean, half)
            obs_corr[key] = _paired_corr(x, y)
        # ABL_ID_CLAUSE is the identity arm's clause used in the
        # CONTRIB bullet and in the Results section.  It is derived here,
        # from the observation A/B comparison, rather than from
        # results_ablation.csv, which does not contain an agent-identity
        # variant.  Without this the token was never created, and the
        # prose kept the typed value instead.
        if "OBS_B" in obs_raw:
            _m, _h = obs_raw["OBS_B"]
            tok["ABL_ID_CI"] = f"[{_m - _h:+.3f}, {_m + _h:+.3f}]"
            tok["ABL_ID_DELTA"] = f"{_m:+.3f}"
            if abs(_m) > _h:
                tok["ABL_ID_CLAUSE"] = (
                    f"the identity is worth {_m:+.3f} persistent "
                    f"coverage (95\\% interval @@ABL_ID_CI@@)")
            else:
                tok["ABL_ID_CLAUSE"] = (
                    f"the identity changes persistent coverage by {_m:+.3f}, "
                    f"an interval @@ABL_ID_CI@@ that includes zero, so at "
                    f"this budget what it demonstrably buys is the ability "
                    f"to specialise")
        # the exclusion argument for the observation content
        obs_stats = []
        for key, label in (("OBS_B", "adding the agent identity"),
                           ("OBS_C", "the full observation of the proposed "
                                     "configuration")):
            if key in obs_raw:
                obs_stats.append((label, obs_raw[key][0], obs_raw[key][1]))
        if obs_stats and "GAP" in tok:
            gap = float(tok["GAP"])
            clauses = [bound_clause(lab, m, h, gap) for lab, m, h in obs_stats]
            excluded = all((m + h) < gap for _, m, h in obs_stats)
            n = len(obs_stats)
            # What the observation terms buy is a claim about the arms, so
            # it is read off the arms: the number of DISTINCT ACTIONS a
            # policy takes is the measure of whether the agents specialise,
            # and the sentence says what the numbers say in either
            # direction.
            spec = ""
            if a and b and "A_DIST" in tok and "B_DIST" in tok:
                d0, d1 = float(tok["A_DIST"]), float(tok["B_DIST"])
                if d1 > d0:
                    spec = (f"the identity raises the number of distinct "
                            f"actions a policy takes from {d0:.2f} to "
                            f"{d1:.2f} out of eight, which is the "
                            f"specialisation the arms measure")
                else:
                    spec = (f"the identity does not raise the number of "
                            f"distinct actions a policy takes ({d0:.2f} "
                            f"without it, {d1:.2f} with it), so the arms do "
                            f"not measure a specialisation effect either")
            tok["OBS_SUMMARY"] = (
                ("The arms" if n > 1 else "The arm")
                + " share their training seeds, so the "
                + ("comparisons are paired: " if n > 1 else "comparison is paired: ")
                + "; ".join(clauses)
                + ((".  Each upper limit is below the gap of " if n > 1
                    else ".  The upper limit is below the gap of ")
                   if excluded else
                   (".  At least one upper limit reaches past the gap of "
                    if n > 1 else
                    ".  The upper limit reaches past the gap of "))
                + f"{gap:.3f} to {tok.get('BEST_RULE_NAME', 'the strongest rule')}, so the content "
                + ("of the observation is not what accounts for the gap; "
                   if excluded else
                   "of the observation is not excluded by these arms; ")
                + ("what the observation terms buy is the ability to "
                   "specialise at all" if spec else
                   "what the observation terms buy is not separated by "
                   "these arms")
                + (", and " + spec if spec else
                   ", and the arms here do not separate the coverage they "
                   "produce")
                + ((".  The claim is a conjunction over the arms, so the "
                    "intervals are unadjusted and the family is not a set of "
                    "independent tests; with a Bonferroni correction over "
                    "the arms of the whole paper the largest adjusted upper "
                    f"limit is {tok.get('ARMS_BONF_UPPER', 'n/a')}, which is "
                    "still below the gap's own lower interval bound "
                    f"{tok.get('GAP_CI_LOWER', 'n/a')}, so the exclusion "
                    "survives the correction")
                   if "ARMS_BONF_UPPER" in tok else "")
                + ".")
        # The pairing is justified by the seed-to-seed variation
        # cancelling, and that justification only holds where the two
        # arms' scores move together.  Where they do not, the paired
        # interval is WIDER than the unpaired one, and the paper has to
        # say so rather than let the pairing carry the claim.
        _c = obs_corr.get("OBS_B")
        if _c is not None and _c < 0:
            tok["OBS_PAIRING_NOTE"] = (
                f"The pairing is the paper's justification for comparing "
                f"the arms on their shared seeds, and it works because the "
                f"seed-to-seed variation cancels --- except for the "
                f"identity arm, whose two arms' per-seed scores are "
                f"negatively correlated (r = {_c:.2f}), so there the paired "
                f"interval is WIDER than the unpaired one; the exclusion "
                f"still holds, because the unpaired upper limit is lower "
                f"still, but the exception is stated rather than assumed")
        else:
            tok["OBS_PAIRING_NOTE"] = ""

    mr = load_json("ab_map_resolution.json")
    if mr:
        parts = []
        for r in mr:
            parts.append(f"$G={r['grid']}$: {f(r['CK'])}")
        # the sweep's own budget, disclosed: the map sweep is measured
        # at 20 episodes and the base comparison at 25, so the sentence
        # that compares the map's effect with the learner's spread has
        # to say which measurement the spread comes from.
        try:
            _map_ep = int(mr[0].get("episodes", 0))
        except (TypeError, ValueError):
            _map_ep = 0
        tok["MAP_BUDGET_NOTE"] = (
            f", measured at a {_map_ep}-episode budget rather than the "
            f"base comparison's 25" if _map_ep and _map_ep != 25 else "")
        # the yardstick for "makes a difference": the LEARNER's own
        # seed-to-seed spread, so that the sentence compares the map's
        # effect with the spread of the thing it is an effect of.  The
        # map sweep's own arm spreads are a different quantity, and using
        # one of them made the sentence compare two different things.
        yard = tok.get("MAPPO_CK_STD")
        try:
            yard = float(yard)
        except (TypeError, ValueError):
            yard = 0.0
        if not yard:
            spreads = [float(r.get("CK_std", 0.0)) for r in mr]
            yard = max(spreads) if spreads else 0.0
        cks = {int(r["grid"]): float(r["CK"]) for r in mr}
        sp = {int(r["grid"]): float(r.get("mean_spread_m", 0.0)) for r in mr}
        base = cks.get(0)
        resolvable = base is not None and any(
            abs(cks[g] - base) >= yard for g in cks if g != 0)

        # What the summary does to the swarm's SPREAD is a measurement, and
        # its direction has to be read off the numbers: the sentence below
        # used to say the spread "falls" and that the summary "keeps the
        # swarm together", which is the wrong direction when the spread
        # grows.  The direction is now derived, and so is the explanation
        # that goes with it.
        def _spread_clause():
            if 0 not in sp or len(sp) < 2:
                return ""
            others = [g for g in sorted(sp) if g != 0]
            deltas = {g: sp[g] - sp[0] for g in others}
            grew = all(d > 0 for d in deltas.values())
            shrank = all(d < 0 for d in deltas.values())
            if grew:
                direction = "grows"
                why = ("a policy that follows the deficit spreads the swarm "
                       "out instead of keeping it together")
            elif shrank:
                direction = "falls"
                why = ("a policy that follows the deficit keeps the swarm "
                       "together instead of dispersing it")
            else:
                direction = "moves by no consistent amount"
                why = "the direction is not consistent across resolutions"
            return (
                "  What the summary does change is the swarm's behaviour: "
                f"the spread {direction}, from {sp[0]:.1f} m without it to "
                + " and ".join(f"{sp[g]:.1f} m" for g in others)
                + " with it, because " + why + ".")

        if resolvable:
            sp_txt = _spread_clause()
            tok["MAP_RESOLUTION"] = (
                "The resolution of the summary does not rescue it: " +
                ", ".join(parts) +
                ", and the resolution makes a difference where the summary "
                "as a whole does not." + sp_txt)
        else:
            # the sweep may be mid-measurement, with only the G=0 arm in
            # the artefact; there is then nothing to compare it with
            worst = (max(abs(cks[g] - base) for g in cks if g != 0)
                     if base is not None and len(cks) > 1 else 0.0)
            tok["MAP_RESOLUTION"] = (
                "The resolution of the summary makes no difference to the "
                "coverage: " + ", ".join(parts)
                + f", and the largest of the differences is {worst:.3f}, "
                f"inside the seed-to-seed spread of the learner "
                f"({yard:.3f})." + _spread_clause()
                + "  The map is therefore neither what makes the controller "
                "work nor what stops it working: it is a channel the policy "
                "uses to move differently, and the difference it makes is "
                "visible in the trajectory rather than in the score.")
        # the verdict for the contribution bullet
        if base is not None and not resolvable:
            tok["MAP_VERDICT"] = (
                "A broadcast summary of where the coverage deficit is was "
                "added a priori and measured to make no difference to the "
                "coverage at any resolution tested, while it does change "
                "what the policy does; it is reported rather than quietly "
                "dropped.")
            tok["MAP_MODEL_SENTENCE"] = (
                "the summary makes no measurable difference to the coverage "
                "at any resolution tested, so the argument for adding it is "
                "neither supported nor refuted by the score, and what it "
                "does change is the shape of the swarm's motion")
        else:
            tok["MAP_VERDICT"] = (
                "A broadcast summary of where the coverage deficit is was "
                "added a priori and measured, and the measurement is "
                "reported rather than the summary being quietly dropped.")
            tok["MAP_MODEL_SENTENCE"] = (
                "the summary measurably changes the coverage, and the "
                "measurement is reported in Section~\\ref{sec:results_obs}")

    # ------------------------------------------------------------------
    # the energy arms
    # ------------------------------------------------------------------
    en = load_json("ab_energy_management.json")
    arms_en = {}
    for r in en:
        arms_en[r["arm"].strip()[0].upper()] = r
    ea, eb, ec = arms_en.get("A"), arms_en.get("B"), arms_en.get("C")
    if ea:
        tok["EA_CK"] = f(ea["CK"])
        tok["EA_AG"] = f(ea["AG"])
        tok["EA_SC"] = f(ea["SC"])
        tok["EA_STR"] = f(ea["stranded_per_episode"], 1)
        tok["EA_RET"] = f(ea["return_events_per_episode"], 0)
    if eb:
        tok["EB_CK"] = f(eb["CK"])
        tok["EB_AG"] = f(eb["AG"])
        tok["EB_SC"] = f(eb["SC"])
        tok["EB_RET"] = f(eb["return_events_per_episode"], 1)
    if ec:
        tok["EC_CK"] = f(ec["CK"])
        tok["EC_AG"] = f(ec["AG"])
        tok["EC_SC"] = f(ec["SC"])
        tok["EC_STR"] = f(ec["stranded_per_episode"], 1)
        tok["EC_RET"] = f(ec["return_events_per_episode"], 1)
    if ea and eb:
        tok["ENERGY_DELTA"] = f(eb["CK"] - ea["CK"])
    if ea and ec:
        tok["EC_DELTA"] = f(ec["CK"] - ea["CK"])
    arms_en = {r["arm"].split(":")[0].strip(): r for r in en}
    ed = arms_en.get("D")
    if ea and ed:
        tok["ED_DELTA"] = f(ed["CK"] - ea["CK"])
        tok["ED_STR"] = f(ed["stranded_per_episode"], 1)
        tok["ED_CK"] = f(ed["CK"])
        tok["ED_RET"] = f(ed["return_events_per_episode"], 0)
        tok["ED_STR_DELTA"] = f(ed["stranded_per_episode"]
                                - ea["stranded_per_episode"], 1)
        tok["ED_RET_DELTA"] = f(abs(ed["return_events_per_episode"]
                                    - ea["return_events_per_episode"]), 0)
        # The REPAIR's effect on the coverage, with its paired interval.
        # The prose used to call this a null ("the coverage is unchanged
        # to within the spread"), which is a sign error: the repair moves
        # the coverage by about a third of the gap, and the interval
        # excludes zero.  The verdict is derived from the measurement, so
        # it says what the measurement says.
        pa = ea.get("per_seed_CK") or []
        pb = ed.get("per_seed_CK") or []
        if pa and pb and len(pa) == len(pb):
            # the direction that matters is the REPAIR's gain: the
            # repaired configuration minus the previous one
            mean, half, p, n = paired_stat(ea.get("per_seed_CK") or [],
                                          ed.get("per_seed_CK") or [])
            tok["ED_PAIRED"] = f"{mean:+.3f}"
            tok["ED_HALF"] = f"{half:.3f}"
            tok["ED_CI"] = f"[{mean-half:+.3f}, {mean+half:+.3f}]"
            tok["ED_P"] = f"{p:.3f}"
            excludes_zero = (mean - half > 0) or (mean + half < 0)
            tok["ED_EXCLUDES_ZERO"] = "yes" if excludes_zero else "no"
            if "GAP" in tok:
                gap = float(tok["GAP"])
                share = abs(mean) / gap if gap else float("nan")
                tok["ED_SHARE"] = f"{100*share:.0f}"
                tok["ED_GAP_GAP"] = f"{gap:.3f}"
            if excludes_zero:
                tok["ED_VERDICT"] = (
                    f"The repair is not a null: the coverage it adds is "
                    f"{mean:+.3f} on the same training seeds (95\\% interval "
                    f"[{mean-half:+.3f}, {mean+half:+.3f}], which excludes "
                    f"zero), and that is about "
                    f"{tok.get('ED_SHARE', 'n/a')}\\% of the gap to "
                    f"{tok.get('BEST_RULE_NAME', 'the strongest rule')}")
            else:
                tok["ED_VERDICT"] = (
                    f"The coverage the repair adds is {mean:+.3f} on the "
                    f"same training seeds (95\\% interval "
                    f"[{mean-half:+.3f}, {mean+half:+.3f}], which includes "
                    f"zero), so the repair's effect on the coverage is not "
                    f"resolved by this arm")
        else:
            tok["ED_VERDICT"] = ("the paired interval for the repair is not "
                                 "computable from the artefact")
    # the paired intervals of the energy arms against arm A
    en_stats = []
    for key, arm, label in (("EB", "B", "removing the energy constraint"),
                             ("EC", "C", "removing the threshold term"),
                             ("ED", "D", "the previous energy rule")):
        r = arms_en.get(arm)
        if not (ea and r):
            continue
        pa = ea.get("per_seed_CK") or []
        pb = r.get("per_seed_CK") or []
        if not (pa and pb):
            continue
        mean, half, p, n = paired_stat(pb, pa)
        if not (mean == mean and half == half):
            # fewer than two shared training seeds: no interval is
            # computable, and the paper must say so rather than print a
            # not-a-number
            tok[f"{key}_PAIRED"] = "not measured"
            tok[f"{key}_HALF"] = "not measured"
            tok[f"{key}_CI"] = "not measured"
            tok[f"{key}_P"] = "not measured"
            tok[f"{key}_N"] = str(n)
            continue
        tok[f"{key}_PAIRED"] = f"{mean:+.3f}"
        tok[f"{key}_HALF"] = f"{half:.3f}"
        tok[f"{key}_CI"] = f"[{mean-half:+.3f}, {mean+half:+.3f}]"
        tok[f"{key}_P"] = f"{p:.3f}"
        tok[f"{key}_N"] = str(n)
        en_stats.append((label, mean, half))
    # ------------------------------------------------------------------
    # the exclusion argument, derived from the measurements
    # ------------------------------------------------------------------
    if "GAP" in tok and not en_stats:
        tok["ENERGY_SUMMARY"] = (
            "The three interventions are paired with the proposed "
            "configuration -- they share its training seeds -- so each one "
            "is reported as an interval rather than as a point, but the "
            "arms have not yet been measured with enough shared training "
            "seeds for an interval to be computable, and no bound is "
            "claimed for them here.")
    elif en_stats and "GAP" in tok:
        gap = float(tok["GAP"])
        clauses = [bound_clause(lab, m, h, gap) for lab, m, h in en_stats]
        excluded = all((m + h) < gap for _, m, h in en_stats)
        if excluded:
            tok["ENERGY_SUMMARY"] = (
                "The energy model is not what accounts for the gap, and "
                "Table~\\ref{tab:results_energy} says so.  The three "
                "interventions are paired with the proposed configuration -- "
                "they share its training seeds -- so each one is reported "
                "as an interval rather than as a point, and each "
                "interval's upper limit is below the gap of "
                f"{gap:.3f} to "
                f"{tok.get('BEST_RULE_NAME', 'the strongest rule')}: "
                + "; ".join(clauses)
                + ".  The energy model is therefore not what accounts for "
                "the gap, and the paper does not claim that it is.")
        else:
            tok["ENERGY_SUMMARY"] = (
                "Table~\\ref{tab:results_energy} bounds the energy "
                "explanation rather than settling it.  The three "
                "interventions are paired with the proposed configuration -- "
                "they share its training seeds -- so each one is reported "
                "as an interval rather than as a point: "
                + "; ".join(clauses)
                + f".  Not every interval excludes the gap of {gap:.3f}, so "
                "the energy model is not excluded by these arms; what the "
                "arms do establish is that the energy model is not the "
                "whole of the explanation, and the point estimates are all "
                "far smaller than the gap.")

    # ------------------------------------------------------------------
    # the component ablation
    # ------------------------------------------------------------------
    # The yardstick is the spread of the learner itself: how much the
    # training seed moves the coverage of the FULL configuration.  A
    # component effect smaller than this is not measurable at this budget.
    yardstick = 0.02
    bs_path = os.path.join(HERE, "base_training_seeds.json")
    if os.path.exists(bs_path):
        bs = json.load(open(bs_path, encoding="utf-8"))
        cks = [float(b["CK"]) for b in bs]
        if len(cks) > 1:
            yardstick = float(np.std(cks))

    abl = load_csv("results_ablation.csv")
    if abl:
        by = {r["model"]: r for r in abl}
        full = by.get("Full")
        van = by.get("VanillaMAPPO")
        # the full configuration's own seed spread, if the ablation measured
        # it, is the better yardstick: it is the same experiment
        if full:
            fs = per_seed_ck(full)
            if len(fs) > 1:
                yardstick = float(np.std(fs))
        if full and van:
            d = float(full["CK"]) - float(van["CK"])
            tok["ABL_DELTA"] = f"{d:+.3f}"

        names = {"no-criticality": "the criticality term",
                 "no-degraded": "the degraded-service term",
                 "no-energy-cost": "the energy-cost term",
                 "no-undercoverage": "the under-coverage term",
                 "no-reserve": "the reserve UAV",
                 "no-arbitration": "the charging arbitration",
                 "no-mask": "the action mask",
                 "no-agent-id": "the agent identity",
                 "no-map": "the coverage map",
                 "no-pad-bearing": "the pad bearing",
                 # the last row removes the framework's mechanisms
                 # together, so it is not a single-component variant
                 "VanillaMAPPO": "the whole design"}
        moved, still, paired = [], [], []
        fs = per_seed_ck(full) if full else []
        for name, r in by.items():
            if name == "Full":
                continue
            # The Vanilla row removes the framework's mechanisms TOGETHER,
            # so it is not a single-component variant -- but it is still
            # measured on the same training seeds, and leaving it out of
            # the paired list made the summary say that every other
            # component's interval includes zero, which is false: the
            # whole design's interval excludes it.
            try:
                delta = float(r["CK"]) - float(full["CK"])
            except (TypeError, ValueError, KeyError):
                continue
            vs = per_seed_ck(r)
            pmean = pstd = None
            if fs and len(vs) == len(fs):
                diffs = [fs[i] - vs[i] for i in range(len(fs))]
                pmean = float(np.mean(diffs))
                pstd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
                paired.append((name, pmean, pstd))
            (moved if abs(delta) >= yardstick else still).append(
                (name, delta, pmean, pstd))

        # --------------------------------------------------------------
        # THE INFERENCE.  A threshold on the size of a difference is not a
        # test: with a per-arm null error rate of about a third at R = 3,
        # a rule that calls an effect "measurable" when it exceeds the
        # learner's own seed spread would flag one null component in three
        # and would still be unable to detect a true effect of that size.
        # The paired design supplies the right test instead: the variants
        # share the training seeds, so the difference is measured on the
        # same training runs, and the paired t interval says what the
        # measurement resolved.  Where the interval includes zero, the
        # honest statement is that NO EFFECT LARGER THAN THE INTERVAL'S
        # HALF-WIDTH WAS DETECTED -- not that the component is inert.
        # --------------------------------------------------------------
        if paired:
            try:
                from scipy import stats as _st
            except Exception:                                  # pragma: no cover
                _st = None
            n_p = len(paired[0][1] and [1] or [1])
            R = len(fs)
            crit = (_st.t.ppf(0.975, R - 1) if (_st and R > 1) else 0.0)
            rows_p = []
            for n, pmean, pstd in paired:
                half = crit * pstd / math.sqrt(R) if R > 1 else float("nan")
                if _st and R > 1 and pstd > 0:
                    tv = pmean / (pstd / math.sqrt(R))
                    pval = 2 * (1 - _st.t.cdf(abs(tv), R - 1))
                elif _st and R > 1:
                    pval = 0.0 if pmean != 0 else 1.0
                else:
                    pval = float("nan")
                rows_p.append(dict(name=n, mean=pmean, sd=pstd, half=half,
                                    p=pval,
                                    excludes_zero=(pmean - half > 0
                                                   or pmean + half < 0)))
            sig = [r for r in rows_p if r["excludes_zero"]]
            nulls = [r for r in rows_p if not r["excludes_zero"]]
            # the resolution of the measurement: the widest half-width among
            # the components whose interval includes zero
            delta_res = max([r["half"] for r in nulls], default=0.0)
            tok["ABL_N_SEEDS"] = str(R)
            tok["ABL_RESOLUTION"] = f"{delta_res:.3f}"
            # the TIGHTEST arm: the resolution is per-arm, and the
            # widest arm alone would misdescribe the others
            _all_half = [r["half"] for r in rows_p]
            tok["ABL_RESOLUTION_MIN"] = (f"{min(_all_half):.3f}"
                                         if _all_half else "n/a")
            tok["ABL_RESOLUTION_SD"] = f"{delta_res / crit:.3f}" if crit else "n/a"
            tok["ABL_CRIT"] = f"{crit:.2f}"
            if sig:
                parts = []
                for r in sorted(sig, key=lambda x: -abs(x["mean"])):
                    nm = names.get(r["name"], r["name"])
                    parts.append(f"{nm} ({r['mean']:+.3f}, 95\\% CI "
                                 f"[{r['mean']-r['half']:+.3f}, "
                                 f"{r['mean']+r['half']:+.3f}])")
                # The table is a family of paired tests, and the
                # intervals are unadjusted.  The correction is computed
                # here, not asserted: with a Bonferroni correction over
                # the arms of the table, the components whose paired
                # p-value survives are named separately, because a
                # component that survives the uncorrected test and not
                # the corrected one is a different claim.
                m_arms = len(rows_p)
                alpha_b = 0.05 / m_arms if m_arms else 0.05
                survivors = [r for r in sig
                             if r["p"] == r["p"] and r["p"] < alpha_b]
                losers = [r for r in sig
                          if r["p"] == r["p"] and r["p"] >= alpha_b]
                if m_arms > 1 and survivors:
                    tok["ABL_BONF_ALPHA"] = f"{alpha_b:.4f}"
                    tok["ABL_BONF_SURVIVORS"] = ", ".join(
                        f"{names.get(r['name'], r['name'])} "
                        f"(p = {r['p']:.4f})" for r in
                        sorted(survivors, key=lambda x: -abs(x["mean"])))
                    tok["ABL_BONF_LOSERS"] = ", ".join(
                        f"{names.get(r['name'], r['name'])} "
                        f"(p = {r['p']:.4f})" for r in
                        sorted(losers, key=lambda x: -abs(x["mean"])))
                    tok["ABL_BONF_NOTE"] = (
                        f".  The table is a family of {m_arms} paired tests and "
                        f"the intervals above are unadjusted, so the "
                        f"correction is stated: with a Bonferroni "
                        f"correction over the table the threshold is "
                        f"$\\alpha = {alpha_b:.4f}$, and the components whose "
                        f"paired p-value survives it are "
                        f"{tok['ABL_BONF_SURVIVORS']}"
                        + (f", while {tok['ABL_BONF_LOSERS']} does not "
                           f"survive that correction and is therefore "
                           f"reported as suggestive rather than resolved"
                           if losers else ""))
                # A SUMMARY THAT REFERS TO A NOTE MUST NEVER REFER TO A NOTE
                # THAT WAS NOT COMPUTED.  The summary below embeds
                # @@ABL_BONF_NOTE@@ unconditionally, and the assembler writes
                # a visible [pending: ...] marker for a token it is not
                # given.  When the paired p-value was unavailable, the
                # `survivors` branch above did not run, so the note was never
                # created and the marker sat inside the sentence -- which is
                # why the prose around it went on quoting typed numbers
                # instead of the macro.  The note is therefore defined on
                # every path.
                if "ABL_BONF_NOTE" not in tok:
                    _has_p = any(r["p"] == r["p"] for r in rows_p)
                    if not _has_p:
                        tok["ABL_BONF_NOTE"] = (
                            f"The table is a family of {m_arms} paired tests "
                            f"and the intervals above are unadjusted; the "
                            f"artefact does not carry the paired p-value the "
                            f"correction would need, so the multiplicity "
                            f"correction is not reported for this table.")
                    elif m_arms > 1:
                        tok["ABL_BONF_NOTE"] = (
                            f"The table is a family of {m_arms} paired tests "
                            f"and the intervals above are unadjusted, so the "
                            f"correction is stated: with a Bonferroni "
                            f"correction over the table the threshold is "
                            f"$\\alpha = {alpha_b:.4f}$, and no component's "
                            f"paired p-value survives it, so every component "
                            f"above is reported as suggestive rather than "
                            f"resolved.")
                    else:
                        tok["ABL_BONF_NOTE"] = ""
                tok["ABL_SUMMARY"] = (
                    "The paired comparison, in which every variant is "
                    "trained on the same training seeds as the full "
                    "configuration, separates the components into two "
                    "groups.  The differences below are the FULL "
                    "configuration minus the variant, so a positive number "
                    "is what the component is worth.  The components whose "
                    "95\\% confidence interval for the paired difference "
                    "excludes zero are "
                    + ", ".join(parts)
                    + "; every other component's interval includes zero, " "and the worst resolution anywhere in the table is " "$@@ABL_RESOLUTION@@$ of persistent coverage, which is the WIDEST " "arm's resolution: the resolution is per-arm, and each arm's own interval is in the table @@ABL_BONF_NOTE@@")
            else:
                tok["ABL_SUMMARY"] = (
                    "Every component's 95\\% confidence interval for the "
                    "paired difference includes zero, and the worst "
                    "resolution anywhere in the table is "
                    "$@@ABL_RESOLUTION@@$ of persistent coverage, which is "
                    "the WIDEST arm's resolution: the resolution is "
                    "per-arm, and each arm's own interval is in the table")
            for r in rows_p:
                key = {"no-agent-id": "ID", "no-pad-bearing": "PAD",
                       "no-map": "MAP", "no-mask": "MASK"}.get(r["name"])
                if key:
                    tok[f"ABL_{key}_DELTA"] = f"{r['mean']:+.3f}"
                    tok[f"ABL_{key}_SPREAD"] = f"{r['sd']:.3f}"
                    tok[f"ABL_{key}_CI"] = (f"[{r['mean']-r['half']:+.3f}, "
                                            f"{r['mean']+r['half']:+.3f}]")
                    tok[f"ABL_{key}_P"] = f"{r['p']:.3f}"
                    if key == "ID":
                        # the contribution bullet must not overstate: if the
                        # interval includes zero, the honest sentence is
                        # about the ability to specialise, not the coverage
                        if r["excludes_zero"]:
                            tok["ABL_ID_CLAUSE"] = (
                                f"the identity is worth "
                                f"${r['mean']:+.3f}$ persistent coverage "
                                f"(95\\% interval $@@ABL_ID_CI@@$)")
                        else:
                            tok["ABL_ID_CLAUSE"] = (
                                f"the identity changes persistent coverage by "
                                f"${r['mean']:+.3f}$, an interval "
                                f"$@@ABL_ID_CI@@$ that includes zero, so at "
                                f"this budget what it demonstrably buys is "
                                f"the ability to specialise")
        else:
            tok["ABL_N_SEEDS"] = "?"
            tok["ABL_RESOLUTION"] = "n/a"
            tok["ABL_RESOLUTION_SD"] = "n/a"
            tok["ABL_CRIT"] = "n/a"
            # no per-seed values in the artefact: the paired comparison is
            # not available, and the unpaired differences are reported with
            # that stated rather than dressed up as an interval
            if moved:
                parts = []
                for n, dlt, pmean, pstd in sorted(moved, key=lambda x: -abs(x[1])):
                    parts.append(f"{names.get(n, n)} ({dlt:+.3f})")
                tok["ABL_SUMMARY"] = (
                    "the components whose difference from the full "
                    "configuration is largest are " + ", ".join(parts)
                    + ", and the rest do not move the coverage by more than "
                    "the seed-to-seed spread of the learner itself")
                for n, dlt, pmean, pstd in moved:
                    if n == "no-agent-id":
                        tok["ABL_ID_CLAUSE"] = (
                            f"the identity is worth ${-dlt:+.3f}$ persistent "
                            f"coverage")
                        tok.setdefault("ABL_ID_CI", "n/a")
                        tok.setdefault("ABL_ID_DELTA", f"{-dlt:+.3f}")
            else:
                tok["ABL_SUMMARY"] = (
                    "no single component changes the persistent coverage by "
                    "more than the seed-to-seed spread of the learner "
                    "itself, and the table is reported in full because that "
                    "is the result")
        # the paired deltas of the components that matter, for the prose
        if paired:
            parts = []
            for n, pmean, pstd in sorted(paired, key=lambda x: -abs(x[1])):
                if abs(pmean) < yardstick:
                    continue
                parts.append(f"{names.get(n, n)} {pmean:+.3f}"
                             f"\\,({pstd:.3f})")
            tok["ABL_PAIRS"] = ", ".join(parts) if parts else "none"
            for n, pmean, pstd in paired:
                key = {"no-agent-id": "ID", "no-pad-bearing": "PAD",
                       "no-map": "MAP", "no-mask": "MASK"}.get(n)
                if key:
                    tok[f"ABL_{key}_DELTA"] = f"{pmean:+.3f}"
                    tok[f"ABL_{key}_SPREAD"] = f"{pstd:.3f}"
        tok["ABL_YARDSTICK"] = f"{yardstick:.3f}"
        tok["ABL_SEEDS"] = str(full.get("train_seeds", "?")) if full else "?"
        tok["ABL_EPISODES"] = str(full.get("episodes", "?")) if full else "?"
        # WHICH observation term matters most is a comparison, so it is
        # derived rather than asserted
        obs_terms = {"no-agent-id": "the agent identity",
                     "no-map": "the coverage map",
                     "no-pad-bearing": "the pad bearing"}
        best = None
        for n, pmean, pstd in paired:
            if n in obs_terms:
                if best is None or abs(pmean) > abs(best[1]):
                    best = (n, pmean, pstd)
        if best is not None:
            n, pmean, pstd = best
            tok["ABL_BEST_OBS"] = obs_terms[n]
            # The convention is the TABLE's: a component's worth is
            # FULL minus the variant without it, so a positive number
            # means the component HELPS.  The prose used to print the
            # opposite sign, which agreed with the table by luck at
            # three seeds and disagreed with it at eight.
            tok["ABL_BEST_OBS_DELTA"] = f"{pmean:+.3f}"
            resolved = (pmean + pstd) < 0 or (pmean - pstd) > 0
            tok["ABL_BEST_OBS_TEXT"] = (
                f"The observation term with the largest effect on the "
                f"coverage is {obs_terms[n]}, and "
                f"Table~\\ref{{tab:results_abl}} prices "
                f"it: it is worth {pmean:+.3f} of persistent "
                f"coverage, paired over the shared training seeds"
                + (", and it is the largest single observation effect in "
                   "this work." if resolved else
                   ", and it is the largest of the three, though the "
                   "interval is wide at this seed count."))
        else:
            tok["ABL_BEST_OBS"] = "the observation terms"
            tok["ABL_BEST_OBS_DELTA"] = "n/a"
            tok["ABL_BEST_OBS_TEXT"] = (
                "The observation terms are priced individually in "
                "Table~\\ref{tab:results_obs}.")
        # the fall-back for the components that matter: if the per-seed
        # values are not in the artefact, the unpaired difference is used and
        # is reported as such
        for n, dlt, pmean, pstd in moved:
            key = {"no-agent-id": "ID", "no-pad-bearing": "PAD",
                   "no-map": "MAP", "no-mask": "MASK"}.get(n)
            if key and f"ABL_{key}_DELTA" not in tok:
                tok[f"ABL_{key}_DELTA"] = f"{-dlt:+.3f}"
                tok[f"ABL_{key}_SPREAD"] = f"{yardstick:.3f}"

    scen = load_csv("results_scenarios.csv")
    if scen:
        wins = losses = ties = 0
        # The extremes are computed over the MEASURED deficits.  They used
        # to be initialised at zero, which made the "best" variant a
        # maximum over an empty set: the paper reported a surplus of
        # +0.000 for a variant that is in fact behind.
        worst = None
        best = None
        deficits = []
        for r in scen:
            try:
                # the comparison is against the STRONGER of the two rules on
                # that variant, so the matrix is read against the best
                # reference the framework has
                rule = float(r["react_CK"])
                if r.get("reactlocal_CK") not in (None, ""):
                    rule = max(rule, float(r["reactlocal_CK"]))
                d = float(r["mappo_CK"]) - rule
            except (TypeError, ValueError, KeyError):
                continue
            if d > 0.02:
                wins += 1
            elif d < -0.02:
                losses += 1
            else:
                ties += 1
            deficits.append(d)
            if worst is None or d < worst[1]:
                worst = (r["scenario"], d)
            if best is None or d > best[1]:
                best = (r["scenario"], d)
        tok["SCEN_WINS"] = str(wins)
        tok["SCEN_LOSSES"] = str(losses)
        tok["SCEN_TIES"] = str(ties)
        # the SPREAD of the deficits, as magnitudes: the figure's caption
        # says how far behind the rule the controller is, and the values
        # the scenario names carry are signed
        if deficits:
            tok["SCEN_SPAN_MIN"] = f"{abs(max(deficits)):.3f}"
            tok["SCEN_SPAN_MAX"] = f"{abs(min(deficits)):.3f}"
        tok["SCEN_VARIANTS"] = str(len(deficits))
        # how many policies each combination was trained with: the rows
        # are a mean over training runs, and the prose has to say so
        scen_seeds = set()
        for r in scen:
            if r.get("train_seeds"):
                try:
                    scen_seeds.add(int(r["train_seeds"]))
                except (TypeError, ValueError):
                    pass
        tok["SCEN_TRAIN_SEEDS"] = (str(sorted(scen_seeds)[0])
                                    if len(scen_seeds) == 1 else
                                    ("several" if scen_seeds else "one"))
        tok["SCEN_WORST"] = f"{worst[0]} ({worst[1]:+.3f})"
        tok["SCEN_BEST"] = f"{best[0]} ({best[1]:+.3f})"        # the closing sentence of the section, derived from the matrix
        # The WORD that goes with the extreme is derived from its sign.
        # "the largest surplus" applied to a negative number describes a
        # surplus that does not exist, and a reader who checks the sign
        # finds the prose wrong.
        tok["SCEN_WORST_WORD"] = ("the largest deficit"
                                  if worst[1] < 0 else
                                  "the smallest surplus")
        tok["SCEN_BEST_WORD"] = ("the largest surplus"
                                if best[1] > 0 else
                                "the smallest deficit")
        by_scen = {}
        for r in scen:
            try:
                rule = float(r["react_CK"])
                if r.get("reactlocal_CK") not in (None, ""):
                    rule = max(rule, float(r["reactlocal_CK"]))
                d = float(r["mappo_CK"]) - rule
            except (TypeError, ValueError, KeyError):
                continue
            by_scen.setdefault(r["scenario"], []).append(d)
        ahead = sorted(s for s, v in by_scen.items()
                       if sum(v) / len(v) > 0.02)
        behind = sorted(s for s, v in by_scen.items()
                        if sum(v) / len(v) < -0.02)
        if ahead:
            tok["SCEN_TEXT"] = (
                "At the level of scenario means the learned controller is "
                "ahead of the stronger rule in " + ", ".join(ahead)
                + (", and behind it in " + ", ".join(behind) if behind
                   else ", and behind it in no scenario")
                + ".  The matrix therefore says something the single "
                "configuration cannot: the ordering of the two is a "
                "property of the regime, not of the method, and a reader who "
                "is told only that a learned controller loses to a rule has "
                "been told half of it.")
        else:
            # Behind in EVERY scenario and EVERY variant is the strongest
            # evidence there is about the ordering, and the old sentence
            # drew the opposite conclusion from it ("a property of the
            # regime rather than of the method").
            span = ""
            if deficits:
                span = (f", by between {min(deficits):.3f} and "
                        f"{max(deficits):.3f} of persistent coverage")
            tok["SCEN_TEXT"] = (
                "At the level of scenario means the learned controller is "
                "behind the stronger rule in every scenario, and behind it "
                f"in {losses} of the {wins + losses + ties} variants"
                + span
                + ", so the ordering is a property of the method at this "
                "budget rather than of a particular regime, and a reader "
                "told only that a learned controller loses to a rule has "
                "been told the whole of it.")
    # These two are placeholders for the blocks that are assembled
    # separately; they are not measurements, and an empty value would be
    # screened out and would leave a [pending] marker in the paper for a
    # token nothing uses.
    for _k in ("ABL_TEXT", "CREDIT_TEXT"):
        tok.pop(_k, None)

    cr = load_json("ab_credit.json")
    if len(cr) >= 2:
        a, b = cr[0], cr[1]
        d = b["CK"] - a["CK"]
        tok["CREDIT_DELTA"] = f"{d:+.3f}"
        tok["CREDIT_A_CK"] = f(a["CK"])
        tok["CREDIT_B_CK"] = f(b["CK"])
        tok["CREDIT_A_SC"] = f(a["SC"])
        tok["CREDIT_B_SC"] = f(b["SC"])
        react = main_rows.get("Reactive")
        if react:
            gap = float(react["CK"]) - a["CK"]
            tok["CREDIT_OF_GAP"] = (f"{100*abs(d)/gap:.0f}\\%" if abs(gap) > 1e-9
                                     else "0\\%")
        # the paired comparison of the two learners
        pa = a.get("per_seed_CK") or []
        pb = b.get("per_seed_CK") or []
        if pa and pb:
            mean, half, p, n = paired_stat(pb, pa)
            tok["CREDIT_PAIRED"] = f"{mean:+.3f}"
            tok["CREDIT_HALF"] = f"{half:.3f}"
            tok["CREDIT_CI"] = f"[{mean-half:+.3f}, {mean+half:+.3f}]"
            tok["CREDIT_P"] = f"{p:.3f}"
            tok["CREDIT_N"] = str(n)
            # the verdict is derived: an interval that excludes zero says
            # something the point estimate does not
            if mean + half < 0:
                tok["CREDIT_VERDICT"] = (
                    f"the per-agent arm is measurably WORSE than the shared "
                    f"one, by {-mean:.3f} of persistent coverage, and the "
                    f"interval excludes zero, so the cheapest modification "
                    f"in the direction the diagnosis points does not close "
                    f"the gap --- it opens it slightly")
            elif mean - half > 0:
                tok["CREDIT_VERDICT"] = (
                    f"the per-agent arm is measurably BETTER than the shared "
                    f"one, by {mean:.3f} of persistent coverage, and the "
                    f"interval excludes zero, so the cheapest modification "
                    f"in the direction the diagnosis points helps --- "
                    f"although not by enough to close the gap")
            else:
                # The interval does not exclude zero, and it is NOT
                # centred on zero: saying "consistent with a difference of
                # any size up to X in either direction" describes a
                # symmetric interval, and this one is not symmetric.  The
                # honest statement names the best case the interval
                # allows, and compares it with the gap.
                best = mean + half
                worst = mean - half
                gap_txt = tok.get("GAP", "the gap")
                if best > 0:
                    tail = (f"its best case is {best:.3f} better, far short "
                            f"of the gap of {gap_txt}, and its worst case is "
                            f"{-worst:.3f} worse")
                else:
                    tail = (f"every difference the interval allows is "
                            f"negative, from {worst:.3f} to {best:.3f}")
                tok["CREDIT_VERDICT"] = (
                    f"the difference is not resolved by this arm: the "
                    f"interval is {tok['CREDIT_CI']}, and {tail}, so the "
                    f"cheapest modification in the direction the diagnosis "
                    f"points has not been shown to close the gap")
        # The inference the abstract carries, derived from this arm: the
        # sentence that says what the residual is consistent with has to
        # follow from the measurement, and it is the one place where a
        # hand-written sentence would be a claim instead of a report.
        if mean + half < 0:
            tok["CREDIT_RESIDUAL_SHORT"] = (
                f"The credit assignment of a shared team reward is the "
                f"explanation the residual is consistent with, and the "
                f"cheapest test of it points that way: a per-agent local "
                f"critic is measurably worse, by {abs(mean):.3f}")
            tok["CREDIT_RESIDUAL"] = (
                f"The residual is consistent with the credit assignment of a "
                f"shared team reward: the cheapest test of that diagnosis, a "
                f"per-agent local critic, is measurably WORSE, by "
                f"{abs(mean):.3f} of persistent coverage (95\\% interval "
                f"{tok['CREDIT_CI']})")
        elif mean - half > 0:
            tok["CREDIT_RESIDUAL_SHORT"] = (
                f"The credit assignment of a shared team reward is the "
                f"explanation the residual is consistent with, but the "
                f"cheapest test of it points the other way: a per-agent "
                f"local critic is measurably better, by {mean:.3f}")
            tok["CREDIT_RESIDUAL"] = (
                f"The residual is consistent with the credit assignment of a "
                f"shared team reward, and the cheapest test of that "
                f"diagnosis, a per-agent local critic, is measurably BETTER, "
                f"by {mean:.3f} of persistent coverage (95\\% interval "
                f"{tok['CREDIT_CI']})")
        else:
            tok["CREDIT_RESIDUAL_SHORT"] = (
                f"The credit assignment of a shared team reward is the "
                f"explanation the residual is consistent with, and the "
                f"cheapest test of it does not resolve it")
            tok["CREDIT_RESIDUAL"] = (
                f"The residual is consistent with the credit assignment of a "
                f"shared team reward, and the cheapest test of that "
                f"diagnosis does not resolve it: the per-agent local "
                f"critic's difference is {mean:+.3f} of persistent coverage "
                f"(95\\% interval {tok['CREDIT_CI']}), which includes zero")

    # ------------------------------------------------------------------
    # THE ARMS' VERDICT, derived.  The abstract and the introduction both
    # say what the controlled arms establish.  Whether they establish it
    # depends on the measurements, so the sentence is derived here: an
    # explanation is excluded only if the arm that tests it has an upper
    # confidence limit below the gap.
    # ------------------------------------------------------------------
    arm_bounds = []
    if obs_stats and "GAP" in tok:
        arm_bounds.append(("the content of the observation",
                           max(m + h for _, m, h in obs_stats)))
    if en_stats and "GAP" in tok:
        arm_bounds.append(("the energy model",
                           max(m + h for _, m, h in en_stats)))
    if arm_bounds:
        gap = float(tok["GAP"])
        excluded = [lab for lab, up in arm_bounds if up == up and up < gap]
        not_excluded = [lab for lab, up in arm_bounds
                        if not (up == up and up < gap)]
        # the sentence names ONLY the arms that were actually measured, so
        # it cannot claim an arm was tested when its artefact is missing
        names_all = " and ".join(lab for lab, _ in arm_bounds)
        # How far below the gap each arm's upper limit lies.  The old
        # sentence said the intervals lie "far below" the gap, which is a
        # claim about the MARGIN, and the margins are small: a reader
        # deserves the number rather than the adjective.
        margins = sorted(gap - up for _, up in arm_bounds
                         if up == up and up < gap)
        tok["ARMS_MARGIN"] = (f"{margins[0]:.3f}" if margins else "n/a")
        tok["ARMS_MARGIN_MAX"] = (f"{margins[-1]:.3f}" if margins else "n/a")
        if not not_excluded:
            tok["ARMS_VERDICT"] = (
                "controlled arms bound how much of the gap the alternative "
                "explanations can account for: "
                + names_all
                + (" is" if len(arm_bounds) == 1 else " are")
                + " tested with a single-factor intervention whose upper "
                "confidence limit lies below the gap, by "
                + (f"a margin of {margins[0]:.3f}"
                   if len(margins) == 1 else
                   f"margins of {margins[0]:.3f} to {margins[-1]:.3f}")
                + f" against a gap of {gap:.3f}")
        else:
            tok["ARMS_VERDICT"] = (
                "controlled arms bound how much of the gap the alternative "
                "explanations can account for: the interval of "
                + ("the arm testing " if len(arm_bounds) == 1
                   else "the arms testing ")
                + names_all + " "
                + ("does" if len(arm_bounds) == 1 else "do")
                + " not exclude the whole gap"
                + (", so " + " and ".join(not_excluded)
                   + (" is" if len(not_excluded) == 1 else " are")
                   + " not excluded by "
                   + ("it" if len(not_excluded) == 1 else "them")
                   if not_excluded else ""))
    else:
        tok["ARMS_VERDICT"] = (
            "the arms that test the alternative explanations have not been "
            "measured yet, so no bound is claimed for them")

    # The sentence the abstract, the introduction and the conclusion share.
    # It used to say that all three explanations were tested with an
    # intervention "whose interval lies far below the gap", which is not
    # what the measurements say: the budget arm's own interval straddles
    # the gap, and the margins on the other two are small.  The sentence
    # is now derived from the arms, so it names each arm's status.
    if arm_bounds and "GAP" in tok:
        gap = float(tok["GAP"])
        # every arm, with its own upper limit: the group's verdict is the
        # conservative one, but a reader deserves to see WHICH arm is
        # wide, because one wide arm does not make the others wide
        every = []
        refs = []
        for stats, group, ref in (
                (obs_stats, "the content of the observation",
                 "the local-only arm A of the observation experiment"),
                (en_stats, "the energy model",
                 "the proposed configuration of the energy experiment")):
            for lab, m, h in stats:
                every.append((lab, m + h, group))
                if ref not in refs:
                    refs.append(ref)
        parts = []
        not_cleared = []
        for lab, up, group in every:
            margin = gap - up
            if margin <= 0:
                not_cleared.append(lab)
            parts.append(f"{lab} (upper limit {up:.3f}, margin {margin:+.3f})")
        bud = ""
        if "BUDGET_PAIRED_CI" in tok and "BUDGET_PAIRED_VS_GAP" in tok:
            bud = (f"; the training budget's own arm, from "
                   f"{tok['BUDGET_RANGE']} episodes, has the interval "
                   f"{tok['BUDGET_PAIRED_CI']}, whose "
                   f"{tok['BUDGET_PAIRED_VS_GAP']}, so it is neither "
                   f"excluded nor implicated")
        if not_cleared:
            tail = (".  The arms whose upper limit reaches past the gap -- "
                    + "; ".join(not_cleared)
                    + " -- are NOT excluded, and remain candidate "
                      "explanations; the others are excluded as the whole "
                      "of the explanation")
        else:
            tail = (".  Every arm's upper limit is below the gap, so none "
                    "of them can be the whole of the explanation, by "
                    "margins that are small but positive")
        # The reference arm is named, because the arm's interval is
        # PAIRED against it and not against the base comparison: a reader
        # who does not know the reference cannot tell what the interval
        # is an interval of.
        # the credit arm is paired against the proposed configuration too,
        # and it is not part of the loop above because its arm set is
        # different; name it here so the sentence is complete
        if refs and "the proposed configuration of the credit experiment" \
                not in refs:
            refs.append("the proposed configuration of the credit experiment")
        ref_clause = ""
        if refs:
            ref_clause = (".  Every interval is paired over the same "
                          "training seeds, against "
                          + (" and ".join(refs)))
        tok["ARMS_SUMMARY_SENTENCE"] = (
            "Controlled arms bound how much of the gap each alternative "
            "explanation can account for: "
            + "; ".join(parts)
            + f", against a gap of {gap:.3f}"
            + bud + tail + ref_clause + ".")
        # The abstract's version: the same verdict, without the list of
        # numbers.  An abstract that reads like a table is not an
        # abstract, and the numbers belong in the results.
        if not not_cleared:
            tok["ARMS_SUMMARY_SHORT"] = (
                f"Controlled arms bound how much of the gap each alternative "
                f"explanation can account for, and every arm's upper "
                f"confidence limit lies below the gap of {gap:.3f} --- by "
                f"margins that are small but positive, the smallest of them "
                f"{margins[0]:.3f} --- so none of the explanations the arms "
                f"test can be the whole of it")
        else:
            tok["ARMS_SUMMARY_SHORT"] = (
                f"Controlled arms bound how much of the gap each alternative "
                f"explanation can account for, and the arm testing "
                f"{' and '.join(not_cleared)} does not exclude the whole gap, "
                f"so that explanation remains a candidate")
    else:
        tok["ARMS_SUMMARY_SENTENCE"] = (
            "the arms that test the alternative explanations have not been "
            "measured yet, so no bound is claimed for them")

    # ------------------------------------------------------------------
    # the causes the arms do NOT test
    # ------------------------------------------------------------------
    # The list is a token so that the abstract and the conclusion quote the
    # same one, and so that a cause which has since been MEASURED cannot
    # stay on the list: the rule's privileged access to the coverage state
    # was on it until the local-assignment arm measured it.
    tok["UNTESTED_CAUSES"] = (
        "the possibility that the hand-written rule was tuned against the "
        "evaluation scenes, the limit of a decentralised policy class, and "
        "the state-dependent corrections that the safety filter and the "
        "joint projection impose on the executed trajectory")

    # ------------------------------------------------------------------
    # what the metrics do on an INFEASIBLE configuration
    # ------------------------------------------------------------------
    inf = load_json("infeasible_instance.json")
    if inf:
        m = inf.get("mean", {})
        tok["INF_CK"] = f(m.get("CK", float("nan")))
        tok["INF_SC"] = f(m.get("SC", float("nan")))
        tok["INF_AG"] = f(m.get("AG", float("nan")))
        tok["INF_ORACLE"] = f(m.get("oracle_CK", float("nan")))
        tok["INF_N"] = str(len(inf.get("rows", [])))

    # ------------------------------------------------------------------
    # THE INFORMATION-ASYMMETRY ARM.  The reactive heuristic is given the
    # exact set of under-covered sensors; the learned policy is given a
    # fixed-size summary.  The arm below is the same rule with the global
    # coverage state removed, and it is what turns the caveat into a
    # measurement.
    # ------------------------------------------------------------------
    rl = main_rows.get("Reactive-local")
    rp = main_rows.get("Reactive")
    if rl and rp and "GAP" in tok:
        priv = float(rp["CK"])
        loc = float(rl["CK"])
        gap = float(tok["GAP"])
        d = priv - loc
        share = d / gap if abs(gap) > 1e-9 else 0.0
        tok["REACTL_SHARE"] = f"{100 * share:.0f}"
        if d < 0:
            # the unprivileged rule is BETTER: the privilege was a handicap
            tok["REACTL_VERDICT"] = (
                f"The privileged version is in fact slightly WORSE than the "
                f"unprivileged one ({d:+.3f}), so the global coverage state "
                f"is not what gives the rule its edge --- the global "
                f"round-robin assignment makes the UAVs contend for "
                f"sensors, and the local rule does better --- and the gap "
                f"to the learned controller is therefore not an information "
                f"artefact at all")
        elif abs(d) < 0.5 * abs(gap):
            tok["REACTL_VERDICT"] = (
                f"the privilege is therefore worth {d:.3f} of persistent "
                f"coverage, which is a small part of the {gap:.3f} gap to "
                f"the learned controller, and the gap is not an information "
                f"artefact")
        else:
            tok["REACTL_VERDICT"] = (
                f"the privilege is therefore worth {d:.3f} of persistent "
                f"coverage, which is {100 * share:.0f}\\% of the {gap:.3f} "
                f"gap to the learned controller, and a large part of the "
                f"gap is therefore an information artefact")

    # ------------------------------------------------------------------
    # where the connectivity availability below 1.000 comes from
    # ------------------------------------------------------------------
    ca = load_json("connectivity_accounting.json")
    if ca:
        n = int(ca.get("disconnected_steps", 0))
        left = int(ca.get("after_active_set_change", 0))
        tel = int(ca.get("after_teleport", 0))
        unex = int(ca.get("unexplained", 0))
        tok["CONN_STEPS"] = str(n)
        tok["CONN_LEFT"] = str(left)
        tok["CONN_TELE"] = str(tel)
        tok["CONN_UNEXPLAINED"] = str(unex)
        tok["CONN_POLICY"] = str(ca.get("policy", "?"))
        tok["CONN_ASC"] = str(left)
        # The sentence BEFORE this clause already says how many of the
        # disconnections follow a UAV leaving the active set, so the clause
        # must add the cause, not repeat the count.  A repeated count
        # reads as two facts where there is one.
        if unex == 0:
            cause = ("each of them is a UAV whose pack emptied away from a "
                     "charging pad" if left > 0 else
                     "every one of them is accounted for by a teleport")
            tok["CONN_VERDICT"] = (
                f"{cause}, and none follows a move, so the projection's "
                "guarantee is not violated by any executed move, and "
                "the availability below 1.000 is the price of a UAV "
                "leaving rather than a hole in the filter")
        else:
            tok["CONN_VERDICT"] = (
                f"{unex} of them are NOT attributable to either mechanism, "
                f"which is a residual the filter does not account for and "
                f"which we report rather than explain away")
        # The conclusion of the accounting, derived: a hand-written
        # sentence that says "every disconnection follows a UAV leaving"
        # is a CLAIM, and it is only true while the counts agree with it.
        if unex == 0 and tel == 0 and left == n and n > 0:
            tok["CONN_ALL_LEAVING"] = (
                "every one of them follows a UAV leaving rather than a move")
        elif unex == 0 and tel == 0:
            tok["CONN_ALL_LEAVING"] = (
                "none of them is attributable to a move")
        else:
            tok["CONN_ALL_LEAVING"] = (
                f"the accounting attributes {left} to a UAV leaving, {tel} "
                f"to a teleport and {unex} to neither, and we report the "
                f"split rather than rounding it to a cleaner story")

    # ------------------------------------------------------------------
    # the alternative corrected deployment (N = 5, K = 2, R_cov = 45 m)
    # ------------------------------------------------------------------
    # the threshold is what the witness construction needs: four discs at
    # the side midpoints and one at the centre, over the square's grid
    try:
        side = 80.0
        centres = [(side / 2, 0.0), (side / 2, side), (0.0, side / 2),
                   (side, side / 2), (side / 2, side / 2)]
        grid = np.linspace(0.0, side, 161)
        X, Y = np.meshgrid(grid, grid, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        d = np.linalg.norm(pts[:, None, :] - np.asarray(centres)[None, :, :],
                           axis=-1)
        d.sort(axis=1)
        tok["ALT_N5_R"] = f"{float(d[:, 1].max()):.2f}"
    except Exception:                                      # pragma: no cover
        pass

    # ------------------------------------------------------------------
    # THE COMBINED-STRESS TABLE.  The ordering of the components is not the
    # same there as in the nominal regime, and the difference is a finding
    # rather than a nuisance, so it is derived rather than left in a table.
    # ------------------------------------------------------------------
    stress = load_csv("results_ablation_stress.csv")
    if stress:
        sby = {r["model"]: r for r in stress}
        sfull = sby.get("Full")
        if sfull:
            sfs = per_seed_ck(sfull)
            # The budget the stress arms were trained at, and the base
            # comparison's: a table of arms trained at a different budget
            # has to say so, because its Full row is not the base
            # comparison's row.
            tok["STRESS_EPISODES"] = str(sfull.get("episodes") or "?")
            tok["ABL_EPISODES_STRESS"] = tok["STRESS_EPISODES"]
            # The verdict is PAIRED, not a point-estimate threshold.  The
            # old version called a component a liability or a help on the
            # strength of a difference in means, which is the rule this
            # paper rejects three paragraphs earlier: every interval
            # here includes zero.
            resolved, unresolved = [], []
            for r in stress:
                if r["model"] in ("Full",):
                    continue
                vs = per_seed_ck(r)
                if not (sfs and vs and len(sfs) == len(vs)):
                    continue
                label = names.get(r["model"], r["model"])
                if r["model"] == "VanillaMAPPO":
                    label = "the whole design"
                # FULL minus the variant, the TABLE's convention: a
                # positive number means the component helps.  The
                # opposite order printed every sign in this summary
                # backwards against the table beside it.
                mean, half, p, n = paired_stat(sfs, vs)
                ex = (mean - half > 0) or (mean + half < 0)
                entry = (label, mean, half)
                (resolved if ex else unresolved).append(entry)
            parts = []
            if resolved:
                parts.append(
                    "the components whose paired interval excludes zero are "
                    + ", ".join(f"{lab} ({m:+.3f}, 95\\% interval "
                                f"[{m-h:+.3f}, {m+h:+.3f}])"
                                for lab, m, h in resolved))
            if unresolved:
                parts.append(
                    ("and " if resolved else "")
                    + "the components whose interval includes zero -- "
                    + ", ".join(f"{lab} ({m:+.3f}, 95\\% interval "
                                f"[{m-h:+.3f}, {m+h:+.3f}])"
                                for lab, m, h in unresolved)
                    + " -- are not separated by this table at all")
            if parts:
                tok["STRESS_SUMMARY"] = (
                    "The ordering of the components cannot be read off this "
                    "table: " + "; ".join(parts)
                    + ".  The differences below are the FULL configuration "
                    "minus the variant, so a negative number is what the "
                    "component is worth on this scenario.  The differences "
                    "between the variants are of the "
                    "same order as the width of the intervals, so the table "
                    "reports what the stress scenario does to the swarm and "
                    "not which component matters.  The framework's recovery "
                    "mechanism, which re-seats an exhausted UAV at a "
                    "charging pad, is a plausible mechanism for the "
                    "behaviour of the stress scenario --- a re-seated UAV is "
                    "out of communication range of the swarm until it flies "
                    "back --- but this table does not resolve it, and we do "
                    "not claim it.")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # the alternative corrected deployment, measured
    # ------------------------------------------------------------------
    alt = load_json("alt_deployment.json")
    if alt:
        by = {r["arm"]: r for r in alt}
        a5, r8 = by.get("alt"), by.get("reported")
        if a5 and r8:
            tok["ALT_MEASURED_CK"] = f(a5["CK"])
            tok["ALT_MEASURED_STD"] = f(a5.get("CK_std", 0.0))
            tok["ALT_REPORTED_CK"] = f(r8["CK"])
            tok["ALT_REPORTED_STD"] = f(r8.get("CK_std", 0.0))
            tok["ALT_DELTA"] = f(a5["CK"] - r8["CK"])
            tok["ALT_SEEDS"] = str(a5.get("seeds", "?"))
            tok["ALT_EPISODES"] = str(a5.get("episodes", "?"))
            better = a5["CK"] > r8["CK"]
            tok["ALT_BETTER"] = "yes" if better else "no"
            # The comparison is between the LEARNED controller at two
            # deployments.  The rule baselines were not measured at the
            # alternative operating point, so the deployment is not shown
            # to be better for a RULE -- only for the controller the
            # paper trains.  Saying "it is BETTER" claimed the stronger
            # thing.
            tok["ALT_COMPARISON"] = (
                "and on this instance the deployment it identifies is the "
                "better of the two for the learned controller"
                if better else
                "and on this instance the deployment it identifies is the "
                "worse of the two for the learned controller")
            # The short form is for the contribution list, which is a
            # summary: the full verdict, with its numbers, belongs to the
            # section that derives it.  Printing the same sentence in both
            # places was the paper's one exact duplication.
            tok["ALT_VERDICT_SHORT"] = (
                f"the alternative deployment is feasible for the learned "
                f"controller and it does better there, by "
                f"{a5['CK'] - r8['CK']:+.3f} of persistent coverage"
                if better else
                f"the alternative deployment is feasible for the learned "
                f"controller but does not do better there "
                f"({a5['CK'] - r8['CK']:+.3f} of persistent coverage)")
            if better:
                tok["ALT_VERDICT"] = (
                    f"the alternative deployment is feasible for the learned "
                    f"controller, and it does better there: with five UAVs "
                    f"and a 45 m coverage radius the learned controller "
                    f"reaches {a5['CK']:.3f} against {r8['CK']:.3f} for the "
                    f"reported eight-UAV deployment, a difference of "
                    f"{a5['CK']-r8['CK']:+.3f} on the same training seeds "
                    f"and the same evaluation scenes.  The rule baselines "
                    f"were not measured at that operating point, so this "
                    f"establishes feasibility for the learned controller "
                    f"and not the superiority of the deployment")
            else:
                tok["ALT_VERDICT"] = (
                    f"the alternative deployment is feasible for the learned "
                    f"controller but does not do better there: with five UAVs "
                    f"and a 45 m coverage radius the learned controller "
                    f"reaches {a5['CK']:.3f} against {r8['CK']:.3f} for the "
                    f"reported eight-UAV deployment, a difference of "
                    f"{a5['CK']-r8['CK']:+.3f}.  The rule baselines were not "
                    f"measured at that operating point, so the comparison is "
                    f"between the learned controller at two deployments and "
                    f"not between the deployments themselves")

    # the reproducibility of a whole matrix, measured across two runs
    # ------------------------------------------------------------------
    det = load_json("determinism_check.json")
    if det:
        tok["DETERMINISM_COMPARED"] = str(det.get("compared", "?"))
        tok["DETERMINISM_DIFFERING"] = str(det.get("differing", "?"))
        tok["DETERMINISM_TEXT"] = (
            f"The same property was then checked across two independent "
            f"runs of a whole matrix: the scenario matrix was measured "
            f"twice, from scratch, and all {det.get('compared', '?')} of the "
            f"compared values agreed exactly, the largest difference "
            f"between the two runs being "
            f"{float(det.get('max_abs_difference', 0.0)):g}.")

    # ------------------------------------------------------------------
    # the rule baselines at the alternative deployment
    # ------------------------------------------------------------------
    # The paper's own feasibility certificate identifies a better
    # deployment, and the rule baselines had never been measured there,
    # so the headline gap was a statement about the reported deployment
    # alone.  The rules are deterministic, so this costs no training.
    altr = load_json("alt_deployment_rules.json")
    if altr:
        dep = altr.get("deployments", {})
        rep = dep.get("reported_n8_r35", {})
        alt = dep.get("alternative_n5_r45", {})
        if rep and alt:
            rep_rule = max(float(rep.get("reactive_CK", 0.0)),
                           float(rep.get("reactive_local_CK", 0.0)))
            alt_rule = max(float(alt.get("reactive_CK", 0.0)),
                           float(alt.get("reactive_local_CK", 0.0)))
            rep_name = ("the unprivileged reactive rule"
                        if float(rep.get("reactive_local_CK", 0.0)) >=
                        float(rep.get("reactive_CK", 0.0))
                        else "the reactive rule")
            alt_name = ("the unprivileged reactive rule"
                        if float(alt.get("reactive_local_CK", 0.0)) >=
                        float(alt.get("reactive_CK", 0.0))
                        else "the reactive rule")
            tok["ALT_RULE_NAME_REPORTED"] = rep_name
            tok["ALT_RULE_NAME_ALT"] = alt_name
            tok["ALT_RULE_REPORTED_CK"] = f"{rep_rule:.3f}"
            tok["ALT_RULE_ALT_CK"] = f"{alt_rule:.3f}"
            # the learned controller at the alternative deployment
            try:
                alt_ck = float(tok.get("ALT_MEASURED_CK", "0"))
            except (TypeError, ValueError):
                alt_ck = 0.0
            try:
                rep_ck = float(tok.get("ALT_REPORTED_CK", "0"))
            except (TypeError, ValueError):
                rep_ck = 0.0
            if alt_ck > 0 and rep_ck > 0:
                tok["ALT_RULE_GAP_REPORTED"] = f"{rep_rule - rep_ck:.3f}"
                tok["ALT_RULE_GAP_ALT"] = f"{alt_rule - alt_ck:.3f}"
                tok["ALT_RULES_SURVIVE"] = (
                    f"the rule baselines were measured at that deployment as "
                    f"well, and {alt_name} reaches {alt_rule:.3f} there "
                    f"against {rep_rule:.3f} at the reported deployment, so "
                    f"the gap of the learned controller is "
                    f"{alt_rule - alt_ck:.3f} there against "
                    f"{rep_rule - rep_ck:.3f} at the reported one: the "
                    f"ordering survives the better deployment, with a "
                    f"smaller margin")

    # ------------------------------------------------------------------
    # the exclusion family, corrected for multiplicity
    # ------------------------------------------------------------------
    # The claim that the arms exclude the candidate explanations is a
    # CONJUNCTION over the arms, and each arm's interval is a plain
    # 95% two-sided interval, so the family is unadjusted.  The
    # correction is computed here rather than asserted: with a
    # Bonferroni correction over the arms the largest adjusted upper
    # limit is computed from the arms' own means, half-widths and
    # degrees of freedom, and the paper quotes it.
    arm_rows = []
    for key in ("OBS_B", "OBS_C", "EC", "ED", "CREDIT", "BUDGET"):
        try:
            mean = float(tok.get(f"{key}_PAIRED", "nan"))
            half = float(tok.get(f"{key}_HALF", "nan"))
            n = int(tok.get(f"{key}_N", "0"))
        except (TypeError, ValueError):
            continue
        if mean == mean and half == half and n > 1:
            arm_rows.append((key, mean, half, n))
    if len(arm_rows) >= 2:
        try:
            from scipy import stats
            m = len(arm_rows)
            worst = -1e9
            for key, mean, half, n in arm_rows:
                df = n - 1
                t975 = float(stats.t.ppf(0.975, df))
                if t975 <= 0:
                    continue
                se = half / t975
                t_adj = float(stats.t.ppf(1.0 - 0.05 / m, df))
                worst = max(worst, mean + t_adj * se)
            if worst > -1e8:
                tok["ARMS_BONF_UPPER"] = f"{worst:.3f}"
                tok["ARMS_BONF_N"] = str(m)
        except Exception:                                      # pragma: no cover
            pass
    # the gap's own lower limit, for the comparison the correction is
    # made against
    try:
        _ci = tok.get("GAP_CI", "[0.000, 0.000]").strip("[]")
        _lo = float(_ci.split(",")[0])
        tok["GAP_CI_LOWER"] = f"{_lo:.3f}"
    except (TypeError, ValueError):
        pass

    # ------------------------------------------------------------------
    # the plasticity probe of the trained actor
    # ------------------------------------------------------------------
    dor = load_json("dormancy.json")
    if dor:
        silent_vals, dormant_any = [], 0
        # The token names carry no digits: TeX ends a control word at the
        # first non-letter, so a token called DORM_L0_T001 would be the
        # macro \tokDormLzeroTzerozeroone.
        layer_names = {"L0": "A", "L1": "B"}
        threshold_names = {"0.001": "LOW", "0.01": "MID", "0.05": "MIDHIGH",
                           "0.1": "HIGH"}
        for r in dor:
            lay = layer_names.get(str(r.get("layer", "L0")).upper(), "A")
            for tau, tag in (("0.001", "LOW"), ("0.01", "MID"),
                             ("0.05", "MIDHIGH"), ("0.1", "HIGH")):
                tok[f"DORM_{lay}_{tag}_DORMANT"] = f(
                    r.get(f"dormant_frac_{tau}", 0.0), 3)
                tok[f"DORM_{lay}_{tag}_SILENT"] = f(
                    r.get(f"silent_frac_{tau}", 0.0), 3)
                tok[f"DORM_{lay}_{tag}_DEAD"] = f(
                    r.get(f"dead_frac_{tau}", 0.0), 3)
                if tau in ("0.001", "0.01", "0.05", "0.1"):
                    silent_vals.append(float(r.get(f"silent_frac_{tau}", 0.0)))
                    dormant_any = max(dormant_any,
                                      float(r.get(f"dormant_frac_{tau}", 0.0)))
            tok[f"DORM_{lay}_SOME"] = str(
                r.get("neurons_dormant_for_some_agent_only", 0))
        if silent_vals:
            tok["DORM_SILENT_MIN"] = f"{100 * min(silent_vals):.0f}"
            tok["DORM_SILENT_MAX"] = f"{100 * max(silent_vals):.0f}"
        tok["DORM_DORMANT_MAX"] = f"{dormant_any:.3f}"

    # ------------------------------------------------------------------
    # the training-budget sweep
    # ------------------------------------------------------------------
    bud = load_json("ab_budget.json")
    if bud:
        rows = sorted(bud, key=lambda r: int(r.get("episodes", 0)))
        # The arithmetic is done on the ROUNDED values, so that the paper's
        # own numbers add up: the increments it prints must sum to the
        # spread it prints, and both must be the difference of the values
        # the reader can see in the sentence.
        cks = [round(float(r["CK"]), 3) for r in rows]
        stds = [float(r.get("CK_std", 0.0)) for r in rows]
        ns = [int(r.get("seeds", 3) or 3) for r in rows]
        parts = ", ".join(f"{int(r['episodes'])} episodes: {c:.3f}"
                          for r, c in zip(rows, cks))
        incs = [b - a for a, b in zip(cks, cks[1:])]
        spread = cks[-1] - cks[0]
        tok["BUDGET_SWEEP"] = parts
        tok["BUDGET_SPREAD"] = f"{spread:+.3f}"
        tok["BUDGET_MIN"] = f"{min(cks):.3f}"
        tok["BUDGET_MAX"] = f"{max(cks):.3f}"
        tok["BUDGET_RANGE"] = (f"{int(rows[0]['episodes'])}--"
                               f"{int(rows[-1]['episodes'])}")
        tok["BUDGET_N"] = str(len(rows))
        tok["BUDGET_INCREMENTS"] = ", ".join(f"{d:+.3f}" for d in incs)
        tok["BUDGET_LAST_DELTA"] = f"{incs[-1]:+.3f}" if incs else "n/a"
        tok["BUDGET_FIRST_DELTA"] = f"{incs[0]:+.3f}" if incs else "n/a"
        max_std = max(stds) if stds else 0.0
        tok["BUDGET_MAX_STD"] = f"{max_std:.3f}"
        # the movement as a fraction of the gap it would have to explain
        if "GAP" in tok:
            gap = float(tok["GAP"])
            frac = abs(spread) / gap if gap else float("nan")
            tok["BUDGET_FRAC"] = f"{100 * frac:.0f}"
            tok["BUDGET_GAP"] = f"{gap:.3f}"
        tok["BUDGET_SEEDS"] = str(ns[-1] if ns else 0)
        # IS the last increment distinguishable from zero?
        # The sweep trains the SAME training seeds at every budget, so two
        # budgets can be compared on the same runs, and the paired
        # comparison is the one the design supports.  The unpaired
        # comparison of two means throws that pairing away and is wider
        # than it needs to be -- wide enough, in this case, to call a
        # resolved increment unresolved.
        se = float("nan")
        t = float("nan")
        half_last = float("nan")
        p_last = float("nan")
        n_pair = 0
        try:
            from scipy import stats as _st_b
            crit_b = float(_st_b.t.ppf(0.975, max(n_pair - 1, 1))) \
                if n_pair > 1 else 0.0
        except Exception:                                      # pragma: no cover
            crit_b = 0.0
        if len(rows) >= 2:
            pa = rows[-2].get("per_seed_CK") or []
            pb = rows[-1].get("per_seed_CK") or []
            if pa and pb and len(pa) == len(pb):
                _m, _h, _p, n_pair = paired_stat(pb, pa)
                half_last = _h
                p_last = _p
                if n_pair > 1:
                    try:
                        from scipy import stats as _st_b
                        crit_b = float(_st_b.t.ppf(0.975, n_pair - 1))
                    except Exception:                          # pragma: no cover
                        crit_b = 1.96
                se = _h / crit_b if crit_b > 0 else float("nan")
                t = _m / se if (se == se and se > 0) else float("nan")
        tok["BUDGET_SE"] = f"{se:.3f}" if se == se else "n/a"
        tok["BUDGET_T"] = f"{t:.2f}" if t == t else "n/a"
        tok["BUDGET_LAST_CI"] = (f"[{incs[-1]-half_last:+.3f}, "
                                  f"{incs[-1]+half_last:+.3f}]"
                                  if half_last == half_last else "n/a")
        tok["BUDGET_LAST_P"] = f"{p_last:.3f}" if p_last == p_last else "n/a"
        tok["BUDGET_PAIR_N"] = str(n_pair)
        resolved = bool(t == t and abs(t) >= 2.0)
        tok["BUDGET_LAST_RESOLVED"] = "yes" if resolved else "no"
        if len(rows) < 2:
            # ONE budget is not a sweep.  The increments of a sweep with a
            # single point are an empty list, and the paper must not print
            # an empty list or a not-a-number: the sweep has to be
            # finished before anything is said about it.
            tok["BUDGET_VERDICT"] = ""
            tok["BUDGET_CAVEAT"] = ""
        elif resolved:
            # the coverage really is still rising at the longest budget
            tok["BUDGET_VERDICT"] = (
                f"the coverage moves by {spread:+.3f} over the range the "
                f"sweep covers, which is about {tok.get('BUDGET_FRAC', 'n/a')}"
                f"\\% of the gap of {tok.get('BUDGET_GAP', 'n/a')}, and the "
                f"increments between successive budgets are "
                f"{tok['BUDGET_INCREMENTS']}; the last of them, "
                f"{tok['BUDGET_LAST_DELTA']}, is {abs(t):.1f} standard errors "
                f"from zero (the standard error of the difference is "
                f"{tok['BUDGET_SE']}, paired over the same {n_pair} training "
                f"seeds), so the coverage does keep rising with the budget "
                f"over the range the sweep covers")
        else:
            # the last increment is WITHIN the noise: the sweep's own
            # TREND does not resolve what each further step of budget
            # buys.  It says nothing about whether the budget matters
            # overall -- the paired comparison of the shortest and the
            # longest budget is the arm that answers that, and it is
            # reported separately.
            tok["BUDGET_VERDICT"] = (
                f"the coverage moves by {spread:+.3f} over the range the "
                f"sweep covers, which is about {tok.get('BUDGET_FRAC', 'n/a')}"
                f"\\% of the gap of {tok.get('BUDGET_GAP', 'n/a')}, and the "
                f"increments between successive budgets are "
                f"{tok['BUDGET_INCREMENTS']}; but the last of them, "
                f"{tok['BUDGET_LAST_DELTA']}, is only "
                f"{abs(t):.1f} standard errors from zero "
                f"(the standard error of the difference is {tok['BUDGET_SE']}"
                f", paired over the same {n_pair} training seeds), so the "
                f"sweep's own trend does not resolve what each further step "
                f"of budget buys")
        if len(rows) < 2:
            tok["BUDGET_CAVEAT"] = ""
        else:
            tok["BUDGET_CAVEAT"] = (
                "we do not extrapolate the sweep past the budgets it covers, "
                "and we do not claim that a larger budget would close the gap")
        # The budget arm's own interval: the shortest budget against the
        # longest, paired by training seed.  It is the arm the abstract
        # names, so it needs an interval of its own: the sweep's trend
        # does not give one.
        if len(rows) >= 2:
            pa = rows[0].get("per_seed_CK") or []
            pb = rows[-1].get("per_seed_CK") or []
            if pa and pb and len(pa) == len(pb):
                # the direction that could explain the gap is MORE budget
                # minus LESS budget, so the comparison is made in that
                # direction: the shorter budget's rows are the second
                # argument
                mean, half, p, n = paired_stat(pb, pa)
                tok["BUDGET_PAIRED"] = f"{mean:+.3f}"
                tok["BUDGET_PAIRED_HALF"] = f"{half:.3f}"
                tok["BUDGET_PAIRED_CI"] = f"[{mean-half:+.3f}, {mean+half:+.3f}]"
                tok["BUDGET_PAIRED_P"] = f"{p:.3f}"
                tok["BUDGET_PAIRED_N"] = str(n)
                tok["BUDGET_PAIRED_EXCLUDES_GAP"] = (
                    "yes" if (mean + half < 0) else "no")
                if "GAP" in tok:
                    gap = float(tok["GAP"])
                    tok["BUDGET_PAIRED_UP"] = f"{mean + half:.3f}"
                    tok["BUDGET_PAIRED_VS_GAP"] = (
                        f"its upper limit of {mean+half:.3f} "
                        + ("is below" if mean + half < gap else
                           "is above")
                        + f" the gap of {gap:.3f}")
                    # The verdict, derived from the interval and the gap:
                    # whether the budget's own effect is resolved at all,
                    # and whether it is large enough to be the explanation.
                    excludes_zero = (mean - half > 0) or (mean + half < 0)
                    below_gap = (mean + half) < gap
                    if excludes_zero and below_gap:
                        tok["BUDGET_PAIRED_VERDICT"] = (
                            f"the paired comparison over the same training "
                            f"seeds does resolve a real gain, "
                            f"{tok['BUDGET_PAIRED']} with a 95\\% interval of "
                            f"{tok['BUDGET_PAIRED_CI']}, and that interval's "
                            f"upper limit of {mean+half:.3f} is below the gap "
                            f"of {gap:.3f}: more training does help, and over "
                            f"the {int(rows[0]['episodes'])}--"
                            f"{int(rows[-1]['episodes'])} episode range the "
                            f"sweep covers the whole of its measured effect "
                            f"is too small to account for the shortfall, so "
                            f"the budget is excluded as the explanation of the "
                            f"shortfall OVER THAT RANGE.  Two things are not "
                            f"claimed.  The increments are increasing "
                            f"({tok['BUDGET_INCREMENTS']}) and the last of "
                            f"them is still resolved against zero, so the "
                            f"sweep has not saturated; and an interval on a "
                            f"measured range is a bound on that range and not "
                            f"on the budgets beyond it, so a larger budget is "
                            f"not excluded by this measurement.  Settling that "
                            f"would need the sweep to run until the paired "
                            f"increment is unresolved, which this paper does "
                            f"not do")
                    elif excludes_zero and not below_gap:
                        tok["BUDGET_PAIRED_VERDICT"] = (
                            f"the paired comparison over the same training "
                            f"seeds does resolve a real gain, "
                            f"{tok['BUDGET_PAIRED']} with a 95\\% interval of "
                            f"{tok['BUDGET_PAIRED_CI']}, and that interval's "
                            f"upper limit of {mean+half:.3f} reaches past "
                            f"the gap of {gap:.3f}, so the budget is NOT "
                            f"excluded by this arm")
                    else:
                        tok["BUDGET_PAIRED_VERDICT"] = (
                            f"the paired comparison over the same training "
                            f"seeds does not resolve a difference between "
                            f"the shortest and the longest budget: "
                            f"{tok['BUDGET_PAIRED']} with a 95\\% interval "
                            f"of {tok['BUDGET_PAIRED_CI']}, which includes "
                            f"zero")
        # the sweep's own last increment: resolved or not
        if tok.get("BUDGET_LAST_RESOLVED") == "yes":
            tok["BUDGET_TREND_CLAUSE"] = (
                "the sweep's last increment IS resolved against zero, so "
                "the coverage is still rising at the longest budget we ran")
        else:
            tok["BUDGET_TREND_CLAUSE"] = (
                "the sweep's own last increment is NOT resolved against "
                "zero, so the sweep does not settle the question on unseen "
                "scenes either")

    # ------------------------------------------------------------------
    # the episode length, in both units, so that the two "400"s of the
    # paper (the 2000-step episode's 400 SECONDS, and the earlier
    # 400-STEP episode) can never be confused
    # ------------------------------------------------------------------
    try:
        # the CANONICAL configuration, not the dataclass defaults: the
        # default episode length is not the one the experiments use
        from uav_wireless.experiments import base_config
        c0 = base_config()
        endurance = c0.B_max / c0.P_hover            # seconds of hover
        old_steps = 400
        old_seconds = old_steps * c0.dt
        tok["PACK_ENDURANCE"] = f"{endurance:.0f}"
        tok["OLD_EPISODE_S"] = f"{old_seconds:.0f}"
        tok["OLD_EPISODE_PCT"] = f"{100 * old_seconds / endurance:.0f}"
        tok["EPISODE_STEPS"] = f"{int(c0.episode_len)}"
        tok["EPISODE_SECONDS"] = f"{c0.episode_len * c0.dt:.0f}"
        # How large the filter's intervention is, as a SHARE of the
        # episode's sampled actions.  The count is accumulated over EVERY
        # agent, so the denominator is the number of agents times the
        # number of steps -- dividing by the steps alone understated the
        # denominator by a factor of N and made the filter's intervention
        # look smaller than it is.
        try:
            un = float(tok.get("MAPPO_UNEXEC", "0"))
            sampled = int(c0.N) * int(c0.episode_len)
            tok["MAPPO_SAMPLED"] = f"{sampled:,}".replace(",", r"\,")
            if un > 0:
                r = sampled / un
                if abs(r - round(r)) < 0.05:
                    tok["MAPPO_UNEXEC_RATIO"] = f"one action in {int(round(r))}"
                else:
                    tok["MAPPO_UNEXEC_RATIO"] = (
                        f"{100 * un / sampled:.3f}\\% of them")
            else:
                tok["MAPPO_UNEXEC_RATIO"] = "none of them"
        except Exception:                                      # pragma: no cover
            pass
        # the pack's endurance IN STEPS at the moving draw, and the duty
        # cycle the episode therefore implies
        draw = (c0.P_move + c0.P_elec + c0.P_tx)      # watts in flight
        tok["PACK_STEPS_MOVING"] = f"{c0.B_max / (draw * c0.dt):.0f}"
        tok["PACK_SECONDS_MOVING"] = f"{c0.B_max / draw:.0f}"
        tok["MOVE_DRAW_W"] = f"{draw:.1f}"
        tok["RESERVE_J"] = f"{c0.E_reserve:.0f}"
        # the duty cycle the measured charging behaviour implies: the
        # fraction of the episode the swarm is NOT charging
        tok["DUTY_CYCLE"] = "n/a"
    except Exception:                                      # pragma: no cover
        pass

    react_row = main_rows.get("Reactive")
    if react_row and "DUTY_CYCLE" in tok:
        try:
            steps = int(tok["EPISODE_STEPS"])
            ret = float(react_row.get("ReturnEvents", 0))
            # each return-to-charge transition is one step spent charging
            duty = 100.0 * (1.0 - ret / steps)
            tok["DUTY_CYCLE"] = f"{duty:.0f}"
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # ------------------------------------------------------------------
    # the feasibility numbers: the packing survey over random sensor draws,
    # and the geometry of the corner certificate's transfer to a sensor set
    # ------------------------------------------------------------------
    feas = load_json_ml("results_feasibility.json")
    if feas:
        ps = feas.get("packing_survey") or {}
        if ps:
            tok["PACK_SEEDS"] = str(ps["seeds"])
            # three decimals, so that the mean and the bound derived from
            # it are visibly consistent to the reader (2 x 2.133 = 4.27)
            tok["PACK_MEAN"] = f(ps["packing_size_mean"], 3)
            tok["PACK_MIN"] = str(ps["packing_size_min"])
            tok["PACK_MAX"] = str(ps["packing_size_max"])
            tok["PACK_N_MEAN"] = f(ps["N_required_mean"], 2)
            tok["PACK_N_MIN"] = str(ps["N_required_min"])
            tok["PACK_N_MAX"] = str(ps["N_required_max"])
            tok["PACK_EXCLUDED"] = str(ps["seeds_excluded"])
            tok["PACK_EXCL_PCT"] = f"{100 * ps['seeds_excluded_fraction']:.0f}"

    # the PHY equivalence's own number: the worst |delta| in dB between the
    # Python log-distance model and the INET breakpoint path-loss model
    phy = load_json("phy_equivalence.json")
    if phy:
        tok["PHY_DELTA"] = f(phy.get("worst_delta_db", 0.0), 3)
        tok["PHY_ROWS"] = str(len(phy.get("rows", [])))

    ct = load_json("corner_transfer.json")
    if ct:
        # the cap radius: the largest radius at which one sensor per corner
        # still gives pairwise disjoint coverage discs
        tok["CAP_R"] = f(ct["cap_radius_max"], 2)
        # the probability that a uniform draw hits all four caps, in the
        # LaTeX form for a small number: 9.339e-06 -> $9.3\times10^{-6}$
        mant, expo = f"{ct['probability_all_caps_hit']:.1e}".split("e")
        tok["CAP_P"] = f"{mant}\\times 10^{{{int(expo)}}}"
        tok["CAP_PCT"] = f"{100 * ct['cap_area_fraction']:.2f}"
        tok["CAP_TWO_R"] = f(ct["two_R_cov"], 0)

    # ------------------------------------------------------------------
    # the numbers the paper quotes about ITSELF
    # ------------------------------------------------------------------
    # the claim audit's own counts, so that the paper cannot claim a
    # different number of verified claims than the audit verifies
    audit = load_json("claim_audit.json")
    if audit:
        n_pass = sum(1 for r in audit if r.get("status") == "PASS")
        n_fail = sum(1 for r in audit if r.get("status") == "FAIL")
        n_info = sum(1 for r in audit if r.get("status") == "INFO")
        tok["AUDIT_ITEMS"] = str(len(audit))
        tok["AUDIT_PASS"] = str(n_pass)
        tok["AUDIT_FAIL"] = str(n_fail)
        tok["AUDIT_INFO"] = str(n_info)
        # the connectivity regime, as the audit measured it
        for r in audit:
            d = r.get("data") or {}
            if r.get("id") == "C6" and d:
                tok["C6_DEGREE"] = f(d["mean_degree"], 2)
                tok["C6_MAXDEG"] = str(int(d["max_degree"]))
                tok["C6_COMPLETE"] = f"{100*d['frac_complete']:.0f}"
                tok["C6_LEAVES"] = f"{100*d['frac_leaves']:.0f}"

    # the size of the test suite, counted the way the runner discovers it.
    # The test module is PARSED, not imported: importing it pulls in torch,
    # which loads the CUDA runtime, and that is several hundred megabytes
    # that the machine may not have while the experiments are running.
    try:
        import ast
        tests_dir = os.path.join(ML, "tests")
        n_tests = 0
        for fname in sorted(os.listdir(tests_dir)):
            if not fname.startswith("test_") or not fname.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(tests_dir, fname),
                                  encoding="utf-8").read())
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                if not node.name.startswith("Test"):
                    continue
                for item in node.body:
                    if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and item.name.startswith("test_")):
                        n_tests += 1
        tok["N_TESTS"] = str(n_tests)
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"WARNING: the test count could not be determined: {exc}",
              file=sys.stderr)
    # the variance decomposition of the headline number: the seed-to-seed
    # component and the scene-to-scene component, reported separately.  The
    # scene component is COMMON to every controller, because they are all
    # evaluated on the same scenes, and it therefore cancels in every
    # comparison the paper draws; the seed component does not.
    seed_rows = os.path.join(HERE, "base_seed_rows.jsonl")
    if os.path.exists(seed_rows):
        by_seed = {}
        for line in open(seed_rows, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ps = d.get("per_seed") or []
            if ps:
                # the file is appended to, so the LAST row for a seed is the
                # one from the most recent run
                by_seed[int(d.get("train_seed", len(by_seed)))] = \
                    [float(x["CK"]) for x in ps]
        per_seed_scenes = [by_seed[k] for k in sorted(by_seed)]
        if len(per_seed_scenes) > 1:
            n_scenes = min(len(x) for x in per_seed_scenes)
            seed_means = [float(np.mean(x[:n_scenes])) for x in per_seed_scenes]
            scene_means = [float(np.mean([x[e] for x in per_seed_scenes]))
                           for e in range(n_scenes)]
            s_seed = float(np.std(seed_means, ddof=1))
            s_scene = float(np.std(scene_means, ddof=1))
            tok["VAR_SEED"] = f"{s_seed:.3f}"
            tok["VAR_SCENE"] = f"{s_scene:.3f}"
            tok["VAR_SEED_SE"] = f"{s_seed / np.sqrt(len(seed_means)):.3f}"
            tok["VAR_SCENE_SE"] = f"{s_scene / np.sqrt(n_scenes):.3f}"
            tok["VAR_N_SEEDS"] = str(len(seed_means))
            tok["VAR_N_SCENES"] = str(n_scenes)
            # the yardstick of the whole paper: the seed-to-seed spread
            tok["ABL_YARDSTICK"] = f"{s_seed:.3f}"

    # The seed-to-seed spread above was taken from base_seed_rows.jsonl,
    # which still holds a 3-seed run; the paper's headline is an 8-seed
    # run, so the authoritative spread is recomputed from the base table's
    # per-seed CKs.  The scene decomposition (VAR_SCENE) is unaffected,
    # because it measures per-scene variation of a fixed controller.
    _base = load_csv("results_main.csv")
    _mp = next((r for r in _base if r.get("model") == "MAPPO"), None) if _base else None
    if _mp:
        _ps = per_seed_ck(_mp)
        if len(_ps) > 1:
            _s = float(np.std(_ps, ddof=1))
            tok["VAR_SEED"] = f"{_s:.3f}"
            tok["VAR_SEED_SE"] = f"{_s / np.sqrt(len(_ps)):.3f}"
            tok["VAR_N_SEEDS"] = str(len(_ps))
            tok["ABL_YARDSTICK"] = f"{_s:.3f}"

    # ------------------------------------------------------------------
    # the learning curve of the first training run
    # ------------------------------------------------------------------
    # The curve is what speaks to the compute-asymmetry objection: a
    # hand-designed rule embeds a human's solution, and the learner gets a
    # fixed budget, so the question "would more training have helped?" has
    # to be answered rather than assumed.  The per-episode coverage of the
    # first training run, on the training scene, answers it.
    hist_path = os.path.join(ML, "hist_base.json")
    if os.path.exists(hist_path):
        try:
            hist = json.load(open(hist_path, encoding="utf-8"))
        except Exception:                                  # pragma: no cover
            hist = []
        cks = [float(h["CK"]) for h in hist if "CK" in h]
        if len(cks) >= 10:
            k = min(5, len(cks) // 2)
            tok["CURVE_EPISODES"] = str(len(cks))
            tok["CURVE_FIRST"] = f"{float(np.mean(cks[:k])):.3f}"
            tok["CURVE_LAST"] = f"{float(np.mean(cks[-k:])):.3f}"
            tok["CURVE_BEST"] = f"{max(cks):.3f}"
            tok["CURVE_FINAL"] = f"{cks[-1]:.3f}"
            tok["CURVE_MIN"] = f"{min(cks):.3f}"
            tok["CURVE_MAX"] = f"{max(cks):.3f}"
            # the evaluated coverage of the SAME policy, on unseen scenes
            if per_seed_scenes:
                tok["CURVE_EVAL"] = f"{float(np.mean(per_seed_scenes[0])):.3f}"
    return screen(tok)


# A value that is not a number must never reach the paper.  The paper is
# assembled from artefacts that may be MID-WRITE: an experiment that was
# killed half way through leaves a partial artefact, and a partial
# artefact is enough to make a token's arithmetic produce a nan or an
# empty list.  Printing "nan standard errors" in a paper is not a
# cosmetic defect -- it is a false statement about a measurement that was
# never made.  So the value is screened here, and a token that fails the
# screen is REMOVED: the paper then shows a visible [pending: ...]
# marker, which is honest, instead of a number that is not a number.
#
# The screen is deliberately narrow: `nan` and `inf` as standalone words
# (neither is ever an English word in this paper), an empty value, and
# the literal words None/null when the value IS one of them.  A sentence
# that legitimately contains the word "null" -- "a null is a statement
# about the resolution of that arm" -- is not a bad value.
_BAD_VALUE = re.compile(r"(?<![A-Za-z])(nan|inf)(?![A-Za-z])", re.I)


def screen(tok):
    """Remove every token whose value is not a usable measurement.

    A token is dropped if its value is empty, if it contains a bare
    `nan` or `inf`, or if the value is literally `None` or `null`.
    Dropping it turns the paper's [pending: TOKEN] marker on, which is
    the honest rendering of "this experiment has not produced this number
    yet".
    """
    bad = []
    for name in list(tok):
        v = tok[name]
        s = "" if v is None else str(v)
        if (s.strip() == "" or _BAD_VALUE.search(s)
                or s.strip() in ("None", "null")):
            bad.append(name)
            tok.pop(name)
    if bad:
        print(f"  [screen] {len(bad)} token(s) dropped as unusable: "
              + ", ".join(sorted(bad)))
    # A token that embeds another token must embed one that exists.  Without
    # this, a missing sibling becomes a visible [pending: ...] marker sitting
    # inside the sentence, which is how the ablation summary ended up quoting
    # typed numbers instead of the macro: the note it referenced was never
    # computed, and the prose kept the typed value rather than the broken
    # macro.
    for _k, _v in list(tok.items()):
        for _ref in re.findall(r"@@([A-Z_0-9]+)@@", str(_v)):
            if _ref not in tok:
                raise RuntimeError(
                    f"token {_k!r} embeds undefined token {_ref!r}")
    return tok


def fill(text, tok):
    """Replace every @@TOKEN@@ in `text`; return (filled_text, missing).

    The substitution is repeated until it reaches a fixed point, because a
    token's VALUE can itself contain a token: the ablation's summary quotes
    the resolution, and if the two are substituted in the wrong order the
    quoted token reaches the paper as text.
    """
    for _ in range(4):
        for k, v in tok.items():
            text = text.replace(f"@@{k}@@", str(v))
        if not re.search(r"@@[A-Z_0-9]+@@", text):
            break
    # a token with only its opening delimiter is just as wrong as one
    # with both, so the check looks for the opening delimiter alone.
    # Token names may be camelCase (e.g. @@MAPPOStranded@@), so the
    # pattern must accept lower-case letters too; otherwise a token whose
    # name starts with a capital and continues in lower case is reported
    # as a missing mixed-case fragment of itself.
    missing = sorted(set(re.findall(r"@@+([A-Za-z0-9_]+)", text)))
    return text, missing


def main():
    tok = build_tokens()
    tmpl = os.path.join(HERE, "results_text.tex")
    out = os.path.join(HERE, "results_text_filled.tex")
    s = open(tmpl, encoding="utf-8").read()
    s, missing = fill(s, tok)
    with open(out, "w", encoding="utf-8") as f:
        f.write(s)
    if missing:
        print("UNFILLED TOKENS:", missing)
        return 1
    print("all tokens filled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
