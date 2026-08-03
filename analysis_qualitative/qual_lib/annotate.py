"""Annotation rubric schemas + validation + intra-annotator agreement.

Annotation itself is performed by Claude in-session (no external API): Claude
reads out/samples/<dim>/batch_NNN.jsonl and writes matching records to
out/annotations/<dim>/batch_NNN.jsonl. This module only defines the schemas
and checks conformity/coverage/agreement.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from qual_lib import provenance

RUBRICS = {
    "messages": {
        "name": "MessageAnnotation",
        "fields": {
            "categories": "list from [inform, request, commit, acknowledge, "
                          "question, noise] (multi-label)",
            "coordination_function": "one of none|inform|request|commit|"
                                     "acknowledge — the PRIMARY function",
            "is_grounded": "bool — consistent with the context window "
                           "(actions/chamber), no invented objects",
            "specificity": "0 (vague) | 1 (some referents) | 2 (actionable, "
                           "names object AND location/direction or addressee "
                           "task)",
            "notes": "short free text (optional)",
        },
        "enums": {"coordination_function":
                  ["none", "inform", "request", "commit", "acknowledge"],
                  "specificity": [0, 1, 2]},
        "required": ["categories", "coordination_function", "is_grounded",
                     "specificity"],
    },
    "beliefs": {
        "name": "BeliefAnnotation",
        "fields": {
            "hallucination": "bool — belief asserts something impossible/"
                             "contradicted by the ground-truth line",
            "stale": "bool — plausibly true earlier but not now",
            "verifiable_claims": "int — claims checkable against ground truth",
            "correct_claims": "int — of those, how many are correct",
            "notes": "short free text (optional)",
        },
        "enums": {},
        "required": ["hallucination", "stale", "verifiable_claims",
                     "correct_claims"],
    },
    "social": {
        "name": "SocialAnnotation",
        "fields": {
            "mentions_bond_values": "bool — reasoning cites bond numbers/"
                                    "directions",
            "values_match_table": "yes|no|approx|absent — cited values vs "
                                  "referenced_bonds",
            "decision_follows_bonds": "yes|no|partial — ask/respond choice "
                                      "consistent with bond ranking",
            "explanation_quality": "0 (generic) | 1 (references state) | "
                                   "2 (bond-specific causal reasoning)",
            "notes": "short free text (optional)",
        },
        "enums": {"values_match_table": ["yes", "no", "approx", "absent"],
                  "decision_follows_bonds": ["yes", "no", "partial"],
                  "explanation_quality": [0, 1, 2]},
        "required": ["mentions_bond_values", "values_match_table",
                     "decision_follows_bonds", "explanation_quality"],
    },
    "failures": {
        "name": "FailureAdjudication",
        "fields": {
            "is_failure": "bool — the window shows genuinely unproductive/"
                          "erroneous behavior",
            "matches_detector": "bool — the failure is of the flagged type "
                                "(false for controls unless a real failure "
                                "coincides)",
            "severity": "0 (cosmetic) | 1 (wastes steps) | 2 (blocks "
                        "progression)",
            "cause_hypothesis": "short free text",
            "notes": "short free text (optional)",
        },
        "enums": {"severity": [0, 1, 2]},
        "required": ["is_failure", "matches_detector", "severity"],
    },
}


def _check_record(dim, rec):
    errs = []
    r = RUBRICS[dim]
    for f in r["required"]:
        if f not in rec:
            errs.append(f"missing:{f}")
    for f, allowed in r["enums"].items():
        if f in rec and rec[f] not in allowed:
            errs.append(f"bad_enum:{f}={rec[f]!r}")
    return errs


def _cohen_kappa(pairs):
    """pairs: [(a, b)] categorical. Returns kappa or None."""
    if len(pairs) < 5:
        return None
    cats = sorted({str(x) for p in pairs for x in p})
    idx = {c: i for i, c in enumerate(cats)}
    n = len(pairs)
    po = sum(1 for a, b in pairs if str(a) == str(b)) / n
    pa = defaultdict(int)
    pb = defaultdict(int)
    for a, b in pairs:
        pa[str(a)] += 1
        pb[str(b)] += 1
    pe = sum((pa[c] / n) * (pb[c] / n) for c in cats)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def validate(args) -> int:
    sdir = args.out / "samples"
    adir = args.out / "annotations"
    report = {}
    for dim in RUBRICS:
        sample_items = {}
        for bf in sorted((sdir / dim).glob("batch_*.jsonl")):
            for it in provenance.read_jsonl(bf):
                sample_items[it["id"]] = it
        ann = {}
        bad = []
        for af in sorted((adir / dim).glob("batch_*.jsonl")):
            for rec in provenance.read_jsonl(af):
                errs = _check_record(dim, rec)
                if errs:
                    bad.append({"id": rec.get("id"), "errors": errs,
                                "file": af.name})
                else:
                    ann[rec.get("id")] = rec
        missing = [i for i in sample_items if i not in ann]
        # agreement on repass
        repass = {r.get("id"): r
                  for r in provenance.read_jsonl(adir / dim / "repass.jsonl")}
        kappas = {}
        cat_fields = [f for f in RUBRICS[dim]["required"]
                      if f not in ("verifiable_claims", "correct_claims")]
        for f in cat_fields:
            pairs = [(ann[i].get(f), repass[i].get(f))
                     for i in repass if i in ann and f in repass[i]]
            k = _cohen_kappa(pairs)
            kappas[f] = (round(k, 3) if k is not None else None,
                         len(pairs))
        report[dim] = {
            "sampled": len(sample_items), "annotated": len(ann),
            "invalid": len(bad), "missing": len(missing),
            "coverage": round(len(ann) / len(sample_items), 4)
            if sample_items else None,
            "kappa": kappas,
        }
        if bad:
            provenance.write_jsonl(adir / dim / "invalid_records.jsonl", bad)
        if missing:
            provenance.write_jsonl(
                adir / dim / "missing_ids.jsonl",
                [{"id": i} for i in missing])
    (args.out / "annotations" / "agreement.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    for dim, r in report.items():
        print(f"[validate] {dim}: {r['annotated']}/{r['sampled']} annotated, "
              f"{r['invalid']} invalid, kappa={r['kappa']}")
    return 0
