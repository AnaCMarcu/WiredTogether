"""Provenance-carrying IO helpers: gzip JSONL, item hashes, manifest."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


def write_jsonl(path: Path, records) -> int:
    """Write records to .jsonl or .jsonl.gz (by suffix). Returns count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    out = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def item_hash(obj) -> str:
    """Stable short id for a sample item (used for annotation resume/joins)."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def fingerprint_run(run) -> dict:
    """Cheap input fingerprint: size+mtime of the files a stage reads."""
    fp = {}
    candidates = [run.path / "final_metrics.json", run.path / "hebbian_snapshots.jsonl",
                  run.path / "config.json"]
    candidates += list(run.llm_logs.values())
    for ep in run.episode_dirs:
        for name in ("messages.jsonl", "step_log.csv", "event_log.jsonl",
                     "episode_summary.json"):
            candidates.append(ep / name)
    for p in candidates:
        try:
            st = p.stat()
            fp[str(p.relative_to(run.path))] = [st.st_size, int(st.st_mtime)]
        except OSError:
            continue
    return fp


def load_manifest(out_root: Path) -> dict:
    p = out_root / "manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_manifest(out_root: Path, manifest: dict) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8"
    )


def needs_update(manifest: dict, run, parser_version: str) -> bool:
    entry = manifest.get(run.run_id)
    if not entry:
        return True
    if entry.get("parser_version") != parser_version:
        return True
    return entry.get("fingerprint") != fingerprint_run(run)
