"""Post-hoc communication metrics for the Five Chambers env.

Reads `runs/<run_id>/episodes/ep_*/messages.jsonl` and `step_log.csv`,
returns a dict suitable for inclusion in `final_metrics.json["comm_metrics"]`.

Metrics implemented (canonical references):

* **Speaker Consistency / Positive Signaling** (Lowe 2019, Eccles 2019):
  per-agent message entropy `H(M_i)` over discretised message clusters,
  and mutual information `I(M_i; chamber_i)` to detect whether messages
  are situation-dependent.

* **Instantaneous Coordination** (Lowe 2019): per-pair
  `I(M_t^sender ; A_{t+1}^receiver)` — pointwise mutual information
  between sender's message and the receiver's next-step action.

* **Per-pair message-count matrix**: who-talks-to-whom, asymmetric N×N
  count of routed messages.

* **Tokens-per-milestone**: mean token count of the W messages preceding
  each milestone fire — a coarse efficiency proxy.

The first two require message clustering. We cluster on sentence-transformer
embeddings with a small K-means (K = 16 by default) so messages that mean
roughly the same thing land in the same bin. The ST instance is the same one
used by ChromaDB elsewhere in the project — load via `ST_MODEL_NAME`.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────
def _read_jsonl(path: Path) -> list:
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _read_step_csv(path: Path) -> list:
    """Step log written by EpisodeLogger as CSV; one row per (step, agent)."""
    import csv
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out


# ─────────────────────────────────────────────────────────────────────────
def _agent_idx(name) -> int:
    if isinstance(name, int):
        return name
    try:
        return int(str(name).split("_")[-1])
    except (ValueError, IndexError):
        return -1


def _cluster_messages(texts: list[str], k: int = 16) -> list[int]:
    """Map each message to a cluster id in [0, K). Falls back to length-bin
    hashing when sentence-transformers / sklearn aren't available — the same
    metric definitions still apply, just with cruder bins."""
    if not texts:
        return []
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        model_name = os.environ.get("ST_MODEL_NAME", "all-MiniLM-L6-v2")
        st = SentenceTransformer(model_name)
        emb = st.encode(texts, show_progress_bar=False)
        n_clusters = min(k, len(texts))
        km = KMeans(n_clusters=n_clusters, n_init=4, random_state=0).fit(emb)
        return [int(c) for c in km.labels_]
    except Exception:
        # Fallback: hash by (first-word, length-bucket).
        out = []
        for t in texts:
            tw = t.split()
            head = (tw[0].lower() if tw else "") + f":{len(tw)//5}"
            out.append(hash(head) % k)
        return out


def _entropy(counts) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)


def _mutual_information(pairs) -> float:
    """MI of a list of (x, y) pairs."""
    if not pairs:
        return 0.0
    n = len(pairs)
    px = Counter(x for x, _ in pairs)
    py = Counter(y for _, y in pairs)
    pxy = Counter(pairs)
    mi = 0.0
    for (x, y), c in pxy.items():
        p_xy = c / n
        p_x  = px[x] / n
        p_y  = py[y] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log(p_xy / (p_x * p_y))
    return mi


# ─────────────────────────────────────────────────────────────────────────
def compute_comm_metrics(run_root, num_agents: int) -> dict:
    """Walk episodes/ep_*/messages.jsonl + step_log.csv, return a dict.

    Returns
    -------
    dict with keys:
      pair_message_count    N×N int matrix   (sender, receiver)
      total_messages        int
      total_tokens          int
      tokens_per_milestone  float            (or None if no milestones)
      speaker_consistency   per-agent {entropy, mi_chamber}
      instantaneous_coord   per-pair float   (pmi(sender_msg, receiver_a_t+1))
      routing_breakdown     {model, hebbian_fallback, random_fallback}
    """
    root = Path(run_root)
    ep_dirs = sorted((root / "episodes").glob("ep_*"))

    all_messages = []     # all message records across episodes
    all_steps    = []     # all step rows across episodes
    for ed in ep_dirs:
        all_messages.extend(_read_jsonl(ed / "messages.jsonl"))
        all_steps.extend(_read_step_csv(ed / "step_log.csv"))

    # Per-agent action lookup keyed on (agent_idx, t) → action.
    action_by = {}
    for row in all_steps:
        try:
            t = int(row.get("step", -1))
            a = _agent_idx(row.get("agent_id"))
        except (TypeError, ValueError):
            continue
        action_by[(a, t)] = row.get("action", "")

    # ── Per-pair message-count matrix (asymmetric) ──
    pair = [[0] * num_agents for _ in range(num_agents)]
    routing = Counter()
    total_tokens = 0
    valid_msgs = 0
    for m in all_messages:
        si = _agent_idx(m.get("sender", -1))
        ri = _agent_idx(m.get("receiver", -1))
        if 0 <= si < num_agents and 0 <= ri < num_agents and si != ri:
            pair[si][ri] += 1
        routing[m.get("routing", "model")] += 1
        total_tokens += int(m.get("tokens", 0))
        if m.get("valid", False):
            valid_msgs += 1

    # ── Speaker Consistency: H(M_i), I(M_i; chamber_i) ──
    # Cluster messages globally so cluster ids are comparable across agents.
    texts = [m.get("text", "") for m in all_messages]
    clusters = _cluster_messages(texts) if texts else []
    speaker_consistency = {}
    for ai in range(num_agents):
        msgs_i = [(c, m.get("chamber") or "?")
                  for m, c in zip(all_messages, clusters)
                  if _agent_idx(m.get("sender", -1)) == ai]
        h = _entropy(Counter(c for c, _ in msgs_i))
        mi_chamber = _mutual_information(msgs_i)
        speaker_consistency[f"agent_{ai}"] = {
            "entropy":    h,
            "mi_chamber": mi_chamber,
            "num_msgs":   len(msgs_i),
        }

    # ── Instantaneous Coordination: per-pair I(M_t^sender ; a_{t+1}^receiver) ──
    inst_coord = {}
    for si in range(num_agents):
        for ri in range(num_agents):
            if si == ri:
                continue
            pairs = []
            for m, c in zip(all_messages, clusters):
                if _agent_idx(m.get("sender")) != si:
                    continue
                if _agent_idx(m.get("receiver")) != ri:
                    continue
                t = m.get("t")
                if t is None:
                    continue
                a_next = action_by.get((ri, int(t) + 1))
                if a_next is None:
                    continue
                pairs.append((c, a_next))
            inst_coord[f"{si}->{ri}"] = _mutual_information(pairs)

    # ── Tokens per milestone unlock (one number per run) ──
    tokens_per_milestone = None
    n_milestones = 0
    for ed in ep_dirs:
        for ev in _read_jsonl(ed / "event_log.jsonl"):
            if ev.get("type") in ("milestone", "comm_milestone"):
                n_milestones += 1
    if n_milestones > 0:
        tokens_per_milestone = total_tokens / n_milestones

    return {
        "total_messages":      len(all_messages),
        "valid_messages":      valid_msgs,
        "total_tokens":        total_tokens,
        "tokens_per_milestone": tokens_per_milestone,
        "pair_message_count":  pair,
        "routing_breakdown":   dict(routing),
        "speaker_consistency": speaker_consistency,
        "instantaneous_coord": inst_coord,
    }
