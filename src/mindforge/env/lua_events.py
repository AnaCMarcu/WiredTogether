"""Lua -> Python event drains.

``craftium.reward()`` never reaches Python in multi-agent mode, so every
reward and event the world mod produces is appended to a JSONL file and
polled here: milestone fires, anvil co-op attempts and deaths. Each poller
keeps a byte offset so repeated calls never re-read a line, and the offsets
are reset per episode.
"""

from __future__ import annotations

import json
import os


class LuaEventsMixin:
    """Event-drain half of :class:`~mindforge.custom_environment_craftium.CraftiumEnvironmentInterface`."""

    def poll_milestone_events(self) -> list:
        """Read any new milestone events from milestone_events.jsonl since the last call.

        Returns a list of dicts, each with keys: step, milestone, contributors, reward.
        The byte-offset is advanced so repeated calls never re-read the same lines.
        If the file is smaller than the tracked offset (episode reset / file cleared),
        the offset is automatically reset to 0.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return []

        path = os.path.join(world_path, "milestone_events.jsonl")
        if not os.path.exists(path):
            self._milestone_file_offset = 0
            return []

        # Detect file truncation / deletion + recreation (episode reset).
        try:
            if os.path.getsize(path) < self._milestone_file_offset:
                self._milestone_file_offset = 0
        except OSError:
            return []

        new_events = []
        try:
            with open(path, "r") as f:
                f.seek(self._milestone_file_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        new_events.append(ev)
                        for raw_name in ev.get("contributors", []):
                            _s = str(raw_name).removeprefix("agent_").removeprefix("agent")
                            try:
                                _aid = int(_s)
                            except ValueError:
                                continue
                            agent_name = f"agent_{_aid}"
                            self._rewards[agent_name] = (
                                self._rewards.get(agent_name, 0.0) + ev.get("reward", 0)
                            )
                    except json.JSONDecodeError:
                        pass
                self._milestone_file_offset = f.tell()
        except OSError:
            pass

        return new_events

    def reset_milestone_offset(self):
        """Anchor the milestone event reader to the current EOF (call after env.reset())."""
        try:
            world_path = self._get_world_path()
        except AttributeError:
            self._milestone_file_offset = 0
            return
        path = os.path.join(world_path, "milestone_events.jsonl")
        try:
            self._milestone_file_offset = os.path.getsize(path)
        except OSError:
            self._milestone_file_offset = 0

    def poll_anvil_coop_events(self) -> list:
        """Read any new anvil-coop diagnostic events since the last call.

        Returns a list of dicts each with keys: step, anvil, row,
        n_active, active. NO reward attached — these are for post-hoc
        analysis of "did the team try to coordinate at the anvils?".
        The Lua side (anvil.lua globalstep) edge-triggers one event per
        anvil per ~ACTIVE_WINDOW ticks when ≥2 agents are punching.

        Mirrors poll_milestone_events()'s file-offset bookkeeping so
        repeated calls never re-read the same lines, and a smaller
        file size (episode reset / file cleared) auto-rewinds the
        offset to 0.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return []
        path = os.path.join(world_path, "anvil_coop_events.jsonl")
        if not os.path.exists(path):
            self._anvil_coop_file_offset = 0
            return []
        try:
            if os.path.getsize(path) < self._anvil_coop_file_offset:
                self._anvil_coop_file_offset = 0
        except OSError:
            return []
        new_events = []
        try:
            with open(path, "r") as f:
                f.seek(self._anvil_coop_file_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        new_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                self._anvil_coop_file_offset = f.tell()
        except OSError:
            pass
        return new_events

    def reset_anvil_coop_offset(self):
        """Anchor the anvil-coop event reader to current EOF (post env.reset())."""
        try:
            world_path = self._get_world_path()
        except AttributeError:
            self._anvil_coop_file_offset = 0
            return
        path = os.path.join(world_path, "anvil_coop_events.jsonl")
        try:
            self._anvil_coop_file_offset = os.path.getsize(path)
        except OSError:
            self._anvil_coop_file_offset = 0

    def poll_death_events(self) -> list:
        """Read any new death / would-die events since the last call.

        Returns a list of dicts each with keys: step, kind ("death" |
        "woulddie"), agent, chamber, reward (a NEGATIVE int: −10 for a
        forgiving-chamber would-have-died, −50 for a real Ch5 permadeath).
        deaths.lua writes these to death_events.jsonl because the server-side
        craftium.reward() it also calls does NOT reach env.step()'s reward
        channel in the multi-agent five-chambers context — so this JSONL is the
        authoritative reward source. The caller (multi_agent_craftium.py) drains
        each event's reward into step_rewards_raw before Hebbian diffusion /
        record_reward, so the penalty propagates into the graph and into
        cumulative_returns / episode_return.

        WOULD-DIE RATE LIMIT (once per agent per EPISODE): the Lua hpchange
        callback fires once per *damage event*, so a continuously attacked agent
        emits dozens of would-die lines — stacking −10s until the penalty swamps
        every positive reward. We cap it hard: each agent is charged the −10
        would-die AT MOST ONCE PER EPISODE. Subsequent would-die lines for an
        already-charged agent are dropped (not returned, not mirrored into
        _rewards). The charged-agent set (self._woulddie_charged) is cleared at
        episode reset via reset_death_offset(). Real Ch5 deaths ("death") are
        one-shot/terminal and are never capped. The file offset still advances
        past every line, so dropped duplicates are not re-read; Lua keeps its
        own full [WOULDDIE] log + would_die_count for diagnostics.

        Mirrors poll_milestone_events()/poll_anvil_coop_events() file-offset
        bookkeeping so repeated calls never re-read the same lines, and a
        smaller file size (episode reset / file cleared) auto-rewinds to 0.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return []
        path = os.path.join(world_path, "death_events.jsonl")
        if not os.path.exists(path):
            self._death_file_offset = 0
            return []
        try:
            if os.path.getsize(path) < self._death_file_offset:
                self._death_file_offset = 0
        except OSError:
            return []
        new_events = []
        try:
            with open(path, "r") as f:
                f.seek(self._death_file_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    _s = str(ev.get("agent", "")).removeprefix("agent_").removeprefix("agent")
                    try:
                        _aid = int(_s)
                    except ValueError:
                        _aid = None
                    if ev.get("kind") == "woulddie" and _aid is not None:
                        if _aid in self._woulddie_charged:
                            continue
                        self._woulddie_charged.add(_aid)
                    new_events.append(ev)
                    if _aid is not None:
                        agent_name = f"agent_{_aid}"
                        self._rewards[agent_name] = (
                            self._rewards.get(agent_name, 0.0) + ev.get("reward", 0)
                        )
                self._death_file_offset = f.tell()
        except OSError:
            pass
        return new_events

    def reset_death_offset(self):
        """Anchor the death-event reader to current EOF (post env.reset()) and
        clear the per-episode would-die charge set so each agent can be charged
        the −10 once again in the new episode."""
        self._woulddie_charged = set()
        try:
            world_path = self._get_world_path()
        except AttributeError:
            self._death_file_offset = 0
            return
        path = os.path.join(world_path, "death_events.jsonl")
        try:
            self._death_file_offset = os.path.getsize(path)
        except OSError:
            self._death_file_offset = 0

    def _get_server_log_path(self):
        """Resolve and cache the path to the server's stderr.txt."""
        if self._server_log_path is None:
            try:
                srv_run_dir = self.env.env.mt_server.run_dir
                self._server_log_path = os.path.join(os.path.abspath(srv_run_dir), "stderr.txt")
            except AttributeError:
                pass
        return self._server_log_path

    def tail_server_log(self, tags=None, print_lines=True):
        """Read any new lines from the server's stderr.txt since the last call.

        Only lines that start with one of the watched tags are returned.
        Calls after reset() automatically re-anchor to the current EOF so old
        log lines from previous runs don't flood the output.

        Args:
            tags: tuple of tag strings to filter on (default: _LOG_TAGS)
            print_lines: if True, print matching lines to stdout

        Returns:
            list[str] of matching new lines (without trailing newline)
        """
        path = self._get_server_log_path()
        if path is None or not os.path.exists(path):
            return []

        tags = tags or self._LOG_TAGS
        matched = []
        try:
            with open(path, "r", errors="replace") as f:
                f.seek(self._server_log_offset)
                for line in f:
                    stripped = line.rstrip()
                    if any(stripped.startswith(t) for t in tags):
                        matched.append(stripped)
                        if print_lines:
                            print(f"  [SRV] {stripped}")
                self._server_log_offset = f.tell()
        except OSError:
            pass
        return matched

    def reset_log_offset(self):
        """Anchor the log tailer to the current EOF (call after env.reset()).

        This prevents lines from a previous episode from appearing in the
        current one.
        """
        path = self._get_server_log_path()
        if path is None:
            self._server_log_offset = 0
            return
        try:
            self._server_log_offset = os.path.getsize(path)
        except OSError:
            self._server_log_offset = 0
