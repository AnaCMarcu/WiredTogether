"""Chamber-state readers for the action prompt.

The Lua mod writes the live puzzle state of each chamber (door locks, anvil
HP and who is punching it, cell/switch status) to flat files in the world
directory. These readers turn those files into the one-line "Chamber state"
block the agent sees each step — the authoritative source the prompt tells
the agent to trust over its own frame.
"""

from __future__ import annotations

import os

from mindforge.agent_modules.chamber_facts import ch4_zombie_count


class ChamberStateMixin:
    """Chamber-state half of :class:`~mindforge.custom_environment_craftium.CraftiumEnvironmentInterface`."""

    def get_chamber_state(self, agentId: int) -> str:
        """Return a one-line LLM-readable summary of the current chamber's
        puzzle state for the given agent. Used to populate the
        ``{chamber_state}`` placeholder in the action-selection prompt so
        the policy can reason about hidden state (anvil HP, who's punching
        what, etc.) instead of inferring it from the visual frame.

        Returns an empty string when there's no chamber-specific state to
        report (e.g., agent is in Ch1 where everything is visible).
        """
        chamber = self.get_chamber(agentId)
        agent_name = f"agent_{agentId}"

        prefix = ""
        warning = self._invalid_action_warning.pop(agent_name, "")
        if warning:
            prefix = warning + " "

        if chamber == "ch1":
            return prefix + self._read_ch1_door_state()
        if chamber == "ch2":
            # Anvil HP + active punchers + Door 2 status.
            parts = [self._read_ch2_anvil_state(), self._read_ch2_door_state()]
            return prefix + " | ".join(p for p in parts if p)
        if chamber == "ch3":
            return prefix + self._read_ch3_state(agentId)
        if chamber == "ch4":
            return prefix + self._read_ch4_door_state()
        return prefix or ""

    def _read_ch1_door_state(self) -> str:
        """Return a one-line Door 1 status string for agents still in Ch1.
        Without this, an agent in Ch1 has no signal that Door 1 has
        unlocked beyond the visual delta (a red bedrock-textured block
        becomes air at the same coords) — the LLM consistently misses it.
        Lua writes {world_path}/door1_state.txt when open_door1() fires."""
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return ""
        if os.path.exists(os.path.join(world_path, "door1_state.txt")):
            return ("Door 1: OPEN — walk north (positive Z) to the opening "
                    "in the north wall and MoveForward through into Chamber 2.")
        return ("Door 1: LOCKED — complete any of M2/M3/M4/M5/M6/M7 "
                "(dig 3 blocks, pick up 3 items, dig 5 wood, kill an "
                "animal, kill 2 animals, or dig 3 stone) to unlock it "
                "for the whole team.")

    def _read_ch2_anvil_state(self) -> str:
        """Parse {world_path}/anvils.txt (written by player_state.lua) into
        a compact human-readable line, including a per-anvil HP delta
        over the last 3 reads.

        The delta is the key closed-loop signal for the LLM: after a
        punch, the agent reads chamber_state on the next prompt — if Δhp
        moved up, its coordination is working; if Δhp is 0, only ONE
        agent is punching and they need a teammate.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return ""
        path = os.path.join(world_path, "anvils.txt")
        try:
            with open(path, "r") as f:
                raw = f.read()
        except (FileNotFoundError, OSError):
            return ""

        from collections import deque

        parts = []
        for line in raw.strip().splitlines():
            fields = line.split("|")
            if len(fields) < 3:
                continue
            kind, hp_str, names = fields[0], fields[1], fields[2]
            try:
                hp = int(hp_str)
            except (TypeError, ValueError):
                hp = None

            delta_str = ""
            if hp is not None:
                hist = self._anvil_hp_history.setdefault(
                    kind, deque(maxlen=4)
                )
                old = hist[0] if hist else hp
                hist.append(hp)
                delta = hp - old
                if delta > 0:
                    delta_str = f" Δhp_last3={delta:+d} (coop working — keep punching)"
                elif delta < 0:
                    delta_str = f" Δhp_last3={delta:+d} (decay — at least 2 agents must punch within ~1s)"
                else:
                    delta_str = " Δhp_last3=0 (no progress — need ≥2 punchers on this anvil at the same time)"

            puncher_names = [n.strip() for n in names.split(",") if n.strip()]
            if puncher_names:
                who = " punchers: " + ", ".join(puncher_names)
            else:
                who = " (idle)"

            parts.append(f"{kind} anvil={hp_str} hp{delta_str}{who}")

        if not parts:
            return ""
        return "Anvils — " + "; ".join(parts)

    def _door_state_file_exists(self, filename: str) -> bool:
        """True iff `{world_path}/<filename>` exists. Used as a presence-
        based flag for door-open events (Lua writes the file on open,
        clear_state_files() deletes it at episode reset).
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return False
        return os.path.exists(os.path.join(world_path, filename))

    def _read_ch2_door_state(self) -> str:
        """Door 2 (Ch2→Ch3) status line. Door 2 opens 20 steps after BOTH
        anvils have been broken (see tick_door2 in doors.lua)."""
        if self._door_state_file_exists("door2_state.txt"):
            return ("Door 2: OPEN — walk north (positive Z) through Door 2 "
                    "to be teleported into your Chamber 3 isolation cell.")
        return ("Door 2: LOCKED — break BOTH purple anvils (M8 + M9) to "
                "open it. Each anvil needs at least 2 agents punching it "
                "within ~1 second to break (solo dig makes zero progress).")

    def _read_ch3_state(self, agentId: int) -> str:
        """Ch3 isolation-cell + communal status.

        Tells the agent both (a) whether their own cell door is OPEN (so
        they can walk out) and (b) whether Door 3 (communal → Ch4) is
        open. Switch rotational mapping: switch in cell i opens the door
        of cell (i+1) mod N; so agent_i's cell is unlocked by the agent
        whose cell index is (i-1) mod N.
        """
        try:
            world_path = self._get_world_path()
        except AttributeError:
            return ""

        # Read the cell-doors file produced by open_cell_door().
        open_cells = set()
        cell_doors_path = os.path.join(world_path, "cell_doors_state.txt")
        try:
            with open(cell_doors_path, "r") as f:
                for line in f.read().strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    head = line.split(":", 1)[0].strip()
                    try:
                        open_cells.add(int(head))
                    except ValueError:
                        continue
        except (FileNotFoundError, OSError):
            pass

        my_cell_open = agentId in open_cells
        cell_line = (
            "Your cell door (cell %d): OPEN — walk north out of your cell "
            "into the communal room." % agentId
            if my_cell_open
            else "Your cell door (cell %d): LOCKED — a teammate must press "
                 "their switch to free you (rotational wiring: switch in "
                 "cell %d opens your door)." % (agentId, (agentId - 1) % self.num_agents)
        )

        door3_line = (
            "Door 3 (communal → Ch4): OPEN — walk north to enter Chamber 4."
            if self._door_state_file_exists("door3_state.txt")
            else "Door 3 (communal → Ch4): LOCKED — all %d agents must be "
                 "in the communal room together to open it."
                 % self.num_agents
        )
        return cell_line + " | " + door3_line

    def _read_ch4_door_state(self) -> str:
        """Door 4 (Ch4→Ch5) status line."""
        if self._door_state_file_exists("door4_state.txt"):
            return ("Door 4: OPEN — walk north into Chamber 5 to fight "
                    "the boss zombie.")
        # Zombie count mirrors what the Lua server actually spawns
        # (min(FC_CH4_MOB_COUNT or num_agents, 6)); previously hardcoded
        # "3", which contradicted chamber_facts' ROOM FACTS at N != 3.
        return ("Door 4: LOCKED — kill all %d zombies in Chamber 4 to open "
                "it (use Dig while facing each zombie; the diamond sword "
                "from Ch2 makes this much faster)."
                % ch4_zombie_count(self.num_agents))
