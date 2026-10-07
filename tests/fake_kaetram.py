"""In-memory stand-in for AgentWorld's ``KaetramGameTools`` + server.

Duck-types the tool methods the harness calls, with result strings in the
same format as the real ones, over a tiny shared world: inventories, a
stick/arrow fletching chain, oak trees, one rat, HP. Lets the executor,
event extraction, progress reward and scheduler run end to end on CPU.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


class FakeWorld:
    def __init__(self):
        self.lock = threading.Lock()
        self.players: Dict[str, Dict[str, Any]] = {}
        self.mobs: Dict[str, Dict[str, Any]] = {
            "m-rat-1": {"instance": "m-rat-1", "name": "Rat", "level": 1, "x": 395, "y": 3,
                        "hitPoints": 6, "maxHitPoints": 6, "aggressive": False},
        }
        self.trees = [{"instance": "t-oak-1", "name": "Oak Tree", "x": 389, "y": 4}]
        self.chat: List[Dict[str, Any]] = []
        self.calls: List[tuple] = []
        self.action_delay: Dict[str, float] = {}

    def player(self, name: str) -> Dict[str, Any]:
        return self.players.setdefault(name, {
            "x": 250, "y": 180, "hp": 69, "maxhp": 69, "items": {}, "skills": {},
        })


class FakeGameTools:
    def __init__(self, world: FakeWorld, base_url: str = "http://fake"):
        self.world = world
        self.base_url = base_url
        self.token: Optional[str] = None
        self.username: Optional[str] = None
        self._last_observation_data: Optional[Dict[str, Any]] = None

    # ── session ────────────────────────────────────────────────────────────
    def _log(self, *call):
        self.world.calls.append((self.username, *call))

    def login_character(self, a):
        self._log("login", a["username"])
        if a["username"] in self.world.players:
            self.token, self.username = f"tok-{a['username']}", a["username"]
            return f"Character {a['username']} logged in successfully. Token obtained."
        return "Failed to login: no such player"

    def create_character(self, a):
        self._log("create", a["username"])
        self.world.player(a["username"])
        self.token, self.username = f"tok-{a['username']}", a["username"]
        return f"Character {a['username']} created successfully. Token obtained."

    def _make_request(self, method, endpoint, data=None, params=None):
        self._log("req", endpoint)
        p = self.world.player(self.username)
        if endpoint == "/ai/observe":
            return {"status": "success", "location": {"x": p["x"], "y": p["y"]}}
        if endpoint == "/ai/setInventory" and data and data.get("clearFirst"):
            p["items"] = {}
        return {"status": "success"}

    def clear_equipment(self):
        self._log("clear_equipment")
        return "Successfully cleared all equipment"

    def set_combat_level(self, a):
        return f"Successfully set all combat skills to level {a['level']}. Total combat level: 3"

    def set_individual_skill_level(self, a):
        self.world.player(self.username)["skills"][a["skill"]] = a["level"]
        return f"Successfully set {a['skill']} to level {a['level']}"

    def teleport_character(self, a):
        p = self.world.player(self.username)
        p["x"], p["y"] = a["x"], a["y"]
        return f"Character teleported to ({a['x']}, {a['y']})"

    def restore_hp_mp(self):
        p = self.world.player(self.username)
        p["hp"] = p["maxhp"]
        return "Successfully restored HP to 69 and MP to 44"

    def give_and_equip_item(self, a):
        return f"Successfully equipped {a['count']}x {a['itemKey']}"

    def set_inventory(self, a):
        p = self.world.player(self.username)
        for it in a["items"]:
            p["items"][it["key"]] = p["items"].get(it["key"], 0) + it["count"]
        return "Successfully set inventory"

    # ── tools ──────────────────────────────────────────────────────────────
    def _delay(self, name):
        d = self.world.action_delay.get(name, 0.0)
        if d:
            time.sleep(d)

    def move_character(self, a):
        self._delay("move_character")
        p = self.world.player(self.username)
        p["x"], p["y"] = a["x"], a["y"]
        return f"Moved to ({a['x']}, {a['y']})"

    def harvest_resource(self, a):
        self._delay("harvest_resource")
        with self.world.lock:
            items = self.world.player(self.username)["items"]
            items["logs"] = items.get("logs", 0) + 1
        return "Successfully harvested 1x logs from Oak Tree"

    def craft_item(self, a):
        self._delay("craft_item")
        with self.world.lock:
            items = self.world.player(self.username)["items"]
            if a["itemKey"] == "stick" and items.get("logs", 0) >= 1:
                items["logs"] -= 1
                items["stick"] = items.get("stick", 0) + 4
                return "Successfully crafted 4x Stick!"
            if (a["itemKey"] == "arrow" and items.get("stick", 0) >= 10
                    and items.get("feather", 0) >= 10):
                items["stick"] -= 10
                items["feather"] -= 10
                items["arrow"] = items.get("arrow", 0) + 10
                return "Successfully crafted 10x Arrow!"
        return f"Failed to craft {a['itemKey']}: missing materials"

    def attack_entity(self, a):
        self._delay("attack_entity")
        with self.world.lock:
            mob = self.world.mobs.get(a["targetInstance"])
            if mob is None:
                return f"Error: Target mob {a['targetInstance']} not found in current environment."
            mob["hitPoints"] = max(0, mob["hitPoints"] - 3)
            if mob["hitPoints"] == 0:
                return f"🏆 VICTORY: Defeated {mob['name']} (Level {mob['level']})"
        return f"Combat with {mob['name']} continues"

    def transfer_items(self, a):
        self._delay("transfer_items")
        with self.world.lock:
            src = self.world.player(self.username)["items"]
            have = src.get(a["itemKey"], 0)
            if have < a["count"]:
                return f"Error: Not enough {a['itemKey']} in inventory. Have {have}, need {a['count']}."
            if a["targetPlayer"] not in self.world.players:
                return "Error: Transfer failed - no such player. Inventory restored to original state."
            src[a["itemKey"]] -= a["count"]
            dst = self.world.players[a["targetPlayer"]]["items"]
            dst[a["itemKey"]] = dst.get(a["itemKey"], 0) + a["count"]
        return (f"✅ Transfer completed: {a['count']}x {a['itemKey']} transferred from "
                f"{self.username} to {a['targetPlayer']} (method: setInventory)")

    def chat(self, a):
        self.world.chat.append({"from": self.username, "text": a["message"],
                                "global": a.get("global", True)})
        return f"Global chat message sent: {a['message']}"

    def sleep(self, a):
        return f"Slept for {a.get('seconds', 1)} second(s)."

    def complete(self, a):
        return f"TASK_COMPLETE: {a.get('response', '')}"

    def enter_portal(self, a):
        return "Entered portal"

    def equip_item(self, a):
        return f"Successfully equipped item {a['index']}"

    def observe_environment(self, a):
        p = self.world.player(self.username)
        r = a.get("radius", 64)
        with self.world.lock:
            others = [{"name": n, "x": q["x"], "y": q["y"],
                       "distanceFrom": abs(q["x"] - p["x"]) + abs(q["y"] - p["y"])}
                      for n, q in self.world.players.items() if n != self.username]
            mobs = [dict(m, distanceFrom=abs(m["x"] - p["x"]) + abs(m["y"] - p["y"]))
                    for m in self.world.mobs.values()]
            self._last_observation_data = {
                "status": "success",
                "location": {"x": p["x"], "y": p["y"]},
                "playerStatus": {"level": 3, "experience": 0, "hitPoints": p["hp"],
                                 "maxHitPoints": p["maxhp"], "mana": 44, "maxMana": 44},
                "inventory": {"items": [{"key": k, "count": c, "name": k}
                                        for k, c in p["items"].items() if c > 0]},
                "players": [o for o in others if o["distanceFrom"] <= r],
                "mobs": [m for m in mobs if m["distanceFrom"] <= r],
                "trees": [t for t in self.world.trees
                          if abs(t["x"] - p["x"]) + abs(t["y"] - p["y"]) <= r],
            }
        return f"Environment observation (radius {r}): ..."

    def get_last_observation_data(self):
        return self._last_observation_data
