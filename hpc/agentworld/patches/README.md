# Patches to AgentWorld

`fetch_agentworld.sh` applies every `*.patch` here, in name order, after
checking out the commit in `../AGENTWORLD_COMMIT`. None yet. Candidates from
the plan, only if P0 shows they are needed (each is a documented deviation
from the benchmark):

- `pause_world.patch` — `/ai/admin/pause` to stop the mob tick between rounds,
  if deaths turn out to be driven by thinking time;
- `spectate.patch` — a client-only `?spectate=<username>` camera for live
  capture, if a second web-client login cannot follow agents.
