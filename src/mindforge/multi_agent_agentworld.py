"""Entry point: MindForge (+ Hebbian) agents in AgentWorld.

Thin wrapper so runs launch the same way as the WIRE runner
(``python -m mindforge.multi_agent_agentworld ...``). Everything lives in
:mod:`agentworld.cli`; the WIRE runner (multi_agent_craftium.py) is untouched.
"""

import sys
from pathlib import Path

# MindForge's modules use runtime-style absolute imports (agent_modules.x),
# like multi_agent_craftium does in production.
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agentworld.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
