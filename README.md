# WiredTogether

Multi-agent reinforcement learning environment built on [Craftium](https://github.com/minosvasilias/craftium) - a Minetest/Luanti-based RL platform.

## 🎯 Quick Start

**Want to test immediately?** → See [QUICKSTART.md](QUICKSTART.md)

**Setting up from scratch?** → See [SETUP_WSL.md](SETUP_WSL.md)

**Check current status** → See [STATUS.md](STATUS.md)

## 📋 Overview

This project provides a **multi-agent OpenWorld environment** using PettingZoo's ParallelEnv API:

- ✅ **4 agents** (configurable) in VoxeLibre open world
- ✅ **Standard RL interface** - Works with any MARL library
- ✅ **Auto-cleanup** - Automatically archives and cleans temp directories
- ✅ **Role-based variant** (optional) - Engineer/Hunter/Guardian with reward shaping

### What's Implemented

#### 1. Basic Multi-Agent OpenWorld
**File**: [src/envs/openworld_parallel.py](src/envs/openworld_parallel.py)

Standard multi-agent OpenWorld environment:
- 4 agents exploring VoxeLibre terrain
- 17 discrete actions per agent (move, dig, place, etc.)
- RGB observations (320x180 per agent)
- Shared rewards from environment milestones

#### 2. Role-Based Multi-Agent (Optional)
**File**: [src/envs/openworld_roles_mvp.py](src/envs/openworld_roles_mvp.py)

Adds role mechanics on top of basic multi-agent:
- **Engineer**: Tools milestones (1.0x), Hunt/Defend (0.5x)
- **Hunter**: Hunt rewards (1.0x), Tools/Defend (0.5x)
- **Guardian**: Defend rewards (1.0x), Tools/Hunt (0.5x)
- Supports multiple agents per role

#### 3. Auto-Cleanup System
**Files**: [src/envs/auto_cleanup_env.py](src/envs/auto_cleanup_env.py), [src/envs/luanti_cleanup.py](src/envs/luanti_cleanup.py)

Automatically manages temporary directories:
- Archives `luanti-run-*` directories to `logs/luanti_runs/`
- Saves debug.txt and log files
- Deletes temp directories on `env.close()`

## 🚀 Usage

### Run Basic Test

```bash
# In WSL with conda environment activated
python scripts/test_openworld.py
```

This will:
1. Create a 4-agent OpenWorld environment
2. Run 100 steps with random actions
3. Display per-agent rewards
4. Clean up temp directories

### Use in Your Code

```python
from src.envs import AutoCleanupOpenWorld

# Create environment
env = AutoCleanupOpenWorld(
    num_agents=4,
    obs_width=320,
    obs_height=180,
    max_steps=10_000,
    auto_cleanup=True,  # Automatic cleanup on close
)

# Reset
observations, infos = env.reset()

# Step
for _ in range(100):
    actions = {agent: env.action_space(agent).sample()
               for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)

# Close (triggers auto-cleanup)
env.close()
```

### Use Role-Based Variant

```python
from src.envs.openworld_roles_mvp import RoleBasedOpenWorld

# Create with custom roles
env = RoleBasedOpenWorld(
    num_agents=6,
    roles=["engineer", "engineer", "hunter", "hunter", "guardian", "guardian"],
)

# Or use default cycling through roles
env = RoleBasedOpenWorld(num_agents=6)  # Auto-cycles through E/H/G
```

## 📦 Environment Details

### Action Space
Discrete(17) per agent:
- **Movement**: forward, backward, left, right
- **Special**: jump, sneak
- **Interaction**: dig, place
- **Inventory**: slot_1 through slot_5
- **Camera**: mouse x+/x-, y+/y-

### Observation Space
Box(0, 255, (180, 320, 3), uint8) - RGB image per agent

### Rewards
**Basic OpenWorld**:
- Tools milestones: 128, 256, 1024, 2048
- Hunt: damage × 0.5
- Defend: damage × 1.0

**Role-Based OpenWorld**:
- In-track actions: Full reward (1.0×)
- Out-of-track actions: Half reward (0.5×)

## 🔧 Setup

### Prerequisites
- **WSL** (Windows Subsystem for Linux) or Linux
- **Conda** (Miniconda or Anaconda)
- **Craftium built** with Luanti binary at `craftium/bin/luanti`

**Note**: Craftium cannot be built on Windows natively due to Unix dependencies. Use WSL.

### Installation

```bash
# 1. Open WSL
wsl

# 2. Navigate to project
cd /mnt/c/Users/marcu/OneDrive/Documente/GitHub/WiredTogether

# 3. Create conda environment
conda env create -f environment.yml

# 4. Activate environment
conda activate wiredtogether

# 5. Install Craftium
pip install -e ./craftium

# 6. Install project
pip install -e .
```

**Full setup guide**: [SETUP_WSL.md](SETUP_WSL.md)

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get running in 5 minutes
- **[SETUP_WSL.md](SETUP_WSL.md)** - Complete setup guide with troubleshooting
- **[STATUS.md](STATUS.md)** - Current project status and next steps
- **[CRAFTIUM_SETUP.md](CRAFTIUM_SETUP.md)** - Craftium-specific setup notes (if exists)

## 🗂️ Project Structure

```
WiredTogether/
├── src/envs/               # Environment implementations
│   ├── openworld_parallel.py      # Basic multi-agent
│   ├── openworld_roles_mvp.py     # Role-based variant
│   ├── auto_cleanup_env.py        # Auto-cleanup wrappers
│   └── luanti_cleanup.py          # Cleanup utilities
├── scripts/                # Test and utility scripts
│   ├── test_openworld.py          # Basic test
│   └── cleanup_luanti_runs.py     # Manual cleanup
├── configs/                # Hydra configuration files
│   └── env/
│       ├── openworld.yaml         # Basic config
│       └── openworld_roles_mvp.yaml  # Role-based config
├── craftium/               # Craftium submodule (built)
├── environment.yml         # Conda environment specification
└── logs/                   # Generated logs and archives
    └── luanti_runs/        # Archived run directories
```

## 🐛 Troubleshooting

### Import Error: "cannot import name 'MarlCraftiumEnv'"
**Solution**: Make sure you're in WSL with conda environment activated, not Git Bash.

### FileNotFoundError: "./bin/luanti"
**Solution**: Craftium needs to be built. Check `craftium/bin/luanti` exists.

### Hanging Processes
**Solution**: Kill manually with `pkill -9 luanti`

**More troubleshooting**: See [SETUP_WSL.md](SETUP_WSL.md#common-issues)

## 🎓 Next Steps

After confirming tests work:

1. **Training pipeline** - Implement PPO/A2C with parameter sharing
2. **Logging** - Add W&B/TensorBoard tracking
3. **Video recording** - Save episode videos for analysis
4. **Hyperparameter tuning** - Optimize learning

## 📄 License

MIT (or whatever license you choose)

## 🙏 Acknowledgments

Built on [Craftium](https://github.com/minosvasilias/craftium) by minosvasilias