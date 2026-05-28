import socket
import sys

print(f"Host: {socket.gethostname()}")
print(f"Python: {sys.version.split()[0]}")
print()

import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  cuda built: {torch.version.cuda}")
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print(f"  matmul OK, result mean={y.mean().item():.4f}")

import gymnasium
print(f"gymnasium {gymnasium.__version__}")

import pettingzoo
print(f"pettingzoo {pettingzoo.__version__}")

import craftium
print(f"craftium from {craftium.__file__}")

import sentence_transformers
print(f"sentence_transformers {sentence_transformers.__version__}")

import transformers
print(f"transformers {transformers.__version__}")

import autogen_agentchat
print(f"autogen_agentchat imported")

print()
print("All imports OK.")
