import socket
import time

print(f"Hello SLURM from {socket.gethostname()}!", flush=True)
time.sleep(60)
print("Done.", flush=True)
