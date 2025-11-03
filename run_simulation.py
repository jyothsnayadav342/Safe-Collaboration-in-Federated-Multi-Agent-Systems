# run_simulation.py
import subprocess
import time
import sys

NUM_CLIENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

# Start server in background
server_proc = subprocess.Popen([sys.executable, "server.py"])
time.sleep(2)  # wait for server to come up

client_procs = []
for i in range(NUM_CLIENTS):
    p = subprocess.Popen([sys.executable, "client.py", str(i)])
    client_procs.append(p)
    time.sleep(0.3)

# wait for clients to finish (they will stop after server completes rounds)
for p in client_procs:
    p.wait()

# terminate server if still running
server_proc.terminate()
print("Simulation completed.")
