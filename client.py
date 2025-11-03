# client.py
import flwr as fl
import sys
import json
import os
from llm_agent import LLMWrapper

class LLMClient(fl.client.NumPyClient):
    def __init__(self, client_id: int):
        self.client_id = client_id
        self.agent = LLMWrapper()
        os.makedirs("logs", 
        exist_ok=True)

    def get_parameters(self, config):
        # Not using ML parameters in this simulation
        return []

    def fit(self, parameters, config):
        # The server passes a prompt via config (if available)
        prompt = config.get("prompt", f"Agent {self.client_id}: Summarize the local dataset insights.")
        # Generate response from LLM wrapper
        response = self.agent.generate_response(prompt)
        # Log communication
        record = {"client_id": self.client_id, "prompt": prompt, "response": response}
        with open(f"logs/client_{self.client_id}.log", "a") as f:
            f.write(json.dumps(record) + "\n")
        # Return (parameters, num_examples, metrics)
        # We return empty params and put message in metrics for server
        return [], 0, {"message": response}

    def evaluate(self, parameters, config):
        # Not used
        return 0.0, 1, {}

def start_client(server_address: str = "127.0.0.1:8080", client_id: int = 0):
    client = LLMClient(client_id)
    print(f"Starting client {client_id} -> connecting to {server_address}")
    fl.client.start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    # usage: python client.py <client_id> [server_address]
    client_id = int(sys.argv[1]) if len(sys.argv) >= 2 else 0
    server_address = sys.argv[2] if len(sys.argv) >= 3 else "127.0.0.1:8080"
    start_client(server_address=server_address, client_id=client_id)
