# server.py
import flwr as fl
from flwr.server.server import Server
from flwr.common import FitRes
import logging

logging.basicConfig(level=logging.INFO)

class Aggregator(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()

    def aggregate_fit(self, rnd, results, failures):
        # results: list of tuples (client_proxy, FitRes)
        messages = []
        for _, fit_res in results:
            # metrics might be empty for actual FL; we used metrics to carry messages
            try:
                msg = fit_res.metrics.get("message")
                if msg:
                    messages.append(msg)
            except Exception:
                pass

        print(f"\n--- Round {rnd} --- Collected {len(messages)} messages ---")
        for i, m in enumerate(messages):
            print(f"[{i}] {m[:200]}")
        # The server can pass a prompt via config to clients in next round
        # return parameters (None) and config for next round
        return None, {"prompt": "Server: please combine and summarize the received insights."}

def start_server(server_address: str = "127.0.0.1:8080", num_rounds: int = 3):
    strategy = Aggregator()
    print(f"Starting Flower server at {server_address} for {num_rounds} rounds...")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()
