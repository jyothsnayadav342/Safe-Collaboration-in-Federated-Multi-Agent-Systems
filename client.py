# client.py
import flwr as fl
import os
import sys
import json
from agents.agent_base import LLM_Agent

# Available LLM models for clients
MODELS = [
    "facebook/bart-large-cnn",
    "t5-base",
    "google/pegasus-xsum",
    "facebook/bart-base",
    "google/flan-t5-base"
]

# Determine which model this client will use based on passed argument
agent_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
model_name = MODELS[agent_id]

# Debug information printed to client terminal at startup
print(f"\n[CLIENT STARTING] → Agent ID: {agent_id} | Model: {model_name} | PID: {os.getpid()}")
print(f"   → This agent is {'MALICIOUS' if agent_id >= 3 else 'BENIGN'} (3 & 4 are malicious)")

# Initialise the LLM agent wrapper
agent = LLM_Agent(agent_id=agent_id, model=model_name)


class SummaryClient(fl.client.NumPyClient):
    """
    Federated client responsible for receiving text,
    generating a summary using the assigned LLM model,
    and returning the summary as metrics
    """

    def fit(self, parameters, config):
        """
        Receives global text and produces a summary using the LLM agent.
        """
        round_num = config.get("round", "?")
        print(f"[ROUND {round_num}] Agent {agent_id} received text (length: {len(config['global_text'])}) → generating summary...")
        
        # Generate summary with metadata 
        output = agent.generate_summary(config["global_text"])

        # Debug print to view output characteristics
        print(
            f"[ROUND {round_num}] Agent {agent_id} DONE → Summary length: {len(output['summary'])} | "
            f"Bias: {output['metadata']['bias_simulated']} | "
            f"Policy: {output['metadata']['policy_violation_simulated']}"
        )
        
        # No model updates so return empty parameter weights
        return [], 0, {"summary_json": json.dumps(output)}


# Start the Flower client
fl.client.start_client(
    server_address="localhost:8080",
    client=SummaryClient().to_client()
)
