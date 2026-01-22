# main.py
import os
import json
from tabulate import tabulate
from agents.agent_base import LLM_Agent
from communication.message_protocol import create_message, transmit_message
from safety_layer.safety_monitor import SafetyMonitor

# Setup directories
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load input text for summarisation
with open("data/input_text.txt", "r", encoding="utf-8") as f:
    input_text = f.read()
    
# Define model pool for 5 agents
models = [
    "facebook/bart-large-cnn",
    "t5-base",
    "google/pegasus-xsum",
    "google/flan-t5-base",
    "google/flan-t5-base"
]

# Initialise LLM agents and safety monitor
agents = [LLM_Agent(i, model=models[i]) for i in range(5)]
monitor = SafetyMonitor(config_path="config.json")

# Store safety inspection results
results = []

# Round Simulation
round_num = 1
for agent in agents:
    # Generate summary from agent
    output = agent.generate_summary(input_text)

    # Create and transmit message
    msg = create_message(agent.id, output["summary"], round_num, metadata=output["metadata"])
    transmit_message(msg)

    # Run safety inspection if summary exists
    if output["summary"]:
        inspected = monitor.inspect(msg, source_text=input_text)
        results.append(inspected)
    else:
        # Handle empty summary / error
        results.append({
            "sender_id": agent.id,
            "round": round_num,
            "content": None,
            "metadata": output["metadata"],
            "safety_report": {
                "summary_decision": "error",
                "reject": True,
                "similarity_score": 0.0,
                "semantic_distance": 1.0,
                "hallucinated": False,
                "notes": [
                    {"toxicity_score": 0.0}, 
                    {"policy_violation_detected": False}, 
                    {"hallucination_detected": False}
                ]
            }
        })

# Save results to JSON
with open("logs/safety_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

# Display results as a table
table_data = []
for s in results:
    r = s.get("safety_report", {})
    meta = s.get("metadata", {})

    table_data.append([
        s["sender_id"],                                           # Agent ID
        meta.get("bias_simulated", False),                        # Simulated Toxicity
        r.get("notes", [{}])[0].get("toxicity_score", 0.0),       # Toxicity Score
        meta.get("policy_violation_simulated", False),            # Simulated Policy Violation
        r.get("notes", [{}])[2].get("hallucination_detected", False),  # Hallucination Detection
        r.get("summary_decision", "N/A")                          # Final Safety Decision
    ])

headers = [
    "Agent ID",
    "Toxicity Simulated",
    "Toxicity",
    "Policy Violation Simulated",
    "Hallucination Simulated",
    "Decision"
]

print("\nSAFETY MONITORING RESULTS\n")
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
print("\nResults saved to logs/safety_results.json\n")




