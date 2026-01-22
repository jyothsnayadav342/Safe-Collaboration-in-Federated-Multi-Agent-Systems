import os
import time
from agents.agent_base import LLM_Agent
from safety_layer.safety_monitor import SafetyMonitor
from utils import load_original_text, aggregate_safe_summaries
from communication.message_protocol import create_message, transmit_message

os.makedirs("logs", exist_ok=True)

# Setup
models = ["facebook/bart-large-cnn", "t5-base", "google/pegasus-xsum", "facebook/bart-base", "google/flan-t5-base"]
agents = [LLM_Agent(i, model=models[i]) for i in range(5)]
monitor = SafetyMonitor(config_path="config.json")

# Round 1 
global_text = load_original_text()
round_num = 1
results_round1 = []

print("ROUND 1 - Running 5 agents one by one...")
for agent in agents:
    print(f"  → Agent {agent.id} generating summary...")
    output = agent.generate_summary(global_text)
    msg = create_message(agent.id, output["summary"], round_num, metadata=output["metadata"])
    transmit_message(msg)
    
    if output["summary"]:
        inspected = monitor.inspect(msg, source_text=global_text)
    else:
        inspected = {"safety_report": {"summary_decision": "error", "reject": True}}
    
    results_round1.append(inspected)
    time.sleep(1) 

# Save round-1 safety results
with open("logs/safety_results.json", "w", encoding="utf-8") as f:
    import json
    json.dump(results_round1, f, indent=4)

# Aggregate safe summaries for Round 2 
safe_texts = [r["content"] for r in results_round1 if r["safety_report"]["summary_decision"] == "safe"]
next_global_text = " ".join(safe_texts) + " AI benefits society by enhancing productivity, enabling smarter decision-making, improving healthcare outcomes, fostering innovation, assisting in education, and supporting environmental monitoring and sustainability efforts."

# Round 2 
round_num = 2
results_round2 = []

print("\nROUND 2 - Running 5 agents again one by one...")
for agent in agents:
    print(f"  → Agent {agent.id} generating summary...")
    output = agent.generate_summary(next_global_text)
    msg = create_message(agent.id, output["summary"], round_num, metadata=output["metadata"])
    transmit_message(msg)
    
    inspected = monitor.inspect(msg, source_text=next_global_text)
    results_round2.append(inspected)
    time.sleep(1)

# Save final results
with open("logs/final_safe_summaries.json", "w", encoding="utf-8") as f:
    import json
    json.dump(results_round2, f, indent=4)

print("\nDone! Check logs/safety_results.json and logs/final_safe_summaries.json")
print("   Run `python federated_aggregator.py` to see the pretty tables.")