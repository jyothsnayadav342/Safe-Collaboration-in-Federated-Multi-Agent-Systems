import json
from agents.agent_base import LLM_Agent
from safety_layer.safety_monitor import SafetyMonitor

# Load safety results from first round
with open("logs/safety_results.json", "r", encoding="utf-8") as f:
    safety_data = json.load(f)

# Initialise agents and safety monitor
models = ["facebook/bart-large-cnn", "t5-base", "google/pegasus-xsum", "facebook/bart-base", "google/flan-t5-base"]
agents = [LLM_Agent(i, model=models[i]) for i in range(5)]
monitor = SafetyMonitor(config_path="config.json")

# Filter safe summaries from previous round
safe_texts = [entry["content"] for entry in safety_data 
              if entry["safety_report"]["summary_decision"] == "safe"]

# Aggregator merges safe summaries and adds 50 words about AI benefits
ai_benefits = (" AI benefits society by enhancing productivity, enabling smarter decision-making, "
               "improving healthcare outcomes, fostering innovation, assisting in education, and supporting "
               "environmental monitoring and sustainability efforts.")
aggregated_text = " ".join(safe_texts) + ai_benefits

# Second round summarisation (same behavior as first round)
second_round_results = []
round_num = 2
for agent in agents:
    # Agent generates summary 
    output = agent.generate_summary(aggregated_text)
    
    # Run safety inspection
    inspected = monitor.inspect({
        "sender_id": agent.id,
        "round": round_num,
        "content": output["summary"],
        "metadata": output["metadata"]
    }, source_text=aggregated_text)
    
    # Append all entries
    second_round_results.append(inspected)


# Save final safe summaries after second round
with open("logs/final_safe_summaries.json", "w", encoding="utf-8") as f:
    json.dump(second_round_results, f, indent=4)
print(f"Second-round safe summaries saved: {len(second_round_results)} entries")

import json
from tabulate import tabulate

# Load second-round results
with open("logs/final_safe_summaries.json", "r", encoding="utf-8") as f:
    second_round_data = json.load(f)

# Prepare table data
table_data = []
for s in second_round_data:
    r = s.get("safety_report", {})
    meta = s.get("metadata", {})

    table_data.append([
        s["sender_id"],                                           # Agent ID
        meta.get("bias_simulated", False),                        # Simulated Toxicity
        r.get("notes", [{}])[0].get("toxicity_score", 0.0),      # Toxicity Score
        meta.get("policy_violation_simulated", False),           # Simulated Policy Violation
        r.get("notes", [{}])[3].get("hallucination_detected", False),  # Hallucination Detection
        r.get("summary_decision", "N/A")                         # Final Safety Decision
    ])

# Define table headers
headers = [
    "Agent ID",
    "Toxicity Simulated",
    "Toxicity",
    "Policy Violation Simulated",
    "Hallucination Simulated",
    "Decision"
]

# Print table
print("\nSECOND-ROUND SAFETY MONITORING RESULTS\n")
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

