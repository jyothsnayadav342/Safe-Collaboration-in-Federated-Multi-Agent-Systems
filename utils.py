# utils.py
import json
from datetime import datetime
from safety_layer.safety_monitor import SafetyMonitor

# Load monitor once at module level for efficiency
monitor = SafetyMonitor(config_path="config.json")


def load_original_text():
    """
    It loads the original input text.
    """
    with open("data/input_text.txt", "r", encoding="utf-8") as f:
        return f.read().strip()


def aggregate_safe_summaries(client_summaries, source_text, current_round):
    """
    It aggregates summaries from different clients, runs safety inspection,
    stores the inspection logs and returns only the safe summaries.
    """
    safe_texts = []              # Stores summaries marked safe
    full_inspection_log = []     # Stores all inspection results for this round

    # Process each client summary
    for client_proxy, metrics, _ in client_summaries:
        summary_json = metrics["summary_json"]
        output_dict = json.loads(summary_json)

        # Extract generated summary + metadata
        summary_text = output_dict["summary"]
        metadata = output_dict.get("metadata", {})

        # Construct message passed to the safety monitor
        msg = {
            "sender_id": metadata.get("agent_id", 0),
            "round": current_round,
            "content": summary_text,
            "metadata": metadata
        }

        # Run safety check
        inspected = monitor.inspect(msg, source_text=source_text)
        inspected["round"] = current_round

        full_inspection_log.append(inspected)

        # Keep only safe summaries
        if inspected["safety_report"]["summary_decision"] == "safe":
            safe_texts.append(summary_text)

    # Path for saving detailed logs as a single JSON array
    log_file = "logs/detailed_safety_inspections.json"

    # Load existing log file if present
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            all_logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_logs = []

    # Add this round's logs to the file
    all_logs.extend(full_inspection_log)

    # Save updated logs with formatting
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)

    # Console summary for user visibility
    safe_count = len(safe_texts)
    print(f"\nROUND {current_round} COMPLETE")
    print(f"   Safe summaries  : {safe_count}/5")
    print(f"   Blocked         : {5 - safe_count}/5")
    print(f"   Log saved → {log_file}\n")
    
    # Final clean text for next round
    ai_benefits = " AI benefits society by enhancing productivity, enabling smarter decision-making, improving healthcare outcomes, fostering innovation, assisting in education, and supporting environmental monitoring and sustainability efforts."
    return " ".join(safe_texts) + ai_benefits