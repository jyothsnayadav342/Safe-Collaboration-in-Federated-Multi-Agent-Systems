# llm_agent.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMWrapper:
    def __init__(self, model_name="distilgpt2"):
        # Keep models small for quick tests
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=100, do_sample=True, top_k=50, top_p=0.95)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # returned text includes the prompt; remove the prompt prefix if desired:
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text
