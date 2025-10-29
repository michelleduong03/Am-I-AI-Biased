# Perplexity Script
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")

# 2. Load tokenizer and model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

# 3. Text you want to evaluate
text = """
NEW YORK (AP) — The head coach of the Portland Trail Blazers and a player for the Miami Heat were arrested Thursday along with more than 30 other people in a takedown of two sprawling gambling operations that authorities said leaked inside information about NBA athletes and rigged poker games backed by Mafia families.

Portland coach Chauncey Billups was charged with participating in a conspiracy to fix high-stakes card games tied to La Cosa Nostra organized crime families that cheated unsuspecting gamblers out of at least $7 million. Heat guard Terry Rozier was accused in a separate scheme of exploiting private information about players to win bets on NBA games.

The two indictments unsealed in New York create a massive cloud for the NBA — which opened its season this week — and show how certain types of wagers are vulnerable to massive fraud in the growing, multibillion-dollar legal sports-betting industry. Joseph Nocella, the top federal prosecutor for the Eastern District of New York, called it “one of the most brazen sports corruption schemes since online sports betting became widely legalized in the United States.”
"""

# 4. Compute perplexity
def compute_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt").to(model.device)
    max_length = getattr(model.config, "n_positions", 1024)
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100  # mask out tokens we don’t predict
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids, use_cache=False)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

# 5. Print perplexity
perplexity = compute_perplexity(text)
print(f"Perplexity: {perplexity:.2f}")