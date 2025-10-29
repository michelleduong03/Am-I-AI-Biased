# https://huggingface.co/meta-llama/Llama-3.2-3B
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
import os

# 1. Load the .env file + Get your secret Hugging Face token
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

# 2. Load the tokenizer and model with the token and trust_remote_code
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    token=hf_token,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    token=hf_token,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, # Recommended for performance
    device_map="auto"           # Automatically use GPU if available
)

# 3. Prepare and tokenize the input
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 4. Generate the output tokens from the model
outputs = model.generate(**inputs, max_new_tokens=200)

# 5. Decode the output tokens back to a string
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)