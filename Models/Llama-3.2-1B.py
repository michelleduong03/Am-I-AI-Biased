# https://huggingface.co/meta-llama/Llama-3.2-1B
import torch
from transformers import pipeline, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()

use_auth_token = os.getenv("HF_TOKEN")

model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=use_auth_token)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=use_auth_token, torch_dtype=torch.bfloat16, device_map="auto")

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    tokenizer=tokenizer,
    device_map="auto"
)

prompt = "Once upon a time"
output = pipe(prompt, max_new_tokens=200, num_return_sequences=1, do_sample=True)

print(output[0]["generated_text"])

#result = pipe("Once upon a time")
#print(result)