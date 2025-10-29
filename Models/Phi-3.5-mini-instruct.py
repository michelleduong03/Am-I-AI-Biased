# https://huggingface.co/microsoft/Phi-3.5-mini-instruct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set a seed for reproducibility
# torch.random.manual_seed(0)

# Load the model and tokenizer with all necessary arguments
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

# Create the high-level pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Define the message you want to send
messages = [
    {"role": "user", "content": "Once upon a time"},
]

# Define generation arguments
generation_args = {
    "max_new_tokens": 400,
    "return_full_text": False,  # So it doesn't repeat your prompt
    "do_sample": True,
    "temperature": 1.2,
    "top_p": 0.8,
    "top_k": 1000,
    "repetition_penalty": 1.1,
}

# Run the pipeline
output = pipe(messages, use_cache=False, **generation_args)
print(output[0]['generated_text'])