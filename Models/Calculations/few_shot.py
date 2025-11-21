# -----------------------------
# 1. Install packages
# -----------------------------
# !pip install "datasets<3.0.0" transformers evaluate accelerate --quiet

# -----------------------------
# 2. Imports
# -----------------------------
import os
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import classification_report
from datasets import load_dataset
import pandas as pd

# -----------------------------
# 3. Load dataset
# -----------------------------
dataset = load_dataset("cardiffnlp/tweet_sentiment_multilingual", "english")

# Convert to pandas for easier manipulation
train_df = dataset["train"].to_pandas()
val_df   = dataset["validation"].to_pandas()
test_df  = dataset["test"].to_pandas()

# Ensure labels are integers
train_df["label"] = train_df["label"].astype(int)
val_df["label"]   = val_df["label"].astype(int)
test_df["label"]  = test_df["label"].astype(int)

# Map numeric labels to sentiment strings
label_map = {0: "negative", 1: "neutral", 2: "positive"}
train_df["sentiment"] = train_df["label"].map(label_map)
val_df["sentiment"]   = val_df["label"].map(label_map)
test_df["sentiment"]  = test_df["label"].map(label_map)

print("Training label distribution:\n", train_df['sentiment'].value_counts())

# -----------------------------
# 4. Helper functions
# -----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")  # Hugging Face token stored in Colab

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_auth_token=HF_TOKEN)
    return tokenizer, model

def get_few_shot_examples(df, k=1):
    examples = []
    for cls in ["positive", "negative", "neutral"]:
        cls_samples = df[df['sentiment'] == cls]
        if len(cls_samples) == 0:
            print(f"Warning: no samples for class {cls}")
            continue
        n = min(k, len(cls_samples))
        examples.extend(cls_samples.sample(n, random_state=42).to_dict('records'))
    return examples

def create_prompt(few_shot_examples, text):
    prompt = ""
    for ex in few_shot_examples:
        prompt += f"Text: {ex['text']}\nSentiment: {ex['sentiment']}\n\n"
    prompt += f"Text: {text}\nSentiment:"
    return prompt

# -----------------------------
# 5. Experiment runner
# -----------------------------
def run_experiment(model_name, k):
    print(f"\n🔵 Running {model_name} with k={k}...\n")
    
    tokenizer, model = load_model(model_name)
    classifier = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    few_shot_examples = get_few_shot_examples(train_df, k=k)
    if len(few_shot_examples) == 0:
        raise ValueError("No few-shot examples found. Check train_df['sentiment'].")
    
    predictions = []
    true_labels = []

    for _, row in val_df.iterrows():
        prompt = create_prompt(few_shot_examples, row['text'])
        output = classifier(prompt, max_new_tokens=10, do_sample=False)
        pred_text = output[0]['generated_text'].split("Sentiment:")[-1].strip().lower()

        if "positive" in pred_text:
            pred = "positive"
        elif "negative" in pred_text:
            pred = "negative"
        else:
            pred = "neutral"

        predictions.append(pred)
        true_labels.append(row['sentiment'])

    report = classification_report(true_labels, predictions, target_names=["negative","neutral","positive"], zero_division=0)
    print(f"✅ {model_name} (k={k}) performance:\n{report}")

# -----------------------------
# 6. Run experiments
# -----------------------------
models = [
    "microsoft/phi-3-mini-4k-instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B"
]

for model_name in models:
    for k in [1, 2]:
        run_experiment(model_name, k)
