# train_mentalbert.py (with transformers version check)

import os
import torch
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
# Verify transformers version
min_required_version = (4, 0, 0)
current_version = tuple(map(int, transformers.__version__.split(".")[:3]))
if current_version < min_required_version:
    raise EnvironmentError(f"Transformers version >= 4.0.0 required, but found {transformers.__version__}. Please upgrade.")



# Check and log device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔧 Using device:", device)

# Optional: Set your Hugging Face token (if needed)
# os.environ["HUGGINGFACE_HUB_TOKEN"] = "hf_your_token_here"

# Load dataset
print("📥 Loading dataset...")
disorder_df = pd.read_csv("data/extracted/anon_disorder_tweets.csv", usecols=["text", "disorder"])
control_df = pd.read_csv("data/extracted/anon_control_tweets.csv", usecols=["text", "disorder"])
df = pd.concat([disorder_df, control_df], ignore_index=True).dropna().sample(frac=1).reset_index(drop=True)

# Encode labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["disorder"])

# Train/test split
#df = df.sample(frac=0.01, random_state=42) # reduce dataset size to make it faster
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# Tokenization
model_name = "mental/mental-bert-base-uncased"  # or replace with 'mental/mental-bert-base-uncased' if approved
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()}).map(tokenize, batched=True)
test_dataset = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()}).map(tokenize, batched=True)

# Model and collator
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_)).to(device)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set save path
save_path = os.path.abspath("saved_models/mentalbert_full")
os.makedirs(save_path, exist_ok=True)

# TrainingArguments optimized for GPU
training_args = TrainingArguments(
    output_dir=save_path,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    fp16=True,
    logging_dir=os.path.join(save_path, "logs"),
    logging_steps=10,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("🚀 Training MentalBERT...")
trainer.train()

# Save model
print("💾 Saving model...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
pd.Series(label_encoder.classes_).to_csv(os.path.join(save_path, "labels.csv"), index=False)

# Evaluate
print("📊 Evaluating model...")
preds = trainer.predict(test_dataset)
pred_labels = preds.predictions.argmax(axis=1)
report = classification_report(test_labels, pred_labels, target_names=label_encoder.classes_)

with open(os.path.join(save_path, "classification_report.txt"), "w") as f:
    f.write(report)

print("✅ MentalBERT training complete.")

