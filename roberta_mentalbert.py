#roberta_mentalbert.py


import os
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report

# Verify transformers version
import transformers
print("Transformers version:", transformers.__version__)

# Load and prepare data
def load_data():
    print("📥 Loading and preparing dataset...")
    disorder_df = pd.read_csv("data/extracted/anon_disorder_tweets.csv", usecols=["text", "disorder"])
    control_df = pd.read_csv("data/extracted/anon_control_tweets.csv", usecols=["text", "disorder"])
    df = pd.concat([disorder_df, control_df], ignore_index=True).dropna().sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=0.01, random_state=42)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["disorder"])
    return df, le

# Tokenization
def tokenize(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

# Train a model
def train_model(model_name, df, label_encoder):
    print(f"\n🚀 Starting training for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )

    train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()}).map(
        lambda x: tokenize(x, tokenizer), batched=True
    )
    test_dataset = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()}).map(
        lambda x: tokenize(x, tokenizer), batched=True
    )

    save_dir = os.path.abspath(f"saved_models_both/{model_name.replace('/', '_')}")
    os.makedirs(save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=save_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        logging_dir=f"{save_dir}/logs",
        logging_steps=10,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()

    print(f"💾 Saving model and tokenizer to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    pd.Series(label_encoder.classes_).to_csv(os.path.join(save_dir, "labels.csv"), index=False)

    print(f"📊 Evaluating {model_name}...")
    preds = trainer.predict(test_dataset)
    pred_labels = preds.predictions.argmax(axis=1)
    report = classification_report(test_labels, pred_labels, target_names=label_encoder.classes_)

    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print(f"✅ Finished training and saving {model_name}\n")

if __name__ == "__main__":
    df, label_encoder = load_data()
    
    for model_name in ["roberta-base"]:
        try:
            train_model(model_name, df, label_encoder)
        except Exception as e:
            print(f"❌ Failed to train {model_name}: {e}")
