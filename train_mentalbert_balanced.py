import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from datasets import Dataset
from sklearn.metrics import classification_report


# Load dataset
print("📥 Loading dataset...")
disorder_df = pd.read_csv("data/extracted/anon_disorder_tweets.csv", usecols=["text", "disorder"])
control_df = pd.read_csv("data/extracted/anon_control_tweets.csv", usecols=["text", "disorder"])
df = pd.concat([disorder_df, control_df], ignore_index=True).dropna().sample(frac=1).reset_index(drop=True)
#df = df.sample(frac=0.01, random_state=42) # reduce dataset size to make it faster

df = df.rename(columns={'disorder': 'label'})



# Train/test split
#df = df.sample(frac=0.01, random_state=42) # reduce dataset size to make it faster

# =========================
# Combined Sampling Strategy
# =========================
grouped = {label: df[df['label'] == label] for label in df['label'].unique()}

target_counts = {
    'control': 20000,
    'depression': 20000,
    'anxiety': 15000,
    'ptsd': 15000,
    'bipolar': 15000,
    'borderline': 15000,
    'panic': 15000
}

balanced_df_list = []
for label, group in grouped.items():
    print(len(group))
    if len(group) > target_counts[label]:
        sampled = group.sample(n=target_counts[label], random_state=42)
    else:
        sampled = resample(group, replace=True, n_samples=target_counts[label], random_state=42)
    balanced_df_list.append(sampled)

balanced_df = pd.concat(balanced_df_list).sample(frac=1, random_state=42).reset_index(drop=True)

# Save balanced dataset locally
balanced_df.to_csv("data/balanced_tweets.csv", index=False)

# Encode labels
label_list = sorted(balanced_df['label'].unique())
label2id = {label: idx for idx, label in enumerate(label_list)}
balanced_df['label_id'] = balanced_df['label'].map(label2id)

# Train/Val split
train_df, val_df = train_test_split(balanced_df, test_size=0.1, stratify=balanced_df['label_id'], random_state=42)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label_id']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("label_id", "labels")
val_dataset = val_dataset.rename_column("label_id", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# =========================
# Model & Class Weights
# =========================
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_df['label_id']), y=train_df['label_id'])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForSequenceClassification.from_pretrained(
    "mental/mental-bert-base-uncased",
    num_labels=len(label_list),
)

# Override loss function to use class weights
class WeightedLossModel(nn.Module):
    def __init__(self, model, weights):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs.logits)

weighted_model = WeightedLossModel(model, class_weights_tensor)

# =========================
# Training Setup
# =========================
# training_args = TrainingArguments(
#     output_dir="./save_model_balanced",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     logging_dir="./logs",
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     num_train_epochs=4,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     fp16=torch.cuda.is_available(),
# )

def compute_metrics(eval_pred):
    from sklearn.metrics import classification_report
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(labels, preds, output_dict=True)
    return {f"f1_{label}": report[label]['f1-score'] for label in report if label.isdigit()}

# trainer = Trainer(
#     model=weighted_model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     compute_metrics=compute_metrics,
# )

save_dir = os.path.abspath(f"save_mentalbert_final")
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
    fp16=True
)

trainer = Trainer(
    model=weighted_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


trainer.train()

# # Save final model
# trainer.save_model("./save_model_balanced")
# tokenizer.save_pretrained("./save_model_balanced")

print(f"💾 Saving model and tokenizer to {save_dir}...")
weighted_model.model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
pd.Series(label_list).to_csv(os.path.join(save_dir, "labels.csv"), index=False)
#target_names=label_list

test_dataset = val_dataset
test_labels = val_df['label_id'].tolist()
model_name = "MentalBERT (Balanced)"
print(f"📊 Evaluating {model_name}...")
preds = trainer.predict(test_dataset)
pred_labels = preds.predictions.argmax(axis=1)
report = classification_report(test_labels, pred_labels, target_names=label_list)

with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
    f.write(report)

print(f"✅ Finished training and saving {model_name}\n")
