
import pandas as pd
import torch
from transformers import RobertaTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# ========== CONFIG ========== #
MODEL_PATH = "/home/nnaher/CAP6640/saved_models/mentalbert"  # or "saved_models/roberta"
TEST_DATA_PATH = "/home/nnaher/CAP6640/data/test_data.csv"
LABEL_NAMES = ["Anxiety", "Depression", "Panic", "PTSD", "Bipolar", "Borderline", "Control"]
BATCH_SIZE = 32
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
df = pd.read_csv(TEST_DATA_PATH)
label2id = {label.lower(): idx for idx, label in enumerate(LABEL_NAMES)}
id2label = {idx: label for label, idx in label2id.items()}

df["label_id"] = df["label"].str.lower().map(label2id)
texts = df["text"].tolist()
labels = df["label_id"].tolist()

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Load model and predict
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

predictions = []
with torch.no_grad():
    for batch in dataloader:
        input_ids, attention_mask, _ = [x.to(DEVICE) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().tolist())

# Evaluation
true_labels = labels
predicted_labels = predictions
print("\n📋 Classification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=LABEL_NAMES))

# Save results
df["predicted"] = [id2label[p] for p in predicted_labels]
df.to_csv("/home/nnaher/CAP6640/evaluation/test_results_mentalbert.csv", index=False)
print("✅ Results saved to test_results_mentalbert.csv")
