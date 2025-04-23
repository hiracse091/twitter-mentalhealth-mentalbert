from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# Config
MODEL_PATH = "/home/nnaher/CAP6640/saved_models/roberta-base"
TEST_DATA_PATH = "/home/nnaher/CAP6640/data/test_data.csv"
LABEL_NAMES = ["Anxiety", "Depression", "Panic", "PTSD", "Bipolar", "Borderline", "Control"]
BATCH_SIZE = 16
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
df = pd.read_csv(TEST_DATA_PATH)
label2id = {label.lower(): idx for idx, label in enumerate(LABEL_NAMES)}
id2label = {idx: label for label, idx in label2id.items()}

df["label_id"] = df["label"].str.lower().map(label2id)
texts = df["text"].tolist()
labels = df["label_id"].tolist()

# Load tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)

# Tokenize data
encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Load model
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# Inference
predictions = []
with torch.no_grad():
    for batch in dataloader:
        input_ids, attention_mask, _ = [x.to(DEVICE) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().tolist())

# Evaluation
df["predicted"] = [id2label[p] for p in predictions]
print("\n📋 Classification Report:\n")
print(classification_report(df["label_id"], predictions, target_names=LABEL_NAMES))

# Save results
df.to_csv("/home/nnaher/CAP6640/evaluation/test_results_roberta.csv", index=False)
print("✅ Results saved to test_results_roberta.csv")
