import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Balanced Dataset
# -----------------------------
data = pd.read_csv('./data/balanced_tweets.csv')  # 🔁 Update path if needed
texts = data['text'].tolist()
labels = data['label'].tolist()

# Encode labels to integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
label_names = label_encoder.classes_

# -----------------------------
# 2. Load Your MentalBERT Model
# -----------------------------
model_path = "./saved_models/mentalbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# -----------------------------
# 3. Dataset Preparation
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = TextDataset(texts, encoded_labels)
loader = DataLoader(dataset, batch_size=16)

# -----------------------------
# 4. Run Inference
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].numpy())

# -----------------------------
# 5. Evaluation
# -----------------------------
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=label_names))

# Save report to file
report_df = pd.DataFrame(classification_report(all_labels, all_preds, target_names=label_names, output_dict=True)).transpose()
report_df.to_csv("./plots/mentalbert_old_classification_report.csv")

# -----------------------------
# 6. Confusion Matrix
# -----------------------------
ConfusionMatrixDisplay.from_predictions(all_labels, all_preds, display_labels=label_names, xticks_rotation=45)
plt.title("Confusion Matrix - MentalBERT w/o Finetuning")
plt.tight_layout()
plt.savefig("./plots/mentalbert_old_confusion_matrix.png")
#plt.show()
