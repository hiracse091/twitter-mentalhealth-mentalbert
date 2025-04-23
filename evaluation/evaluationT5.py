import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load your results
df = pd.read_csv("t5_zero_shot_results.csv")

# Clean predicted and true labels (standardize capitalization)
df["Prediction"] = df["Prediction"].str.strip().str.capitalize()
df["label"] = df["disorder"].str.strip().str.capitalize()

# Evaluation metrics
y_true = df["label"]
y_pred = df["Prediction"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision (macro): {precision:.3f}")
print(f"Recall (macro): {recall:.3f}")
print(f"F1 Score (macro): {f1:.3f}")

# Confusion matrix
labels = sorted(df["label"].unique())
conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

# Plot the confusion matrix
# sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("T5 Zero-Shot - Confusion Matrix")
# plt.tight_layout()
#plt.show()


df = pd.read_csv("t5_few_shot_results.csv")

# Clean predicted and true labels (standardize capitalization)
df["Prediction"] = df["Prediction"].str.strip().str.capitalize()
df["label"] = df["disorder"].str.strip().str.capitalize()

# Evaluation metrics
y_true = df["label"]
y_pred = df["Prediction"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision (macro): {precision:.3f}")
print(f"Recall (macro): {recall:.3f}")
print(f"F1 Score (macro): {f1:.3f}")

# Confusion matrix
labels = sorted(df["label"].unique())
conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

# Plot the confusion matrix
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("T5 Few-Shot - Confusion Matrix")
plt.tight_layout()
plt.savefig("../plots/T5FewShotConfuctionMatrix.png")
plt.show()
