import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Function to parse a classification report
def parse_classification_report(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines[2:]:  # Skip header lines
        parts = re.split(r'\s{2,}', line.strip())
        if len(parts) == 5:
            label, precision, recall, f1, support = parts
            try:
                data.append([label, float(precision), float(recall), float(f1), int(support)])
            except ValueError:
                continue
    return pd.DataFrame(data, columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])

# Load classification reports
roberta_df = parse_classification_report("../evaluation/classification_report_roberta.txt")
mentalbert_df = parse_classification_report("../evaluation/classification_report_mentalbert.txt")
mentalbert_finetuned_df = parse_classification_report("../evaluation/classification_report_finetuned_mentalbert.txt")

# Add model identifiers
roberta_df['Model'] = 'RoBERTa'
mentalbert_df['Model'] = 'MentalBERT'
mentalbert_finetuned_df['Model'] = 'MentalBERT_Finetuned'

# Combine and filter
combined_df = pd.concat([roberta_df, mentalbert_df, mentalbert_finetuned_df], ignore_index=True)
target_classes = ['anxiety', 'bipolar', 'borderline', 'control', 'depression', 'panic', 'ptsd']
combined_df = combined_df[combined_df['Class'].isin(target_classes)]

# Plotting
sns.set(style="whitegrid")
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

# Precision
ax1 = fig.add_subplot(gs[0, 0])
sns.barplot(data=combined_df, x='Class', y='Precision', hue='Model', ax=ax1)
ax1.set_title('Precision Comparison by Class')
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylabel('Precision')

# Recall
ax2 = fig.add_subplot(gs[0, 1])
sns.barplot(data=combined_df, x='Class', y='Recall', hue='Model', ax=ax2)
ax2.set_title('Recall Comparison by Class')
ax2.tick_params(axis='x', rotation=45)
ax2.set_ylabel('Recall')

# F1-Score
ax3 = fig.add_subplot(gs[1, :])
sns.barplot(data=combined_df, x='Class', y='F1-Score', hue='Model', ax=ax3)
ax3.set_title('F1-Score Comparison by Class')
ax3.tick_params(axis='x', rotation=45)
ax3.set_ylabel('F1-Score')

# Layout and show
plt.tight_layout()
plt.savefig("../plots/model_performance_comparison.png")
plt.show()


# ---- Create bar plots for Precision, Recall, F1-Score ----
# sns.set(style="whitegrid")
# fig = plt.figure(figsize=(18, 10))

# # Grid layout: 2 columns, 2 rows (top row for Precision & Recall, bottom row spans both for F1)
# gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

# # --- Precision Plot ---
# ax1 = fig.add_subplot(gs[0, 0])
# sns.barplot(data=combined_df, x='Class', y='Precision', hue='Model', ax=ax1)
# ax1.set_title('Precision Comparison by Class')
# ax1.set_xlabel('')
# ax1.set_ylabel('Precision')
# ax1.tick_params(axis='x', rotation=45)
# ax1.legend(loc='upper right')

# # --- Recall Plot ---
# ax2 = fig.add_subplot(gs[0, 1])
# sns.barplot(data=combined_df, x='Class', y='Recall', hue='Model', ax=ax2)
# ax2.set_title('Recall Comparison by Class')
# ax2.set_xlabel('')
# ax2.set_ylabel('Recall')
# ax2.tick_params(axis='x', rotation=45)
# ax2.legend(loc='upper right')

# # --- F1-Score Plot (full width on second row) ---
# ax3 = fig.add_subplot(gs[1, :])
# sns.barplot(data=combined_df, x='Class', y='F1-Score', hue='Model', ax=ax3)
# ax3.set_title('F1-Score Comparison by Class')
# ax3.set_xlabel('Class')
# ax3.set_ylabel('F1-Score')
# ax3.tick_params(axis='x', rotation=45)
# ax3.legend(loc='upper right')

# # --- Final layout ---
# plt.tight_layout()
# plt.savefig('../plots/model_performance_comparison3.png')  # Save the plot
# plt.show()
