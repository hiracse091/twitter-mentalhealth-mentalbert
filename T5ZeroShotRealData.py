import pandas as pd
from transformers import pipeline

# ===== USER SETTINGS =====
DATASET_PATH = "data/extracted/anon_disorder_tweets.csv"  # update with your actual file
TEXT_COLUMN = "text"               # column name for tweet text
LABEL_COLUMN = "disorder"             # column name for true label
NUM_SAMPLES = 50                   # number of tweets to test
OUTPUT_FILE = "t5_zero_shot_results.csv"
# ==========================

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Sample N rows
sample_df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna().sample(n=NUM_SAMPLES, random_state=42)
tweets = sample_df[TEXT_COLUMN].tolist()

# Load T5 model
classifier = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device=0,  # GPU
    max_length=128
)

# Build prompt
def format_prompt(tweet):
    return (
        "Classify the following tweet into one of the following mental health conditions: "
        "Anxiety, Depression, PTSD, Bipolar, Borderline, Panic. "
        f"Tweet: {tweet} Output:"
    )
prompts = [format_prompt(t) for t in tweets]

# Get predictions
outputs = classifier(prompts)
preds = [o['generated_text'].strip().capitalize() for o in outputs]

# Save predictions with true labels
sample_df["Prediction"] = preds
sample_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved predictions to {OUTPUT_FILE}")
