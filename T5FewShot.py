import pandas as pd
from transformers import pipeline

# ===== FILE & SETTINGS =====
DATASET_PATH = "data/extracted/anon_disorder_tweets.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "disorder"
NUM_FEW_SHOT = 2  # per label
NUM_SAMPLES = 20  # test samples
OUTPUT_FILE = "t5_few_shot_results.csv"
FEW_SHOT_FILE = "few_shot_example.txt"
# ===========================

# Load dataset
df = pd.read_csv(DATASET_PATH)
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

# Filter for target labels
valid_labels = ["anxiety", "depression", "ptsd", "bipolar", "borderline", "panic", "control"]
df = df[df[LABEL_COLUMN].str.lower().isin(valid_labels)]

# Normalize labels
df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip().str.capitalize()

# Load few-shot examples from file
with open(FEW_SHOT_FILE, "r") as f:
    few_shot_prompt = f.read().strip()

# Sample test tweets (excluding few-shot examples)
sample_df = df.sample(n=NUM_SAMPLES, random_state=42)
tweets = sample_df[TEXT_COLUMN].tolist()

# Format each prompt
def format_prompt(tweet):
    intro = "Classify the following tweet into one of these categories: Anxiety, Depression, PTSD, Bipolar, Borderline, Panic.\n\n"
    test_tweet = f'Tweet: "{tweet}"\nLabel:'
    return intro + few_shot_prompt + "\n\n" + test_tweet

prompts = [format_prompt(tweet) for tweet in tweets]


# Load FLAN-T5 model
classifier = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device=0,  # Use GPU
    max_length=128
)

# Run predictions
outputs = classifier(prompts)
preds = [output['generated_text'].strip().capitalize() for output in outputs]

# Save predictions alongside true labels
sample_df["Prediction"] = preds
sample_df.to_csv(OUTPUT_FILE, index=False)
print(f"[✓] Saved few-shot predictions to {OUTPUT_FILE}")
