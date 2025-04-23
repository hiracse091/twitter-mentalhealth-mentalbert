import pandas as pd

# Load test set
df = pd.read_csv("data/test_data.csv")

# Get unique disorder labels (excluding control)
disorders = df["label"].unique()
disorders = [d for d in disorders if d.lower() != "control"]

# Sample N candidate tweets per disorder
N = 10  # Change this to sample more if needed
samples = []

for disorder in disorders:
    subset = df[df["label"] == disorder].sample(n=N, random_state=42)
    subset["disorder"] = disorder
    samples.append(subset)

# Combine and save to CSV
sample_df = pd.concat(samples).reset_index(drop=True)
sample_df.to_csv("data/few_shot_candidates.csv", index=False)

print("✅ Saved few_shot_candidates.csv with 10 tweets per disorder for manual review.")
