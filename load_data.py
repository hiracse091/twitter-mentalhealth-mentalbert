import os
import kagglehub
import tarfile
import pandas as pd
from pathlib import Path

# Step 1: Set up local data directory
project_root = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
data_dir = project_root / "data"
os.environ["KAGGLEHUB_DIR"] = str(data_dir)  # ✅ Set Kagglehub to use local folder

# Step 2: Download dataset (uses KAGGLEHUB_DIR)
download_path = kagglehub.dataset_download("rrmartin/twitter-mental-disorder-tweets-and-musics")
print("✅ Dataset downloaded to:", download_path)

# Step 3: Extract .tar.xz files
extract_path = data_dir / "extracted"
extract_path.mkdir(exist_ok=True)

for file in os.listdir(download_path):
    if file.endswith(".tar.xz"):
        file_path = os.path.join(download_path, file)
        print(f"📦 Extracting {file}...")
        with tarfile.open(file_path, mode="r:xz") as tar:
            tar.extractall(path=extract_path)

# Step 4: List extracted files
print("\n📂 Extracted files:")
for root, dirs, files in os.walk(extract_path):
    for name in files:
        print(" -", os.path.join(root, name))

# Step 5: Preview any tabular/text data
print("\n🔍 Previewing Data Files:")
for root, dirs, files in os.walk(extract_path):
    for name in files:
        if name.endswith((".csv", ".tsv", ".txt")):
            file_path = os.path.join(root, name)
            try:
                df = pd.read_csv(file_path, nrows=5, sep=None, engine='python')
                print(f"\n🔹 {name}")
                print("   Shape:", pd.read_csv(file_path).shape)
                print("   Columns:", df.columns.tolist())
                print(df.head())
            except Exception as e:
                print(f"⚠️ Could not preview {name}: {e}")
