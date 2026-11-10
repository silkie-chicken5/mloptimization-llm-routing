"""Download RouterBench (0-shot) into data/raw/. Run from the repo root."""
import os
import pandas as pd
from huggingface_hub import hf_hub_download

RAW_DIR = "data/raw"
FILENAME = "routerbench_0shot.pkl"

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    target = os.path.join(RAW_DIR, FILENAME)

    if os.path.exists(target):
        print(f"Already downloaded: {target}")
        return

    path = hf_hub_download(
        repo_id="withmartian/routerbench",
        filename=FILENAME,
        repo_type="dataset",
        local_dir=RAW_DIR,
    )
    print(f"Saved to: {path}")


    



if __name__ == "__main__":
    main()
    df = pd.read_pickle("/Users/juliahansen/Documents/MBAn Courses/ML/mloptimization-llm-routing/python/data/raw/routerbench_0shot.pkl")
    print(df.shape)
    print(df.columns.tolist())
