import pandas as pd
from datasets import load_dataset
import os

def download_welfake():
    print("Downloading WELFake dataset from Hugging Face...")
    ds = load_dataset('davanstrien/WELFake', split='train')
    
    print("Converting to pandas DataFrame...")
    df = ds.to_pandas()
    
    # Save to CSV
    output_path = "WELFake_Dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} records to {output_path}")

if __name__ == "__main__":
    download_welfake()
