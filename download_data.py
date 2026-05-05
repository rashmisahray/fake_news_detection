import pandas as pd
from datasets import load_dataset
import os

def download_and_merge():
    print("1. Downloading WELFake dataset...")
    try:
        ds_welfake = load_dataset('davanstrien/WELFake', split='train')
        df_welfake = ds_welfake.to_pandas()
        # Combine title and text, or just use text if title is missing
        df_welfake['text'] = df_welfake['title'].fillna('') + ' ' + df_welfake['text'].fillna('')
        df_welfake = df_welfake[['text', 'labels']].rename(columns={'labels': 'label'})
        df_welfake['source'] = 'WELFake'
        print(f"   Loaded {len(df_welfake)} WELFake records.")
    except Exception as e:
        print(f"Error loading WELFake: {e}")
        df_welfake = pd.DataFrame(columns=['text', 'label', 'source'])

    print("2. Downloading FakeNewsNet dataset...")
    try:
        ds_fnn = load_dataset('rickstello/FakeNewsNet', split='train')
        df_fnn = ds_fnn.to_pandas()
        # Map real=0 -> Fake (1), real=1 -> Real (0)
        df_fnn['label'] = df_fnn['real'].apply(lambda x: 0 if x == 1 else 1)
        df_fnn['text'] = df_fnn['title'].fillna('')
        df_fnn = df_fnn[['text', 'label']]
        df_fnn['source'] = 'FakeNewsNet'
        print(f"   Loaded {len(df_fnn)} FakeNewsNet records.")
    except Exception as e:
        print(f"Error loading FakeNewsNet: {e}")
        df_fnn = pd.DataFrame(columns=['text', 'label', 'source'])

    print("3. Downloading LIAR dataset...")
    try:
        # Load from GitHub raw content
        url = 'https://raw.githubusercontent.com/Tariq60/LIAR-PLUS/master/dataset/tsv/train2.tsv'
        df_liar_raw = pd.read_csv(url, sep='\t', header=None)
        
        # Columns: 1 is label, 2 is statement
        df_liar = pd.DataFrame()
        df_liar['text'] = df_liar_raw[3].fillna('')
        
        # Map labels
        fake_labels = ['pants-fire', 'false', 'barely-true']
        real_labels = ['half-true', 'mostly-true', 'true']
        
        def map_liar_label(l):
            if str(l).lower() in fake_labels: return 1
            if str(l).lower() in real_labels: return 0
            return 1 # Default to fake if unknown (shouldn't happen)
            
        df_liar['label'] = df_liar_raw[2].apply(map_liar_label)
        df_liar['source'] = 'LIAR'
        print(f"   Loaded {len(df_liar)} LIAR records.")
    except Exception as e:
        print(f"Error loading LIAR: {e}")
        df_liar = pd.DataFrame(columns=['text', 'label', 'source'])

    print("Merging datasets...")
    df_combined = pd.concat([df_welfake, df_fnn, df_liar], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Clean up empty texts
    df_combined = df_combined[df_combined['text'].str.strip().astype(bool)]
    
    output_path = "Combined_Dataset.csv"
    df_combined.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_combined)} combined records to {output_path}")
    print(df_combined['source'].value_counts())

if __name__ == "__main__":
    download_and_merge()
