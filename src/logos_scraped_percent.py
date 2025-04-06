import os
import pandas as pd

df = pd.read_parquet("./data/logos.snappy.parquet")

df['filename'] = df['domain'].apply(lambda d: f"{d.replace('.', '_')}.png")

downloaded_files = set(os.listdir("logos"))

df['downloaded'] = df['filename'].apply(lambda f: f in downloaded_files)

total_domains = len(df)
downloaded_count = df['downloaded'].sum()
percent_downloaded = (downloaded_count / total_domains) * 100

print(f"✅ Downloaded logos: {downloaded_count} / {total_domains} ({percent_downloaded:.2f}%)")
