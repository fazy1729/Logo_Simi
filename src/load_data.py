import pandas as pd
import os

def load_dataset(file_path="./data/logos.snappy.parquet"):
    df = pd.read_parquet(file_path)
    print(df.head())  
    print(df.columns) 
    return df

if __name__ == "__main__":
    df = load_dataset()
