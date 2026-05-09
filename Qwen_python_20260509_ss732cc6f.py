import pandas as pd
import re

def load_and_preprocess_data(filepath: str = "top_rated_tv.csv") -> pd.DataFrame:
    """Memuat dataset TV dan melakukan preprocessing teks."""
    df = pd.read_csv(filepath)
    
    # Hapus baris tanpa nama & isi overview kosong
    df = df.dropna(subset=["name"])
    df["overview"] = df["overview"].fillna("")
    
    # Cleaning teks: lowercase, hapus simbol, rapikan spasi
    df["overview"] = (
        df["overview"]
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    
    # Reset index agar pencarian berdasarkan posisi aman
    df = df.reset_index(drop=True)
    return df