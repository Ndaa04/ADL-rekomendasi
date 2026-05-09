import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def compute_content_similarity(df: pd.DataFrame, max_features: int = 5000) -> np.ndarray:
    """Content-Based: TF-IDF pada kolom overview + Cosine Similarity."""
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(df["overview"])
    return cosine_similarity(tfidf_matrix)

def compute_collaborative_similarity(df: pd.DataFrame) -> np.ndarray:
    """Item-Based CF: vote_average, vote_count, popularity + Cosine Similarity."""
    features = df[["vote_average", "vote_count", "popularity"]].fillna(0)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    return cosine_similarity(features_scaled)

def get_hybrid_recommendations(
    df: pd.DataFrame, 
    cb_sim: np.ndarray, 
    cf_sim: np.ndarray, 
    show_name: str, 
    cf_weight: float = 0.2, 
    cb_weight: float = 0.8, 
    top_n: int = 10
) -> tuple[pd.DataFrame | None, str | None]:
    """Menggabungkan skor CF & CB dengan bobot tertentu dan mengembalikan top-N rekomendasi."""
    mask = df["name"].str.lower() == show_name.lower()
    if not mask.any():
        return None, f"Show '{show_name}' tidak ditemukan di dataset."
        
    idx = df.index[mask][0]
    
    # Weighted Hybrid Score
    hybrid_sim = (cf_weight * cf_sim[idx]) + (cb_weight * cb_sim[idx])
    
    # Ambil top_n (skip index 0 karena itu dirinya sendiri)
    similar_indices = hybrid_sim.argsort()[::-1][1:top_n+1]
    
    result = df.iloc[similar_indices][["name", "vote_average", "vote_count"]].copy()
    result["similarity_score"] = hybrid_sim[similar_indices]
    result = result.sort_values(by="similarity_score", ascending=False)
    
    return result, None