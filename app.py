import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os

# ==========================================
# 1. DATA LOADER
# ==========================================
@st.cache_data
def load_and_preprocess_data(filepath: str = "top_rated_tv.csv") -> pd.DataFrame:
    if not os.path.exists(filepath):
        st.error(f"File '{filepath}' tidak ditemukan. Pastikan dataset berada di folder yang sama.")
        st.stop()
        
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["name"])
    df["overview"] = df["overview"].fillna("")
    
    df["overview"] = (
        df["overview"]
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    return df.reset_index(drop=True)

# ==========================================
# 2. RECOMMENDER ENGINE
# ==========================================
@st.cache_resource
def build_similarity_matrices(df: pd.DataFrame, max_features: int = 5000):
    # Content-Based (TF-IDF)
    tfidf = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(df["overview"])
    cb_sim = cosine_similarity(tfidf_matrix)
    
    # Collaborative Filtering (Numeric Features)
    features = df[["vote_average", "vote_count", "popularity"]].fillna(0)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    cf_sim = cosine_similarity(features_scaled)
    
    return cb_sim, cf_sim

def get_hybrid_recommendations(df, cb_sim, cf_sim, show_name, cf_weight=0.2, cb_weight=0.8, top_n=10):
    mask = df["name"].str.lower() == show_name.lower()
    if not mask.any():
        return None, f"Show '{show_name}' tidak ditemukan di dataset."
        
    idx = df.index[mask][0]
    hybrid_sim = (cf_weight * cf_sim[idx]) + (cb_weight * cb_sim[idx])
    
    similar_indices = hybrid_sim.argsort()[::-1][1:top_n+1]
    result = df.iloc[similar_indices][["name", "vote_average", "vote_count", "popularity"]].copy()
    result["similarity_score"] = hybrid_sim[similar_indices]
    return result.sort_values(by="similarity_score", ascending=False), None

# ==========================================
# 3. UTILS
# ==========================================
def find_show_by_keyword(df, keyword, context_chars=100):
    keyword_lower = keyword.lower()
    mask = df['overview'].str.contains(keyword_lower, case=False, na=False)
    results = df[mask].copy()

    if len(results) == 0:
        return None, f"Kata '{keyword}' tidak ditemukan di overview manapun."

    def highlight_context(text, kw, chars=context_chars):
        idx = text.lower().find(kw)
        if idx == -1: return text
        start = max(0, idx - chars)
        end = min(len(text), idx + len(kw) + chars)
        return f"...{text[start:end]}..."

    results['overview_snippet'] = results['overview'].apply(lambda x: highlight_context(x, keyword_lower))
    return results[['name', 'overview_snippet', 'vote_average', 'vote_count']], None

# ==========================================
# 4. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="📺 TV Show Recommender", page_icon="📺", layout="wide")
st.title("Sistem Rekomendasi TV Show (Hybrid)")
st.caption("Item-Based CF + Content-Based (TF-IDF) | Weighted Hybrid")

# Load Data & Compute Similarities (Cached)
with st.spinner("Memuat dataset & membangun matriks kemiripan..."):
    df = load_and_preprocess_data()
    cb_sim, cf_sim = build_similarity_matrices(df)  # ✅ Pastikan baris ini tepat seperti ini

# Sidebar Controls
st.sidebar.header("⚙️ Pengaturan Bobot")
cf_w = st.sidebar.slider("Bobot Collaborative Filtering (CF)", 0.0, 1.0, 0.2, 0.1)
cb_w = st.sidebar.slider("Bobot Content-Based Filtering (CB)", 0.0, 1.0, 0.8, 0.1)
top_n = st.sidebar.slider("Jumlah Rekomendasi", 5, 20, 10)

# Normalize weights to sum = 1.0
total_w = cf_w + cb_w
cf_norm = cf_w / total_w if total_w > 0 else 0.5
cb_norm = cb_w / total_w if total_w > 0 else 0.5

# Main UI
tab1, tab2 = st.tabs(["🎯 Rekomendasi Berdasarkan Judul", "🔍 Pencarian Keyword"])

with tab1:
    show_options = sorted(df["name"].tolist())
    selected_show = st.selectbox("Pilih TV Show:", show_options)
    
    if st.button("🚀 Generate Rekomendasi", type="primary", use_container_width=True):
        result, error = get_hybrid_recommendations(df, cb_sim, cf_sim, selected_show, cf_norm, cb_norm, top_n)
        if error:
            st.error(error)
        else:
            st.success(f"✅ Top {top_n} rekomendasi untuk *{selected_show}*")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(result, use_container_width=True, hide_index=True)
            with col2:
                st.metric("Avg Rating", f"{result['vote_average'].mean():.2f}")
                st.metric("Avg Popularity", f"{result['popularity'].mean():.1f}")
            st.info(f"⚖️ Bobot aktif: CF {cf_norm:.0%} | CB {cb_norm:.0%}")

with tab2:
    keyword = st.text_input("Masukkan kata kunci (contoh: 'lawyer', 'space', 'family'):")
    if keyword:
        results, error = find_show_by_keyword(df, keyword)
        if error:
            st.warning(error)
        else:
            st.success(f"🔍 Ditemukan `{len(results)}` show mengandung *'{keyword}'*")
            st.dataframe(results, use_container_width=True, hide_index=True)