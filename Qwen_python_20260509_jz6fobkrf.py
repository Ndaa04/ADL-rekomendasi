import streamlit as st
import pandas as pd
from data_loader import load_and_preprocess_data
from recommender import compute_content_similarity, compute_collaborative_similarity, get_hybrid_recommendations
from utils import find_show_by_keyword

st.set_page_config(page_title="📺 TV Show Recommender", page_icon="📺", layout="wide")
st.title("Sistem Rekomendasi TV Show (Hybrid)")
st.caption("Metode: Item-Based CF + Content-Based (TF-IDF) dengan Weighted Hybrid")

# 1. Load Data
with st.spinner("Memuat dan membersihkan dataset..."):
    df = load_and_preprocess_data()

# 2. Cache Similarity Matrices (Hanya dihitung sekali per sesi/server)
@st.cache_resource
def load_similarities(dataframe):
    cb = compute_content_similarity(dataframe)
    cf = compute_collaborative_similarity(dataframe)
    return cb, cf

with st.spinner("Membangun matriks kemiripan (CF & CB)..."):
    cb