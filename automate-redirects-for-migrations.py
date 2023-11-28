import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Function Definitions

def load_data(file):
    return pd.read_csv(file)

def find_exact_matches(origin_df, destination_df, cols):
    matched = origin_df[origin_df[cols].apply(tuple, 1).isin(destination_df[cols].apply(tuple, 1))]
    remaining_origin = origin_df.drop(matched.index)
    return matched, remaining_origin

def process_and_match(origin_df, destination_df, cols):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Vectorize origin
    origin_vectors = model.encode(origin_df[cols].astype(str).agg(' '.join, axis=1).tolist(), show_progress_bar=True)

    # Vectorize destination
    destination_vectors = model.encode(destination_df[cols].astype(str).agg(' '.join, axis=1).tolist(), show_progress_bar=True)

    # FAISS
    index = faiss.IndexFlatL2(origin_vectors.shape[1])
    index.add(np.array(destination_vectors).astype('float32'))

    # Search
    _, indices = index.search(np.array(origin_vectors).astype('float32'), 1)
    destination_df['matched_index'] = indices[:,0]
    destination_df['similarity'] = 1  # Placeholder for similarity score

    # Combine
    matched_df = pd.concat([
        origin_df.reset_index(drop=True),
        destination_df.iloc[destination_df['matched_index']].reset_index(drop=True)
    ], axis=1)

    return matched_df

# Streamlit App

st.title("Automate Redirect URL Matching for Site Migrations")

uploaded_origin = st.file_uploader("Choose a file for origin URLs")
uploaded_destination = st.file_uploader("Choose a file for destination URLs")

if uploaded_origin is not None and uploaded_destination is not None:
    origin_df = load_data(uploaded_origin)
    destination_df = load_data(uploaded_destination)

    # Select columns for matching
    st.write("Select columns for matching:")
    selected_cols = st.multiselect("Columns", origin_df.columns.tolist(), default=origin_df.columns.tolist())

    if st.button("Run"):
        # Exact matching
        exact_matches, remaining_origin = find_exact_matches(origin_df, destination_df, selected_cols)

        # Process and match remaining URLs
        similarity_matches = process_and_match(remaining_origin, destination_df, selected_cols)

        # Combine results
        final_matches_df = pd.concat([exact_matches, similarity_matches], ignore_index=True)
        
        st.write("Matching Complete")
        st.dataframe(final_matches_df)
        st.download_button("Download Matched URLs", final_matches_df.to_csv().encode('utf-8'), "matched_urls.csv", "text/csv", key='download-csv')
