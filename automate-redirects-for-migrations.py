# redirect_matchmaker_streamlit.py

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Title
st.title("Automated Redirect Matchmaker")

# File upload
uploaded_origin = st.file_uploader("Upload origin.csv", type="csv")
uploaded_destination = st.file_uploader("Upload destination.csv", type="csv")

def process_and_match(origin_df, destination_df, selected_columns):
    # Combine selected columns into a single text column for vectorization
    origin_df['combined_text'] = origin_df[list(selected_columns)].fillna('').apply(lambda x: ' '.join(x), axis=1)
    destination_df['combined_text'] = destination_df[list(selected_columns)].fillna('').apply(lambda x: ' '.join(x), axis=1)

    # Use a pre-trained model for embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Vectorize the combined text
    origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
    destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

    # Create a FAISS index
    dimension = origin_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(destination_embeddings.astype('float32'))

    # Perform the search for the nearest neighbors
    D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)

    # Calculate similarity score
    similarity_scores = 1 - (D / np.max(D))

    # Create the output DataFrame
    matches_df = pd.DataFrame({
        'origin_url': origin_df['Address'],
        'matched_url': destination_df['Address'].iloc[I.flatten()].values,
        'similarity_score': np.round(similarity_scores.flatten(), 4)
    })

    return matches_df

if uploaded_origin is not None and uploaded_destination is not None:
    origin_df = pd.read_csv(uploaded_origin)
    destination_df = pd.read_csv(uploaded_destination)

    # Find common columns
    common_columns = list(set(origin_df.columns) & set(destination_df.columns))
    
    # Column selection
    selected_columns = st.multiselect("Select columns for matching", common_columns)

    if st.button("Start Matching"):
        if selected_columns:
            result_df = process_and_match(origin_df, destination_df, selected_columns)
            st.write(result_df)
            st.download_button("Download Matches", result_df.to_csv(index=False), file_name="output.csv")
        else:
            st.warning("Please select at least one column to continue.")
