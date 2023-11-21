# redirect_matchmaker_streamlit.py
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("Automated Redirect Matchmaker v3.0.4")

# Initialize Sentence Transformer Model
model_name = 'all-MiniLM-L6-v2'  # Example model, replace with the model you are using
model = SentenceTransformer(model_name)

@st.cache
def calculate_similarity(text1, text2):
    embeddings1 = model.encode([text1], convert_to_tensor=True)
    embeddings2 = model.encode([text2], convert_to_tensor=True)
    cosine_scores = cosine_similarity(embeddings1, embeddings2)
    return cosine_scores[0][0]

def process_files(origin_df, destination_df):
    processed_pairs = set()
    results = []

    for _, row_origin in origin_df.iterrows():
        origin_url = row_origin['url']  # Replace 'url' with the appropriate column name
        for _, row_destination in destination_df.iterrows():
            destination_url = row_destination['url']  # Replace 'url' with the appropriate column name
            pair = (origin_url, destination_url)

            if pair not in processed_pairs:
                similarity_score = calculate_similarity(origin_url, destination_url)
                results.append({'origin_url': origin_url, 'matched_url': destination_url, 'similarity_score': similarity_score})
                processed_pairs.add(pair)

    return pd.DataFrame(results)

# Streamlit interface
# st.title("Automated Redirect Matching for Site Migrations")

# File upload
st.sidebar.title("Upload Files")
origin_file = st.sidebar.file_uploader("Upload Origin URLs", type=['csv'])
destination_file = st.sidebar.file_uploader("Upload Destination URLs", type=['csv'])

if origin_file and destination_file:
    origin_df = pd.read_csv(origin_file)
    destination_df = pd.read_csv(destination_file)

    if st.button("Run Matching"):
        with st.spinner("Processing..."):
            result_df = process_files(origin_df, destination_df)
            st.write(result_df)
            result_df.to_csv('matched_urls.csv', index=False)
            st.sidebar.download_button(
                label="Download Matched URLs as CSV",
                data=result_df.to_csv().encode('utf-8'),
                file_name='matched_urls.csv',
                mime='text/csv',
            )
