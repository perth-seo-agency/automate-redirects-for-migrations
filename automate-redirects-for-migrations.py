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

# Function to process and match URLs based on similarity
def process_and_match(origin_df, destination_df, selected_similarity_columns):
    if not selected_similarity_columns:
        raise ValueError("No columns selected for similarity matching.")

    # Validate that selected columns exist in both dataframes
    valid_columns = [col for col in selected_similarity_columns if col in origin_df.columns and col in destination_df.columns]
    if not valid_columns:
        raise ValueError("Selected columns do not exist in both origin and destination dataframes.")

    # Combine selected columns into a single text column for vectorization
    origin_df['combined_text'] = origin_df[valid_columns].fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1)
    destination_df['combined_text'] = destination_df[valid_columns].fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1)

    # ... [rest of the function remains the same]

# Function to find exact matches based on user-selected columns
def find_exact_matches(origin_df, destination_df, exact_match_columns):
    # ... [function remains the same]

# Main function to control the flow
def main():
    if uploaded_origin is not None and uploaded_destination is not None:
        # Load the dataframes
        origin_df = pd.read_csv(uploaded_origin)
        destination_df = pd.read_csv(uploaded_destination)

        # ... [column selection and button logic remains the same]

        if st.button("Match URLs"):
            try:
                with st.spinner('Processing...'):
                    # Find exact matches first
                    origin_df, exact_matches_df = find_exact_matches(origin_df, destination_df, exact_match_columns)

                    # Process and match URLs based on similarity for non-exact matches
                    similarity_matches_df = process_and_match(origin_df, destination_df, selected_similarity_columns)

                    # Combine exact and similarity-based matches
                    final_matches_df = pd.concat([exact_matches_df, similarity_matches_df])

                    st.write(final_matches_df)

                    # Download link for the matches
                    st.download_button(
                        label="Download Matches as CSV",
                        data=final_matches_df.to_csv(index=False).encode('utf-8'),
                        file_name='matched_urls.csv',
                        mime='text/csv',
                    )
            except ValueError as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
