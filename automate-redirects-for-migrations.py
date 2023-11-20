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
    origin_df['combined_text'] = origin_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)
    destination_df['combined_text'] = destination_df[selected_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)

    # Use a pre-trained model for embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Vectorize the combined text
    origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
    destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

    # Create a FAISS index
    dimension = origin_embeddings.shape[1]  # The dimension of vectors
    faiss_index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
    faiss_index.add(destination_embeddings.astype('float32'))  # Add destination vectors to the index

    # Perform the search for the nearest neighbors
    D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)  # k=1 finds the closest match

    # Identify exact matches (where distance is 0)
    exact_matches = D.flatten() == 0

    # Calculate similarity score (1 - normalized distance)
    # We normalize the distances only for non-exact matches
    max_distance = np.max(D[~exact_matches]) if np.max(D[~exact_matches]) > 0 else 1
    similarity_scores = 1 - (D / max_distance)

    # Set similarity score for exact matches to 1
    similarity_scores[exact_matches] = 1

    # Create the output DataFrame
    matches_df = pd.DataFrame({
        'origin_url': origin_df['Address'],
        'matched_url': destination_df['Address'].iloc[I.flatten()].values,
        'similarity_score': np.round(similarity_scores.flatten(), 4)  # Rounded for better readability
    })

    return matches_df

# Main function to control the flow
def main():
    if uploaded_origin is not None and uploaded_destination is not None:
        # Load the dataframes
        origin_df = pd.read_csv(uploaded_origin)
        destination_df = pd.read_csv(uploaded_destination)

        # Column selection
        st.write("Select the columns you want to include for similarity matching:")
        selected_columns = st.multiselect('Columns', options=list(origin_df.columns.intersection(destination_df.columns)))

        if st.button("Match URLs"):
            with st.spinner('Processing...'):
                # Process and match URLs
                matches_df = process_and_match(origin_df, destination_df, selected_columns)
                st.write(matches_df)

                # Download link for the matches
                st.download_button(
                    label="Download Matches as CSV",
                    data=matches_df.to_csv(index=False).encode('utf-8'),
                    file_name='matched_urls.csv',
                    mime='text/csv',
                )

if __name__ == "__main__":
    main()
