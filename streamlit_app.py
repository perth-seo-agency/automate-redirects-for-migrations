import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from logger_setup import setup_logger
import csv

# Initialize logger
logger = setup_logger()
logger.info("Main script started.")

# Title
st.title("Automated Redirect Matchmaker v4.0.2")

# File upload
uploaded_origin = st.file_uploader("Upload origin.csv", type="csv")
uploaded_destination = st.file_uploader("Upload destination.csv", type="csv")

# Function to process and match URLs based on similarity
def process_and_match(origin_df, destination_df, selected_similarity_columns, excluded_urls):
   if not selected_similarity_columns:
       raise ValueError("No columns selected for similarity matching.")

   # Validate that selected columns exist in both dataframes
   valid_columns = [col for col in selected_similarity_columns if col in origin_df.columns and col in destination_df.columns]
   if not valid_columns:
       raise ValueError("Selected columns do not exist in both origin and destination dataframes.")

   # Exclude URLs that were already matched using exact matching
   origin_df = origin_df[~origin_df['Address'].isin(excluded_urls)]

   # Combine selected columns into a single text column for vectorization
   origin_df['combined_text'] = origin_df[valid_columns].fillna('').astype(str).apply(' '.join, axis=1)
   destination_df['combined_text'] = destination_df[valid_columns].fillna('').astype(str).apply(' '.join, axis=1)

   # Use a pre-trained model for embedding
   model = SentenceTransformer('all-MiniLM-L6-v2')

   # Vectorize the combined text
   origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
   destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

   # Normalize embeddings to unit length for cosine similarity
   origin_embeddings = origin_embeddings / np.linalg.norm(origin_embeddings, axis=1, keepdims=True)
   destination_embeddings = destination_embeddings / np.linalg.norm(destination_embeddings, axis=1, keepdims=True)

   # Create a FAISS index for Inner Product (cosine similarity)
   dimension = origin_embeddings.shape[1]  # The dimension of vectors
   faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
   faiss_index.add(destination_embeddings.astype('float32'))  # Add destination vectors to the index

   # Perform the search for the nearest neighbors
   D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)  # k=1 finds the closest match

   # Convert distances to similarity scores (1 - distance for L2, direct output for IP)
   similarity_scores = D.flatten()

   # Create the output DataFrame
   matches_df = pd.DataFrame({
       'origin_url': origin_df['Address'],
       'matched_url': destination_df['Address'].iloc[I.flatten()].values,
       'similarity_score': np.round(similarity_scores, 4)  # Rounded for better readability
   })

   return matches_df


# Function to find exact matches based on user-selected columns
def find_exact_matches(origin_df, destination_df, exact_match_columns):
   exact_matches_list = []
   exact_matched_urls = []  # List to store URLs that were matched exactly
   for col in exact_match_columns:
       if col == 'Address':  # Handle Address column separately
           matched_rows = origin_df[origin_df[col] == destination_df[col]]
           for _, row in matched_rows.iterrows():
               exact_matched_urls.append(row['Address'])  # Add the URL to the list of matched URLs
               exact_matches_list.append({
                   'origin_url': row['Address'],
                   'matched_url': destination_df[destination_df[col] == row[col]].iloc[0]['Address'],
                   'similarity_score': 1.0
               })
           origin_df = origin_df[origin_df[col] != destination_df[col]]
       else:
           matched_rows = origin_df[origin_df[col].isin(destination_df[col])]
           for _, row in matched_rows.iterrows():
               exact_matched_urls.append(row['Address'])  # Add the URL to the list of matched URLs
               exact_matches_list.append({
                   'origin_url': row['Address'],
                   'matched_url': destination_df[destination_df[col] == row[col]].iloc[0]['Address'],
                   'similarity_score': 1.0
               })
           origin_df = origin_df[~origin_df[col].isin(destination_df[col])]

   exact_matches_df = pd.DataFrame(exact_matches_list)
   return origin_df, exact_matches_df, exact_matched_urls


# Main function to control the flow
def main():
   if uploaded_origin is not None and uploaded_destination is not None:
       # Load the dataframes
       origin_df = pd.read_csv(uploaded_origin)
       destination_df = pd.read_csv(uploaded_destination)

       # Exact match column selection
       st.write("Select columns for exact match pairing:")
       exact_match_columns = st.multiselect('Exact Match Columns', options=list(origin_df.columns))

       # Similarity match column selection
       st.write("Select the columns you want to include for similarity matching:")
       selected_similarity_columns = st.multiselect('Similarity Match Columns', options=list(origin_df.columns))

       if st.button("Match URLs"):
           try:
               with st.spinner('Processing...'):
                   # Find exact matches first
                   origin_df, exact_matches_df, exact_matched_urls = find_exact_matches(origin_df, destination_df,
                                                                                     exact_match_columns)

                   # Process and match URLs based on similarity for non-exact matches
                   similarity_matches_df = process_and_match(origin_df, destination_df, selected_similarity_columns,
                                                             exact_matched_urls)

                   # Combine exact and similarity-based matches
                   final_matches_df = pd.concat([exact_matches_df, similarity_matches_df])

                   # Ensure each URL is only included once in the final matches
                   final_matches_df = final_matches_df.drop_duplicates(subset=['origin_url'], keep='first')

                   # Create a list of URLs that were not matched
                   no_match_urls = list(set(origin_df['Address']) - set(final_matches_df['origin_url']))

                   # Output unmatched URLs to a CSV file
                   with open('no-matches.csv', 'w', newline='') as csvfile:
                       csvwriter = csv.writer(csvfile)
                       csvwriter.writerow(['Address'])
                       for url in no_match_urls:
                           csvwriter.writerow([url])

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