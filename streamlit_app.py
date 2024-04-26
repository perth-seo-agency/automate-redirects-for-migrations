import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from logger_setup import setup_logger, logging_decorator
import csv

# Initialize logger
logger = setup_logger()

@logging_decorator
def validate_input_files(origin_df, destination_df):
    required_columns = ['Address']
    for col in required_columns:
        if col not in origin_df.columns or col not in destination_df.columns:
            raise ValueError(f"Required column '{col}' missing in input files.")

@logging_decorator
def process_and_match(origin_df, destination_df, selected_similarity_columns, excluded_urls, similarity_threshold):
    if not selected_similarity_columns:
        raise ValueError("No columns selected for similarity matching.")

    valid_columns = [col for col in selected_similarity_columns if col in origin_df.columns and col in destination_df.columns]
    if not valid_columns:
        raise ValueError("Selected columns do not exist in both origin and destination dataframes.")

    origin_df = origin_df[~origin_df['Address'].isin(excluded_urls)]

    origin_df['combined_text'] = origin_df[valid_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    destination_df['combined_text'] = destination_df[valid_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
    destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

    origin_embeddings = origin_embeddings / np.linalg.norm(origin_embeddings, axis=1, keepdims=True)
    destination_embeddings = destination_embeddings / np.linalg.norm(destination_embeddings, axis=1, keepdims=True)

    dimension = origin_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(destination_embeddings.astype('float32'))

    D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)

    similarity_scores = D.flatten()

    matches_df = pd.DataFrame({
        'origin_url': origin_df['Address'],
        'matched_url': destination_df.iloc[I.flatten()]['Address'].values,
        'similarity_score': np.round(similarity_scores, 4)
    })

    matches_df = matches_df[matches_df['similarity_score'] >= similarity_threshold]

    return matches_df

@logging_decorator
def find_exact_matches(origin_df, destination_df, exact_match_columns):
    exact_matches_list = []
    exact_matched_urls = []
    for col in exact_match_columns:
        if col == 'Address':
            matched_rows = origin_df[origin_df[col].isin(destination_df[col])]
            for _, row in matched_rows.iterrows():
                exact_matched_urls.append(row['Address'])
                exact_matches_list.append({
                    'origin_url': row['Address'],
                    'matched_url': destination_df[destination_df[col] == row[col]].iloc[0]['Address'],
                    'similarity_score': 1.0
                })
            origin_df = origin_df[~origin_df[col].isin(destination_df[col])]
        else:
            matched_rows = origin_df[origin_df[col].isin(destination_df[col])]
            for _, row in matched_rows.iterrows():
                exact_matched_urls.append(row['Address'])
                exact_matches_list.append({
                    'origin_url': row['Address'],
                    'matched_url': destination_df[destination_df[col] == row[col]].iloc[0]['Address'],
                    'similarity_score': 1.0
                })
            origin_df = origin_df[~origin_df[col].isin(destination_df[col])]

    exact_matches_df = pd.DataFrame(exact_matches_list)
    return origin_df, exact_matches_df, exact_matched_urls

@logging_decorator
def display_unmatched_urls(unmatched_urls):
    st.subheader("Unmatched URLs")
    if unmatched_urls:
        st.write(pd.DataFrame({'Unmatched URL': unmatched_urls}))
        st.download_button(
            label="Download Unmatched URLs as CSV",
            data=pd.DataFrame({'Unmatched URL': unmatched_urls}).to_csv(index=False).encode('utf-8'),
            file_name='unmatched_urls.csv',
            mime='text/csv',
        )
    else:
        st.write("All URLs matched successfully.")

@logging_decorator
def main():
    st.title("Automated Redirect Matchmaker v4.0.2")

    uploaded_origin = st.file_uploader("Upload origin.csv", type="csv")
    uploaded_destination = st.file_uploader("Upload destination.csv", type="csv")

    if uploaded_origin is not None and uploaded_destination is not None:
        origin_df = pd.read_csv(uploaded_origin)
        destination_df = pd.read_csv(uploaded_destination)

        try:
            validate_input_files(origin_df, destination_df)

            st.subheader("Exact Match Columns")
            exact_match_columns = st.multiselect('Select columns for exact match pairing:', options=list(origin_df.columns))

            st.subheader("Similarity Match Columns")
            selected_similarity_columns = st.multiselect('Select columns for similarity matching:', options=list(origin_df.columns))

            similarity_threshold = st.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

            if st.button("Match URLs"):
                with st.spinner('Finding exact matches...'):
                    origin_df, exact_matches_df, exact_matched_urls = find_exact_matches(origin_df, destination_df, exact_match_columns)

                with st.spinner('Performing similarity matching...'):
                    similarity_matches_df = process_and_match(origin_df, destination_df, selected_similarity_columns, exact_matched_urls, similarity_threshold)

                final_matches_df = pd.concat([exact_matches_df, similarity_matches_df])
                final_matches_df = final_matches_df.drop_duplicates(subset=['origin_url'], keep='first')

                st.subheader("Matched URLs")
                st.write(final_matches_df)

                st.download_button(
                    label="Download Matches as CSV",
                    data=final_matches_df.to_csv(index=False).encode('utf-8'),
                    file_name='matched_urls.csv',
                    mime='text/csv',
                )

                unmatched_urls = list(set(origin_df['Address']) - set(final_matches_df['origin_url']))
                display_unmatched_urls(unmatched_urls)

        except ValueError as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()