# Automate Redirect URL Matching for Site Migrations ⚡️🔀
[By Daniel Emery]([url](https://www.linkedin.com/in/dpe1/))

---

### 👉🏼 What It Is
This is a Python tool for Streamlit that automates redirect mappings during site migrations by matching URLs from an old site to a new site based on content similarity. Users can interactively choose the relevant columns from their CSV data for URL matching through a web interface.

### 👉🏼 What It's Made With:
- `faiss-cpu`: A library for efficient similarity search and clustering of dense vectors.
- `sentence-transformers`: A Python framework for state-of-the-art sentence, text, and image embeddings.
- `pandas`: An open-source data manipulation and analysis library.
- `streamlit`: An open-source app framework for Machine Learning and Data Science projects.

### 👉🏼 How You Use It:
1. Prepare `origin.csv` and `destination.csv` with the page URL in the first column, followed by titles, meta descriptions, and headings. Remove unwanted URLs and duplicates.
2. Run the Streamlit app using the Redirect Matchmaker Script.
3. Upload the origin.csv and destination.csv files through the Streamlit interface.
4. Select columns for matching using the interactive Streamlit widgets.
5. Click "Run" to initiate the matching process.
6. The app will process the data and create `output.csv` with matched URLs and a similarity score.
7. Review and manually correct any inaccuracies in `output.csv` directly in the app.
8. Download `output.csv` directly from the Streamlit app interface.

*Note: Ensure only URLs with a 200 status code and without UTM parameters, etc., are used.*

---
