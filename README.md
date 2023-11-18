# Automate Redirect URL Matchming for Site Migrations ⚡️🔀
[By Daniel Emery]([url](https://www.linkedin.com/in/dpe1/))

---

</br>

### 👉🏼 What It Is
This is a Python tool for Google Colab that automates redirect mappings during site migrations by matching URLs from an old site to a new site based on content similarity. Users can interactively choose the relevant columns from their CSV data for URL matching.

### 👉🏼 What It's Made With:
- `faiss-cpu`: A library for efficient similarity search and clustering of dense vectors.
- `sentence-transformers`: A Python framework for state-of-the-art sentence, text, and image embeddings.
- `pandas`: An open-source data manipulation and analysis library.
- `ipywidgets`: An interactive widget library for Jupyter notebooks.

### 👉🏼 How You Use It:
1. Prepare `origin.csv` and `destination.csv` with the page URL in the first column, followed by titles, meta descriptions, and headings. Remove unwanted URLs and duplicates.
2. Open the Redirect Matchmaker Script in Google Colab.
3. Install libraries by running Cell #1.
4. Upload the origin.csv and destination.csv files when prompted by Cell #2.
5. Select columns for matching in Cell #3 using the displayed widget.
6. Click "Let's Go!" to initiate the matching process.
7. Cell #4 will processes the data and creates `output.csv` with matched URLs and a similarity score.
8. Review and manually correct any inaccuracies in `output.csv`.
9. Download `output.csv` automatically from the browser's download directory after the script completion.

*Note: Ensure only URLs with 200 status code and without UTM parameters, etc, are used.*
