# Abstracts Clustering Project

## Overview

This project automates the process of clustering research abstracts. It covers downloading and preprocessing text data, applying TF-IDF vectorization, dimensionality reduction, and clustering algorithms, and finally generating visualizations and reports of the clusters.

The solution helps in analyzing the abstracts by extracting the most relevant topics and words for each cluster, using techniques like:
- **TF-IDF Vectorization** 
- **Dimensionality Reduction (SVD and t-SNE)**
- **Clustering (K-Means)** 
- **Silhouette Score for cluster evaluation**

## Features

- **Data Downloading and Extraction**: Automates data fetching and extraction from a specified source.
- **Preprocessing**: Cleans and preprocesses the text data for clustering.
- **Clustering**: Uses DBScan to cluster abstracts based on TF-IDF and dimensionality-reduction techniques.
- **Word Analysis**: Extracts the top words per cluster.
- **PDF and Graph Generation**: Creates visual and textual reports for each clustering solution.

## Project Structure

```bash
abstracts_clustering/
│
├── data/                           # Directory for storing the data
├── output/                         # Directory for saving generated graphs, PDFs, and pickle files
├── prototype/prototype.ipynb       # Original prototype created as a Interactive Python Notebook file 
├── app.log                         # Log file
├── main.py                         # Main orchestration script
├── download_unzip_data.py          # Handles data download and extraction
├── preprocessing_abstracts.py      # Abstracts preprocessing class
├── vectorize_and_clustering.py     # Handles vectorization, dimensionality reduction, and clustering
├── solution_builder.py             # Builds clustering solutions and evaluates them
├── get_common_words_from_clusters.py # Extracts top words from clusters
└── requirements.txt                # Python dependencies
```

## Getting Started
### Prerequisites
To run this project, ensure you have the following installed:

- Python 3.8 or higher
- Required libraries from `requirements.txt`

You can install the necessary libraries by running:

```bash
pip install -r requirements.txt
```

### Running the Project
Simply run the main script:
```bash
python main.py
```
1) **Download the Data and Preprocess Abstracts.**
The project will automatically download and preprocess the data if not already available.
2) **Execute the Clustering.**
3) **Results.** After running the script, you will find:
    - **Pickle files** for saved models.
    - **Scatter plot images** visualizing clustered abstracts.
    - **PDF reports** containing cluster-specific data and top words.

## Customization
You can tweak the clustering process by adjusting the following parameters in 
[vectorize_and_clustering.py](https://github.com/Lalo-Rodriguez/abstracts_clustering/blob/main/vectorize_and_clustering.py):
- **n_features_tfid:** Number of features for TF-IDF vectorization.
- **n_components_truncatesvd:** Number of components for dimensionality reduction with SVD.
- **n_components_tsne:** Number of dimensions for t-SNE visualization.
- **epsilon_parameter:** List of cluster sizes to evaluate.

## Logging 
All actions and errors are logged in the `app.log` file. 
This helps track the steps and diagnose issues if the script doesn't run as expected.

## Future Improvements
- Improve PDF formatting and visualizations for large datasets.