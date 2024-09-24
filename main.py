import os
import logging
from download_unzip_data import DataDownloader
from preprocessing_abstracts import AbstractPreprocessor
from vectorize_and_clustering import ClusteringForAbstracts

# Define the json data file path
data_file = 'data/all_data.json'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()  # Optional: log to console as well
    ]
)

# Get the logger for the main script
logger = logging.getLogger(__name__)


def log_program_start():
    """Log a special message indicating the program has started again."""
    logger.info('='*50)
    logger.info('Main script start')
    logger.info('='*50 + '\n')


def main():
    """Main function to orchestrate the download, preprocessing, clustering, and word counting."""
    log_program_start()  # Log program start

    # Check if the data file exists; if not, download, unzip the data and create the json file
    if not os.path.isfile(data_file):
        data_downloader = DataDownloader()
        data_downloader.automate_download_and_unzip()

    # Preprocess the abstracts
    preprocessor = AbstractPreprocessor(json_file=data_file)
    preprocessed_abstracts = preprocessor.preprocess_abstracts()

    # Cluster the preprocessed abstracts
    clustering_algorithm = ClusteringForAbstracts(n_features_tfid=1000, n_components_truncatesvd=100)
    cluster_solution = clustering_algorithm.get_clusters(preprocessed_abstracts=preprocessed_abstracts)
    cluster_solution.generate_pickle_file()
    cluster_solution.generate_graph()
    cluster_solution.generate_pdf()


if __name__ == '__main__':
    main()
