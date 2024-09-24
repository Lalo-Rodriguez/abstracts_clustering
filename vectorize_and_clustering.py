from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
import numpy as np
import logging
import os
from solution_builder import ClusterSolution

# Get the logger for this module
logger = logging.getLogger(__name__)


class ClusteringForAbstracts:
    """
    A class for clustering research abstracts using TF-IDF, dimensionality reduction,
    and k-means clustering with silhouette score analysis.

    """
    def __init__(self,
                 n_features_tfid: int = 1000,
                 n_components_truncatesvd: int = 100):
        """
        Initializes the clustering class with the specified parameters.

        Args:
            n_features_tfid: Number of features to keep from TF-IDF (default 1000).
            n_components_truncatesvd: Number of components for dimensionality reduction with SVD (default 100).
        """
        self.n_features_tfid = n_features_tfid
        self.n_components_truncatesvd = n_components_truncatesvd

        self.output_dir = 'output'  # Create folder to save the vectorizer and pipeline as joblib files
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_dbscan_parameters(self, x_data) -> dict:
        neighbors = NearestNeighbors(n_neighbors=5).fit(x_data)
        neigh_dist, neigh_ind = neighbors.kneighbors(x_data)
        sort_neigh_dist = np.sort(neigh_dist, axis=0)
        k_dist = sort_neigh_dist[:, 4]

        kneedle = KneeLocator(x=range(1, len(neigh_dist) + 1), y=k_dist, S=1.0,
                              curve='concave', direction='increasing', online=True)

        parameter_variables = {'epsilon': kneedle.knee_y,
                               'min_samples': 2 * self.n_components_truncatesvd}

        return parameter_variables

    def get_clusters(self, preprocessed_abstracts: list) -> ClusterSolution:
        """
        Executes the clustering process: TF-IDF vectorization, dimensionality reduction,
        DBScan clustering, and returns clustering results.

        Args:
            preprocessed_abstracts: List of preprocessed research abstracts.

        Returns:
            dict: A ClusterSolution object with the data.
        """
        logging.info('Initializing the clustering algorithm.')

        try:
            # Step 1: TF-IDF vectorization
            tfidf_vectorizer = TfidfVectorizer(max_features=self.n_features_tfid)
            tfidf_data = tfidf_vectorizer.fit_transform(preprocessed_abstracts)

            # Step 2: Pipeline for Dimensionality reduction with TruncatedSVD and Normalization
            svd = TruncatedSVD(n_components=self.n_components_truncatesvd)
            normalizer = Normalizer(copy=False)
            svd_and_normalizer = make_pipeline(svd, normalizer)
            x_data = svd_and_normalizer.fit_transform(tfidf_data)

            # Step 3: Calculate parameters for the DBScan clustering
            dbscan_parameters = self._get_dbscan_parameters(x_data)

            # Step 4: DBScan algorithm
            cluster_algorithm = DBSCAN(eps=dbscan_parameters['epsilon'],
                                       min_samples=dbscan_parameters['min_samples'])
            predictions = cluster_algorithm.fit_predict(x_data)

            # Step 5: Create dictionary with models
            models_to_save = {
                'tfidf_vectorizer': tfidf_vectorizer,
                'svd_and_normalizer': svd_and_normalizer,
                'cluster_algorithm': cluster_algorithm
            }

            solution = ClusterSolution(algorithm_name="DBScan",
                                       abstracts=preprocessed_abstracts,
                                       normalized_x_data=x_data,
                                       labels=predictions,
                                       models_used=models_to_save,
                                       n_components_tsne=2)
            return solution

        except Exception as e:
            logging.error(f"An error occurred during clustering: {e}")
            raise
