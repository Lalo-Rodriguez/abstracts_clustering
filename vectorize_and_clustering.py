from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import logging
import numpy as np
from solution_builder import ClusterSolution

# Get the logger for this module
logger = logging.getLogger(__name__)


class ClusteringForAbstracts:
    """
    A class for clustering research abstracts using TF-IDF, dimensionality reduction,
    and k-means clustering with silhouette score analysis.

    Attributes:
        n_features_tfid: Number of features for TF-IDF vectorization.
        n_components_truncatesvd: Number of components for TruncatedSVD.
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

        self.k_min = 4
        self.k_max = 20
        self.n_solutions = 3

    def _elbow_method(self, data: np.array) -> int:
        """
        Private function that returns an int representing the k parameter (Of K-Means algorithm)
        where the elbow_point is located.

        Args:
            data: Data from the research abstracts after the tf-idf vectorization,
            truncated SVD and Normalization processes.

        Returns:
             An int representing the k parameter (Of K-Means algorithm) where the elbow_point is located.
        """
        inertia = []

        for k in range(self.k_min, self.k_max + 1):
            # Apply KMeans
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)

            # Store inertia (within-cluster sum of squares)
            inertia.append(kmeans.inertia_)

        # Generate the range of k values
        k_values = list(range(self.k_min, self.k_max + 1))

        # Automatically detect the elbow using KneeLocator
        kn = KneeLocator(k_values, inertia, curve='convex', direction='decreasing')
        elbow_point = kn.knee

        return elbow_point

    def _silhouette_score_for_range(self, data: np.array, elbow_point: int) -> dict:
        """
        Private function that returns the n-higher silhouette score solutions near the
        elbow_point based on the self.n_solutions parameter defined in the vectorize_and_clustering
        builder.

        Args:
            data: Data from the research abstracts after the tf-idf vectorization,
            truncated SVD and Normalization processes.
            elbow_point: An int representing the k parameter (Of K-Means algorithm) where the
            elbow_point is located. Used as the starting point of the local search.

        Returns:
             A dictionary with the sorted best models got after near the elbow_point. Using the
             silhouette_score.
        """
        k_min = max(elbow_point - self.n_solutions, 2)  # Ensure k_min is at least 2
        k_max = elbow_point + self.n_solutions

        # Dictionary to store model details for each k
        kmeans_results = {}
        # List to store silhouette scores for each k
        silhouette_scores = []

        for k in range(k_min, k_max + 1):
            # Apply KMeans
            kmeans = KMeans(n_clusters=k)
            cluster_labels = kmeans.fit_predict(data)

            # Calculate silhouette score
            score = silhouette_score(data, cluster_labels)

            # Store the model, labels, and score in kmeans_results dictionary
            kmeans_results[k] = {
                "model": kmeans,
                "labels": cluster_labels,
                "silhouette_score": score
            }
            silhouette_scores.append((k, score))

        sorted_scores = sorted(silhouette_scores, key=lambda x: x[1], reverse=True)
        # Select the top N k values
        best_k_values = sorted_scores[:self.n_solutions]

        # Create a dictionary to store the top N configurations
        best_kmeans = {k: kmeans_results[k] for k, _ in best_k_values}

        return best_kmeans

    def _find_best_kmeans_k(self, data: np.array) -> dict:
        """
        Private function that returns the n-higher silhouette score solutions near the
        elbow_point based on the self.n_solutions parameter defined in the vectorize_and_clustering
        builder.

        Args:
            data: Data from the research abstracts after the tf-idf vectorization,
            truncated SVD and Normalization processes.

        Returns:
             A dictionary with the sorted best models got after near the elbow_point. Using the
             silhouette_score.
        """
        elbow_point = self._elbow_method(data)
        best_kmeans = self._silhouette_score_for_range(data, elbow_point)

        return best_kmeans

    def get_clusters(self, preprocessed_abstracts: list) -> list:
        """
        Executes the clustering process: TF-IDF vectorization, dimensionality reduction,
        KMeans clustering, and returns clustering results with silhouette scores.

        Args:
            preprocessed_abstracts: List of preprocessed research abstracts.

        Returns:
            dict: Dictionary of clustering results, including silhouette scores,
                  cluster labels, and visualization of clusters.
        """
        logging.info('Initializing the clustering algorithm.')

        solution_list = []
        try:
            # Step 1: TF-IDF vectorization
            tfidf_vectorizer = TfidfVectorizer(max_features=self.n_features_tfid)
            tfidf_data = tfidf_vectorizer.fit_transform(preprocessed_abstracts)

            # Step 2: Dimensionality reduction with TruncatedSVD and Normalization
            svd = TruncatedSVD(n_components=self.n_components_truncatesvd)
            normalizer = Normalizer(copy=False)
            svd_and_normalizer = make_pipeline(svd, normalizer)
            x_data = svd_and_normalizer.fit_transform(tfidf_data)

            # Step 3: Get n-solutions based on the elbow_method and silhouette_score_for_range
            kmeans_options = self._find_best_kmeans_k(x_data)

            # Step 4: Build the solutions based on the k-means options
            for key, variables in kmeans_options.items():
                models_to_save = {
                    'tfidf_vectorizer': tfidf_vectorizer,
                    'svd_and_normalizer': svd_and_normalizer,
                    'cluster_algorithm': variables['model']
                }

                solution_instance = ClusterSolution(algorithm_name=f'kmeans_{key}_clusters',
                                                    abstracts=preprocessed_abstracts,
                                                    normalized_x_data=x_data,
                                                    labels=variables['labels'],
                                                    models_used=models_to_save,
                                                    n_components_tsne=2)
                solution_list.append(solution_instance)

            return solution_list

        except Exception as e:
            logging.error(f"An error occurred during clustering: {e}")
            raise
