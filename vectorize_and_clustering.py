from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import logging
from solution_builder import ClusterSolution

# Get the logger for this module
logger = logging.getLogger(__name__)


class ClusteringForAbstracts:
    """
    A class for clustering research abstracts using TF-IDF, dimensionality reduction,
    and k-means clustering with silhouette score analysis.

    Attributes:
        n_alternatives: Number of alternative clustering solutions to evaluate.
        n_features_tfid: Number of features for TF-IDF vectorization.
        n_components_truncatesvd: Number of components for TruncatedSVD.
        n_components_tsne: Number of components for t-SNE dimensionality reduction.
        image_folder: Directory for saving cluster visualizations.
        clusters_for_test: List of cluster sizes to evaluate using silhouette scores.
    """
    def __init__(self,
                 n_alternatives: int = 3,
                 n_features_tfid: int = 1000,
                 n_components_truncatesvd: int = 100,
                 n_components_tsne: int = 2):
        """
        Initializes the clustering class with the specified parameters.

        Args:
            n_alternatives: Number of top alternative clustering solutions to save (default 3).
            n_features_tfid: Number of features to keep from TF-IDF (default 1000).
            n_components_truncatesvd: Number of components for dimensionality reduction with SVD (default 100).
            n_components_tsne: Number of dimensions for t-SNE visualization (default 2).
        """
        self.n_alternatives = n_alternatives
        self.n_features_tfid = n_features_tfid
        self.n_components_truncatesvd = n_components_truncatesvd
        self.n_components_tsne = n_components_tsne

        self.clusters_for_test = [4, 5, 6, 7, 8, 9, 10]
        if self.n_alternatives > len(self.clusters_for_test):
            self.n_alternatives = 3

    def _get_biggest_n_scores(self, x_data: list, preprocessed_abstracts: list) -> list:
        """
        Computes silhouette scores for different cluster sizes and returns top n results.

        Args:
            x_data: A List of Dimensionality-reduced data.
            preprocessed_abstracts: List of preprocessed research abstracts.

        Returns:
            dict: A dictionary of top n clustering solutions sorted by silhouette score.
        """
        solution_list = []
        for n_clusters in self.clusters_for_test:
            solution_instance = ClusterSolution(KMeans(n_clusters=n_clusters), algorithm_name="kmeans")
            solution_instance.build_cluster_solution(x_data, preprocessed_abstracts)
            solution_list.append(solution_instance)

        top_n_objects = sorted(solution_list,
                               key=lambda obj: obj.get_evaluation_score(),
                               reverse=True)[:self.n_alternatives]
        return top_n_objects

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

        try:
            # Step 1: TF-IDF vectorization
            vectorizer = TfidfVectorizer(max_features=self.n_features_tfid)
            tfidf_matrix = vectorizer.fit_transform(preprocessed_abstracts)

            # Step 2: Dimensionality reduction with TruncatedSVD and Normalization
            svd = TruncatedSVD(n_components=self.n_components_truncatesvd)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)
            x_data = lsa.fit_transform(tfidf_matrix)

            # Step 3: t-SNE for dimensionality reduction to 2D/3D space
            logging.info(f"Applying TSNE with {self.n_components_tsne} components.")
            x_data = TSNE(n_components=self.n_components_tsne).fit_transform(x_data)

            # Step 4: Get the biggest "n" score for the num_cluster parameter
            kmeans_options = self._get_biggest_n_scores(x_data, preprocessed_abstracts)

            return kmeans_options

        except Exception as e:
            logging.error(f"An error occurred during clustering: {e}")
            raise
