from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import logging

# Get the logger for this module
logger = logging.getLogger(__name__)


def _convert_numpy_to_int(obj):
    if isinstance(obj, dict):
        return {_convert_numpy_to_int(k): _convert_numpy_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_int(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_to_int(i) for i in obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


class WordCounter:
    """
    A class to find the most frequent words in clusters of text data.

    Attributes:
    -----------
    num_top_words : int
        The number of top words to extract from each cluster.

    Methods:
    --------
    get_top_words_each_cluster(cluster_dict: dict) -> dict:
        Returns a dictionary with the top words for each cluster.
    """
    def __init__(self, num_top_words: int = 20):
        """
        Initializes the WordCounter with the specified number of top words to extract.

        Parameters:
        -----------
        num_top_words : int, optional
            Number of top words to extract from each cluster. Default is 20.
        """
        self.num_top_words = num_top_words

    def _extract_top_words(self, clustered_texts: list) -> list:
        """
        Private method to extract the top words and their counts from the provided texts.

        Parameters:
        -----------
        clustered_texts : list of str
            A list of text documents from a single cluster.

        Returns:
        --------
        list of tuples:
            A list of tuples where each tuple contains a word and its frequency in the cluster.
        """
        if not clustered_texts:
            logging.warning("Empty text list provided for word extraction.")
            return []

        vectorizer = CountVectorizer(stop_words='english')
        x_data = vectorizer.fit_transform(clustered_texts)
        word_counts = np.asarray(x_data.sum(axis=0)).flatten()  # Get word counts across all texts
        vocab = vectorizer.get_feature_names_out()  # List of words (vocabulary)

        # Combine words with their corresponding counts
        word_freq = [(word, word_counts[idx]) for idx, word in enumerate(vocab)]

        # Sort and extract the top N words
        top_words = sorted(word_freq, key=lambda x: x[1], reverse=True)[:self.num_top_words]
        return top_words

    def get_top_words_each_cluster(self, cluster_dict: dict) -> dict:
        """
        Extracts the most frequent words for each cluster of texts.

        Parameters:
        -----------
        cluster_dict : dict
            A dictionary where keys are cluster IDs and values are lists of texts.

        Returns:
        --------
        dict:
            A dictionary where each key is a cluster ID, and the value is a list of the top words and their frequencies.
        """
        if not cluster_dict:
            logging.warning("Empty cluster dictionary provided.")
            return {}

        words_dict = {}
        for key, values in cluster_dict.items():
            if key not in words_dict:
                words_dict[key] = []
            words_dict[key] = self._extract_top_words(values)

        converted_data = _convert_numpy_to_int(words_dict)

        return converted_data
