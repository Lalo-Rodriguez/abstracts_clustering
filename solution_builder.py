from sklearn.metrics import silhouette_score
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import logging
import os
from get_common_words_from_clusters import WordCounter

logger = logging.getLogger(__name__)


class ClusterSolution:
    def __init__(self, clusterer, algorithm_name: str):
        self.clusterer = clusterer
        self.algorithm_name = algorithm_name

        self.output_dir = 'output'
        os.makedirs(self.output_dir, exist_ok=True)
        self.img_file = f'{self.output_dir}/{self.algorithm_name}_{self.clusterer.n_clusters}.png'
        self.pdf_file = f'{self.output_dir}/{self.algorithm_name}_{self.clusterer.n_clusters}.pdf'
        self.pickle_file = f'{self.output_dir}/{self.algorithm_name}_{self.clusterer.n_clusters}.pkl'

        self._evaluation_score = None
        self._cluster_labels = None
        self._centers = None
        self._clustered_data = None
        self._preprocessed_abstracts = None
        self._x_data = None
        self._word_dictionary = None

    def build_cluster_solution(self, x_data: list, preprocessed_abstracts: list):
        """
        Builds a clustering solution by fitting the clusterer on the given data and extracting relevant information.

        Parameters:
        -----------
        x_data : list of array-like
            The input data used for clustering.
        preprocessed_abstracts : list of str
            A list of preprocessed abstracts that correspond to the input data points.
        """
        self._x_data = x_data
        self._cluster_labels = self.clusterer.fit_predict(self._x_data)
        self._centers = self.clusterer.cluster_centers_
        self._evaluation_score = silhouette_score(self._x_data, self._cluster_labels)

        # Group abstracts by cluster labels
        clustered_data = {}
        for i, label in enumerate(self._cluster_labels):
            clustered_data.setdefault(label, []).append(preprocessed_abstracts[i])
        self._clustered_data = clustered_data

        word_counter = WordCounter(num_top_words=20)
        self._word_dictionary = word_counter.get_top_words_each_cluster(self._clustered_data)

    def get_evaluation_score(self) -> float:
        """
        Getter function for the evaluation score private variable
        :return: evaluation score
        """
        return self._evaluation_score

    def generate_pickle_file(self):
        """
        Generates a pickle file with the model for posterior usage.

        """
        logging.info(f'Saving the pickle file for {self.clusterer.n_clusters} clusters.')

        with open(self.pickle_file, 'wb') as file:
            pickle.dump(self.clusterer, file)

        logging.info(f'pkl file saved as {self.pickle_file}')

    def generate_graph(self):
        """
        Creates a scatter plot of clustered data and saves it as an image.

        """
        logging.info(f'Saving the scatter plot for {self.clusterer.n_clusters} clusters.')

        # Create the scatter plot
        plt.figure(figsize=(9, 7))

        # Use the cluster labels to color each point
        colors = cm.get_cmap('tab10', self.clusterer.n_clusters)(self._cluster_labels / self.clusterer.n_clusters)
        plt.scatter(self._x_data[:, 0], self._x_data[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)

        # Draw white circles at cluster centers
        plt.scatter(self._centers[:, 0], self._centers[:, 1], marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(self._centers):
            plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        plt.title(f"Visualization of the clustered data (n_clusters = {self.clusterer.n_clusters})")
        plt.xlabel("Feature space for the 1st feature")
        plt.ylabel("Feature space for the 2nd feature")

        # Save the figure and close plot
        plt.savefig(f'{self.img_file}')
        plt.close()
        logging.info(f'img file saved as {self.img_file}')

    def generate_pdf(self):
        """Generate a PDF containing cluster images and their word frequency data.

        """
        # Create a PDF document
        logging.info(f'Generating pdf file for the model with {self.clusterer.n_clusters} clusters.')

        w, h = letter
        c = canvas.Canvas(self.pdf_file, pagesize=letter)

        # Set the initial position for the PDF
        x, y = 1 * inch, 10 * inch  # Starting position
        line_height = 0.2 * inch  # Height of each line
        space_between_clusters = 0.5 * inch  # Space between clusters
        image_height = 5 * inch  # Height of the image

        # Sort the outer dictionary by number of clusters and inner dictionaries by cluster number
        sorted_data = dict(sorted(self._word_dictionary.items()))

        # Write the title
        c.drawString(x, y, text=f'Alternative with {self.clusterer.n_clusters} clusters:')
        y -= (line_height + image_height)  # Move down for the title

        # Add the image
        if os.path.exists(self.img_file):
            c.drawImage(self.img_file, x, y, width=5 * inch, height=image_height)  # Adjust size as needed
            y -= line_height  # Move down after the image

        for cluster_num, word_freq in sorted_data.items():
            # Write the cluster heading
            cluster_heading = f'Cluster {cluster_num}:'
            if y < 1 * inch:  # Check if there's enough space for the next line
                c.showPage()
                y = 10 * inch  # Reset y position for the new page

            c.drawString(x, y, text=cluster_heading)
            y -= line_height  # Move down for the cluster heading

            for word, freq in word_freq:
                if y < 1 * inch:  # Check if there's enough space for the next line
                    c.showPage()
                    y = 10 * inch  # Reset y position for the new page
                c.drawString(x + 0.5 * inch, y, text=f'{word} {freq}')
                y -= line_height  # Move down for the next cluster

        # Save the PDF
        c.save()
        logging.info(f'pdf file saved as {self.pdf_file}')
