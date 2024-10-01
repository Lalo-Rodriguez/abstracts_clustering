import logging
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from sklearn.manifold import TSNE
from get_common_words_from_clusters import WordCounter

logger = logging.getLogger(__name__)


class ClusterSolution:
    def __init__(self,
                 algorithm_name: str,
                 abstracts: list,
                 normalized_x_data: np.array,
                 labels: np.array,
                 models_used: dict,
                 n_components_tsne: int = 2):
        self.algorithm_name = algorithm_name
        self.abstracts = abstracts
        self.normalized_x_data = normalized_x_data
        self.labels = labels
        self.models_used = models_used
        self.n_components_tsne = n_components_tsne

        self.output_dir = 'output'
        os.makedirs(self.output_dir, exist_ok=True)
        self.img_file = f'{self.output_dir}/{self.algorithm_name}.png'
        self.pdf_file = f'{self.output_dir}/{self.algorithm_name}.pdf'
        self.pickle_file = f'{self.output_dir}/{self.algorithm_name}.pkl'

        self.words_in_clusters = self._build_cluster_word_dictionary()

    def _build_cluster_word_dictionary(self) -> dict:
        """
        Function used to get the num_top
        :return: A dictionary with the num_top_words in each cluster and their frequency
        """
        clustered_data = {}
        for i, label in enumerate(self.labels):
            clustered_data.setdefault(label, []).append(self.abstracts[i])

        word_counter = WordCounter(num_top_words=20)
        word_dictionary = word_counter.get_top_words_each_cluster(clustered_data)
        return word_dictionary

    def generate_pickle_file(self):
        """
        Generates a pickle file with the model for posterior usage.

        """
        logging.info(f'Saving the pickle file with the models used.')

        with open(self.pickle_file, 'wb') as file:
            pickle.dump(self.models_used, file)

        logging.info(f'pkl file saved as {self.pickle_file}')

    def generate_graph(self):
        """
        Creates a scatter plot of clustered data and saves it as an image.

        """
        centers = self.models_used['cluster_algorithm'].cluster_centers_
        logging.info(f'Saving the scatter plot for {self.algorithm_name} algorithm.')

        # Create the scatter plot
        plt.figure(figsize=(9, 7))

        x_2d_data = TSNE(n_components=2).fit_transform(self.normalized_x_data)
        df = pd.DataFrame(x_2d_data, columns=['Var1', 'Var2'])
        p = sns.scatterplot(data=df, x='Var1', y='Var2', hue=self.labels, legend="full", palette="deep")
        sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.), title='Clusters')

        # Draw white circles at cluster centers
        plt.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            plt.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        plt.title(f"Visualization of the clustered data using the {self.algorithm_name} algorithm)")
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
        logging.info(f'Generating pdf file with the clusters created by the {self.algorithm_name} algorithm.')

        w, h = letter
        c = canvas.Canvas(self.pdf_file, pagesize=letter)

        # Set the initial position for the PDF
        x, y = 1 * inch, 10 * inch  # Starting position
        line_height = 0.2 * inch  # Height of each line
        space_between_clusters = 0.5 * inch  # Space between clusters
        image_height = 5 * inch  # Height of the image

        # Sort the outer dictionary by number of clusters and inner dictionaries by cluster number
        sorted_data = dict(sorted(self.words_in_clusters.items()))

        # Write the title
        c.drawString(x, y, text=f'Clusters created by the {self.algorithm_name} algorithm')
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
