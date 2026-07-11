import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import pickle
from numpy.linalg import norm

class Database:
    def __init__(self):
        self.image_ids = []
        self.caption_ids = []
        self.image_to_caption = {}
        self.caption_to_image = {}
        self.caption_to_text = {}
        self.embedding_matrix = None
        self.image_embeddings = {}

    # Function to download an image
    @staticmethod
    def download_image(img_url: str) -> Image:
        response = requests.get(img_url)
        return Image.open(BytesIO(response.content))

    # Step 1: Convert Image Descriptors to Embeddings
    @staticmethod
    def convert_to_embeddings(descriptors, embedding_matrix):
        return np.dot(descriptors, embedding_matrix)

    # Step 2: Create the Image Database
    def create_image_database(self, image_ids: list, embeddings: np.array):
        self.image_embeddings = {image_id: embedding for image_id, embedding in zip(image_ids, embeddings)}

    # Save the image database to a file
    @staticmethod
    def save_image_database(database, filename):
        with open(filename, 'wb') as f:
            pickle.dump(database, f)

    # Load the image database from a file
    @staticmethod
    def load_image_database(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    # Custom cosine similarity function
    @staticmethod
    def cosine_similarity_manual(a, b):
        dot_product = np.dot(a, b)
        norm_a = norm(a)
        norm_b = norm(b)
        return dot_product / (norm_a * norm_b)

    @staticmethod
    def cosine_similarity_matrix(matrix, vector):
        dot_products = np.dot(matrix, vector)
        norm_matrix = norm(matrix, axis=1)
        norm_vector = norm(vector)
        return dot_products / (norm_matrix * norm_vector)

    # Step 3: Query the Database
    def query_database(self, caption_embedding, top_k=5):
        image_ids = list(self.image_embeddings.keys())
        embeddings = np.array(list(self.image_embeddings.values()))
        
        similarities = self.cosine_similarity_matrix(embeddings, caption_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [image_ids[i] for i in top_indices]

    # Step 4: Display Images
    @staticmethod
    def display_images(image_urls: list):
        fig, axes = plt.subplots(1, len(image_urls), figsize=(15, 5))
        for ax, url in zip(axes, image_urls):
            img = Database.download_image(url)
            ax.imshow(img)
            ax.axis('off')
        plt.show()

    @staticmethod
    def generate_random_descriptor_and_matrices(num_images, descriptor_dimension, embedding_dimension):
        descriptors = np.random.rand(num_images, descriptor_dimension)
        embedding_matrix = np.random.rand(descriptor_dimension, embedding_dimension)
        return descriptors, embedding_matrix
