import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Set
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


class Text2Img:
    def __init__(self, collection_name: str = 'images'):
        self.collection_name = collection_name

        # Initialize encoder models for image and text
        self.text_encoder = SentenceTransformer("clip-ViT-B-32", device="cpu")

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")

    def search(self, text: str) -> List[Dict[str, str]]:
        """
        Search function for the vector database
        Args:
            text: text used in the search

        Returns:
            List of payloads (images paths more exactly)
        """
        # Convert text query into vector
        vector = self.text_encoder.encode(text).tolist()

        # Use `vector` to search for closest images in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,
            with_payload=True,
            limit=5,
        )
        # Retrieve payload results
        payloads = [hit.payload for hit in search_result]

        return payloads

    def avg_precision_at_k(self, test_dataset: List[str], k: int = 5) -> Tuple[float, Dict[str, Set[str]]]:
        """
        Computes precition@k metric for a custom set of queries
        Args:
            test_dataset: dataset of text queries
            k: parameter of the metric

        Returns: tuple represented by the result of the metric precision@k
                and a mapping between one search query and a list of common images in both ANN and full kNN
        """
        # Initialize precision and common indexes mapping
        precisions = []
        common_images_mapping = {}

        print("Evaluating custom dataset...")
        for item in tqdm(test_dataset):
            # Convert text query into vector
            vector = self.text_encoder.encode(item).tolist()

            # Get approximate results
            ann_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=k,
            )

            # Get full results
            knn_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=k,
                search_params=models.SearchParams(
                    exact=True,  # Turns on the exact search mode
                ),
            )

            # Get ids of the search results
            ann_ids = set(item.id for item in ann_result)
            knn_ids = set(item.id for item in knn_result)

            # Get common indexes
            common_indexes = ann_ids.intersection(knn_ids)

            # Get common images path
            mask = np.isin(list(common_indexes), list(knn_ids))
            common_results = np.array(knn_result)[mask]
            common_images = [res.payload['path'] for res in common_results]
            common_images_mapping[item] = set(common_images)

            # Compute precision
            precision = len(common_indexes) / k
            precisions.append(precision)

        return sum(precisions) / len(precisions), common_images_mapping
