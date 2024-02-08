from typing import List, Dict
from qdrant_client import QdrantClient
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
