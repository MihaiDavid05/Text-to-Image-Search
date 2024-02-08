import os.path
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

tqdm.pandas()


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


def calculate_embedding(model, image_path: str) -> Optional[List[float]]:
    """
    Compute embeddings for an image
    Args:
        model: Image embedding model instance
        image_path:  path for the image to be embedded

    Returns:
        None or the embedding

    """
    try:
        image = Image.open(image_path)
        return model.encode(image).tolist()
    except:
        print(f"Error when embedding image {image_path}")
        return None


def build_image_embeddings(df: pd.DataFrame, save_path: str = 'docs'):
    """
    Builds and save image embeddings
    Args:
        df: DataFrame with all images information
        save_path: path to save the embeddings

    Returns: DataFrame with embeddings added

    """

    # Check existence of directory and files
    embeddings_file = "images_embeddings.parquet"
    embeddings_path = os.path.join(save_path, embeddings_file)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        # Check for existence of embeddings
        if os.path.isfile(embeddings_path):
            print("Embeddings were already created")
            return

    # Set model for embedding the images
    image_model = SentenceTransformer("clip-ViT-B-32")

    # Check for torch and CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            image_model = image_model.to('cuda')
        else:
            print("CUDA is not available. Using CPU instead.")
    except ModuleNotFoundError:
        print("torch is not even installed")

    # Compute embeddings
    df["embedding"] = df["path"].progress_apply(lambda x: calculate_embedding(image_model, x))
    df["embedding"] = df["embedding"].replace({None: np.nan})
    df = df.dropna(subset=["embedding"])

    # Save to parquet file
    df.to_parquet(embeddings_path)


def update_db_collection(collection_name: str = 'images', vectors_dir_path: str = 'docs'):
    """
    Create a Qdrant collection
    Args:
        collection_name: name of the collection to store the vector db points
        vectors_dir_path: paths to the directory containing the embeddings file
    """

    embeddings_path = os.path.join(vectors_dir_path, "images_embeddings.parquet")
    if not os.path.isfile(embeddings_path):
        print("Embeddings are not in the directory you specified or were not created!")
        return

    # Initialize Qdrant client
    qdrant_client = QdrantClient("http://localhost:6333")

    # Load all vectors into memory
    im_df = pd.read_parquet(embeddings_path)

    # Create payloads and vectors
    paths = im_df['path'].values
    payloads = [{'path': p} for p in paths]
    vectors = list(map(list, im_df["embedding"].tolist()))

    # Create collection
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    # Update data to collection
    qdrant_client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payloads,
        ids=None,
        batch_size=256,
    )

    print(f"There are {qdrant_client.count(collection_name)} points "
          f"in the collection {collection_name}")
