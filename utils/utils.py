import io
import os
import zipfile
import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm
from typing import List, Dict
from fastapi import Response
from PIL import Image
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

tqdm.pandas()


def zip_files(filenames: List[Dict[str, str]]) -> Response:
    """
    Function that prepares an archive with all the matched images
    Args:
        filenames: response from Qdrant vector similarity search

    Returns: fastAPI response

    """
    # Define archive name
    zip_filename = "images.zip"

    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    for entry in filenames:
        fpath = entry['path']

        # Calculate path for file in zip
        fdir, fname = os.path.split(fpath)

        # Add file, at correct path
        zf.write(fpath, fname)

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

    return resp


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


def build_image_embeddings(df: pd.DataFrame, save_path: str = 'resources'):
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

    print("Building image embeddings...")

    # Set model for embedding the images
    image_model = SentenceTransformer("clip-ViT-B-32")

    # Check for torch and CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            image_model = image_model.to('cuda')
        else:
            print("CUDA is not available. Using CPU instead. This may take some time...")
    except ModuleNotFoundError:
        print("Torch is not installed")

    # Compute embeddings
    df["embedding"] = df["path"].progress_apply(lambda x: calculate_embedding(image_model, x))
    df["embedding"] = df["embedding"].replace({None: np.nan})
    df = df.dropna(subset=["embedding"])

    # Save to parquet file
    df.to_parquet(embeddings_path)

    print("Finished")


def update_db_collection(collection_name: str = 'images', vectors_dir_path: str = 'resources'):
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
    print(f"There are {len(im_df)} images.")

    # Create payloads and vectors
    paths = im_df['path'].values
    payloads = iter([{'path': p} for p in paths])
    vectors = iter(list(map(list, im_df["embedding"].tolist())))

    print("Populating Qdrant collection with the embeddings...")

    # (Re-)create collection
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

    print(f"There are {qdrant_client.count(collection_name)} points created "
          f"in the Qdrant collection named '{collection_name}'")
