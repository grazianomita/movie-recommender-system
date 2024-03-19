import os
import faiss
import numpy as np

from src import logger


class FaissIndex:
    def __init__(self, embedding_size: int=768):
        """
        Initialize a flas Faiss Index.

        :param embedding_size: embedding size
        """
        self.index = faiss.IndexFlatIP(embedding_size)
        logger.info("Index created.")

    def add(self, embeddings: np.ndarray) -> None:
        """
        Add embeddings to the index.

        :param embeddings: embeddings
        :return: None
        """
        self.index.add(embeddings)
        logger.info("Embeddings added to the index.")

    def search(self, embedding: np.array, k: int=5) -> (np.array, np.array):
        """
        Search the k closest embeddings to embedding.

        :param embedding: search embedding
        :param k: top k closest embeddings
        :return: distances and indices of the top k closest embeddings
        """
        distances, indices = self.index.search(np.array([embedding]), k)
        return distances, indices

    def save(self, path: str) -> None:
        """
        Save index to path.

        :param path: filepath where the index will be stored
        :return: None
        """
        faiss.write_index(self.index, path)
        logger.info(f"Index written to {path}.")

    def load(self, path: str) -> None:
        """
        Load an index from path, if it exists.

        :param path: filepath where the index will be retrieved from
        :return: None
        """
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            logger.info(f"Index retrieved from {path}.")


def get_index(path, embeddings, embeddings_size=768, refresh=False):
    """
    Get an index from the specified path.
    If not present or the refresh flag is set, a new index is created using the specified embeddings and saved to path.

    :param path: index path
    :param embeddings: embeddings in the index
    :param embeddings_size: size of the embeddings
    :param refresh: True to force reconstruction of the index
    :return: index
    """
    index = FaissIndex(embeddings_size)
    if refresh or not os.path.exists(path):
        index.add(np.array(embeddings).astype("float32"))
        index.save(index, path)
    else:
        index.load(path)
    return index
