import os
import numpy as np

from src import logger
from transformers import pipeline


class FeatureExtractor:
    CHECKPOINT = "bert-base-uncased"
    EMBEDDING_SIZE = 768
    MAX_INPUT_SIZE = 512

    def __init__(self):
        """
        Create a pipeline for feature extraction.
        """
        self.pipeline = pipeline(
            "feature-extraction",
            framework="pt",
            model=FeatureExtractor.CHECKPOINT,
            padding=True,
            truncation=True
        )
        logger.info(f"Feature extractor: {FeatureExtractor.CHECKPOINT} initialized.")

    def run(self, text_list: list[str]) -> np.ndarray:
        """
        Extract features/embeddings from a lit of text.

        :param text_list: list of input texts
        :return: embeddings for the input list
        """
        embeddings = []
        for text in text_list:
            embedding = self.pipeline(text, return_tensors="pt")[0].numpy().mean(axis=0)
            embeddings.append(embedding)
        logger.info(f"Computed {len(text_list)} embedding(s).")
        return np.array(embeddings)


def get_repo_movies_embeddings(text_list: list[str], embeddings_filepath: str, recompute: bool=False) -> np.ndarray:
    """
    Retrieve the embeddings for the list of movies in text_list.
    Embeddings are computed only if the file specified in embeddings_filepathis missing or the recompute flag is set.

    :param text_list: list of texts representing the movies in the processed data?
    :param embeddings_filepath: filepath of embeddings
    :param recompute: True ro recompute embeddings
    :return: embeddings corresponding to text_list
    """
    if os.path.exists(embeddings_filepath) and not recompute:
        logger.info(f'Loading existing embeddings from {embeddings_filepath}.')
        embeddings = np.load(embeddings_filepath)
    else:
        m = FeatureExtractor()
        logger.info(f'Using {m.CHECKPOINT} to compute embeddings. This might take some time.')
        embeddings = m.run(text_list)
        np.save(embeddings_filepath, embeddings)
        logger.info(f'Embeddings stored in {embeddings_filepath}.')
    return embeddings
