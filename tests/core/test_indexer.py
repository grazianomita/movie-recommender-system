import os
import tempfile
import numpy as np

from unittest.mock import patch
from src.core.indexer import FaissIndex


@patch('src.core.indexer.logger')
def test_faiss_index(mock_logger):
    # Create index
    embedding_size = 768
    faiss_index = FaissIndex(embedding_size)
    assert faiss_index.index is not None
    mock_logger.info.assert_called_once_with("Index created.")
    # Add embeddings
    embeddings = np.random.rand(10, embedding_size).astype(np.float32)
    faiss_index.add(embeddings)
    mock_logger.info.assert_called_with("Embeddings added to the index.")
    # Test search
    test_embedding = np.random.rand(embedding_size).astype(np.float32)
    distances, indices = faiss_index.search(test_embedding, k=3)
    assert distances.shape == (1, 3)
    assert indices.shape == (1, 3)
    # Save and load index
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, 'index.index')
        faiss_index.save(index_path)
        assert os.path.exists(index_path)
        faiss_index.index = None
        faiss_index.load(index_path)
        assert faiss_index.index is not None
