import os
import tempfile

import numpy as np

from unittest.mock import patch, Mock, MagicMock

from src.core.embeddings import FeatureExtractor, get_repo_movies_embeddings


def embedding():
    return np.ones((FeatureExtractor.EMBEDDING_SIZE,))


@patch('src.core.embeddings.pipeline')
@patch('src.core.embeddings.logger')
def test_feature_extractor(mock_logger, mock_pipeline):
    mock_pipeline.return_value = MagicMock()
    feature_extractor = FeatureExtractor()
    embeddings = feature_extractor.run(["text1", "text2"])
    print(embeddings)
    mock_pipeline.assert_called_once_with(
        "feature-extraction",
        framework="pt",
        model=FeatureExtractor.CHECKPOINT,
        padding=True,
        truncation=True
    )
    assert embeddings.shape[0] == 2


def test_get_existing_embeddings():
    with tempfile.NamedTemporaryFile(suffix='.npy') as tmp_file:
        known_embeddings = np.array([[1, 2, 3], [4, 5, 6]])
        np.save(tmp_file.name, known_embeddings)
        embeddings = get_repo_movies_embeddings(["text1", "text2"], tmp_file.name, recompute=False)
        np.testing.assert_array_equal(embeddings, known_embeddings)


@patch('src.core.embeddings.FeatureExtractor')
def test_get_not_exist_or_recompute_embeddings(mock_feature_extractor):
    known_embeddings = np.array([[1, 2, 3], [4, 5, 6]])
    mock_instance = mock_feature_extractor.return_value
    mock_instance.run.return_value = known_embeddings
    with tempfile.NamedTemporaryFile(suffix='.npy') as tmp_file:
        embeddings = get_repo_movies_embeddings(["text1", "text2"], tmp_file.name, recompute=True)
        mock_instance.run.assert_called_once_with(["text1", "text2"])
        mock_instance.run.assert_called_once()
        assert os.path.exists(tmp_file.name)
        np.testing.assert_array_equal(embeddings, known_embeddings)
