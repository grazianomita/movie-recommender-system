import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.core.indexer import FaissIndex
from src.core.embeddings import FeatureExtractor
from src.core.reco import RecommendationEngine


def mock_faiss_index():
    mock_index = MagicMock()
    mock_index.search.return_value = (np.array([[1, 2, 3, 4, 5]]), np.array([[0, 1, 2, 3, 4]]))
    return mock_index


def mock_feature_extractor():
    mock_extractor = MagicMock()
    mock_extractor.run.return_value = [np.array([1, 2, 3])]
    return mock_extractor


@patch('src.core.reco.logger')
def test_recommendation_engine(mock_logger):
    index = mock_faiss_index()
    feature_extractor = mock_feature_extractor()
    df = pd.DataFrame({'name': ['movie1', 'movie2', 'movie3', 'movie4', 'movie5']})
    engine = RecommendationEngine(index, feature_extractor, df)
    assert engine.index == index
    assert engine.feature_extractor == feature_extractor
    assert engine.df.equals(df)
    recommendations = engine.recommend('test_text', k=5)
    np.testing.assert_array_equal(recommendations, ['movie1', 'movie2', 'movie3', 'movie4', 'movie5'])
