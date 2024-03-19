import pandas as pd

from src import logger
from src.core.indexer import FaissIndex
from src.core.embeddings import FeatureExtractor


class RecommendationEngine:

    def __init__(self, index: FaissIndex, feature_extractor: FeatureExtractor, df: pd.DataFrame):
        """
        Initialize a movie recommendation engine.

        :param index: faiss index
        :param feature_extractor: feature extractor used to compute embeddings
        :param df: reference movies dataframe associated to the index
        """
        self.index = index
        self.feature_extractor = feature_extractor
        self.df = df

    def recommend(self, text: str, k: int=5):
        """
        Recommended the top K movies according to the provided text.
        Text is expected to be a string containing the title of a movie for which you want
        a recommendation and a short description.

        :param text: input text for which you want a movie recommendation
        :param k: number of recommendations
        :return: top k recommendations
        """
        embedding = self.feature_extractor.run([text])[0]
        _, indices = self.index.search(embedding, k)
        res = self.df.iloc[indices[0]]['name'].values
        logger.info(f"Recommendations for: {text}.")
        logger.info(f"Recommended indices: {indices}.")
        logger.info(f"Recommended movies: {res}.")
        return res
