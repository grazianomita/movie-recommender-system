from src.core.data_loader import MoviesDataLoader
from src.core.embeddings import get_repo_movies_embeddings, FeatureExtractor
from src.core.indexer import get_index
from src.core.parser import CustomParser
from src.core.reco import RecommendationEngine


def main() -> list:
    """
    Retrieve the top K movie recommendations based on the given query.
    This function utilizes a feature extractor to compute a word embedding for the query.
    Subsequently, the generated embedding is employed to run a similarity search across all movies stored in the index.
    If embeddings and/or the index are not present at the path specified in the arguments, this function will rebuild
    them accordingly.

    :return: top k movie recommendations
    """
    args = CustomParser.parse_args()
    MoviesDataLoader.get(args['data_dir'])
    df = MoviesDataLoader.process(args['data_dir'], args['data_dir'], min_release_year=1900)
    feature_extractor = FeatureExtractor()
    embeddings = get_repo_movies_embeddings(df['text'].values, embeddings_filepath=args['embeddings_path'], recompute=False)
    index = get_index(args['faiss_index_path'], embeddings)
    reco_engine = RecommendationEngine(index, feature_extractor, df)
    return reco_engine.recommend(args['query'], k=args['k'])


if __name__ == '__main__':
    recommendations = main()
    for r in recommendations:
        print(f'> {r}')
