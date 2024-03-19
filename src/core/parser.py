import argparse


class CustomParser:
    DEFAULT_K = 5
    DEFAULT_DATA_DIR = "data"
    DEFAULT_EMBEDDINGS_PATH = "data/embeddings.npy"
    DEFAULT_FAISS_INDEX_PATH = "data/movies.index"

    @staticmethod
    def parse_args() -> dict:
        """
        This method parses input arguments using the argparse module.
        It defines arguments for specifying the query text, the number of recommended movies,
        as well as options for the data directory, embeddings file path, and Faiss index file path.

        :return: dictionary containing the parsed input arguments
        """
        parser = argparse.ArgumentParser(description='Movie recommender system')
        inputs = parser.add_argument_group('Query params')
        inputs.add_argument('-q', '--query', help='<Required> Text used to recommend movies', required=True)
        inputs.add_argument('-k', help='Number of recommended movies (default=5)', required=False, type=int, default=CustomParser.DEFAULT_K)
        options = parser.add_argument_group('Input')
        options.add_argument('--data-dir', help='Data directory', required=False, default=CustomParser.DEFAULT_DATA_DIR)
        options.add_argument('--embeddings-path', help='embeddings filepath', required=False, default=CustomParser.DEFAULT_EMBEDDINGS_PATH)
        options.add_argument('--faiss-index-path', help='faiss index filepath', required=False, default=CustomParser.DEFAULT_FAISS_INDEX_PATH)
        return vars(parser.parse_args())
