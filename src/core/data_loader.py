import os
import tarfile
import subprocess
import pandas as pd

from src import logger
from src.util.utils import delete_file, extract_year, extract_values


class MoviesDataLoader:
    ARCHIVE_NAME = ''
    URL = f'https://www.cs.cmu.edu/~ark/personas/data/{ARCHIVE_NAME}'
    MOVIE_METADATA_NAME = 'MovieSummaries/movie.metadata.tsv'
    MOVIE_SUMMARY_NAME = 'MovieSummaries/plot_summaries.txt'
    MOVIE_FINAL_NAME = 'movies.csv'

    @staticmethod
    def get(dst_dir: str) -> None:
        """
        Get the data at MoviesDataLoader.URL and stores it into dst_dir.

        :param dst_dir: destination directory
        :return: None
        """
        MoviesDataLoader.__download_archive(dst_dir)

    @staticmethod
    def __extract_tar_gz(archive_path: str, dst_dir: str) -> None:
        """
        Extract movies data from archive_path.
        Useless files are deleted.

        :param archive_path: archive path
        :param dst_dir: files are extracted into dst_dir
        :return: None
        """
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=dst_dir)
            logger.info(f"Extraction successful. Files extracted to {dst_dir}")
        except tarfile.TarError as e:
            logger.error(f"Error extracting {archive_path}: {e}")
        delete_file(archive_path)
        for filename in os.listdir(os.path.join(dst_dir, 'MovieSummaries')):
            if filename not in ['plot_summaries.txt', 'movie.metadata.tsv']:
                delete_file(os.path.join(dst_dir, 'MovieSummaries', filename))

    @staticmethod
    def __download_archive(dst_dir: str) -> None:
        """
        Download the archive from URL, only if the extracted required data is not already present in dst_dir.

        :param dst_dir: files are extracted into dst_dir
        :return: None
        """
        movie_final_path = os.path.join(dst_dir, MoviesDataLoader.MOVIE_FINAL_NAME)
        archive_path = os.path.join(dst_dir, MoviesDataLoader.ARCHIVE_NAME)
        os.makedirs(dst_dir, exist_ok=True)
        try:
            if not os.path.exists(movie_final_path):
                if not os.path.exists(archive_path):
                    subprocess.run(['wget', MoviesDataLoader.URL, '--no-check-certificate', '-O', archive_path], check=True)
                    logger.info(f"Downloaded {MoviesDataLoader.URL} successfully.")
                else:
                    logger.info(f"Archive already downloaded.")
                MoviesDataLoader.__extract_tar_gz(archive_path, dst_dir)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading {MoviesDataLoader.URL}: {e}")

    @staticmethod
    def process(src_dir: str, dst_dir: str, min_release_year: int=2000) -> pd.DataFrame:
        """
        Movies data stored in src_dir is processed.
        Processed data is stored into dst_dir with the name specified in `filename`.

        :param src_dir: directory where raw movies data is stored
        :param dst_dir: destination where processed movies data is stored
        :param min_release_year: movies with release_year < min_release_year are filtered out
        :return: processed movies data as pandas dataframe
        """
        df_output_path = os.path.join(dst_dir, MoviesDataLoader.MOVIE_FINAL_NAME)
        if not os.path.exists(df_output_path):
            movie_metadata_path = os.path.join(src_dir, MoviesDataLoader.MOVIE_METADATA_NAME)
            movie_summary_path = os.path.join(src_dir, MoviesDataLoader.MOVIE_SUMMARY_NAME)
            df_movie_metadata = MoviesDataLoader.__process_movie_metadata(movie_metadata_path)
            df_movie_summary = MoviesDataLoader.__process_movie_summary(movie_summary_path)
            df_movie_metadata = df_movie_metadata[df_movie_metadata['release_date'] >= min_release_year]
            df = df_movie_metadata.merge(df_movie_summary, on=['wikipedia_id'])
            df['text'] = df['name'] + '. ' + df['summary']
            df = df[['wikipedia_id', 'name', 'text']].reset_index()
            df.to_csv(df_output_path, index=False)
            logger.info(f'{df_output_path} processed.')
        else:
            df = pd.read_csv(df_output_path)
            logger.info(f'{df_output_path} already present on disk.')
        return df

    @staticmethod
    def __process_movie_metadata(path: str) -> pd.DataFrame:
        """
        Simple processing applied to the movie metadata file

        :param path: filepath of the movie metadata file
        :return: processed pandas dataframe
        """
        df = pd.read_csv(
            path,
            sep='\t',
            header=None,
            names=['wikipedia_id', 'freebase_id', 'name', 'release_date', 'box_office_revenue', 'runtime_minutes',
                   'languages', 'countries', 'genres']
        )
        df['genres'] = df['genres'].apply(extract_values)
        df['languages'] = df['languages'].apply(extract_values)
        df['countries'] = df['countries'].apply(extract_values)
        df['release_date'] = df['release_date'].apply(extract_year)
        logger.info(f'{path} processed.')
        return df

    @staticmethod
    def __process_movie_summary(path: str) -> pd.DataFrame:
        """
        Simple processing applied to the movie summary file

        :param path: filepath of the movie summary file
        :return: processed pandas dataframe
        """
        df = pd.read_csv(
            path,
            sep='\t',
            header=None,
            names=['wikipedia_id', 'summary']
        )
        logger.info(f'{path} processed.')
        return df
