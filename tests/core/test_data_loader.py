import os
import tempfile
import pandas as pd

from unittest.mock import patch
from src.core.data_loader import MoviesDataLoader


@patch('src.core.data_loader.MoviesDataLoader._MoviesDataLoader__download_archive')
def test_get(mock_download_archive):
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_download_archive.return_value = None
        MoviesDataLoader.get(tmpdir)
        mock_download_archive.assert_called_once_with(tmpdir)


@patch('src.core.data_loader.tarfile')
@patch('src.core.data_loader.logger')
@patch('src.core.data_loader.os.listdir')
@patch('src.core.data_loader.delete_file')
def test_extract_tar_gz(mock_delete_file, mock_listdir, mock_logger, mock_tarfile):
    mock_tarfile.open.return_value.__enter__.return_value.extractall.side_effect = lambda path: None
    mock_tarfile.TarError = FileNotFoundError  # or any other suitable exception class
    mock_listdir.return_value = ['file1.txt', 'file2.txt', 'plot_summaries.txt', 'movie.metadata.tsv']
    with tempfile.TemporaryDirectory() as tmpdir:
        MoviesDataLoader._MoviesDataLoader__extract_tar_gz('/path/to/archive.tar.gz', tmpdir)
    mock_tarfile.open.assert_called_once_with('/path/to/archive.tar.gz', 'r:gz')
    mock_logger.info.assert_called_once()
    mock_listdir.assert_called_once_with(os.path.join(tmpdir, 'MovieSummaries'))
    mock_delete_file.assert_any_call('/path/to/archive.tar.gz')
    mock_delete_file.assert_any_call(os.path.join(tmpdir, 'MovieSummaries', 'file1.txt'))
    mock_delete_file.assert_any_call(os.path.join(tmpdir, 'MovieSummaries', 'file2.txt'))


@patch('src.core.data_loader.os.makedirs')
@patch('src.core.data_loader.os.path.exists')
@patch('src.core.data_loader.subprocess.run')
@patch('src.core.data_loader.MoviesDataLoader._MoviesDataLoader__extract_tar_gz')
@patch('src.core.data_loader.logger')
def test_download_archive(mock_logger, mock_extract_tar_gz, mock_subprocess_run, mock_path_exists, mock_makedirs):
    mock_path_exists.side_effect = [False, False]   # neither the archive nor the final movie path exists
    with tempfile.TemporaryDirectory() as tmpdir:
        MoviesDataLoader._MoviesDataLoader__download_archive(tmpdir)
    mock_makedirs.assert_called_once_with(tmpdir, exist_ok=True)
    mock_path_exists.assert_any_call(os.path.join(tmpdir, 'movies.csv'))
    mock_logger.info.assert_called_once()
    mock_extract_tar_gz.assert_called_once()


@patch('src.core.data_loader.logger')
@patch('src.core.data_loader.MoviesDataLoader._MoviesDataLoader__process_movie_metadata')
@patch('src.core.data_loader.MoviesDataLoader._MoviesDataLoader__process_movie_summary')
def test_process(mock_process_summary, mock_process_metadata, mock_logger):
    mock_process_metadata.return_value = pd.DataFrame({'wikipedia_id': [1, 2], 'name': ['M1', 'M2'], 'release_date': [2000, 2001]})
    mock_process_summary.return_value = pd.DataFrame({'wikipedia_id': [1, 2], 'summary': ['S1', 'S2']})
    with tempfile.TemporaryDirectory() as tmpdir:
        df = MoviesDataLoader.process(tmpdir, tmpdir)
    assert isinstance(df, pd.DataFrame)
    mock_process_metadata.assert_called_once()
    mock_process_summary.assert_called_once()
    mock_logger.info.assert_called()
