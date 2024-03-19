import os
import pandas as pd

from src import logger


def delete_file(file_path: str) -> None:
    """
    Delete a file.

    :param file_path: path of the files to be deleted
    :return: None or it raises an exception
    """
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"File {file_path} successfully deleted.")
        except OSError as e:
            logger.error(f"Error deleting {file_path}: {e.strerror}.")


def extract_year(date_str: str) -> int:
    """
    Extract the year from a date string.

    :param date_str: string containing the date
    :return: year if extraction succeds, None otherwise
    """
    try:
        return pd.to_datetime(date_str).year
    except ValueError:
        return None


def extract_values(string_repr: str) -> list[str]:
    """
    Extract values from a string representation of a dictionary.

    :param string_repr: string representation of a dictionary
    :return: values within the dictionary
    """
    import ast
    dictionary = ast.literal_eval(string_repr)
    return list(dictionary.values())
