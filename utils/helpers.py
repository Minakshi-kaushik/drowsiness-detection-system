import os


def create_directory(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def file_exists(path):
    """
    Check if file exists.
    """
    return os.path.exists(path)
