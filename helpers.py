import os


def ensure_directory(path):
    """
    Create a directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def file_exists(path):
    """
    Check whether a file exists.
    """
    return os.path.isfile(path)


def get_project_root():
    """
    Return the root directory of the project.
    """
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )