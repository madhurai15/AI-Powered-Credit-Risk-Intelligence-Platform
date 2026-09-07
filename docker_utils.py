import os


def is_running_in_docker():
    """
    Check whether the application is running inside Docker.
    """
    return os.path.exists("/.dockerenv")


def get_data_path():
    """
    Return the data directory path used by the application.
    """
    if is_running_in_docker():
        return "/app/data"

    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data"
        )
    )


def get_model_path():
    """
    Return the model directory path used by the application.
    """
    if is_running_in_docker():
        return "/app/models"

    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "models"
        )
    )