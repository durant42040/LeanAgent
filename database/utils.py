import shutil
import time
from pathlib import Path

from loguru import logger


def safe_remove_dir_path(dir_path):
    """
    Safely removes a directory if it exists.

    Attempts to remove the directory multiple times in case of permission errors.

    Args:
        dir_path (Path): Path object representing the directory to remove

    Raises:
        PermissionError: If the directory cannot be removed after multiple attempts

    Returns:
        None
    """
    if dir_path.exists():
        logger.warning(f"{dir_path} already exists. Removing it now.")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                shutil.rmtree(dir_path)
                break
            except PermissionError as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Wait a bit before retrying
                else:
                    logger.error(
                        f"Failed to remove {dir_path} after {max_retries} attempts: {e}"
                    )
                    raise 