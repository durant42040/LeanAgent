"""
Utility functions for filesystem operations.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Union

from loguru import logger


def remove_dir(dir_path: Union[str, Path]) -> None:
    """
    Safely removes a directory path if it exists, with retries.

    This function attempts to delete a directory and retries multiple times if permission errors occur,
    which can happen if files are temporarily locked by another process or if the directory
    contains read-only files.

    Args:
        dir_path (Union[str, Path]): Path to the directory to be removed.

    Raises:
        PermissionError: If the directory cannot be removed after multiple retries
                         due to permission issues.

    Example:
        ```
        remove_dir('/path/to/directory')
        remove_dir(Path('/path/to/directory'))
        ```
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

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
