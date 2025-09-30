import os
from typing import List


def find_files_with_substring(
    base_path: str,
    substring: str,
) -> List[str]:
    """
    Find all the files in any subdirectory of the given base path that contain the specified substring in their name.

    :param base_path: The base path to start searching from.
    :param substring: The substring to look for in the file names.

    :return: A list of file paths that match the criteria.
    """
    matching_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if substring in file:
                matching_files.append(os.path.join(root, file))
    return matching_files
