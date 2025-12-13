import shutil
from pathlib import Path

import zenodo_get as zg


def zenodo_get_and_unzip(zenodo_id: str, download_file: str, destination_dir: str) -> None:
    """Download a dataset from Zenodo, unzip it, and organize it into the specified directory.

    Parameters
    ----------
    zenodo_id : str
        The Zenodo record ID to download.
    download_file : str
        The name of the zip file to download.
    destination_dir : str
        The directory where the dataset will be stored.

    Behavior
    --------
    - Check if the target dataset folder exists; if not, create it.
    - Download the dataset zip file from Zenodo using the provided ID.
    - Unzip the downloaded file into the destination directory.
    - Remove the zip file after extraction.
    - Print progress and error messages during the process.

    """
    dest_dir = Path(destination_dir)

    try:
        zg.download(zenodo_id, output_dir=str(dest_dir))
        print(f"Downloaded {dest_dir}.")

        # Unzip the downloaded file
        print(f"Unzipping {download_file}...")
        shutil.unpack_archive(dest_dir / download_file, destination_dir)
        print("Unzipping complete.")

        # Clean up the downloaded zip file
        # Path.unlink(dest_dir / download_file)
        # print(f"Removed {download_file}.")

    except Exception as e:
        print(f"Error downloading or unzipping the dataset: {e}")
