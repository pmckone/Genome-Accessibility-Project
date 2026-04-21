import os
import requests
import zipfile
import numpy as np
import pyBigWig

def download_and_extract_zenodo(record_id, destination_folder=DATA_DIR):
    """Download and extract data.zip from a Zenodo record with resume support."""
    base_url = f"https://zenodo.org/api/records/{record_id}"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data  = response.json()
        files = data.get('files', [])

        if not files:
            print(f"No files found for record: {record_id}")
            return None

        os.makedirs(destination_folder, exist_ok=True)
        file_url   = files[0]['links']['self']
        zip_path   = os.path.join(destination_folder, 'data.zip')
        total_size = files[0].get('size', 0)

        # Check how much we already have for resume
        downloaded = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0

        if downloaded >= total_size:
            print(f"data.zip already fully downloaded, skipping.")
        else:
            if downloaded > 0:
                print(f"Resuming download from {downloaded / 1e6:.1f} MB...")
            else:
                print(f"Downloading data.zip ({total_size / 1e6:.1f} MB)...")

            headers = {'Range': f'bytes={downloaded}-'}
            with requests.get(file_url, stream=True, headers=headers) as r:
                r.raise_for_status()
                with open(zip_path, 'ab') as f:  # 'ab' = append bytes
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = downloaded / total_size * 100
                        print(f"  {percent:.1f}% ({downloaded / 1e6:.1f} MB)", end='\r')
            print(f"\nDownload complete.")

        # Check zip is valid before extracting
        print(f"Extracting to {destination_folder}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(destination_folder)
        print(f"Extraction complete.")

        os.remove(zip_path)
        print(f"Removed data.zip.")

        return destination_folder

    except Exception as e:
        print(f"Error downloading: {e}")
        return None