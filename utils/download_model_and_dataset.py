"""For downloading the food model and dataset."""

import os

import requests
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download


def download_file(url, local_path):
    """Downloads a specific file."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def download_model():
    """Downloads the model."""
    model_path = "/workspace/models/nateraw/food"
    os.makedirs(model_path, exist_ok=True)

    api = HfApi()
    model_files = api.list_repo_files("nateraw/food")

    for file in model_files:
        if not file.endswith("/"):  # Skip directories
            url = f"https://huggingface.co/nateraw/food/resolve/main/{file}"
            local_path = os.path.join(model_path, file)
            print(f"Downloading {file}...")
            download_file(url, local_path)

    print(f"Model downloaded to: {model_path}")


def download_dataset():
    """Downloads the food dataset."""
    dataset_path = "/workspace/food101_data"
    os.makedirs(dataset_path, exist_ok=True)

    snapshot_download(
        repo_id="ethz/food101",
        repo_type="dataset",
        local_dir=dataset_path,
        ignore_patterns=["*.gitattributes", "*.md"],
        local_dir_use_symlinks=False,
    )

    print(f"Dataset downloaded to: {dataset_path}")


if __name__ == "__main__":
    download_model()
    download_dataset()
