#!/usr/bin/env python3
import argparse
import os
import zipfile

import requests
from tqdm import tqdm


def download_glove(output_dir="data", model_name="glove.6B"):
    """
    Download and unzip the GloVe word embeddings

    Parameters:
    -----------
    output_dir: Directory for saving the data
    model_name: GloVe model name ('glove.6B', 'glove.42B.300d', 'glove.840B.300d', etc.)
    """
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, f"{model_name.replace('.', '-')}")
    os.makedirs(model_dir, exist_ok=True)

    # Define the URL for downloading the ZIP file
    base_url = "https://nlp.stanford.edu/data"
    zip_filename = f"{model_name}.zip"
    download_url = f"{base_url}/{zip_filename}"

    # Download the ZIP file
    zip_path = os.path.join(output_dir, zip_filename)
    if not os.path.exists(zip_path):
        print(f"Downloading {download_url}...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()  # Make sure the request was successful

        # Get the total file size
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB

        with open(zip_path, "wb") as f, tqdm(
            desc=zip_filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))
    else:
        print(f"{zip_path} Already exists. Skipping download.")

    # Extract the ZIP file
    print(f"Unpacking {zip_path} to {model_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(model_dir)

    print(f"GloVe model saved to {model_dir}")

    # List the available files
    files = os.listdir(model_dir)
    print(f"Available files: {', '.join(files)}")

    return model_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GloVe word embeddings")
    parser.add_argument(
        "--output_dir", default="data", help="Directory for saving data"
    )
    parser.add_argument(
        "--model",
        default="glove.6B",
        help="GloVe model name ('glove.6B', 'glove.42B.300d', etc.)",
    )
    args = parser.parse_args()

    download_glove(args.output_dir, args.model)
