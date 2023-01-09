import gzip
import json
import logging
import os
import pickle as pkl
import zipfile
from typing import Any

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as reader:
        return pkl.load(reader)


def save_pickle(path: str, obj: object) -> None:
    with open(path, "wb") as writer:
        pkl.dump(obj, writer)


def save_json(path: str, obj: object, compressed: bool = False, indent: int = None) -> None:
    if compressed:
        with gzip.open(path, 'wt', encoding='UTF-8') as zipfile:
            json.dump(obj, zipfile, indent=indent)
    else:
        with open(path, "w") as writer:
            json.dump(obj, writer, indent=indent)


def load_json(path: str, compressed: bool = False) -> dict:
    if compressed:
        with gzip.open(path, 'rt', encoding='UTF-8') as zipfile:
            return json.load(zipfile)
    else:
        with open(path, "r") as reader:
            return json.load(reader)


def read_qrels(file: str, for_pytrec: bool = True) -> dict:
    qrels = {}
    with open(file) as reader:
        for line in reader:
            (query_id, _, document_id, _) = line.split()
            if for_pytrec:
                qrels[query_id] = {
                    document_id: 1
                }
            else:
                qrels[query_id] = document_id
    return qrels


def download_file(url, file_location):
    """
        Downloads a file from url to file_location.
    """

    if os.path.exists(file_location):
        log.info(f"Data already downloaded to {file_location}")
        return

    with open(file_location, "wb") as handle:

        headers = requests.head(url)

        if headers.status_code != 200:
            raise ValueError(f"non-200 status code for {url}")

        response = requests.get(url, stream=True)

        chunk_size = 1024
        try:
            iters = int(headers.headers["Content-Length"]) // chunk_size
        except:
            iters = None
        uu = url if len(url) < 30 else url[:30] + "..."
        for data in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading {uu} to {file_location}",
                         total=iters, leave=False):
            handle.write(data)
        log.info("Finished downloading file")


def unzip(file_location, folder_location):
    """
        Unzips a file.
    """
    if os.path.exists(folder_location):
        log.info(f"Data already unzipped to {folder_location}")
        return

    with zipfile.ZipFile(file_location, 'r') as zip_ref:
        zip_ref.extractall(folder_location)
