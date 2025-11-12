import os
import gzip
import requests
import io
import pandas as pd
import numpy as np
from urllib.parse import urljoin
from time import sleep

# ----------------------------
# 0) Settings
DOWNLOAD_DIR = "data_gse39582"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

SERIES = "GSE39582"
SERIES_MATRIX_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE39nnn/GSE39582/matrix/GSE39582_series_matrix.txt.gz"
ANNOTATION_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL570"  # GPL platform page (we'll download annotation separately if available)
ANNOTATION_FILE = "GPL570_annotation.csv"  # path to mapping file (to be downloaded or provided)
TARGET_GENES = 600  # at least 500 genes; you can set larger (e.g., 1000, 2000)
SEED = 42

# ----------------------------
# 1) Download data (series matrix)
def download_file(url, dest_path, retries=3, sleep_sec=5):
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
            print(f"Downloaded: {dest_path}")
            return True
        except Exception as e:
            print(f"Download failed ({e}), attempt {attempt+1}/{retries}. Retrying in {sleep_sec}s...")
            sleep(sleep_sec)
    return False

matrix_path = os.path.join(DOWNLOAD_DIR, "GSE39582_series_matrix.txt.gz")
if not os.path.exists(matrix_path):
    ok = download_file(SERIES_MATRIX_URL, matrix_path)
    if not ok:
        raise SystemExit("Failed to download series matrix. Check network or URL.")