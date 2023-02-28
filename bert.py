import zipfile
import urllib.request
import os

import warnings
warnings.simplefilter("ignore")
STANFORD_SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
DATA_DIR = "./data/snli/"
def download_and_extract(dataset_url, data_dir):
    print("Downloading and extracting ...")
    filename = "snli.zip"
    urllib.request.urlretrieve(dataset_url, filename)
    with zipfile.ZipFile(filename) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(filename)
    print("Completed!")

download_and_extract(STANFORD_SNLI_URL, DATA_DIR)