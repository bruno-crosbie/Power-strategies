"""Step 1: download (or refresh) every data source into data/raw.

    py download.py               # everything
    py download.py gas oil       # just some sources
"""
import sys

from lib.sources import download_all

if __name__ == "__main__":
    download_all(set(sys.argv[1:]) or None)
