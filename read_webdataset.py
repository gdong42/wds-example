
import webdataset as wds
import argparse
import os
from glob import glob

def read_webdataset(dataset_path):
    """
    Reads and iterates through a webdataset.
    """
    # Create a dataset object
    # Use glob to expand the wildcard path
    shard_list = glob(os.path.join(dataset_path, 'shard-*.tar'))
    dataset = wds.WebDataset(shard_list)

    # Iterate through the dataset
    for i, sample in enumerate(dataset):
        print(f"--- Sample {i} ---")
        print(f"Key: {sample['__key__']}")
        print(f"Text: {sample['txt'].decode('utf-8').strip()}")
        print(f"Image size: {len(sample['jpg'])} bytes")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a WebDataset.")
    parser.add_argument("--path", required=True, help="Path to the webdataset shards.")
    args = parser.parse_args()
    
    read_webdataset(args.path)
