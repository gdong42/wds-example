
# WebDataset Parallel Processing Example

This project provides a practical, end-to-end example of how to use the `webdataset` library to efficiently pack large-scale multimodal datasets in parallel. It is designed to be a starting point for anyone looking to build efficient data pipelines for deep learning models.

## Features

- **Parallel Data Sharding**: Demonstrates how to split a large dataset into smaller `.tar` shards using multiple processes.
- **Multimodal Data Handling**: Shows how to pack different data modalities (in this case, images and text) into a single `webdataset`.
- **Efficient Data Loading**: Includes a script to demonstrate how to read the generated `webdataset` for use in a training pipeline.
- **Scalable**: The parallel processing script can be easily adjusted to scale with the number of available CPU cores.

## Project Structure

```
.
├── source_data/              # Directory for raw source data (images, text, etc.)
├── webdataset_output/        # Directory for the generated .tar shards (ignored by git)
├── create_webdataset.py      # Python script to create a single webdataset shard
├── run_parallel.sh           # Shell script to run the creation script in parallel
├── read_webdataset.py        # Python script to read and verify the created webdataset
├── .gitignore                # Git ignore file
└── README.md                 # This file
```

## Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for environment management.
- Python 3.8+

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd wds-example
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n wds-example python=3.10
    conda activate wds-example
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install webdataset
    ```

### Usage

1.  **Prepare your data (Optional):**
    The project comes with a script to generate dummy data. If you want to use your own data, place your image and text files in the `source_data` directory. Ensure that for each image `sample-N.jpg`, there is a corresponding `sample-N.txt`.

2.  **Generate Dummy Data:**
    If you don't have your own data, you can create 10 sample files by running:
    ```bash
    mkdir -p source_data
    for i in {0..9}; do
      touch source_data/sample-$i.jpg
      echo "This is the description for sample $i." > source_data/sample-$i.txt
    done
    ```

3.  **Create the WebDataset in Parallel:**
    This will run the packing process using 4 parallel workers.
    ```bash
    chmod +x run_parallel.sh
    ./run_parallel.sh
    ```
    This will create several `.tar` files (shards) in the `webdataset_output` directory.

4.  **Verify the Dataset:**
    To ensure the dataset was created correctly, you can read and print its contents:
    ```bash
    python read_webdataset.py --path webdataset_output
    ```
    This will iterate through all the shards and print the key, text, and image size for each sample.

## How It Works

-   **`create_webdataset.py`**: This script is the core of the packing process. It takes a `--shard-id` and `--total-shards` as arguments, which allows it to process a specific subset of the source data. This is the key to parallelization.

-   **`run_parallel.sh`**: This script orchestrates the parallel execution. It launches multiple instances of `create_webdataset.py` in the background, each with a unique `shard-id`. The `wait` command ensures that the script waits for all background jobs to complete.

-   **`read_webdataset.py`**: This script demonstrates how to consume the data. It uses a glob pattern (`shard-*.tar`) to load all the shards into a single `wds.WebDataset` object, which can then be iterated over seamlessly.
