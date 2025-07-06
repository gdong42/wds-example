
import ray
import os
import argparse
import shutil

# Import the core function from our original script
from create_webdataset import create_webdataset

# Turn the original function into a Ray remote task
@ray.remote
def create_webdataset_task(source_path, output_path, shard_id, total_shards):
    create_webdataset(source_path, output_path, shard_id, total_shards)

def main(source, output, num_workers):
    # Clean the output directory
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)

    # Initialize Ray
    ray.init()

    print(f"Starting parallel processing with {num_workers} workers using Ray...")

    # Start the remote tasks
    results = []
    for i in range(num_workers):
        results.append(create_webdataset_task.remote(source, output, i, num_workers))

    # Wait for all tasks to complete
    ray.get(results)

    # Shutdown Ray
    ray.shutdown()

    print("All shards created successfully with Ray.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create WebDataset shards in parallel using Ray.")
    parser.add_argument("--source", required=True, help="Path to the source data directory.")
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers.")
    
    args = parser.parse_args()
    
    main(args.source, args.output, args.num_workers)
