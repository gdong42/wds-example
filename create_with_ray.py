import ray
import os
import argparse
import shutil
import json
import glob

# Import the core function from our original script
from create_webdataset import create_webdataset_with_index

# Turn the original function into a Ray remote task
@ray.remote
def create_webdataset_task(source_path, output_path, shard_id, total_shards):
    """Ray remote task for creating WebDataset with index."""
    return create_webdataset_with_index(source_path, output_path, shard_id, total_shards)

def merge_shard_indexes(output_path, num_shards):
    """
    Merge individual shard JSONL index files into a master index file.
    """
    master_index_file = os.path.join(output_path, 'master_index.jsonl')
    shard_summary_file = os.path.join(output_path, 'shard_summary.json')
    
    all_records = []
    shard_info = {}
    total_samples = 0
    
    print("Merging shard indexes...")
    
    for shard_id in range(num_shards):
        shard_index_file = os.path.join(output_path, f'shard-{shard_id:06d}.jsonl')
        
        if os.path.exists(shard_index_file):
            shard_records = []
            with open(shard_index_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line.strip())
                        all_records.append(record)
                        shard_records.append(record)
            
            # Collect shard statistics
            shard_info[f'shard-{shard_id:06d}'] = {
                'shard_id': shard_id,
                'num_samples': len(shard_records),
                'tar_file': os.path.join(output_path, f'shard-{shard_id:06d}.tar'),
                'index_file': shard_index_file,
                'sample_keys': [r['key'] for r in shard_records]
            }
            total_samples += len(shard_records)
            print(f"  Shard {shard_id}: {len(shard_records)} samples")
        else:
            print(f"  Warning: Shard index file {shard_index_file} not found")
    
    # Write master index file
    with open(master_index_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Write shard summary
    summary = {
        'total_shards': num_shards,
        'total_samples': total_samples,
        'master_index_file': master_index_file,
        'shards': shard_info,
        'dataset_stats': {
            'avg_samples_per_shard': total_samples / num_shards if num_shards > 0 else 0,
            'class_distribution': get_class_distribution(all_records)
        }
    }
    
    with open(shard_summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Master index created: {master_index_file}")
    print(f"Shard summary created: {shard_summary_file}")
    print(f"Total samples across all shards: {total_samples}")
    
    return master_index_file, shard_summary_file

def get_class_distribution(records):
    """Get class distribution from index records."""
    class_counts = {}
    for record in records:
        if 'metadata' in record and 'class' in record['metadata']:
            label = record['metadata']['class'].get('label')
            if label:
                class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

def load_index(index_file):
    """
    Load index file and return records.
    """
    records = []
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

def find_sample_by_key(index_file, key):
    """
    Find a sample by key in the index file.
    """
    records = load_index(index_file)
    for record in records:
        if record['key'] == key:
            return record
    return None

def main(source, output, num_workers, create_master_index=True):
    # Clean the output directory
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)

    # Initialize Ray
    ray.init()

    print(f"Starting parallel WebDataset creation with indexing using {num_workers} workers...")

    # Start the remote tasks
    results = []
    for i in range(num_workers):
        results.append(create_webdataset_task.remote(source, output, i, num_workers))

    # Wait for all tasks to complete and collect results
    shard_results = ray.get(results)
    
    # Shutdown Ray
    ray.shutdown()

    print("All shards created successfully with Ray.")
    
    # Print shard statistics
    total_samples = sum(shard_results)
    print(f"Total samples processed: {total_samples}")
    for i, count in enumerate(shard_results):
        print(f"  Shard {i}: {count} samples")
    
    # Create master index if requested
    if create_master_index:
        master_index_file, shard_summary_file = merge_shard_indexes(output, num_workers)
        
        # Verify the created files
        print(f"\nCreated files:")
        for i in range(num_workers):
            tar_file = os.path.join(output, f'shard-{i:06d}.tar')
            jsonl_file = os.path.join(output, f'shard-{i:06d}.jsonl')
            if os.path.exists(tar_file):
                print(f"  {tar_file} ({os.path.getsize(tar_file)} bytes)")
            if os.path.exists(jsonl_file):
                print(f"  {jsonl_file} ({os.path.getsize(jsonl_file)} bytes)")
        
        print(f"  {master_index_file} ({os.path.getsize(master_index_file)} bytes)")
        print(f"  {shard_summary_file} ({os.path.getsize(shard_summary_file)} bytes)")
        
        return master_index_file, shard_summary_file
    
    return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create WebDataset shards with external indexing in parallel using Ray.")
    parser.add_argument("--source", required=True, help="Path to the source data directory.")
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--no-master-index", action="store_true", help="Skip creating master index file.")
    
    args = parser.parse_args()
    
    main(args.source, args.output, args.num_workers, create_master_index=not args.no_master_index)
