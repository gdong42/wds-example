#!/usr/bin/env python3
"""
Example script demonstrating how to read WebDataset using external JSONL index files.
This shows both direct access using offset information and streaming methods.
"""

import json
import os
import tarfile
import io
from PIL import Image
import webdataset as wds
import argparse

def load_index(index_file):
    """Load index file and return records."""
    records = []
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

def load_shard_summary(summary_file):
    """Load shard summary file."""
    with open(summary_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_sample_by_offset(tar_path, png_offset, png_size, cls_offset, cls_size):
    """
    Read a sample directly from tar file using offset information.
    This is more efficient than extracting the entire tar.
    """
    with open(tar_path, 'rb') as f:
        # Read PNG data
        f.seek(png_offset)
        png_data = f.read(png_size)
        
        # Read CLS data
        f.seek(cls_offset)
        cls_data = f.read(cls_size)
        
        return png_data, cls_data

def read_sample_by_key_with_index(index_file, key):
    """
    Read a specific sample using index file.
    """
    records = load_index(index_file)
    
    for record in records:
        if record['key'] == key:
            if 'tar_offsets' in record:
                # Extract tar file path from wds_path
                wds_path = record['wds_path']
                if wds_path.startswith('wds://'):
                    tar_path = wds_path.split('#')[0][6:]  # Remove 'wds://' prefix
                    
                    png_offset = record['tar_offsets']['png']['offset']
                    png_size = record['tar_offsets']['png']['size']
                    cls_offset = record['tar_offsets']['cls']['offset']
                    cls_size = record['tar_offsets']['cls']['size']
                    
                    png_data, cls_data = read_sample_by_offset(tar_path, png_offset, png_size, cls_offset, cls_size)
                    
                    # Decode data
                    image = Image.open(io.BytesIO(png_data))
                    class_label = cls_data.decode('utf-8').strip()
                    
                    return {
                        'key': key,
                        'image': image,
                        'class': class_label,
                        'metadata': record['metadata']
                    }
            else:
                print(f"Warning: No tar offset information for key {key}")
                return None
    
    print(f"Key {key} not found in index")
    return None

def stream_webdataset_with_index(output_dir, batch_size=1):
    """
    Stream WebDataset using traditional WebDataset methods with index information.
    """
    # Find all tar files
    tar_files = []
    for i in range(10):  # Assume max 10 shards
        tar_path = os.path.join(output_dir, f'shard-{i:06d}.tar')
        if os.path.exists(tar_path):
            tar_files.append(tar_path)
    
    if not tar_files:
        print("No tar files found")
        return
    
    print(f"Found {len(tar_files)} tar files")
    
    # Create WebDataset
    dataset = wds.WebDataset(tar_files)
    
    # Process samples
    def decode_sample(sample):
        key = sample['__key__']
        image = Image.open(io.BytesIO(sample['png']))
        class_label = sample['cls'].decode('utf-8').strip()
        return {
            'key': key,
            'image': image,
            'class': class_label
        }
    
    # Stream and process
    processed_dataset = dataset.map(decode_sample)
    
    count = 0
    for sample in processed_dataset:
        print(f"Sample {count}: key={sample['key']}, class={sample['class']}, image_size={sample['image'].size}")
        count += 1
        if count >= 10:  # Limit output for demo
            break
    
    print(f"Processed {count} samples")

def analyze_dataset_with_index(summary_file):
    """
    Analyze dataset using summary file.
    """
    summary = load_shard_summary(summary_file)
    
    print("Dataset Analysis:")
    print(f"  Total shards: {summary['total_shards']}")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Average samples per shard: {summary['dataset_stats']['avg_samples_per_shard']:.2f}")
    
    print("\nClass Distribution:")
    for class_name, count in summary['dataset_stats']['class_distribution'].items():
        percentage = (count / summary['total_samples']) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    print("\nShard Information:")
    for shard_name, shard_info in summary['shards'].items():
        print(f"  {shard_name}: {shard_info['num_samples']} samples")

def query_samples_by_class(index_file, class_label, limit=5):
    """
    Query samples by class label using index.
    """
    records = load_index(index_file)
    
    matching_samples = []
    for record in records:
        if (record.get('metadata', {}).get('class', {}).get('label') == class_label):
            matching_samples.append(record)
            if len(matching_samples) >= limit:
                break
    
    print(f"Found {len(matching_samples)} samples with class '{class_label}':")
    for sample in matching_samples:
        print(f"  Key: {sample['key']}, Shard: {sample['shard_id']}")
        if 'metadata' in sample:
            img_meta = sample['metadata'].get('image', {})
            print(f"    Image: {img_meta.get('width')}x{img_meta.get('height')} {img_meta.get('format')}")
    
    return matching_samples

def main():
    parser = argparse.ArgumentParser(description="Read WebDataset using external index files")
    parser.add_argument("--output-dir", required=True, help="Path to the WebDataset output directory")
    parser.add_argument("--action", choices=['stream', 'analyze', 'query', 'read-sample'], 
                       default='analyze', help="Action to perform")
    parser.add_argument("--key", help="Sample key to read (for read-sample action)")
    parser.add_argument("--class", help="Class label to query (for query action)")
    parser.add_argument("--limit", type=int, default=5, help="Limit for query results")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    master_index_file = os.path.join(output_dir, 'master_index.jsonl')
    shard_summary_file = os.path.join(output_dir, 'shard_summary.json')
    
    # Check if files exist
    if not os.path.exists(master_index_file):
        print(f"Error: Master index file not found: {master_index_file}")
        return
    
    if args.action == 'analyze':
        if os.path.exists(shard_summary_file):
            analyze_dataset_with_index(shard_summary_file)
        else:
            print(f"Warning: Shard summary file not found: {shard_summary_file}")
            print("Analyzing from master index file...")
            records = load_index(master_index_file)
            print(f"Total samples: {len(records)}")
    
    elif args.action == 'stream':
        stream_webdataset_with_index(output_dir)
    
    elif args.action == 'query':
        if not args.class:
            print("Error: --class argument required for query action")
            return
        query_samples_by_class(master_index_file, args.class, args.limit)
    
    elif args.action == 'read-sample':
        if not args.key:
            print("Error: --key argument required for read-sample action")
            return
        
        sample = read_sample_by_key_with_index(master_index_file, args.key)
        if sample:
            print(f"Successfully read sample: {sample['key']}")
            print(f"  Class: {sample['class']}")
            print(f"  Image size: {sample['image'].size}")
            print(f"  Image format: {sample['image'].format}")
            if 'metadata' in sample:
                print(f"  Metadata: {sample['metadata']}")
        else:
            print(f"Failed to read sample with key: {args.key}")

if __name__ == "__main__":
    main() 