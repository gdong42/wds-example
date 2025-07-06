import webdataset as wds
import os
import argparse
from glob import glob

def create_webdataset(source_path, output_path, shard_id, total_shards):
    """
    Creates a webdataset shard from a subset of the source data.
    """
    # Find all sample base names in the source directory
    source_files = sorted(glob(os.path.join(source_path, '*.png')))
    
    # Determine the subset of files for this shard
    files_per_shard = (len(source_files) + total_shards - 1) // total_shards
    start_index = shard_id * files_per_shard
    end_index = min(start_index + files_per_shard, len(source_files))
    
    shard_files = source_files[start_index:end_index]

    # Configure the output file name
    output_file = os.path.join(output_path, f'shard-{shard_id:06d}.tar')

    # Create the webdataset
    with wds.TarWriter(output_file) as sink:
        for i, img_path in enumerate(shard_files):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            cls_path = os.path.join(source_path, base_name + '.cls')
            
            # 检查文件是否存在
            if not os.path.exists(cls_path):
                print(f"Warning: {cls_path} not found, skipping {base_name}")
                continue
            
            with open(img_path, 'rb') as img_file, open(cls_path, 'rb') as cls_file:
                sink.write({
                    "__key__": base_name,
                    "png": img_file.read(),
                    "cls": cls_file.read()
                })

    print(f"Shard {shard_id} created with {len(shard_files)} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create WebDataset shards.")
    parser.add_argument("--source", required=True, help="Path to the source data directory.")
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument("--shard-id", type=int, required=True, help="The ID for this shard.")
    parser.add_argument("--total-shards", type=int, required=True, help="The total number of shards.")
    
    args = parser.parse_args()
    
    create_webdataset(args.source, args.output, args.shard_id, args.total_shards)
