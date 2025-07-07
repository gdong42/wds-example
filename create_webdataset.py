import webdataset as wds
import os
import argparse
import json
import tarfile
from glob import glob
from PIL import Image
import io

def create_webdataset_with_index(source_path, output_path, shard_id, total_shards):
    """
    Creates a webdataset shard from a subset of the source data with external JSONL index.
    """
    # Find all sample base names in the source directory
    source_files = sorted(glob(os.path.join(source_path, '*.png')))
    
    # Determine the subset of files for this shard
    files_per_shard = (len(source_files) + total_shards - 1) // total_shards
    start_index = shard_id * files_per_shard
    end_index = min(start_index + files_per_shard, len(source_files))
    
    shard_files = source_files[start_index:end_index]

    # Configure the output file names
    output_file = os.path.join(output_path, f'shard-{shard_id:06d}.tar')
    index_file = os.path.join(output_path, f'shard-{shard_id:06d}.jsonl')

    # Index records to be written to JSONL
    index_records = []
    
    # Create the webdataset with index tracking
    with wds.TarWriter(output_file) as sink:
        for i, img_path in enumerate(shard_files):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            cls_path = os.path.join(source_path, base_name + '.cls')
            
            # Check if files exist
            if not os.path.exists(cls_path):
                print(f"Warning: {cls_path} not found, skipping {base_name}")
                continue
            
            # Read files and get metadata
            with open(img_path, 'rb') as img_file:
                img_data = img_file.read()
                
            with open(cls_path, 'rb') as cls_file:
                cls_data = cls_file.read()
            
            # Get image metadata
            try:
                img = Image.open(io.BytesIO(img_data))
                width, height = img.size
                format_name = img.format
            except Exception as e:
                print(f"Warning: Could not read image metadata for {img_path}: {e}")
                width, height, format_name = None, None, None
            
            # Get class label
            try:
                class_label = cls_data.decode('utf-8').strip()
            except:
                class_label = None
            
            # Get file sizes
            img_size = len(img_data)
            cls_size = len(cls_data)
            
            # Write to webdataset
            sample_dict = {
                "__key__": base_name,
                "png": img_data,
                "cls": cls_data
            }
            sink.write(sample_dict)
            
            # Create index record
            # Using wds:// protocol with offset information
            wds_path = f"wds://{os.path.abspath(output_file)}#{base_name}"
            
            index_record = {
                "key": base_name,
                "wds_path": wds_path,
                "shard_id": shard_id,
                "sample_index": i,
                "metadata": {
                    "image": {
                        "width": width,
                        "height": height,
                        "format": format_name,
                        "size_bytes": img_size,
                        "filename": os.path.basename(img_path)
                    },
                    "class": {
                        "label": class_label,
                        "size_bytes": cls_size,
                        "filename": os.path.basename(cls_path)
                    }
                },
                "source_files": {
                    "image": os.path.abspath(img_path),
                    "class": os.path.abspath(cls_path)
                }
            }
            
            index_records.append(index_record)
    
    # After creating the tar file, get actual offsets from the tar file
    try:
        with tarfile.open(output_file, 'r') as tar:
            # Update index records with actual tar offsets
            for record in index_records:
                key = record["key"]
                # Find tar members for this key
                png_member = None
                cls_member = None
                
                for member in tar.getmembers():
                    if member.name == f"{key}.png":
                        png_member = member
                    elif member.name == f"{key}.cls":
                        cls_member = member
                
                if png_member and cls_member:
                    # Update wds_path with actual offset information
                    # Format: wds://path/to/file.tar#{key}#{png_offset}_{png_size}#{cls_offset}_{cls_size}
                    wds_path_with_offset = (
                        f"wds://{os.path.abspath(output_file)}#{key}#"
                        f"png:{png_member.offset}_{png_member.size}#"
                        f"cls:{cls_member.offset}_{cls_member.size}"
                    )
                    record["wds_path"] = wds_path_with_offset
                    record["tar_offsets"] = {
                        "png": {"offset": png_member.offset, "size": png_member.size},
                        "cls": {"offset": cls_member.offset, "size": cls_member.size}
                    }
    
    except Exception as e:
        print(f"Warning: Could not extract tar offsets for shard {shard_id}: {e}")
    
    # Write index file
    with open(index_file, 'w', encoding='utf-8') as f:
        for record in index_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Shard {shard_id} created with {len(shard_files)} samples.")
    print(f"Index file created: {index_file}")
    return len(index_records)

# Keep the original function for backward compatibility
def create_webdataset(source_path, output_path, shard_id, total_shards):
    """
    Creates a webdataset shard from a subset of the source data (original function).
    """
    return create_webdataset_with_index(source_path, output_path, shard_id, total_shards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create WebDataset shards with external JSONL index.")
    parser.add_argument("--source", required=True, help="Path to the source data directory.")
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    parser.add_argument("--shard-id", type=int, required=True, help="The ID for this shard.")
    parser.add_argument("--total-shards", type=int, required=True, help="The total number of shards.")
    parser.add_argument("--with-index", action="store_true", default=True, help="Create external index (default: True).")
    
    args = parser.parse_args()
    
    if args.with_index:
        create_webdataset_with_index(args.source, args.output, args.shard_id, args.total_shards)
    else:
        create_webdataset(args.source, args.output, args.shard_id, args.total_shards)
