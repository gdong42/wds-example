import ray
import webdataset as wds
import argparse
import os
from glob import glob
import logging
import traceback

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(path, num_workers):
    print(f"Starting parallel read with Ray Data from path: {path}")

    # 1. Discover all the shard files
    shard_files = glob(f"{path}/shard-*.tar")
    if not shard_files:
        print("No shard files found. Exiting.")
        return

    print(f"Found {len(shard_files)} shard files")

    # 2. 创建Ray Dataset，每个元素是一个分片文件路径
    # 这是Ray Data的核心：将文件路径作为数据集
    ds = ray.data.from_items(shard_files)

    # 3. 定义处理函数，用于处理每个分片文件
    def process_shard_file(batch):
        """
        处理一个batch的分片文件路径
        batch: {"item": ["/path/to/shard1.tar", "/path/to/shard2.tar", ...]}
        """
        results = {"__key__": [], "png": [], "cls": []}
        
        for shard_file in batch["item"]:
            try:
                # 使用webdataset读取单个分片文件
                dataset = wds.WebDataset([shard_file], shardshuffle=False)
                for sample in dataset:
                    try:
                        results["__key__"].append(sample["__key__"])
                        results["png"].append(sample["png"])
                        results["cls"].append(sample["cls"].decode("utf-8"))
                    except (KeyError, UnicodeDecodeError) as e:
                        logger.warning(f"Failed to decode sample from {shard_file}: {e}")
                        continue
            except Exception as e:
                logger.error(f"Failed to process shard {shard_file}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
        
        return results

    # 4. 使用Ray Data的map_batches进行并行处理
    # 这里每个batch包含一个分片文件，确保每个worker处理完整的分片
    processed_ds = ds.map_batches(
        process_shard_file,
        batch_size=1,  # 每个batch处理一个分片文件
        num_cpus=1,    # 每个任务使用1个CPU
        # 设置并行度
        concurrency=num_workers
    )

    # 5. 收集并显示结果
    print(f"\n--- Verifying Samples ---")
    count = 0
    
    try:
        # 遍历处理后的数据集
        for batch in processed_ds.iter_batches():
            # 每个batch包含多个样本
            for i in range(len(batch["__key__"])):
                count += 1
                key = batch["__key__"][i]
                cls = batch["cls"][i]
                png_size = len(batch["png"][i])
                
                print(f"Key: {key}, Class: {cls.strip()}, Image Size: {png_size}")
                
                if count >= 10:  # 只显示前10个样本
                    print("... (showing first 10 samples)")
                    break
            
            if count >= 10:
                break
                
    except Exception as e:
        logger.error(f"Error during iteration: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    print(f"\nSuccessfully read {count} samples using Ray Data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a WebDataset in parallel using Ray Data.")
    parser.add_argument("--path", required=True, help="Path to the webdataset shards directory.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers for reading.")
    args = parser.parse_args()

    try:
        # 初始化Ray
        ray.init()
        
        # 运行主函数
        main(args.path, args.num_workers)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        # 清理Ray资源
        ray.shutdown()
