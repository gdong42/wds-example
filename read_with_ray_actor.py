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
    print(f"Starting parallel read with Ray Actor from path: {path}")

    # 1. 发现所有的分片文件
    shard_files = glob(f"{path}/shard-*.tar")
    if not shard_files:
        print("No shard files found. Exiting.")
        return

    print(f"Found {len(shard_files)} shard files")

    # 2. 将分片文件分成多个批次，每个worker处理一部分
    files_per_worker = (len(shard_files) + num_workers - 1) // num_workers
    
    # 3. 定义处理函数，每个worker处理一组分片文件
    @ray.remote
    def process_shards(shard_batch):
        """处理一组分片文件"""
        results = []
        try:
            # 为每个分片文件创建WebDataset
            for shard_file in shard_batch:
                try:
                    dataset = wds.WebDataset([shard_file], shardshuffle=False)
                    for sample in dataset:
                        try:
                            results.append({
                                "__key__": sample["__key__"],
                                "png": sample["png"],
                                "cls": sample["cls"].decode("utf-8")
                            })
                        except (KeyError, UnicodeDecodeError) as e:
                            logger.warning(f"Failed to decode sample from {shard_file}: {e}")
                            continue
                except Exception as e:
                    logger.error(f"Failed to process shard {shard_file}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in process_shards: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
        return results

    # 4. 分配任务给workers
    tasks = []
    for i in range(num_workers):
        start_idx = i * files_per_worker
        end_idx = min(start_idx + files_per_worker, len(shard_files))
        worker_shards = shard_files[start_idx:end_idx]
        
        if worker_shards:  # 只有当有文件需要处理时才创建任务
            task = process_shards.remote(worker_shards)
            tasks.append(task)

    # 5. 等待所有任务完成并收集结果
    try:
        all_results = ray.get(tasks)
        
        # 合并所有结果
        combined_results = []
        for worker_results in all_results:
            combined_results.extend(worker_results)
        
        # 6. 显示结果
        print(f"\n--- Verifying Samples ---")
        for i, sample in enumerate(combined_results[:10]):  # 只显示前10个样本
            key = sample["__key__"]
            cls = sample["cls"]
            png_size = len(sample["png"])
            
            print(f"Key: {key}, Class: {cls.strip()}, Image Size: {png_size}")
        
        if len(combined_results) > 10:
            print("... (showing first 10 samples)")
        
        print(f"\nSuccessfully read {len(combined_results)} samples using Ray Actor.")
        
    except Exception as e:
        logger.error(f"Error collecting results: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a WebDataset in parallel using Ray Actor.")
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