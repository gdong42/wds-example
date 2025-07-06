import ray
import argparse
import os
from glob import glob
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(path, num_workers):
    print(f"Starting parallel read with Ray Data WebDataset from path: {path}")

    # 1. 发现所有的分片文件
    shard_files = glob(f"{path}/shard-*.tar")
    if not shard_files:
        print("No shard files found. Exiting.")
        return

    print(f"Found {len(shard_files)} shard files")

    # 2. 使用Ray Data的内置read_webdataset函数
    # 这是最简单和推荐的方法
    try:
        ds = ray.data.read_webdataset(
            shard_files,
            # 设置并行度
            override_num_blocks=num_workers
        )
        
        print(f"Dataset schema: {ds.schema()}")
        
        # 3. 迭代处理数据
        sample_count = 0
        for batch in ds.iter_batches(batch_size=10):
            for i in range(len(batch["__key__"])):
                sample_count += 1
                key = batch["__key__"][i]
                
                # 适配新的数据格式
                if "cls" in batch:
                    cls_data = batch["cls"][i]
                    if isinstance(cls_data, bytes):
                        cls_text = cls_data.decode("utf-8").strip()
                    else:
                        cls_text = str(cls_data).strip()
                else:
                    cls_text = "N/A"
                
                # 获取图像信息
                if "png" in batch:
                    img_data = batch["png"][i]
                    if hasattr(img_data, 'shape'):
                        img_info = f"PNG array shape: {img_data.shape}"
                    else:
                        img_info = f"PNG data size: {len(img_data)} bytes"
                else:
                    img_info = "No PNG data"
                
                print(f"Key: {key}, Class: {cls_text}, Image: {img_info}")
                
                if sample_count >= 10:  # 只显示前10个样本
                    print("... (showing first 10 samples)")
                    break
            
            if sample_count >= 10:
                break
                
        print(f"\nSuccessfully read {sample_count} samples using Ray Data WebDataset.")
        
    except Exception as e:
        logger.error(f"Error reading WebDataset: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a WebDataset in parallel using Ray Data WebDataset.")
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
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        # 清理Ray资源
        ray.shutdown() 