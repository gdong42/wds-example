import argparse
import os
import tarfile
import io
from glob import glob
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(path, num_workers, output_path=None):
    print(f"Starting production-scale parallel read with Spark from path: {path}")
    
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, count, avg, max as spark_max, min as spark_min
        from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"Error: Missing dependencies. Please install: pip install pyspark pillow numpy")
        print(f"Import error: {e}")
        return

    # 1. 发现所有的分片文件
    shard_files = glob(f"{path}/shard-*.tar")
    if not shard_files:
        print("No shard files found. Exiting.")
        return

    print(f"Found {len(shard_files)} shard files")

    # 2. 创建Spark会话（生产配置）
    spark = SparkSession.builder \
        .appName("WebDataset Reader - Production") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()

    try:
        sc = spark.sparkContext
        sc.setLogLevel("WARN")

        # 3. 定义处理单个tar文件的函数（返回轻量级元数据）
        def process_tar_file_metadata(tar_path):
            """处理单个tar文件，只返回元数据，不加载完整图像"""
            samples = []
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    file_groups = {}
                    for member in tar.getmembers():
                        if member.isfile():
                            name_parts = member.name.split('.')
                            if len(name_parts) >= 2:
                                key = name_parts[0]
                                ext = name_parts[1]
                                
                                if key not in file_groups:
                                    file_groups[key] = {}
                                file_groups[key][ext] = member
                    
                    # 只处理元数据，不加载完整图像
                    for key, files in file_groups.items():
                        if 'png' in files and 'cls' in files:
                            try:
                                # 读取图像尺寸（不加载完整图像）
                                png_data = tar.extractfile(files['png']).read()
                                img = Image.open(io.BytesIO(png_data))
                                width, height = img.size
                                channels = len(img.getbands())
                                
                                # 读取类别标签
                                cls_data = tar.extractfile(files['cls']).read().decode('utf-8').strip()
                                
                                samples.append({
                                    'key': key,
                                    'class': int(cls_data),
                                    'height': height,
                                    'width': width,
                                    'channels': channels,
                                    'original_size': len(png_data),
                                    'shard_file': os.path.basename(tar_path)
                                })
                                
                            except Exception as e:
                                logger.warning(f"Failed to process sample {key}: {e}")
                                continue
                                
            except Exception as e:
                logger.error(f"Failed to process tar file {tar_path}: {e}")
                
            return samples

        # 4. 使用Spark并行处理，创建DataFrame（避免collect）
        print("Processing tar files in parallel...")
        shard_rdd = sc.parallelize(shard_files, numSlices=len(shard_files))
        samples_rdd = shard_rdd.flatMap(process_tar_file_metadata)
        
        # 5. 转换为DataFrame进行分布式分析
        print("Converting to DataFrame for distributed analysis...")
        
        # 定义Schema
        schema = StructType([
            StructField("key", StringType(), True),
            StructField("class", IntegerType(), True),
            StructField("height", IntegerType(), True),
            StructField("width", IntegerType(), True),
            StructField("channels", IntegerType(), True),
            StructField("original_size", LongType(), True),
            StructField("shard_file", StringType(), True)
        ])
        
        # 创建DataFrame
        df = spark.createDataFrame(samples_rdd, schema)
        
        # 6. 分布式分析（不使用collect）
        print("\n--- Production-Scale Analysis ---")
        
        # 总样本数
        total_samples = df.count()
        print(f"Total samples: {total_samples}")
        
        # 基本统计（分布式计算）
        print("\nBasic Statistics:")
        df.select("height", "width", "original_size").describe().show()
        
        # 类别分布（分布式计算）
        print("\nClass Distribution (top 20):")
        df.groupBy("class").count().orderBy(col("count").desc()).limit(20).show()
        
        # 图像尺寸分布
        print("\nImage Size Distribution:")
        df.groupBy("height", "width").count().orderBy(col("count").desc()).limit(10).show()
        
        # 分片文件分布
        print("\nShard File Distribution:")
        df.groupBy("shard_file").count().show()
        
        # 7. 可选：将结果写入文件（分布式写入）
        if output_path:
            print(f"\nWriting results to {output_path}...")
            df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
            print("Results written successfully!")
        
        # 8. 只取样本进行验证（而不是collect全部）
        print("\n--- Sample Verification (first 10 samples) ---")
        sample_df = df.limit(10)
        samples = sample_df.collect()  # 只collect少量样本
        
        for sample in samples:
            print(f"Key: {sample['key']}, Class: {sample['class']}, "
                  f"Size: {sample['height']}x{sample['width']}x{sample['channels']}, "
                  f"Original Size: {sample['original_size']}")
        
        print(f"\nSuccessfully processed {total_samples} samples using production-scale Spark.")
        
        # 9. 可选：缓存DataFrame以供后续使用
        df.cache()
        print(f"DataFrame cached for future operations.")
        
    except Exception as e:
        logger.error(f"Error in Spark processing: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read a WebDataset in parallel using Spark (Production Version).")
    parser.add_argument("--path", required=True, help="Path to the webdataset shards directory.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers for reading.")
    parser.add_argument("--output-path", help="Optional path to save results as CSV.")
    args = parser.parse_args()

    try:
        main(args.path, args.num_workers, args.output_path)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise 