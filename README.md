# WebDataset Parallel Processing Examples

This project provides comprehensive examples of how to efficiently process WebDataset format data using various parallel computing frameworks. It demonstrates multiple approaches for both **creating** and **reading** WebDataset files at scale, making it an excellent starting point for building efficient data pipelines for deep learning models.

## 🚀 Features

- **Multiple Creation Methods**: Shell script and Ray-based parallel WebDataset creation
- **Diverse Reading Approaches**: Ray Actor, Ray Data, Ray Data WebDataset, and Spark implementations
- **Production-Ready**: Includes production-scale considerations and best practices
- **Multimodal Data Support**: Handles images and text data efficiently
- **Scalable Architecture**: All methods can scale from single machine to distributed clusters
- **Performance Optimized**: Avoids common pitfalls like memory overflow and network bottlenecks

## 📁 Project Structure

```
.
├── source_data/                    # Raw source data (images, text, etc.)
├── webdataset_output/              # Generated .tar shards (ignored by git)
├── create_webdataset.py            # Core WebDataset creation script
├── create_with_ray.py              # Ray-based parallel creation
├── run_parallel.sh                 # Shell script for parallel creation
├── read_webdataset.py              # Simple WebDataset reader
├── read_with_ray_actor.py          # Ray Actor-based reader
├── read_with_ray_data.py           # Ray Data pipeline reader
├── read_with_ray_data_webdataset.py # Ray Data WebDataset reader (recommended)
├── read_with_spark.py              # Apache Spark reader (production-scale)
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Conda (recommended) or pip

### Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd wds-example

# Create and activate conda environment
conda create -n wds-example python=3.10
conda activate wds-example

# Install dependencies
pip install webdataset ray[default] pyspark pillow numpy
```

## 📊 Data Preparation

### Using Real Data
Place your image and text files in the `source_data` directory. For each image `N.png`, ensure there's a corresponding `N.cls` file.

### Generate Sample Data (for testing)
```bash
mkdir -p source_data
for i in {0..9}; do
  # Create dummy PNG files (replace with real images)
  touch source_data/sample-$i.png
  echo "$((RANDOM % 1000))" > source_data/sample-$i.cls
done
```

## 🔧 Creating WebDatasets

### Method 1: Shell Script (Simple)
```bash
chmod +x run_parallel.sh
./run_parallel.sh
```

### Method 2: Ray Framework (Recommended)
```bash
python create_with_ray.py --source source_data --output webdataset_output --num-workers 4
```

## 📖 Reading WebDatasets

### 1. Simple Reader
Basic WebDataset reading without parallelization:
```bash
python read_webdataset.py --path webdataset_output
```

### 2. Ray Actor (Manual Parallelization)
Uses Ray remote functions for manual task distribution:
```bash
python read_with_ray_actor.py --path webdataset_output --num-workers 4
```

**Best for**: Fine-grained control over parallel execution

### 3. Ray Data Pipeline (Declarative)
Uses Ray Data's declarative data processing pipeline:
```bash
python read_with_ray_data.py --path webdataset_output --num-workers 4
```

**Best for**: Complex data transformation pipelines

### 4. Ray Data WebDataset (Recommended for ML)
Uses Ray Data's built-in WebDataset support with automatic image decoding:
```bash
python read_with_ray_data_webdataset.py --path webdataset_output --num-workers 4
```

**Best for**: Machine learning workflows, automatic image preprocessing

### 5. Apache Spark (Production Scale)
Production-ready implementation that avoids common scaling pitfalls:
```bash
python read_with_spark.py --path webdataset_output --num-workers 4
```

**Best for**: Large-scale production deployments, data analytics

## 🎯 Method Comparison

| Method | Complexity | Performance | Best Use Case | Scalability |
|--------|------------|-------------|---------------|-------------|
| Simple Reader | Low | Basic | Development/Testing | Single Machine |
| Ray Actor | Medium | Good | Custom Logic | Multi-Machine |
| Ray Data | Medium | Good | Data Pipelines | Multi-Machine |
| Ray Data WebDataset | Low | Excellent | ML Training | Multi-Machine |
| Apache Spark | High | Excellent | Production/Analytics | Enterprise Scale |

## 🔍 Key Features by Method

### Ray Data WebDataset
- ✅ Automatic image decoding to numpy arrays
- ✅ Built-in WebDataset format support
- ✅ Seamless integration with ML frameworks
- ✅ Optimized for computer vision workflows

### Apache Spark
- ✅ Production-scale data processing
- ✅ Advanced DataFrame analytics
- ✅ SQL query support
- ✅ Avoids memory overflow issues
- ✅ Distributed statistics and aggregations

## 🏭 Production Considerations

### Memory Management
- **Avoid**: `collect()` operations on large datasets
- **Use**: Distributed operations (`count()`, `describe()`, `groupBy()`)
- **Implement**: Proper data partitioning strategies

### Scalability Best Practices
- Process metadata instead of full images when possible
- Use appropriate batch sizes for your hardware
- Configure memory settings based on data size
- Implement proper error handling and retry logic

## 🚀 Performance Tips

1. **For ML Training**: Use `read_with_ray_data_webdataset.py`
2. **For Data Analysis**: Use `read_with_spark.py`
3. **For Custom Logic**: Use `read_with_ray_actor.py`
4. **For Development**: Use `read_webdataset.py`

## 📈 Scaling Guidelines

| Data Size | Recommended Method | Configuration |
|-----------|-------------------|---------------|
| < 1GB | Any method | Default settings |
| 1GB - 10GB | Ray Data WebDataset | 4-8 workers |
| 10GB - 100GB | Ray Data WebDataset | 8-16 workers |
| 100GB+ | Apache Spark | Cluster deployment |

## 🔧 Configuration Examples

### Ray Configuration
```bash
# Local multi-core
python read_with_ray_data_webdataset.py --path webdataset_output --num-workers 8

# Cluster deployment (requires Ray cluster setup)
ray start --head --port=10001
python read_with_ray_data_webdataset.py --path webdataset_output --num-workers 32
```

### Spark Configuration
```bash
# Local mode
python read_with_spark.py --path webdataset_output --num-workers 4

# Cluster mode (requires Spark cluster)
spark-submit --master spark://master:7077 read_with_spark.py --path webdataset_output
```

## 🤝 Contributing

Feel free to submit issues and pull requests to improve these examples!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).