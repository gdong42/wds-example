#!/bin/bash

# Number of parallel processes
NUM_WORKERS=4

# Source and output directories
SOURCE_DIR="source_data"
OUTPUT_DIR="webdataset_output"

# Clear the output directory before starting
rm -f $OUTPUT_DIR/*.tar

# Run the processes in parallel
for (( i=0; i<$NUM_WORKERS; i++ ))
do
  python create_webdataset.py \
    --source $SOURCE_DIR \
    --output $OUTPUT_DIR \
    --shard-id $i \
    --total-shards $NUM_WORKERS &
done

# Wait for all background processes to finish
wait

echo "All shards created."
