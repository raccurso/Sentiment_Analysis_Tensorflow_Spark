#!/bin/bash

# Copy NLTK corpus in all nodes
gsutil cp gs://main_tfm/init/nltk_data.zip /tmp/nltk_data.zip
sudo unzip /tmp/nltk_data.zip -d /usr/share
export NLTK_DATA=/usr/share/nltk_data

# Initialization scripts to be executed only by the master node.
ROLE=$(/usr/share/google/get_metadata_value attributes/dataproc-role)
if [[ "${ROLE}" == 'Master' ]]; then
  # Create folders in hdfs to store the model, logs and dataset
  hdfs dfs -mkdir hdfs://tfm-m/user/hdfs/model
  hdfs dfs -mkdir hdfs://tfm-m/user/hdfs/logs
  hdfs dfs -mkdir hdfs://tfm-m/user/hdfs/data
  # Copy datasets
  hdfs dfs -cp gs://main_tfm/data/amazon_reviews_us_PC_v1_00.tsv hdfs://tfm-m/user/hdfs/data/amazon_reviews_us_PC_v1_00.tsv
fi
