#!/bin/bash

# Copy NLTK corpus in all nodes
gsutil cp gs://tfm_init/nltk_data.zip /tmp/nltk_data.zip
sudo unzip /tmp/nltk_data.zip -d /usr/share
export NLTK_DATA=/usr/share/nltk_data

# Initialization scripts to be executed only by the master node.
ROLE=$(/usr/share/google/get_metadata_value attributes/dataproc-role)
if [[ "${ROLE}" == 'Master' ]]; then
  # Create folders in hdfs to store the model, logs and dataset
  hdfs dfs -mkdir hdfs://tfm-m/user/rodrigo_accurso/logs
  hdfs dfs -mkdir hdfs://tfm-m/user/rodrigo_accurso/model
  hdfs dfs -mkdir hdfs://tfm-m/user/rodrigo_accurso/data
  # Copy datasets
  hdfs dfs -cp gs://tfm_main/data/amazon_reviews_us_PC_v1_00_red.tsv.gz hdfs://tfm-m/user/rodrigo_accurso/data/amazon_reviews_us_PC_v1_00_red.tsv.gz
  hdfs dfs -cp gs://tfm_main/data/amazon_reviews_us_PC_v1_00.tsv hdfs://tfm-m/user/rodrigo_accurso/data/amazon_reviews_us_PC_v1_00.tsv
fi
