import numpy as np
import pandas as pd
import tensorflow as tf
import time
from datetime import datetime
from tensorflowonspark import TFNode
from tensorflow.python.lib.io import file_io
import model

# Constants
EVAL_X_FILENAME = 'eval_x.csv'
EVAL_Y_FILENAME = 'eval_y.csv'
TRAIN_FILENAME = 'train.csv'
TRAIN_SAMPLE_FILENAME = 'train_sample.csv'
TEST_FILENAME = 'test.tsv'
DICTIONARY_FILENAME = 'dictionary.json'
SAMPLE_LENGHT = 100
DICTIONARY_LENGHT = 10000
SEED = 1234
TRAIN_SPLIT = 0.9
EVAL_SPLIT = 0.05
TEST_SPLIT = 0.05


# SessionRunHook to terminate RDD feeding if the training loop exits before the entire RDD is consumed.
class StopFeedHook(tf.estimator.SessionRunHook):
    def __init__(self, tf_feed):
        self._tf_feed = tf_feed

    def end(self, session):
        self._tf_feed.terminate()
        self._tf_feed.next_batch(1)


# Load the evaluation data from GS and store it in a Pandas dataframe
def load_eval_data(eval_folder):
	with file_io.FileIO(eval_folder + EVAL_X_FILENAME, 'r') as f:
		x_eval = pd.read_csv(f, sep=",", header=0).values.astype('float32')
	with file_io.FileIO(eval_folder + EVAL_Y_FILENAME, 'r') as f2:
		y_eval = pd.read_csv(f2, sep=",", header=0).values.astype('int32')
	return x_eval, y_eval


'''
Description:
  Main function executed by workers when training the model with Tensorflow.

Input parameters:
  - args: Input parameters from the main function
  - crx: Spark context

Details:
  - Create the Tensorflow estimator from Keras model
  - Set up all configurations needed by Tensorflow estimator
  - Define input functions for training and evaluation
  - Execute training and export final model
'''
def main_fun(args, ctx):
	print('<<<<<<<<<<<<<<<<<< JOB NAME: ' + ctx.job_name + ' >>>>>>>>>>>>>>>>>>>>>')

	tf.logging.set_verbosity(tf.logging.INFO)
	
	# Set this config for logs and checkpoints
	run_config = tf.estimator.RunConfig(
					model_dir=args.model_folder,
					tf_random_seed=None,
					save_summary_steps=100,
					save_checkpoints_steps=100,
					session_config=None,
					log_step_count_steps=100,
					keep_checkpoint_max=5,        
					keep_checkpoint_every_n_hours=1)
	
	# Create the Keras model and convert it in an estimator
	keras_model = model.get_model(DICTIONARY_LENGHT, args.num_classes, SAMPLE_LENGHT)
	estimator = tf.keras.estimator.model_to_estimator(keras_model, model_dir=args.model_folder, config=run_config)
		
	tf_feed = TFNode.DataFeed(ctx.mgr)
	
	# Function to be called for feeding Tensorflow during training from Spark RDD
	# Returns a tuple with an array of features and an array of labels for batch-size elements
	def rdd_generator():
		while not tf_feed.should_stop():
			batch = tf_feed.next_batch(1)
			if len(batch) > 0:
				record = batch[0]
				data_array = np.array(record).astype('float32')[:(-1*args.num_classes)]
				labels_array = np.array(record).astype('int32')[(-1*args.num_classes):]				
				yield (data_array, labels_array)
			else:
				return
	
	# Input function of training data for Tensorflow estimator 
	def train_input_fn():
		ds = tf.data.Dataset.from_generator(rdd_generator,
											(tf.float32, tf.int32),
											(tf.TensorShape([SAMPLE_LENGHT]), tf.TensorShape([args.num_classes])))
		ds = ds.batch(args.batch_size)
		return ds
		
	# Add a hook to terminate the RDD data feed when the session ends
	train_hooks = [StopFeedHook(tf_feed)]

	# Load the evaluation dataset from GS in Pandas dataframe
	x_eval, y_eval = load_eval_data(args.output_folder)
	
	# Input function of evaluation data for Tensorflow estimator 
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
											x={"features_layer_input": x_eval},
											y=y_eval,
											num_epochs=1,
											shuffle=False)
	
	# Estimator settings and export model
	feature_spec = {'features_layer_input': 
					tf.placeholder(tf.float32, shape=[None, SAMPLE_LENGHT])}
	exporter = tf.estimator.FinalExporter("serving", 
						serving_input_receiver_fn=tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec))
	train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, 
						max_steps=args.steps, hooks=train_hooks)
	eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, exporters=exporter, steps=100, 
						throttle_secs=0, start_delay_secs=0)
	
	# Train and evaluate model with Tensorflow estimator
	tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
	
	# Wait for all other nodes to complete (via done files)
	done_dir = "{}/done".format(ctx.absolute_path(args.model_folder))
	print("Writing done file to: {}".format(done_dir))
	tf.gfile.MakeDirs(done_dir)
	with tf.gfile.GFile("{}/{}".format(done_dir, ctx.task_index), 'w') as done_file:
		done_file.write("done")
	
	for i in range(60):
		if len(tf.gfile.ListDirectory(done_dir)) < len(ctx.cluster_spec['worker']):
			print("{} Waiting for other nodes {}".format(datetime.now().isoformat(), i))
			time.sleep(1)
		else:
			print("{} All nodes done".format(datetime.now().isoformat()))
		break

'''
Description:
  Main function for data processing and model training.

Input parameters:
  --mode: mode of execution. Allowed values:
      - all: executes the NLP pipeline to process raw data and train the model
	  - nlp: only executes the NLP pipeline and stores processed datasets
	  - train: train the model from processed datasets
  --train_file: file with training data
  --output_folder: folder to store the datasets and the dictionary in output
  --model_folder: folder to write model and checkpoints
  --output_hdfs: folder to write train and test datasets in hdfs
  --batch_size: number of records per batch
  --epochs: number of epochs of training data  
  --steps: number of steps for training
  --tensorboard: launch tensorboard process
  --train_sample_fract: fraction of train sample for evaluating training accuracy

Details:
  - Parse input parameters
  - If mode = all:
      - Execute the NLP pipeline to process raw data
	  - Stores preocessed datasets and dictionary to output folder
	  - Train the model with Keras
  - If mode = nlp:
      - Only execute the NLP pipeline to process raw data
	  - Stores preocessed datasets and dictionary to output folder
  - If mode = train:
      - Only train the model with Keras
	  Note: this mode can be executed only after "all" mode in order to have 
	        the preprocessed dataset in hdfs.
'''
if __name__ == '__main__':
	import argparse
	from pyspark.context import SparkContext
	from pyspark.conf import SparkConf
	import pyspark
	from tensorflowonspark import TFCluster
	import nlp_pipeline as nlp
	import json
	
	# Start Spark context and print resources in use
	sc = SparkContext(conf=SparkConf().setAppName("distributed_nlp"))
	executors = sc._conf.get("spark.executor.instances")
	num_executors = int(executors) if executors is not None else 1
	print('{} spark.executor.instances: {} '.format(datetime.now(), num_executors))
	print('{} spark.task.cpus: {}'.format(datetime.now(), sc._conf.get("spark.task.cpus")))
	print('{} spark.executor.cores: {}'.format(datetime.now(), sc._conf.get("spark.executor.cores")))
	
	# Parse input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", help="mode of execution (all/nlp/train)")	
	parser.add_argument("--train_file", help="file with training data")
	parser.add_argument("--output_folder", help="folder to store the datasets and the dictionary in output")
	parser.add_argument("--model_folder", help="folder to write model and checkpoints", default='')
	parser.add_argument("--output_hdfs", help="folder to write train and test datasets in hdfs", default='')
	parser.add_argument("--batch_size", help="number of records per batch", type=int, default=100)
	parser.add_argument("--epochs", help="number of epochs of training data", type=int, default=None)
	parser.add_argument("--steps", help="number of steps for training", type=int, default=100)
	parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
	parser.add_argument("--train_sample_fract", help="fraction of train sample for evaluating training accuracy", type=float, default=0)
	parser.add_argument("--num_classes", help="number of classes", type=int, default=5)
	args = parser.parse_args()
	print("{} Parameters: {}".format(datetime.now(), args))
	
	if (args.mode == 'all') or (args.mode == 'nlp'):
		# 'all' or 'nlp' mode. Execute the NLP pipeline in order to process raw data and get the final datasets for training
		(df, dict_map, test_raw_data) = nlp.nlp_pipeline(args.train_file, SAMPLE_LENGHT, DICTIONARY_LENGHT, 
														 num_executors, None, TEST_SPLIT, args.num_classes, sc)

		# Split processed dataframe into train and evaluation datasets
		(train_df, eval_df) = df.randomSplit([TRAIN_SPLIT, EVAL_SPLIT], seed=SEED)
		
		# Store the evaluation dataset in GS separating X and Y columns.
		print('{} Collecting eval data to pandas'.format(datetime.now()))
		eval_pd = eval_df.toPandas()
		print('{} Storing eval dataset to GS'.format(datetime.now()))
		with file_io.FileIO(args.output_folder + EVAL_X_FILENAME, 'w') as f:
			eval_pd.iloc[:,:(-1*args.num_classes)].to_csv(f, sep=",", header=0, index=False)
		with file_io.FileIO(args.output_folder + EVAL_Y_FILENAME, 'w') as f:
			eval_pd.iloc[:,(-1*args.num_classes):].to_csv(f, sep=",", header=0, index=False)

		# Store preprocessed training dataset in Hadoop for model tuning
		print('{} Storing train dataset to Hadoop'.format(datetime.now()))
		train_df.write.format('csv').save(args.output_hdfs + TRAIN_FILENAME, sep=',', header=False)

		# Store raw test dataset in Hadoop for testing accuracy  (same format as raw source)
		print('{} Storing test dataset to Hadoop'.format(datetime.now()))
		test_raw_data.write.format('csv').save(args.output_hdfs + TEST_FILENAME, sep='\t', header=True)
		
		# Store the dictionary word-index
		print('{} Storing dictionary to GS'.format(datetime.now()))
		with file_io.FileIO(args.output_folder + DICTIONARY_FILENAME, 'w') as f:
			json.dump(dict_map, f)

		# Save a sample of training sample in Hadoop for evaluating training accuracy 
		if args.train_sample_fract > 0:
			train_pd = train_df.sample(withReplacement=False, fraction=args.train_sample_fract).toPandas()
			with file_io.FileIO(args.output_folder + TRAIN_SAMPLE_FILENAME, 'w') as f:
				train_pd.to_csv(f, sep=",", header=0, index=False)		
	elif args.mode == 'train':
		# 'train' mode. Read preprocessed train data with word indexes and train the model
		sql = pyspark.SQLContext(sc)
		train_df = sql.read.csv(args.train_file, sep=",", inferSchema=True, header=False)
	else:
		print('Wrong value for parameter mode. Allowed: all, nlp or train')
		quit()		

	if (args.mode == 'all') or (args.mode == 'train'):
		# Cache the dataset before training
		train_rdd = train_df.rdd.cache()
		
		# Train the model in the Spark cluster
		print('{} Starting model training'.format(datetime.now()))
		num_ps = 1
		cluster = TFCluster.run(sc, main_fun, args, num_executors, num_ps, args.tensorboard, TFCluster.InputMode.SPARK, 
								log_dir=args.model_folder, master_node='master', reservation_timeout=60)
		cluster.train(train_rdd, args.epochs, feed_timeout=4800)
	
		print('{} End model training'.format(datetime.now()))
		cluster.shutdown()

	print('{} End program'.format(datetime.now()))
