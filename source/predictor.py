import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from tensorflow.python.lib.io import file_io
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import pyspark
import nlp_pipeline as nlp
import argparse
from datetime import datetime
import math

# Constants
SAMPLE_LENGHT = 100
DICTIONARY_LENGHT = 10000
MAX_PREDICTIONS = 10000

# Load pre-processed data (word indexes) and labels stored in one file
def load_one_file(file_name, num_classes):
	with file_io.FileIO(file_name, 'r') as f:
		eval = pd.read_csv(f, sep=",", header=0)
    # Split features and labels columns
	x_eval = eval.iloc[:,:(-1*num_classes)].values.astype('float32')
	y_eval = eval.iloc[:,(-1*num_classes):].values.astype('int32')
	return (x_eval, y_eval)

# Load pre-processed data (word indexes)
def load_csv(file_name):
	with file_io.FileIO(file_name, 'r') as f:
		data = pd.read_csv(f, sep=",", header=0)
	return data

# Execute the NLP pipeline over raw data with PySpark
def execute_NLP(input_file, dict_file, num_classes):
	sc = SparkContext(conf=SparkConf().setAppName("predictor"))
	executors = sc._conf.get("spark.executor.instances")
	num_executors = int(executors) if executors is not None else 1
	(df, dict_none, df_none) = nlp.nlp_pipeline(input_file, SAMPLE_LENGHT, DICTIONARY_LENGHT, 
												num_executors, dict_file, None, num_classes, sc)
	return df

# Execute predictions with pre-trained model.
# Predictions are done in batches of 10.000 in order to avoid an error for the limit of resources
def predict(model_folder, x):
	predict_fn = predictor.from_saved_model(model_folder)
	total = math.ceil(x.shape[0] / MAX_PREDICTIONS)
	all_preds = ()
	for i in np.arange(total):
		params = {'features_layer_input': x[i*MAX_PREDICTIONS:(i+1)*MAX_PREDICTIONS]}
		predictions = predict_fn(params)
		if len(all_preds) == 0:
			all_preds = predictions['output_layer']
		else:
			all_preds = np.concatenate((all_preds, predictions['output_layer']))	
	return all_preds

# Calculate and print the accuracy
def eval(predictions, y):
	pred_index = tf.keras.backend.eval(tf.argmax(predictions, axis=1))
	label_index = tf.keras.backend.eval(tf.argmax(y, axis=1))
	df = pd.concat([pd.Series(pred_index, name='pred'), pd.Series(label_index, name='lab')], axis=1)
	# Calculate accuracy
	df['diff'] = df.apply(lambda x: 1 if x['pred'] == x['lab'] else 0, axis=1)
	acc = df['diff'].sum() / len(df)
	print('Accuracy = {}'.format(acc))

# Save predictions into the output file
def save_predictions(predictions, output_file):
	pred_index = tf.keras.backend.eval(tf.argmax(predictions, axis=1))
	# Sum 1 to calculate stars from index of OHE
	stars = pred_index + 1
	df = pd.DataFrame(data=stars)
	with file_io.FileIO(output_file, 'w') as f:
		df.to_csv(f, sep=',', index=False, header=0)


'''
Description:
  Main function for evaluating training and making predictions from raw data.

Input parameters:
  --mode: mode of execution. Values:
      - eval: execute predictions over preprocessed eval dataset and calculate the accuracy
	  - test: execute predictions over raw test dataset and calculate the accuracy
	  - pred: execute predictions over raw dataset without labels
  --model_folder: folder where the model has been exported
  --input_file: input file for evaluating the model
  --labels_file: file with the labels of the samples in the input_file
  --output_file: file where the predictions will be stored
  --dict_file: dictionary file for "test" and "pred" mode

Details:
  - Parse input parameters
  - Load input data
  - If mode = "eval":
	  - Load pre-processed data (word indexes)
  - If mode = "test" or "pred":  
      - Load raw data
	  - Executes the NLP raw data
  - Executes predictions with pre-trained model
  - If mode = "eval" or "test":
      - Calculate and print the accuracy of predictions
  - If output_file is not None:
      - Save predictions into the output file
'''
if __name__ == '__main__':
	#Parse input arguments
	parser = argparse.ArgumentParser()	
	parser.add_argument("--mode", help="mode of execution (eval/test/pred)")
	parser.add_argument("--model_folder", help="folder where the model has been exported")
	parser.add_argument("--input_file", help="file with features")
	parser.add_argument("--labels_file", help="file with labels for eval", default='')
	parser.add_argument("--output_file", help="file to store predictions", default='')	
	parser.add_argument("--dict_file", help="dictionary file", default='')	
	parser.add_argument("--num_classes", help="number of classes", type=int, default=5)
	args = parser.parse_args()

	print('{} Loading data'.format(datetime.now()))	
	if args.mode == 'eval':
	    # Load pre-processed data (word indexes) and labels
		if args.labels_file == '':
			(x, y) = load_one_file(args.input_file, args.num_classes)
		else:
			x = load_csv(args.input_file)
			y = load_csv(args.labels_file)
	elif args.mode == 'test':
		# Load raw data and execute the NLP pipeline
		df = execute_NLP(args.input_file, args.dict_file, args.num_classes)
		data = df.toPandas()
		# Split features and labels columns
		x = data.iloc[:,:(-1*args.num_classes)].values.astype('float32')
		y = data.iloc[:,(-1*args.num_classes):].values.astype('int32')			
	elif args.mode == 'pred':
		# Load raw data and execute the NLP pipeline
		df = execute_NLP(args.input_file, args.dict_file)
		x = df.toPandas()
	else:
		print('Wrong value for parameter mode. Allowed: eval, test or pred')
		quit()
	
	# Execute predictions with pre-trained model
	print('{} Starting predictions'.format(datetime.now()))
	predictions = predict(args.model_folder, x)
	
	# Calculate the accuracy for eval and test mode
	if args.mode == 'eval' or args.mode == 'test':
		eval(predictions, y)

	# Save predicted values
	if args.output_file != '':
		save_predictions(predictions, args.output_file)
	print('{} End predictions'.format(datetime.now()))