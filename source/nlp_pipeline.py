import numpy as np
import pyspark
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import json
from datetime import datetime
from tensorflow.python.lib.io import file_io

# Constants
DICTIONARY_OFFSET = 1
TEXT_PADDING = 0
SEED = 1234

# Dictionary for one hot encoding of label (star rating)
label_OHE = {1: [1, 0, 0, 0, 0],
             2: [0, 1, 0, 0, 0],
             3: [0, 0, 1, 0, 0],
             4: [0, 0, 0, 1, 0],
             5: [0, 0, 0, 0, 1]}

# Clean data by removing murkups, urls, emails, etc.			 
def clean_data(sample):
	REMOVE_MARKUP = '<[^>]+>'
	REMOVE_URLS_1 = 'http\S+'
	REMOVE_URLS_2 = 'www.\S+'
	REMOVE_HTML_CODES = '&#[0-9]+;'
	REMOVE_EMAIL = '[A-Za-z0-9\.\-+_]+@[A-Za-z0-9\.\-+_]+\.[A-Za-z0-9]+'
	
	if sample[0] and len(sample[0]) > 0:
		clean_text = re.sub(REMOVE_MARKUP, ' ', sample[0])
		clean_text = re.sub(REMOVE_URLS_1, '', clean_text)
		clean_text = re.sub(REMOVE_URLS_2, '', clean_text)
		clean_text = re.sub(REMOVE_HTML_CODES, '', clean_text)
		clean_text = re.sub(REMOVE_EMAIL, '', clean_text)
	else:
		clean_text = ''
	return (clean_text.lower(), sample[1])

# Split sentences in tokens
def tokenizer(sample):    
	text = nltk.word_tokenize(sample[0])
	return (text, sample[1])

# Remove stop words and punctiuation from token lists
def remove_stop_words_and_punc(sample):    
	# Stop words in English from NLTK
	stop_words = set(stopwords.words('english'))
	# list of punctuation
	punctuation = list(string.punctuation)
	# Remove stop words
	no_stop_words = [word for word in sample[0] if not word in stop_words]
	# Remove punctuation
	no_punct = [''.join(character for character in word if character not in punctuation) for word in no_stop_words] 
	# Remove empty words
	clean_text = [word for word in no_punct if word != '']
	return (clean_text, sample[1])

# Replace each word with its lemma
def lemmatize(sample):    
	# Get the lemma of each words
	lemmatizer = WordNetLemmatizer()
	lemmatized_text = [lemmatizer.lemmatize(word, pos="v") for word in sample[0]]
	return (lemmatized_text, sample[1])

# Create a dictionary map of word - index
def create_dictionary(rdd, dict_length):
	from pyspark.ml.feature import CountVectorizer
	df = rdd.toDF(['text', 'rating'])
	filled_df = df.na.fill(0)    
	cv = CountVectorizer(inputCol="text", outputCol="vectors", vocabSize=dict_length)
	model_cv = cv.fit(filled_df)
	dictionary = {k: v + DICTIONARY_OFFSET for v, k in enumerate(model_cv.vocabulary)}
	return dictionary

# Load an existing dictionary of a pre-trained model
def load_dictionary(dict_file):
	with file_io.FileIO(dict_file, 'r') as f:
		dictionary = json.load(f)
	return dictionary
		
# Replace each word with its index in the dictionary
def replace_word(sample, dict_broad):
	# Look for the index of each word in the dictionary
	word_indexes = [dict_broad.value.get(word, 0) for word in sample[0]]
	return (word_indexes, sample[1])

# Trim or pad each sample to match the sample_lenght
def trim_or_pad_samples(sample, sample_lenght):
	lenght = len(sample[0])
	if lenght >= sample_lenght:
		text = sample[0][:sample_lenght]
	else:
		text = np.pad(sample[0], (0, sample_lenght - lenght), 'constant', constant_values=(TEXT_PADDING,TEXT_PADDING))\
				.tolist()
	return (text, sample[1])


'''
Description:
  Main function for Natural Language Processing.

Input parameters:
  - filename: input file with raw data in Hadoop
  - max_length: number of words that the output file will have per sample
  - dict_length: number of words for the dictionary
  - num_executors: number of Spark executors 
  - dict_file: dictionary file used for predictions over a pre-trained model
  - test_split: fraction of raw samples to keep apart for model testing (from 0 to 1)
  - sc: Spark contect
  
Return:
  - Processed dataframe with word indexes. Shape [number of samples, max_length].
  - Dictionary map with word - index
  - Test dataframe with raw data reserved for testing the model (only if test_split is <> None)

Details:
  - Load raw data from Hadoop in TSV format
  - Clean text by removing markups, links, email, etc.
  - Split sentences in tokens and removes puctuation and stop words with NLTK library
  - Get the root word (lemma)
  - Create the dictionary map of word - index
  - Replace words with indexes from dictionary
  - Pad or trim each samples to max_length elements
  - One hot encoding of labels (star rating)
'''
def nlp_pipeline(filename, max_length, dict_length, num_executors, dict_file, test_split, sc):    
	sql = pyspark.SQLContext(sc)
	# Load csv file with raw samples
	print('{} Loading input data'.format(datetime.now()))
	if test_split == None:
		start_data = sql.read.csv(filename, sep="\t", inferSchema=True, header=True)
		test_raw_data = None
	else:
		# Take apart test data from the beginning (if needed)
		(start_data, test_raw_data) = sql.read.csv(filename, sep="\t", inferSchema=True, header=True) \
										 .randomSplit([1-test_split, test_split], seed=SEED)		
	
	raw_data = start_data.repartition(num_executors) \
						 .rdd \
						 .cache()
	
	# Create a new RDD with a tuple of text and label
	data = raw_data.map(lambda sample: (sample.review_body, sample.star_rating))
	
	# Clean data by removing murkups, urls, emails, etc.
	print('{} Cleaning text'.format(datetime.now()))
	clean_rdd = data.map(lambda sample: clean_data(sample))
	
	# Split sentences in tokens
	print('{} Splitting text in tokens'.format(datetime.now()))
	words_rdd = clean_rdd.map(lambda sample: tokenizer(sample))
	
	# Remove stop words and punctuations from sentences
	print('{} Removing stop words and punctuation'.format(datetime.now()))
	clean_words_rdd = words_rdd.map(lambda sample: remove_stop_words_and_punc(sample))
	
	# Get the lemma of each words
	print('{} Lemmatization'.format(datetime.now()))
	lemma_words_rdd = clean_words_rdd.map(lambda sample: lemmatize(sample)) \
									 .cache()
	
	if dict_file == None:
		# Create a dictionary of distinct words and index
		print('{} Creating dictionary'.format(datetime.now()))
		dictionary = create_dictionary(lemma_words_rdd, dict_length)
	else:
		print('{} Loading dictionary'.format(datetime.now()))
		dictionary = load_dictionary(dict_file)
	
	# Broadcast the dictionary in order to have it in all workers
	dict_broad = sc.broadcast(dictionary)
	
	# Substitute words with its index
	print('{} Substituting words with indexes'.format(datetime.now()))
	index_rdd = lemma_words_rdd.map(lambda sample: replace_word(sample, dict_broad))
		
	# Pad or trim samples in order to have the same lenght.
	print('{} Padding arrays'.format(datetime.now()))
	padded_rdd = index_rdd.map(lambda sample: trim_or_pad_samples(sample, max_length))
	
	# One hot encoding of label
	print('{} One hot encoding labels'.format(datetime.now()))
	final_rdd = padded_rdd.map(lambda sample: sample[0] + label_OHE[sample[1]])
	
	# Convert de RDD to a Dataframe and fills with 0 null values
	final_df = final_rdd.toDF() \
						.na.fill(0)
	
	return final_df, dictionary, test_raw_data