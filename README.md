# Sentiment Analysis with Tensorflow and Spark
This project consists in the design and development of a predictive model for the sentiment analysis over the public dataset of Amazon reviews. The architecture is made up of the open source frameworks Spark and Tensorflow, used for natural language processing (NLP) and deep learning. The connection between both frameworks has been possible thanks to the Yahoo library TensorflowOnSpark. Finally, the Google Cloud Platform has been chosen as the infrastructure for distributed processing.

The solution executes in Spark both steps of NLP pipeline and model training. The RDD made during the first step is used to feed the distributed training process of the classification model, which structure is composed of 1 dimensional convolutional neural networks and fully connected layers.

## Dataset
The dataset contains about 6,9 millions of Amazon reviews written by buyers of PC category's products. Its size is 3,4 GB and its format is TSV. It can be downloaded in the following page:

[https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_PC_v1_00.tsv.gz]

The dataset has been split in three parts:

![alt text](https://github.com/raccurso/Sentiment_Analysis_Tensorflow_Spark/blob/master/images/dataset.jpg)

## Model
The predictive model has been built and trained using the following libraries:
* Tensorflow Keras
* Tensorflow Estimator
* TensorflowOnSpark by Yahoo

The structure is made up of three 1D convolutional neural network and 3 totally connected layers. They are followed by dropout layers in order to reduce the overfitting over the training dataset.

The binary classifier is showed in the following picture.

![alt text](https://github.com/raccurso/Sentiment_Analysis_Tensorflow_Spark/blob/master/images/Binary_model.jpg)

## Results
The following pictures show the accuracy obtained in the binary and multiclass classifiers.

![alt text](https://github.com/raccurso/Sentiment_Analysis_Tensorflow_Spark/blob/master/images/Accuracy.jpg)

The training process has been done in Google Cloud Platform, using the Dataproc service. The benchmark results of the training phase, using from 6 to 20 instances, is shown below.

![alt text](https://github.com/raccurso/Sentiment_Analysis_Tensorflow_Spark/blob/master/images/Benchmark.jpg)
