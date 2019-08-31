import tensorflow as tf


'''
Description:
  Creates the Keras model 

Input parameters:
  - dict_length: number of words for the dictionary
  - num_classes: number of classes (or labels)

Return:
  - Keras model

Details:
  - Create a sequential model
  - Add layers to the model
  - Set the optimizer, loss and metrics
  - Compile the model
'''
def get_model(dictionary_lenght, num_classes, sample_lenght):
	if num_classes == 2:
		return conv_binary(dictionary_lenght, num_classes, sample_lenght)
	else:
		return conv_multi(dictionary_lenght, num_classes, sample_lenght)


def conv_binary(dictionary_lenght, num_classes, sample_lenght):
	# Create the Keras model
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Embedding(dictionary_lenght + 1, 128, input_length=sample_lenght, name='features_layer'))
	model.add(tf.keras.layers.Conv1D(128, 3, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.AveragePooling1D(2))
	model.add(tf.keras.layers.Conv1D(256, 3, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.AveragePooling1D(2))
	model.add(tf.keras.layers.Conv1D(256, 3, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer'))
	# Print the summary of the layers in the model
	model.summary()
	# Compile the model
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])

	return model


def conv_multi(dictionary_lenght, num_classes, sample_lenght):
	# Create the Keras model
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Embedding(dictionary_lenght + 1, 128, input_length=sample_lenght, name='features_layer'))
	model.add(tf.keras.layers.Conv1D(128, 3, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.AveragePooling1D(2))
	model.add(tf.keras.layers.Conv1D(256, 3, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.AveragePooling1D(2))
	model.add(tf.keras.layers.Conv1D(256, 3, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(256, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.2))
	model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer'))
	# Print the summary of the layers in the model
	model.summary()
	# Compile the model
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])

	return model