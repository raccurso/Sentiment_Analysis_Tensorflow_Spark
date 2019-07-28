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
def get_model(dictionary_lenght, num_classes):
	return conv_model2(dictionary_lenght, num_classes)


def base_model(dictionary_lenght, num_classes):
	# Create the Keras model
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Embedding(dictionary_lenght + 1, 128, name='features_layer'))
	model.add(tf.keras.layers.Conv1D(64, 7, activation='relu'))
	model.add(tf.keras.layers.GlobalMaxPooling1D())
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer'))
	# Print the summary of the layers in the model
	model.summary()
	# Compile the model
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])

	return model

def conv_model2(dictionary_lenght, num_classes):
	# Create the Keras model
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Embedding(dictionary_lenght + 1, 128, name='features_layer'))
	model.add(tf.keras.layers.Conv1D(256, 3, activation='relu'))
	model.add(tf.keras.layers.MaxPooling1D(5))
	model.add(tf.keras.layers.Conv1D(256, 3, activation='relu'))
	model.add(tf.keras.layers.GlobalMaxPooling1D())
	model.add(tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer'))
	# Print the summary of the layers in the model
	model.summary()
	# Compile the model
	model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
				  loss='categorical_crossentropy',
				  metrics=['accuracy'])

	return model