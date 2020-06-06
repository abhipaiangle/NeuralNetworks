import tensorflow as tf
import numpy as np
import json
import sys
#getting the data as a dictionary

def get_dict(filename):
	with open(filename) as f:
		data = json.load(f)
	return data

def process_data(data):
	X = np.zeros((len(data['flag'], 2)))
	Y = np.zeros((X.shape[0], ))
	X[:, 0] = np.array(data['input1'])
	X[:,1] = np.array(data['input2'])
	Y = np.array(data['flag'])
	return X, Y

def model_create():
	model = tf.keras.models.Sequential()
	model.add(units = 3, activation = 'relu', input_shape = (2,))
	model.add(units = 2, activation = 'relu')
	model.add(units = 1, activation = 'sigmoid')
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	return model


def main():
	filename = str(sys.argv[1])
	data = get_dict(filename)
	X, Y = process_data(data)
	model = model_create()
	model.fit(X, Y, validation_split = 0.2, epochs = 60)
	print("Fitting has been done")
	model.save("mymodel")

if __name__ == '__main__':
	main()