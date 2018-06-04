import numpy
import sys
import os
from keras.preprocessing import image
from keras.models import model_from_json

# don't show tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
	if not sys.argv[1]:
		print('Wrong file name!')
		exit()


def make_prediction(img_path):

	# remake image to array

	img = image.load_img(img_path, target_size=(48, 48))
	x = image.img_to_array(img)
	x /= 255
	x = numpy.expand_dims(x, axis=0)

	# make model from files

	json_file = open("../Models/MLP/model.json", "r")
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("../Models/MLP/model.h5")
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# make predict

	loaded_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
	prediction = loaded_model.predict(x, verbose=1)
	classes = ['Beware, children!', 'Overtaking prohibited', 'Speed limit', 'Stop is forbidden', 'Straight movement']

	return classes[numpy.argmax(prediction)]


img_path = '../Examples/' + sys.argv[1]

print('FOUND: ' + make_prediction(img_path))



