import numpy as np
import argparse
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import model_from_json

def main():
	run(make_args())


def make_args():
	ap = argparse.ArgumentParser()
	ap.add_argument('--image', help='Path to image file')
	args = vars(ap.parse_args())
	return args

def run(args):
	img = image.load_img(args['image'], target_size=(48, 48))
	imageArray = image.img_to_array(img)
	imageArray /= 255
	imageArray = np.expand_dims(imageArray, axis=0)
	model = make_model()
	result = predict(model, imageArray)
	found = image.load_img('classes/' + result['image'] + '.jpg')
	f = plt.figure()
	f.add_subplot(1,2, 1)
	plt.title('found ' + result['image'] + ' acc - ' + result['acc'] + '%')
	plt.imshow(img)
	f.add_subplot(1,2, 2)
	plt.imshow(found)
	plt.show(block=True)
	return True

def make_model():
	json_model = open('models/model36-1.json', 'r')
	loaded_json_model = json_model.read()
	json_model.close()
	mlp_model = model_from_json(loaded_json_model)
	mlp_model.load_weights('models/weights36-1.h5')
	mlp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return mlp_model


def predict(mlpModel, imageArray):
	prediction = mlpModel.predict(imageArray, verbose=0)
	classes = ['1.1', '1.22', '1.33', '1.8', '2.1',
	           '2.2', '2.3.1', '2.5', '3.1', '3.18.1',
	           '3.24.10', '3.20', '3.24.5', '3.24.20', '3.24.5',
	           '3.24.50', '3.24.60', '3.24.70', '3.24.80', '3.24.40',
	           '3.28', '3.27', '4.1.1', '4.3', '5.16',
	           '5.19.1', '5.20', '5.5', '5.6', '6.3.2',
	           '6.4', '7.3', '7.4', '3.21', '3.25', 'trash']
	best = []
	for predict in prediction[0]:
		best.append(round(predict*100, 5))
	return {'image' : classes[np.argmax(prediction)], 'acc': str(best[np.argmax(best)])}

if __name__ == '__main__':
	main()
