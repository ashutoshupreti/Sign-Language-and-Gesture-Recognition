import json
import pickle
import gzip
import os
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from skimage.io import imread
from skimage.viewer import ImageViewer
from skimage.transform import resize
from time import time
from skimage import color

	

"""
	This is an example of a valid submission to the web server
	Please note that the functions have dummy implementations
	that loosely model what should happen in them.

	Pay close attention to the save and load functions as they will
	be of importance to you. You may use them as a template for your
	save and load functions.

	The great people who want to use Theano / Tensorflow / Keras
	will need to implement their load and save functions carefully

	Please do not keep print statements in your code.
	If you want to print debug messages, use the logging module
"""

class GestureRecognizer(object):

	"""class to perform gesture recognition"""

	def __init__(self, data_directory):

		"""
			data_directory : path like /home/ashutosh/mlproj/dataset/
			includes the dataset folder with '/'

			Initialize all your variables here
		"""

		# unpickling binary and multiclass classifier
		f1 = open("clf_bin.txt",'r')
		clf = pickle.load(f1)
		f1.close()

		f2 = open("multi_clf.txt",'r')
		multi_clf = pickle.load(f2)
		f2.close()

		self.data_directory = data_directory
		self.bin_clf = 	clf						# hand / non-hand classifier (binary classifier)
		self.multi_clf = multi_clf				# gesture recognizer 	(multiclass classifier)

	# Implementing sliding window	
	def sliding_window(self,image, stepSize, windowSize):
		for y in xrange(0, image.shape[0],stepSize):
			for x in xrange(0, image.shape[1], stepSize):
				if (y+windowSize[1]<image.shape[0] and x+windowSize[0]<image.shape[1]):
					yield (x, y, image[y:y + windowSize[1], x:x+windowSize[0]])

	# Implementing Image Pyramid on various scales
	def pyramid_scale(self,image, scale, num_iter):
		if(scale>1):
			yield image

		while (num_iter>0):
			num_iter = num_iter - 1
			w = int(image.shape[1]/scale)
			h = int(image.shape[0]/scale)
			image = resize(image,(h,w))
			yield image

	def train(self, train_list):

		"""
			train_list : list of users to use for training
			eg ["user_1", "user_2", "user_3"]

			The train function should train all your classifiers,
			both binary and multiclass on the given list of users
		"""



	def recognize_gesture(self, image):

		"""
			image : a 320x240 pixel RGB image in the form of a numpy array

			This function locates the hand and classify the gesture.

			returns : (position, labels)

			position : a tuple of (x1,y1,x2,y2) coordinates of bounding box
					   x1,y1 is top left corner, x2,y2 is bottom right

			labels : a list of top 5 character predictions
						eg - ['A', 'B', 'C', 'D', 'E']
		"""
		list_letters = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

		image = color.rgb2gray(image)
		counter = 0
		fd_list = []
		pos = []
		counter_list = []
		for x in range(0,2):
			if x == 0:
				scale = 1.1			#SCALE DOWN	
			elif x == 1:
				scale = 0.9			#SCALE UP

			for rescaled_image in self.pyramid_scale(image,scale,3):
				counter+=1
				for (x,y,window) in self.sliding_window(rescaled_image,10,(125,125)):
					fd = hog(window,feature_vector=True,pixels_per_cell=(8,8),cells_per_block=(2,2),visualise=False,transform_sqrt = False)
					fd_list.append(fd)
					pos.append((x,y))
					counter_list.append(counter)

		fd_list = np.array(fd_list)
		prob = self.bin_clf.predict_proba(fd_list)
		prob = np.array(prob)	
		index = np.argmax(prob,axis=0)
		index = index[1]
		x,y = pos[index]

		if(counter_list[index]<5):
			x = int(x*pow(1.1,counter_list[index]-1))
			y = int(y*pow(1.1,counter_list[index]-1))
			size = int(125*pow(1.1,counter_list[index]-1)) 
		else:
			x = int(x*pow(0.9,counter_list[index]-4))
			y = int(y*pow(0.9,counter_list[index]-4))
			size = int(125*pow(0.9,counter_list[index]-4))
		pick = [x,y,x+size,y+size]


		image1 = image[pick[1]:pick[3],pick[0]:pick[2]]
		image1 = resize(image1,(125,125))
		final_fd=hog(image1,feature_vector=True,pixels_per_cell=(8,8),cells_per_block=(2,2),visualise=False,transform_sqrt = False)
		prob = self.multi_clf.predict_proba(final_fd)

		char = []
		for i in range(0,5):
			index = np.argmax(prob,axis=1)
			if(index<=8):
				char.append(chr(index+65))
			else:
				char.append(chr(index+66))
			prob[0][index] = -1
			
		return tuple(pick),char

	def save_model(self, **params):

		"""
			save your GestureRecognizer to disk.
		"""

		self.version = params['version']
		self.author = params['author']

		file_name = params['name']

		pickle.dump(self, gzip.open(file_name, 'wb'))
		# We are using gzip to compress the file
		# If you feel compression is not needed, kindly take lite

	@staticmethod		# similar to static method in Java
	def load_model(**params):

		"""
			Returns a saved instance of GestureRecognizer.

			load your trained GestureRecognizer from disk with provided params
			Read - http://stackoverflow.com/questions/36901/what-does-double-star-and-star-do-for-parameters
		"""

		file_name = params['name']
		return pickle.load(gzip.open(file_name, 'rb'))

		# People using deep learning need to reinitalize model, load weights here etc.

if __name__ == "__main__":

	"""
		This is an example of how to save and load models
	"""

	gr = GestureRecognizer("/home/ashutosh/mlproject/dataset")
	gr.train(["user_1", "user_2", "user_3"])

	# Now the GestureRecognizer is trained. We can save it to disk

	gr.save_model(	name = "gr.pkl.gz",
					version = "0.0.1",
					author = "Ashutosh"
				 )

	# Now the GestureRecognizer is saved to disk
	# It will be dumped as a compressed .gz file


	new_gr = GestureRecognizer.load_model(name = "gr.pkl.gz") # automatic dict unpacking

	print (new_gr.version) # will print "0.0.1"
