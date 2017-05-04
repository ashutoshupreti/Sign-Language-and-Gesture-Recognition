import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.transform import resize
import pandas as pd
import numpy as np
from skimage import io,color
import glob
from sklearn import svm
import os
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.neighbors import KNeighborsClassifier as KN
from imblearn.under_sampling import RandomUnderSampler
import random
import pickle


def sliding_window(image, stepSize, windowSize):
	for y in xrange(0, image.shape[0],stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			if (y+windowSize[1]<image.shape[0] and x+windowSize[0]<image.shape[1]):
				yield (x, y, image[y:y + windowSize[1], x:x+windowSize[0]])


user_list = [3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19]
group_pos = []					#GROUPING POSITIVE SAMPLES
group_neg = []					#GROUPING NEGATIVE SAMPLES
groupy_pos = []		
groupy_neg = []
group_neg_undersampled = []		#GROUPING UNDERSAMPLED NEGATIVE SAMPLES
groupy_neg_undersampled = []


def image_load(path,label):
	for user_num in user_list:

		data = pd.read_csv(path+"user_"+str(user_num)+"/user_"+str(user_num)+"_loc.csv")
		x_left = data['top_left_x'].tolist()
		x_right = data['bottom_right_x'].tolist()
		y_left = data['top_left_y'].tolist()
		y_right = data['bottom_right_y'].tolist()
		counter = -1

		for filename in glob.glob(path+"user_"+str(user_num)+"/"+str(label)+".jpg"):
			counter = counter + 1
			current_file = io.imread(filename)
			current_file = color.rgb2gray(current_file)
			cropped_pos =  current_file[ y_left[counter] : y_right[counter] + 1, x_left[counter] : x_right[counter] + 1]
			cropped_pos = resize(cropped_pos, (125, 125))
			fd_pos = hog(cropped_pos,feature_vector=True,pixels_per_cell=(8,8),cells_per_block=(2,2),visualise=False,transform_sqrt = False)
			group_pos.append(fd_pos)
			groupy_pos.append([1])

			for (x_left_frame, y_left_frame, moving_frame) in sliding_window(current_file, stepSize=25, windowSize=(125,125)):
				
				x11=x_left[counter]
				x12=x_right[counter]
				y11=y_left[counter]
				y12=y_right[counter]
				x21=x_left_frame
				x22=x_left_frame+125
				y21=y_left_frame
				y22=y_left_frame+125

				x1_box = max(x11,x21)
				y1_box = max(y11,y21)
				x2_box = min(x12,x22)
				y2_box = min(y12,y22)
				w = max(0,x2_box-x1_box+1)
				h = max(0,y2_box-y1_box+1)
				area = (x_right[counter] - x_left[counter] + 1) * (y_right[counter] - y_left[counter] + 1)
				overlap = float(w*h)/area

				if(overlap>0.4):
					continue

				fd_neg = hog(moving_frame,feature_vector=True,pixels_per_cell=(8,8),cells_per_block=(2,2),visualise=False,transform_sqrt = False)
				group_neg.append(fd_neg)
				groupy_neg.append([0])		


list_letters = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

def file_load(path):
	for ch in list_letters:
		for i in range(0,10):
			image_load(path,ch+str(i))

path = "/home/ml/Documents/ml_ashutosh/dataset/"
file_load(path)																																																														

#UNDERSAMPLING NEGATIVE SAMPLES
for i in range(0,6000):
	rand = random.randint(0,87839)
	group_neg_undersampled.append(group_neg[rand])
	groupy_neg_undersampled.append([0])
																																																																																																																																																																																																																																																																																																																																																																								

group_pos = np.array(group_pos)
group_neg = np.array(group_neg_undersampled)
groupy_pos = np.array(groupy_pos)
groupy_neg = np.array(groupy_neg_undersampled)

trainX = np.vstack([group_pos,group_neg])
trainY = np.vstack([groupy_pos,groupy_neg])

f1 = open("trainX.txt",'wb') 
f2 = open("trainY.txt",'wb')
pickle.dump(trainX,f1) 
pickle.dump(trainY,f2)
f1.close()
f2.close()

print "pickling done"

