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
import random
import pickle
from skimage.io import imread

#INITIALIZE VARIABLES
user_list = [3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19]
list_letters = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
x_start = []
x_end = []
y_start = []
y_end = []
group1=[]
group2=[]
group3=[]
group4=[]
group1y=[]
group2y=[]
group3y=[]
group4y=[]

def image_load(path,user_num):
	
	counter = -1
	for ch in list_letters:
		for j in range(0,10):
			counter+=1
			for filename in glob.glob(path+"user_"+str(user_num)+"/"+ch+str(j)+".jpg"):
				current_file = io.imread(filename)
				current_file = color.rgb2gray(current_file)
				cropped_pos =  current_file[ y_start[counter] : y_end[counter] + 1, x_start[counter] : x_end[counter] + 1]
				cropped_pos = resize(cropped_pos, (125, 125))
				fd_pos = hog(cropped_pos,feature_vector=True,pixels_per_cell=(8,8),cells_per_block=(2,2),visualise=False,transform_sqrt = False)
				if(user_num==3 or user_num==4 or user_num==5 or user_num==6):
					group1.append(fd_pos)
					group1y.append([1])
				elif(user_num==7 or user_num==9 or user_num==10 or user_num==11):
					group2.append(fd_pos)
					group2y.append([1])
				elif(user_num==12 or user_num==13 or user_num==14 or user_num==15):
					group3.append(fd_pos)
					group3y.append([1])
				elif(user_num==16 or user_num==17 or user_num==18 or user_num==19):
					group4.append(fd_pos)
					group4y.append([1])

				for i in range(0,2):
					while(1):
						x = random.randint(0,195)
						y = random.randint(0,115)

						x11=x_start[counter]
						x12=x_end[counter]
						y11=y_start[counter]
						y12=y_end[counter]
						x21=x
						x22=x+125
						y21=y
						y22=y+125

						x1_box = max(x11,x21)
						y1_box = max(y11,y21)
						x2_box = min(x12,x22)
						y2_box = min(y12,y22)
						w = max(0,x2_box-x1_box+1)
						h = max(0,y2_box-y1_box+1)
						area = (x_end[counter] - x_start[counter] + 1) * (y_end[counter] - y_start[counter] + 1)
						overlap = float(w*h)/area
						
						if(overlap>0.4):
							continue

						frame = current_file[y:y+125,x:x+125]
						fd_neg = hog(frame,feature_vector=True,pixels_per_cell=(8,8),cells_per_block=(2,2),visualise=False,transform_sqrt = False)
						if(user_num==3 or user_num==4 or user_num==5 or user_num==6):
							group1.append(fd_neg)
							group1y.append([0])
						elif(user_num==7 or user_num==9 or user_num==10 or user_num==11):
							group2.append(fd_neg)
							group2y.append([0])
						elif(user_num==12 or user_num==13 or user_num==14 or user_num==15):
							group3.append(fd_neg)
							group3y.append([0])
						elif(user_num==16 or user_num==17 or user_num==18 or user_num==19):
							group4.append(fd_neg)
							group4y.append([0])
						break

def file_load(path):
	for user_num in user_list:
		
		data = pd.read_csv(path+"user_"+str(user_num)+"/user_"+str(user_num)+"_loc.csv")
		global x_start 
		x_start = data['top_left_x'].tolist()
		global x_end 
		x_end = data['bottom_right_x'].tolist()
		global y_start 
		y_start = data['top_left_y'].tolist()
		global y_end 
		y_end = data['bottom_right_y'].tolist()
		image_load(path,user_num)
		print "done with user: " + str(user_num)

path = "/home/ml/Documents/ml_ashutosh/dataset/"
file_load(path)

group1 = np.array(group1)
group2 = np.array(group2)
group3 = np.array(group3)
group4 = np.array(group4)
group1y = np.array(group1y)
group2y = np.array(group2y)
group3y = np.array(group3y)
group4y = np.array(group4y) 

trainX = np.vstack([group1,group2,group3,group4])
trainY = np.vstack([group1y,group2y,group3y,group4y])
clf = rf(n_estimators=500,max_features='auto',n_jobs=-1,criterion='gini',verbose=1)
clf.fit(trainX,trainY)

f1 = open("clf_bin.txt",'wb')
pickle.dump(clf,f1)
f1.close()
print "done pickling!!!"


