import os
import cv2
import numpy as np
import pandas as pd
import glob
import cPickle
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from sklearn.metrics import log_loss
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
PIXELS=64
BATCH_SIZE=64
NB_EPOCH=20
C_SIZE=64
CHANNELS=1


def save_model_weights(model, file_name ):
	model.save_weights(file_name)

def load_model_weights(model, file_name ):
	model.load_weights(file_name)
	return model
	

def create_network():
	model=Sequential()
	#layer 1
	model.add(Convolution2D(10,3 ,3,input_shape=(1,PIXELS,PIXELS) ))
	model.add(Activation('relu'))
	#~ model.add(MaxPooling2D(pool_size=(2 , 2)))
	
	
	model.add(Convolution2D(15 , 3, 3))
	model.add(Activation('relu'))
	#~ model.add(Dropout(0.2))
	#~ 
	#~ model.add(Convolution2D(10 , 3, 3, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
	#~ model.add(Activation('relu'))
	#~ model.add(Dropout(0.2))
	#~ 
	model.add(Flatten())
	model.add(Dense(512 ))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	#layer 7
	model.add(Dense(512 , W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	#layer 8
	model.add(Dense(10))
	model.add(Activation('softmax'))
	
	sgd = SGD(lr=0.01, decay=0.001, momentum=0.9, nesterov=False)
	#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer='sgd')
	return model

def load_test_images():
	print("Loading test data")
	test_X=[]
	filepath=os.path.join('..','test','*.jpg')
	print(filepath)
	file_list=glob.glob(filepath)
	print(len(file_list))
	for f in file_list:
		X=cv2.imread(f,0) # reading as a Greyscale image -> 0
		X=cv2.resize(X,(PIXELS,PIXELS))
		test_X.append(X)
	
	fp1=open("../data/array_list_test.txt",'w')
	cPickle.dump(test_X,fp1)
	print("Loading test data done")
	return test_X


def load_driver_data():
	print("Loading driver Data")
	dr=dict()
	path=("../data/driver_imgs_list.csv")
	fp=open(path , 'r')
	line=fp.readline()
	while(1):
		line=fp.readline()
		if line =='':
			break
		else:
			arr=line.strip().split(',')
			dr[arr[2]]=arr[0]
	fp.close()
	return dr
	
		
def load_images():
	
	print("Loading images")
	train_X=[]
	train_Y=[]
	
	driver_data= load_driver_data()
	driver_id=[]
	for i in range(10):
		print('load folder c{}'.format(i))
		filepath=os.path.join('..','train','c'+str(i),'*.jpg')
		file_list=glob.glob(filepath)
		print(len(file_list))
		for f in file_list:
			base_file_name = os.path.basename(f)
			X=cv2.imread(f,0) # reading as a Greyscale image -> 0
			X=cv2.resize(X,(PIXELS,PIXELS))
			train_X.append(X)
			train_Y.append(i)
			driver_id.append(driver_data[base_file_name])
			
	
	unique_drivers_list= sorted(list(set(driver_id)))
	fp1=open("../data/array_list_X.txt",'w')
	fp2=open("../data/array_list_Y.txt",'w')
	cPickle.dump(train_X,fp1)
	cPickle.dump(train_Y,fp2)
	print("loading images done")
	return train_X,train_Y,driver_id, unique_drivers_list

def train_and_valid_data(X , Y , driver_id , unique_list_train , unique_list_valid):
	X_train=[]
	Y_train=[]
	X_valid=[]
	Y_valid=[]
	length = len(X)
	for i in range (0,length):
		if driver_id[i] in unique_list_train :
			X_train.append(X[i])
			Y_train.append(Y[i])
			
		else:
			if driver_id[i] in unique_list_valid:
				X_valid.append(X[i])
				Y_valid.append(Y[i])
	
	print(len(X_train))
	print(len(Y_train))
	print(len(X_valid))
	print(len(Y_valid))
	
	return X_train , Y_train , X_valid , Y_valid
	
	
def transform_train_data(train_X , train_Y):
	print("transforming data")
	length=len(train_X)
	print(length)
	train_X = np.array(train_X , dtype=np.uint8)
	train_Y = np.array(train_Y , dtype=np.uint8)
	train_X = train_X.reshape(length, CHANNELS, PIXELS , PIXELS)
	train_Y = np_utils.to_categorical(train_Y , 10)
	train_X = train_X.astype('float32')
	train_X /= 255
	return train_X , train_Y
	
def transform_test_data(test_X):
	print("transforming data")
	length=len(test_X)
	print(length)
	test_X = np.array(test_X , dtype=np.uint8)
	test_X = test_X.reshape(length , CHANNELS, PIXELS , PIXELS)
	test_X = test_X.astype('float32')
	test_X /= 255
	return test_X
	
def prepare_data():
	train_X , train_Y , driver_id ,unique_drivers_list = load_images()
	
	# prepare data for training and validation
	unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024','p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049','p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072','p075']
	print(unique_list_train)
	
	unique_list_valid=['p081']
	
	train_X , train_Y = transform_train_data(train_X , train_Y)
	
	train_X , train_Y, valid_X , valid_Y= train_and_valid_data(train_X , train_Y , driver_id, unique_list_train , unique_list_valid)
	
	return train_X , train_Y, valid_X , valid_Y
	
	
def test_prediction( model ,test_X ):
	prediction=model.predict_proba(test_X,verbose=1)
	print(prediction)
	return prediction
	
	
def create_submission(file_path ,prediction):
	file_name=[]
	for path, subdirs, files in os.walk(r'../test'):
		for filename in files:
			file_name.append(filename)
	
	col=['amg','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']	
	submission = pd.DataFrame({"c0":prediction[:,0],"c1":prediction[:,1],"c2":prediction[:,2],"c3":prediction[:,3],"c4":prediction[:,4],"c5":prediction[:,5],"c6":prediction[:,6],"c7":prediction[:,7],"c8":prediction[:,8],"c9":prediction[:,9],"amg":file_name})
	submission.to_csv(file_path, index=False)

	

def train_and_validate_model(train_X , train_Y, valid_X , valid_Y):
	model = create_network()
	print(model.summary())
	train_X=np.array(train_X)
	#print(len(train_X))
	print(train_X.shape)
	train_Y=np.array(train_Y)
	#print(len(train_Y))
	print(train_Y.shape)
	valid_X=np.array(valid_X)
	#print(len(valid_X))
	print(valid_X.shape)
	valid_Y=np.array(valid_Y)
	#print(len(valid_Y))
	print(valid_Y.shape)
	checkpointer = ModelCheckpoint(filepath="../data/weights_1_2.hdf5", monitor='val_loss', verbose=1, save_best_only=True)
	model.fit(train_X, train_Y, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,show_accuracy=True, verbose=1, validation_data=(valid_X, valid_Y), callbacks=[checkpointer])
	print("saving weights")
	model.save_weights('../data/model_weights.h5')
	
	predictions_valid = model.predict(valid_X, batch_size=128, verbose=1)
	score = log_loss(valid_Y, predictions_valid)
	print('Score log_loss %d ' %score)
	
	test_X = load_test_images()
	test_X = transform_test_data(test_X)
	
	prediction = test_prediction( model ,test_X)
	
	file_name = "../data/submission_overall.csv"
	create_submission(file_name, prediction)
	
	weights = "../data/weights_1_2.hdf5"
	model = load_model_weights( model , weights) 
	
	prediction = test_prediction( model ,test_X)
	
	file_name = "../data/submission_best.csv"
	create_submission(file_name, prediction)


'''
	Main call to function
'''
train_X , train_Y, valid_X , valid_Y = prepare_data()	

train_and_validate_model(train_X , train_Y, valid_X , valid_Y)
