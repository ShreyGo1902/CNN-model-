import numpy as np
import os
from sklearn.utils import shuffle
from tqdm import tqdm
import tflearn
import tensorflow as tf
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from keras.utils import np_utils
import matplotlib.pyplot as plt


benign_train = np.load('benign_train_preprocessed_final.npy')
benign_test  = np.load('benign_test_preprocessed.npy')

#plt.imshow(benign_train[0],cmap=plt.get_cmap('gray'))
#plt.show()

malignant_train = np.load('malignant_train_preprocessed_final.npy')
malignant_test  = np.load('malignant_test_preprocessed.npy')

y1      = np.zeros((len(benign_train),), dtype=np.int)
y2      = np.ones((len(malignant_train),), dtype=np.int)
y_train = np.concatenate((y1,y2))

y1     = np.zeros((len(benign_test),), dtype=np.int)
y2     = np.ones((len(malignant_test),), dtype=np.int)
y_test = np.concatenate((y1,y2))

X_train = np.concatenate((benign_train,malignant_train))
X_test  = np.concatenate((benign_test,malignant_test))

X_train,y_train  = shuffle(X_train,y_train)
X_test,y_test    = shuffle(X_test,y_test)


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
      

X_train = X_train.astype('float32')
#X_train/=255
X_test = X_test.astype('float32')
#X_test/=255

convnet = input_data(shape=[None, 64, 64, 3], name='input')

convnet = conv_2d(convnet, 32, filter_size=1,strides=1, padding='same',  activation='relu')
convnet = conv_2d(convnet, 32, filter_size=1,strides=1, padding='same',  activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32,filter_size=1, strides=1,padding='same', activation='relu')
convnet = conv_2d(convnet, 32, filter_size=1,strides=1, padding='same',activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = dropout(convnet,0.4)

convnet = conv_2d(convnet, 32,filter_size=1, strides=1,padding='same', activation='relu')
convnet = conv_2d(convnet, 32, filter_size=1,strides=1,padding='same',activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32,filter_size=1, strides=1,padding='same', activation='relu')
convnet = conv_2d(convnet, 32,filter_size=1, strides=1,padding='same', activation='relu')
convnet = max_pool_2d(convnet, 2)
convet = dropout(convnet,0.4)

convnet = fully_connected(convnet, 32, activation='relu')
convnet = dropout(convnet, 0.3)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', loss='categorical_crossentropy', name='targets')


model = tflearn.DNN(convnet)


model.fit({'input': X_train}, {'targets': y_train}, n_epoch=5, validation_set=({'input': X_test}, {'targets': y_test}), 
    snapshot_step=500, show_metric=True, run_id='Cancer')


scores = model.evaluate(X_test,y_test)
print(scores)

result = np.zeros(len(scores))
for i in range(len(scores)):
    a  = np.argmax(scores[i])
    result[i] = a
    
