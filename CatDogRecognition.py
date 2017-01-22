#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:35:32 2017

@author: xuanwei
"""

from pandas import DataFrame
import cv2
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Convolution2D,MaxPooling2D,Flatten,Dropout
from keras.regularizers import l2
import os
from keras.optimizers import SGD,RMSprop,Adagrad

rootDir="/home/xuanwei/job/learnCNN/CatDogRecognition/train"
listImg=os.listdir(rootDir)
numOfTrainImage=len(listImg)
zoomImageArr=np.empty([numOfTrainImage,32,32,3])
dic_classes={'cat':0,'dog':1}
labels=np.empty([numOfTrainImage])
for i in range(numOfTrainImage):
    rawImageArr=cv2.imread(rootDir+'/'+listImg[i])
    zoomImageArr[i,:,:,:]=cv2.resize(rawImageArr,(32,32),interpolation=cv2.INTER_CUBIC)
    labels[i]=dic_classes[listImg[i].split('.')[0]]

model=Sequential()
model.add(Convolution2D(8,3,3,border_mode='same',input_shape=(32,32,3),activation='relu'))
model.add(Convolution2D(16,3,3,border_mode='same',activation='relu'))
model.add(MaxPooling2D())
#model.add(Convolution2D(16,3,3,border_mode='same',activation='relu'))
model.add(Convolution2D(32,3,3,border_mode='same',activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128,init='normal',W_regularizer=l2(0.01),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,init='normal',activation='sigmoid'))

#sgd=SGD(lr=0.01,decay=1e-7,momentum=0.9,nesterov=True)
optimizer = RMSprop(lr=0.001)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.fit(zoomImageArr,labels,batch_size=100,nb_epoch=20,shuffle=True,verbose=1,validation_split=0.2)

#test
rootDirTest="/home/xuanwei/job/learnCNN/CatDogRecognition/test"
listImgTest=os.listdir(rootDirTest)
numOfTestImage=len(listImgTest)
zoomImageArrTest=np.empty([numOfTestImage,32,32,3])
res_index=np.empty([numOfTestImage])
for j in range(numOfTestImage):
    rawImageArrTest=cv2.imread(rootDir+'/'+listImg[j])
    zoomImageArrTest[j,:,:,:]=cv2.resize(rawImageArrTest,(32,32),interpolation=cv2.INTER_CUBIC)
    res_index[j]=int(listImgTest[j].split('.')[0])
labels_test=model.predict_classes(zoomImageArrTest,batch_size=100, verbose=1)
#predictions = model.predict(zoomImageArrTest, verbose=0)

labels_test=np.column_stack((res_index,labels_test))
frame=DataFrame(labels_test,columns=['id','label'],dtype='int32')
frame=frame.sort_values(by='id')
frame.to_csv('/home/xuanwei/job/learnCNN/CatDogRecognition/test_res.csv',header=True,index=False)
#def showPic(i):
#    im=Image.open(rootDir+'/'+listImg[i])
#    return im
#for i in range(10):
#    showPic(i)
#    if predictions[i,0]>0.5:
#        print("i am %s sure this is dog"%predictions[i,0])
#    else:
#        print("i am %s sure this is cat"%(1-predictions[i,0]))
#       
