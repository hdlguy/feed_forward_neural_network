import keras
import numpy as np
import pandas as pd 
import tensorflow as tf

data= pd.read_csv("./input/train.csv")
datat=pd.read_csv("./input/test.csv")

X_train=data.iloc[:,1:785]
y_train=data.iloc[:,0]
yt=keras.utils.to_categorical(y_train,10)
X_test=datat.iloc[:,0:785]
X_test

from keras import Sequential
from keras.layers import Dense
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(32, activation='sigmoid', kernel_initializer='random_normal', input_dim=784))
#Output Layer
classifier.add(Dense(10, activation='softmax', kernel_initializer='random_normal'))
#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
classifier.fit(X_train,yt, batch_size=10, epochs=20)

y_test=classifier.predict(X_test)
y_test1=np.argmax(y_test,axis=1)
np.size(y_test1)
id1=list(range(28001))
id1=id1[1:28001]
my_submission = pd.DataFrame({'ImageId':id1, 'Label': y_test1})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


