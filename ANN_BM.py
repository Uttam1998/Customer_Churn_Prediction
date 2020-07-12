#Artificial Neural Networks

#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X  = dataset.iloc[:,3:13].values
y =  dataset.iloc[:,13].values

#encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder1 = LabelEncoder()
X[:,1] = labelencoder1.fit_transform(X[:,1])
labelencoder2 = LabelEncoder()
X[:,2] = labelencoder2.fit_transform(X[:,2])

#splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#importing the keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising the NN
classifier = Sequential()
#adding the input and hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation ='relu',input_dim = 10))
#adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation ='relu'))
#adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation ='sigmoid'))
#compiling the model
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'] )
#fitting the NN on the dataset
classifier.fit(X_train,y_train,batch_size = 10, nb_epoch = 200)
#predicting the test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

acc = (1540+146)/2000
print(acc)

















