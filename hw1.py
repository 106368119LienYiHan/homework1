
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import preprocessing
from sklearn.pipeline import Pipeline


# In[2]:


#import data

data_1 = pd.read_csv('/Users/yihan/Documents/homework1/train-v3.csv', header=0, names=['id','price','sale_yr','sale_month','sale_day','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15'])
X_train = data_1.drop(['id', 'price', 'sale_month', 'sale_day', 'view'], axis = 1)
X_train['sale_yr'] = list(map(int, X_train['sale_yr']))
Y_train = data_1['price'].values

data_2 = pd.read_csv('/Users/yihan/Documents/homework1/valid-v3.csv', header=0, names=['id','price','sale_yr','sale_month','sale_day','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15'])
X_valid = data_2.drop(['id', 'price', 'sale_month', 'sale_day', 'view'], axis = 1)
X_valid['sale_yr'] = list(map(int, X_valid['sale_yr']))
Y_valid = data_2['price'].values

data_3 = pd.read_csv('/Users/yihan/Documents/homework1/test-v3.csv', header=0, names=['id','sale_yr','sale_month','sale_day','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15'])
X_test = data_3.drop(['id', 'sale_month','sale_day', 'view'], axis = 1)
X_test['sale_yr'] = list(map(int, X_test['sale_yr']))


# In[3]:


#normalize

X_train = preprocessing.scale(X_train)
X_valid = preprocessing.scale(X_valid)
X_test = preprocessing.scale(X_test)


# In[4]:


#build model

model = Sequential()
model.add(Dense(80, input_dim = X_train.shape[1], kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(250, input_dim = 80, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(100, input_dim = 250, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(80, input_dim = 100, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(X_train.shape[1], input_dim = 40, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'normal'))
    
model.compile(loss = 'MAE', optimizer = 'adam', metrics=['accuracy'])


# In[5]:


#train

model.fit(X_train, Y_train, epochs = 285, batch_size = 32, verbose=0, validation_data = (X_valid, Y_valid))


# In[6]:


#predict

Y_predict = model.predict(X_test)
np.savetxt('/Users/yihan/Documents/homework1/test2.csv', Y_predict, delimiter = ',')

