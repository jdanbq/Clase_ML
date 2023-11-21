#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.svm import SVR


# In[2]:


path = os.getcwd()
#names = ['CARVAJAL','LAS FERIAS','SAN CRISTOBAL','PUENTE ARANDA','MIN AMBIENTE','TUNAL','SUBA', 'GUAYMARAL', 'KENNEDY','USAQUEN']
names = ['GUAYMARAL']


# In[3]:


pm25_train = pd.read_csv(path + '/DATA/pm25_train.csv', index_col = 0, parse_dates = True)
pm25_train = pm25_train.loc['2015-01-01':'2020-12-31']

pm25_comp = pd.read_csv(path + '/DATA/llenado_PM25.csv', index_col = 0, parse_dates = True)
pm25_comp = pm25_comp.loc['2015-01-01 00:00:00':'2020-12-31 23:00:00']


# In[ ]:


for name in names:

    scaler = StandardScaler()
    scaler.fit(pm25_comp[name].values.reshape(-1, 1))

    train_data = scaler.transform(pm25_train[name].values.reshape(-1, 1))

    x_train = []
    y_train = []

    for i in np.arange(120, len(train_data)):
        x_train.append(train_data[i-120:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    svr_rbf = SVR(kernel= 'rbf', C = 10**5, gamma = 0.1,)
    model = svr_rbf.fit(x_train, y_train)

    pickle.dump(model, open(str(name) + '.sav', 'wb'))

