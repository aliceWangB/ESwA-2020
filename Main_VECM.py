# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:43:49 2020

@author: Sergio
"""

import os
import pickle
import numpy as np
import pandas as pd
from os import listdir
from statsmodels.tsa.api import VECM
from sklearn.metrics import mean_squared_error



#==============================================================================
#                        Importing Traning and Testing samples
#==============================================================================
crr_wd        = os.getcwd()
data_dir_trte = os.path.join(crr_wd,'Data\\3_Train_Test\\Log_Return_Stan')

TrainTest = {}
for trte_stock in listdir(data_dir_trte):
    key_df            = trte_stock.split('.')[0].split('Stock_')[1]
    TrainTest[key_df] = pickle.load(open('Data\\3_Train_Test\\Log_Return_Stan\\' + str(trte_stock), 'rb'))
keys_stocks = list(TrainTest.keys())

data = {}
for stock in keys_stocks:
    data[stock] = TrainTest[stock]['Train']['Set']
data = pd.DataFrame(data)
data = np.array(data)



#==============================================================================
#                     Vector Error Correction Model (VECM)
#==============================================================================
data_test = np.zeros([280,10,24])
count     = 0    

for stock in keys_stocks:    
    #======================================================================
    #                            Data Embedding
    #======================================================================
    X_test = TrainTest[stock]['Test']['X_te']
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    for i in range(280):
        data_test[i, :, count] = X_test[i, :, 0]
    count = count + 1


    #======================================================================
    #                            Predictions
    #======================================================================
pred = np.zeros([280, 1, 24]) 
for i in range(280):
    vecm          = VECM(endog = np.concatenate((data, data_test[i, :, :])), k_ar_diff = 3, coint_rank = 0, deterministic = 'ci')
    res           = vecm.fit()
    pred[i, :, :] = res.predict(steps=1)


    #==========================================================================
    #                               Saving Results
    #==========================================================================
count = 0  
for stock in keys_stocks:
    y_test = TrainTest[stock]['Test']['y_te']
    y_pred = pred[:, 0, count][:, np.newaxis]
    mse    = mean_squared_error(y_test, y_pred)
    
    Results_VECM                           = {}
    
    Results_VECM['Figure 1']               = {}
    Results_VECM['Figure 1']['Desired']    = y_test
    Results_VECM['Figure 1']['Prediction'] = y_pred
    
    Results_VECM['Table 1']                = {}
    Results_VECM['Table 1']['MSE']         = mse
    
    count = count + 1
    
    pickle.dump(Results_VECM, open('Results\\Results_Log_Return_Stan\\Results_VECM_' + stock + '.pkl', 'wb'))


