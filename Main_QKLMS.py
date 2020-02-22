# -*- coding: utf-8 -*-
"""
The following are the functions on this script:
    
    

Created on Mon Oct 22 11:42:34 2018

@author: Sergio GARCIA-VEGA
sergio.garcia-vega@manchester.ac.uk
The University of Manchester, Manchester, UK
BigDataFinance, Work Package 1, Research Project 1
Id: Main_QKLMS.py
"""

import os
import pickle
import numpy as np
from os import listdir
import Functions.Aux_funcs as aux



#==============================================================================
#                        Importing Traning and Testing samples
#==============================================================================
crr_wd = os.getcwd()
data_dir_trte  = os.path.join(crr_wd,'Data\\3_Train_Test\\Log_Return_Stan')

TrainTest = {}
for trte_stock in listdir(data_dir_trte):
    key_df = trte_stock.split('.')[0].split('Stock_')[1]
    TrainTest[key_df] = pickle.load(open('Data\\3_Train_Test\\Log_Return_Stan\\' + str(trte_stock), 'rb'))

    
keys_stocks = list(TrainTest.keys())

count = 1

for model in keys_stocks:
        
    #==========================================================================
    #                          Filter Configuration
    #==========================================================================
    epsilon    = 0.4 #0.06
    N_tr       = TrainTest[model]['Parameters']['NTrS']
    N_te       = TrainTest[model]['Parameters']['NTeS']
    eta        = TrainTest[model]['Parameters']['LR']
    sigma      = TrainTest[model]['Parameters']['KS']
    
    
    #==========================================================================
    #                              Data Embedding
    #==========================================================================
    X_train = TrainTest[model]['Train']['X_tr'] 
    y_train = TrainTest[model]['Train']['y_tr'] 
    
    X_test = TrainTest[model]['Test']['X_te']
    y_test = TrainTest[model]['Test']['y_te']
    
    
    #==========================================================================
    #                       Kernel Least-Mean-Square (KLMS)
    #==========================================================================
    err_train     = np.zeros((N_tr,1))    
    y_train_pred  = np.zeros((N_tr,1))   
    mse_test      = np.zeros((N_tr,1))   
    num_samp_pred = np.zeros((N_te,1))
    
    
    #==========================================================================
    #                              Initialization
    #==========================================================================
    err_train[0]    = y_train[0]                   #Init: error (training)
    y_train_pred[0] = 0                            #Init: prediction (training)
    dicti           = X_train[0,:][np.newaxis, :]  #Init: dictionary (training)
    alphas          = eta*y_train[0]               #Init: weights (training)
    mse_test[0]     = np.mean(y_test**2)           #Init: MSE (testing)
    Dicti_size      = np.zeros((N_tr,1))
    Dicti_size[0]   = len(dicti)
    
    n=1
    while n < N_tr:
        print('Stock: ', count,'/','24', ' - Training: ', n+1,'/',N_tr)
        
        "%%%%%%%%%%%%%%%%%%%%% TRANING %%%%%%%%%%%%%%%%%%%%%"
        
        "Input vector"
        u_train         = X_train[n,:][np.newaxis, :] 
        y_train_pred[n] = np.float(np.dot(aux.gaus_kernel(u_train, dicti,sigma), alphas))
        err_train[n]    = y_train[n] - y_train_pred[n]
        Dicti_size[n]   = len(dicti)
        
        "Quantization approach"
        mindist_val  = np.min(np.linalg.norm(u_train-dicti, axis=1))
        mindist_arg  = np.argmin(np.linalg.norm(u_train-dicti, axis=1))
        if mindist_val <= epsilon:
            alphas[mindist_arg] = alphas[mindist_arg] + eta*err_train[n]
        else:
            dicti      = np.append(dicti,u_train,axis=0)
            alphas     = np.append(alphas,eta*err_train[n])
        n = n + 1
        
        
    "%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%"
    y_test_pred = np.zeros((N_te,1))
    i = 0
    while i < N_te:
        "Input vector"
        u_test           = X_test[i,:][np.newaxis, :]
        num_samp_pred[i] = len(dicti)
        
        #Prediction (testing)
        y_test_pred[i] = np.float(np.dot(aux.gaus_kernel(u_test,dicti,sigma), alphas))
        i = i + 1  

    
    #Performance measure: MSE
    err_test = y_test - y_test_pred
    mse_test = np.mean(err_test**2)
    count    = count + 1
                              
    
    
    #==========================================================================
    #                              Saving Results
    #==========================================================================
    Results_QKLMS = {}
    
    Results_QKLMS['Figure 1']                  = {}
    Results_QKLMS['Figure 1']['Desired']       = y_test 
    Results_QKLMS['Figure 1']['Prediction']    = y_test_pred
    
    Results_QKLMS['Table 1']                     = {}
    Results_QKLMS['Table 1']['MSE']              = mse_test
    Results_QKLMS['Table 1']['Num_Samp']         = num_samp_pred
    
    pickle.dump(Results_QKLMS, open('Results\\Results_Log_Return_Stan\\Results_QKLMS_' + model + '.pkl', 'wb'))
    
            
    
