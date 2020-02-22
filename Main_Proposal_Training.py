# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 08:51:42 2018

@author: Sergio GARCIA-VEGA
sergio.garcia-vega@manchester.ac.uk
The University of Manchester, Manchester, UK
BigDataFinance, Work Package 1, Research Project 1
Id: Main_Proposal_Training.py
"""

import os
import time
import pickle
import numpy as np
from os import listdir
from scipy.stats import entropy
import Functions.Aux_funcs as aux
from change_finder import ChangeFinder
from sklearn.neighbors import KernelDensity


tic = time.time()



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

for m in keys_stocks:
       
    #==========================================================================
    #                          Filter Configuration
    #==========================================================================
    delta   = 10  
    epsilon = 0.0001 #0.00002
    N_tr    = TrainTest[m]['Parameters']['NTrS']
    eta     = TrainTest[m]['Parameters']['LR']
    sigma   = TrainTest[m]['Parameters']['KS']
    
    
    #==========================================================================
    #                              Data Embedding
    #==========================================================================
    X_train = TrainTest[m]['Train']['X_tr'] 
    y_train = TrainTest[m]['Train']['y_tr'] 
    
    
    #==========================================================================
    #                       Kernel Least-Mean-Square (KLMS)
    #==========================================================================
    err_train    = np.zeros((N_tr,1))    #Vector with errors (training)
    y_train_pred = np.zeros((N_tr,1))    #Vector with predictions (training)


    #==========================================================================
    #                              Initialization
    #==========================================================================
    err_train[0]    = y_train[0]                   #Init: error (training)
    y_train_pred[0] = 0                            #Init: prediction (training)
    dicti           = X_train[0,:][np.newaxis, :]  #Init: dictionary (training)
    alphas          = eta*y_train[0]               #Init: weights (training)
    dicti_tr        = dicti[0,-1]
    num_clusters    = 1 
    
    Set_Weights                         = {}
    Set_Weights[str(num_clusters)]      = alphas
    Set_Dictionaries                    = {}
    Set_Dictionaries[str(num_clusters)] = dicti
    Set_Dict_tr                         = {}
    Set_Dict_tr[str(num_clusters)]      = dicti_tr
    
    
    #==========================================================================
    #                         Change Detection Point
    #==========================================================================
    cf        = ChangeFinder()
    scores    = np.zeros((N_tr,1))
    scores[0] = cf.update(y_train[0])
    
    
    y_grid    = np.linspace(-0.5, 1.5, 500)
    
    n=1
    while n < N_tr:
        print('Stock: ', count,'/24', ' - Training: ', n+1,'/', N_tr)
        
        "%%%%%%%%%%%%%%%%%%%%% TRANING %%%%%%%%%%%%%%%%%%%%%"
        
        "Input vector"
        u_train         = X_train[n,:][np.newaxis, :]
                
        size     = np.zeros(len(Set_Dict_tr))
        kullback = np.ones(len(Set_Dict_tr))
        for i in Set_Dict_tr:
            prev_dict_ytr = Set_Dict_tr[str(i)].reshape(-1,1)
            curr_dict_ytr = np.append(prev_dict_ytr, u_train[0,-1])[:,np.newaxis]
                    
            kde_prev = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(prev_dict_ytr)
            kde_curr = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(curr_dict_ytr)
            
            den_prev = np.exp(kde_prev.score_samples(y_grid[:, None]))
            den_curr = np.exp(kde_curr.score_samples(y_grid[:, None]))
                    
            kullback[int(i)-1] = entropy(pk=den_prev, qk=den_curr)
            
            size[int(i)-1] = np.shape(Set_Dictionaries[str(i)])[0]
            
            del prev_dict_ytr, curr_dict_ytr, kde_prev, kde_curr, den_prev, den_curr
        
        entr_near_cluster = np.min(kullback)
        near_cluster      = np.argmin(kullback) + 1  
        
        
        
        "Compute change point direction"
        scores[n] = cf.update(y_train[n])
        
        "Compute the output of the nearest cluster and error"
        dicti     = Set_Dictionaries[str(near_cluster)]
        alphas    = Set_Weights[str(near_cluster)]
        dicti_tr = Set_Dict_tr[str(near_cluster)]
        
        "Prediction (training)"
        y_train_pred[n] = np.float(np.dot(aux.gaus_kernel(u_train, dicti, sigma), alphas))
        "Error (training)"
        err_train[n]    = y_train[n] - y_train_pred[n]
        
        
        if scores[n] < delta:
                            
            if entr_near_cluster <= epsilon:
                "Quantization approach"
                mindist_arg                    = np.argmin(np.linalg.norm(u_train-dicti, axis=1))
                alphas[mindist_arg]            = alphas[mindist_arg] + eta*err_train[n]
                Set_Weights[str(near_cluster)] = alphas
            else:
                #"Update the weights of the nearest cluster"
                alphas                         = np.append(alphas,eta*err_train[n])
                Set_Weights[str(near_cluster)] = alphas
                #"Update nearest cluster"
                dicti                               = np.append(dicti,u_train,axis=0)
                Set_Dictionaries[str(near_cluster)] = dicti
                #"Update dictionary of ytrains"
                dicti_tr                       = np.append(dicti_tr, u_train[0,-1])[:,np.newaxis]
                Set_Dict_tr[str(near_cluster)] = dicti_tr
        
        else:
            num_clusters = num_clusters + 1
    
            "Knowledge transfer"
            alphas                              = np.append(alphas, eta*err_train[n])
            Set_Weights[str(num_clusters)]      = alphas
            
            dicti                               = np.append(dicti, u_train, axis=0)
            Set_Dictionaries[str(num_clusters)] = dicti
            
            dicti_tr                       = np.append(dicti_tr, u_train[0,-1])[:,np.newaxis]
            Set_Dict_tr[str(num_clusters)] = dicti_tr
            
            
        n = n + 1
    
    count = count + 1

    toc = time.time()
    print(toc-tic, 'sec Elapsed')


    #==============================================================================
    #                              Saving Results
    #==============================================================================
    Model_Stock = {}
    
    Model_Stock['Dictionary'] = Set_Dictionaries
    Model_Stock['Weights']    = Set_Weights
    Model_Stock['y_trains']   = Set_Dict_tr

    pickle.dump(Model_Stock, open('Data\\Models\\Log_Return_Stan\\Model_Stock_' + m + '.pkl', 'wb'))
