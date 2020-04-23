# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:42:34 2018

@author: Sergio GARCIA-VEGA
sergio.garcia-vega@manchester.ac.uk
The University of Manchester, Manchester, UK
BigDataFinance, Work Package 1, Research Project 1
Id: Main_NICE.py

"""

import os
import time
import pickle
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import Functions.Aux_funcs as aux

tic = time.time()



#==============================================================================
#                        Importing Traning and Testing samples
#==============================================================================
crr_wd = os.getcwd()
data_dir_trte  = os.path.join(crr_wd,'Data')

TrainTest = {}
for trte_stock in listdir(data_dir_trte):
    key_df = trte_stock.split('.')[0].split('Stock_')[1]
    TrainTest[key_df] = pickle.load(open('Data\\' + str(trte_stock), 'rb'))

    
keys_stocks = list(TrainTest.keys())

count = 1

for model in keys_stocks:
        
    #==========================================================================
    #                          Filter Configuration
    #==========================================================================
    epsilon = 0.06
    N_tr    = TrainTest[model]['Parameters']['NTrS']
    N_te    = TrainTest[model]['Parameters']['NTeS']
    eta     = TrainTest[model]['Parameters']['LR']
    sigma   = TrainTest[model]['Parameters']['KS']
    d_c     = 2*sigma  #centroid distance threshold
    
    
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
    err_train    = np.zeros((N_tr,1))   
    y_train_pred = np.zeros((N_tr,1))    
    mse_test     = np.zeros((N_tr,1))   
    labels       = np.zeros((1,N_te)) 
    
    
    #==========================================================================
    #                              Initialization
    #==========================================================================
    err_train[0]    = y_train[0]                   #Init: error (training)
    y_train_pred[0] = 0                            #Init: prediction (training)
    dicti           = X_train[0,:][np.newaxis, :]  #Init: dictionary (training)
    alphas          = eta*y_train[0]               #Init: weights (training)
    mse_test[0]     = np.mean(y_test**2)           #Init: MSE (testing)
    
    num_clusters  = 1 
    
    Set_Weights                    = {}
    Set_Weights[str(num_clusters)] = alphas
    
    Set_Dictionaries                    = {}
    Set_Dictionaries[str(num_clusters)] = dicti
    
    Set_Centroids                    = {}
    Set_Centroids[str(num_clusters)] = dicti
    
    Ef_Size_Clusters = {}
    Ef_Size_Clusters[str(num_clusters)]  = 1
    
    
    
    num_samp_pred = np.zeros((N_te,1)) 
    
    
    n=1
    while n < N_tr:
        print('Stock: ', count,'/','24', ' - Training: ', n+1,'/',N_tr)
        
        "%%%%%%%%%%%%%%%%%%%%% TRANING %%%%%%%%%%%%%%%%%%%%%"
        
        "Input vector"
        u_train         = X_train[n,:][np.newaxis, :]
        
        "Compute minimum CENTROID distance"
        dist_c   = np.zeros(len(Set_Centroids))
        for i in Set_Centroids:
            Centroid  = Set_Centroids[str(i)]
            dist_c[int(i)-1] = np.linalg.norm(u_train-Centroid)
        d_min_c      = np.min(dist_c)
        
        "Select nearest cluster"
        near_cluster = np.argmin(dist_c) + 1
        
        "Compute the output of the nearest cluster and error"
        dicti  = Set_Dictionaries[str(near_cluster)]
        alphas = Set_Weights[str(near_cluster)]    
        #Prediction (training)
        y_train_pred[n] = np.float(np.dot(aux.gaus_kernel(u_train, dicti,sigma), alphas))
        #Compute error (training)
        err_train[n]    = y_train[n] - y_train_pred[n]
        
        
        
        "------------> NICE Algorithm <------------"
        if d_min_c < d_c:
                    
            "Quantization approach"
            mindist_val  = np.min(np.linalg.norm(u_train-dicti, axis=1))
            mindist_arg  = np.argmin(np.linalg.norm(u_train-dicti, axis=1))
            
            if mindist_val <= epsilon:
                alphas[mindist_arg]            = alphas[mindist_arg] + eta*err_train[n]
                Set_Weights[str(near_cluster)] = alphas
            else:
                #"Update the weights of the nearest cluster"
                alphas                         = np.append(alphas,eta*err_train[n])
                Set_Weights[str(near_cluster)] = alphas
                #"Update nearest cluster"
                dicti                               = np.append(dicti,u_train,axis=0)
                Set_Dictionaries[str(near_cluster)] = dicti
                #"Update centroid of nearest cluster"
                s_near        = Ef_Size_Clusters[str(near_cluster)]
                centroid_near = Set_Centroids[str(near_cluster)]
                Set_Centroids[str(near_cluster)] = (s_near*centroid_near + u_train)/(s_near + 1)
                #"Update effective size"
                Ef_Size_Clusters[str(near_cluster)] = s_near + 1
                
        else:
            num_clusters = num_clusters + 1
            "Create new cluster"
            Set_Centroids[str(num_clusters)]    = u_train
            Ef_Size_Clusters[str(num_clusters)] = 1
            "Knowledge transfer"
            alphas                         = np.append(alphas,eta*err_train[n])
            Set_Weights[str(num_clusters)] = alphas
            
            dicti                               = np.append(dicti,u_train,axis=0)
            Set_Dictionaries[str(num_clusters)] = dicti
        
        n = n + 1
           
    
    "%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%"
    y_test_pred = np.zeros((N_te,1))
    i = 0
    while i < N_te:
        "Input vector"
        u_test = X_test[i,:][np.newaxis, :]

        "Select nearest cluster"
        dist_c   = np.zeros(len(Set_Centroids))
        for j in Set_Centroids:
            Centroid  = Set_Centroids[str(j)]
            dist_c[int(j)-1] = np.linalg.norm(u_test-Centroid)
        near_cluster = np.argmin(dist_c) + 1
        labels[0,i]  = near_cluster
        #"Select Dictionary and Weigths of nearest cluster"
        dicti  = Set_Dictionaries[str(near_cluster)]
        alphas = Set_Weights[str(near_cluster)]
        
        num_samp_pred[i] = len(dicti)

        "Prediction (testing)"
        y_test_pred[i] = np.float(np.dot(aux.gaus_kernel(u_test,dicti,sigma), alphas))
        i = i + 1  
    
    #Performance measure: MSE
    err_test = y_test - y_test_pred
    mse_test = np.mean(err_test**2) 
    count    = count + 1
        
    
    #==============================================================================
    #                              Saving Results
    #==============================================================================
    Results_NICE = {}
    
    Results_NICE['Figure 1']                  = {}
    Results_NICE['Figure 1']['Desired']       = y_test 
    Results_NICE['Figure 1']['Prediction']    = y_test_pred
    
    Results_NICE['Table 1']                     = {}
    Results_NICE['Table 1']['MSE']              = mse_test
    Results_NICE['Table 1']['Num_Samp']         = num_samp_pred
    
    pickle.dump(Results_NICE, open('Results\\NICE\\Results_NICE_' + model + '.pkl', 'wb'))
   


