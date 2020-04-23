# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 08:51:42 2018

@author: Sergio GARCIA-VEGA
sergio.garcia-vega@manchester.ac.uk
The University of Manchester, Manchester, UK
BigDataFinance, Work Package 1, Research Project 1
Id: Main_Proposal_Testing.py
"""

import os
import time
import pickle
import numpy as np
from os import listdir
from scipy.stats import entropy
import Functions.Aux_funcs as aux
from sklearn.neighbors import KernelDensity




tic = time.time()

#==============================================================================
#                                Importing Models
#==============================================================================
crr_wd       = os.getcwd()
data_dir_mod = os.path.join(crr_wd,'Data\\Models\\Log_Return_Stan')

Models = {}
for mod_stock in listdir(data_dir_mod):
    key_df         = mod_stock.split('.')[0].split('Stock_')[1]
    Models[key_df] = pickle.load(open('Data\\Models\\Log_Return_Stan\\' + mod_stock, 'rb'))
    
keys_stocks = list(Models.keys())
keys_DE     = keys_stocks[0:8]
keys_UK     = keys_stocks[8:16]
keys_US     = keys_stocks[16:]

count = 1

for model in keys_stocks:
    
    Set_Weights      = {}
    Set_Dictionaries = {}
    Set_Dict_tr      = {}
    label_markets    = {}
    tick_markets     = {}
    
    k=1
    for i in Models:
        curr_weights = Models[i]['Weights']
        curr_dicti   = Models[i]['Dictionary']
        curr_ytrains = Models[i]['y_trains']
        for j in curr_weights:        
            if len(curr_weights[j]) <= 10:
                continue
            else:
                Set_Weights[str(k)]      = curr_weights[j]
                Set_Dictionaries[str(k)] = curr_dicti[j]
                Set_Dict_tr[str(k)]      = curr_ytrains[j]            
                if i in keys_DE:
                    label_markets[str(k)] = 1
                elif i in keys_UK:
                    label_markets[str(k)] = 2
                else:
                    label_markets[str(k)] = 3                
                tick_markets[str(k)] = i
                k = k+1
                
    #==========================================================================
    #                    Importing Traning and Testing samples
    #==========================================================================

    data_dir_trte  = os.path.join(crr_wd,'Data')
    
    TrainTest = {}
    for trte_stock in listdir(data_dir_trte):
        key_df = trte_stock.split('.')[0].split('Stock_')[1]
        TrainTest[key_df] = pickle.load(open('Data\\' + str(trte_stock), 'rb'))
    
        
    #==========================================================================
    #                              Data Embedding
    #==========================================================================
    X_test = TrainTest[model]['Test']['X_te']
    y_test = TrainTest[model]['Test']['y_te']
    N_te   = TrainTest[model]['Parameters']['NTeS']
    sigma  = TrainTest[model]['Parameters']['KS']
    
    #==========================================================================
    #                       Kernel Least-Mean-Square (KLMS)
    #==========================================================================
    ev_pred       = np.zeros((1,N_te)) 
    labels        = np.zeros((1,N_te))
    num_samp_pred = np.zeros((N_te,1))
    y_grid        = np.linspace(-0.5, 1.5, 500)
    
    
    "%%%%%%%%%%%%%%%%%%%%% TESTING %%%%%%%%%%%%%%%%%%%%%"
    y_test_pred = np.zeros((N_te,1))
    i = 0
    while i < N_te:
        print('Stock: ', count,'/24', ' - Testing: ', i+1,'/', N_te)
        
        "Input vector"
        u_test = X_test[i,:][np.newaxis, :]
        
        "Select nearest cluster"
        kullback = np.ones(len(Set_Dict_tr))
        for j in Set_Dict_tr:
            prev_dict_yte      = Set_Dict_tr[str(j)]
            curr_dict_yte      = np.append(prev_dict_yte, u_test[0,-1])[:,np.newaxis]
            kde_prev           = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(prev_dict_yte)
            kde_curr           = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(curr_dict_yte)
            den_prev           = np.exp(kde_prev.score_samples(y_grid[:, None]))
            den_curr           = np.exp(kde_curr.score_samples(y_grid[:, None]))
            kullback[int(j)-1] = entropy(pk=den_prev, qk=den_curr)
            del prev_dict_yte, curr_dict_yte, kde_prev, kde_curr, den_prev, den_curr
        near_cluster = np.argmin(kullback) + 1
        labels[0,i]  = near_cluster
        
        dicti  = Set_Dictionaries[str(near_cluster)]
        alphas = Set_Weights[str(near_cluster)]
        
        num_samp_pred[i] = len(dicti)
    
        "Prediction (testing)"
        y_test_pred[i] = np.float(np.dot(aux.gaus_kernel(u_test,dicti,sigma), alphas))
        i = i + 1  
    
    #Saving predictions
    ev_pred = y_test_pred.T
    
    #Performance measure: MSE
    err_test = y_test - y_test_pred
    mse_test = np.mean(err_test**2)
    
    print('Testing MSE: ' + str(mse_test))
    
    toc = time.time()
    print(toc-tic, 'sec Elapsed')
    
    labs     = np.zeros((N_te))
    Tik_labs = {}
    
    for i in range(N_te):
        labs[i]     = label_markets[str(int(labels[0,i]))]
        Tik_labs[str(i)] = tick_markets[str(int(labels[0,i]))]
    count = count + 1
    
    #==========================================================================
    #                              Saving Results
    #==========================================================================
    Results_Proposal = {}
    
    Results_Proposal['Figure 1']                  = {}
    Results_Proposal['Figure 1']['Desired']       = y_test 
    Results_Proposal['Figure 1']['Prediction']    = y_test_pred
    Results_Proposal['Figure 1']['Labels_Market'] = labs
    
    Results_Proposal['Table 1']                     = {}
    Results_Proposal['Table 1']['MSE']              = mse_test
    Results_Proposal['Table 1']['Num_Samp']         = num_samp_pred
    Results_Proposal['Table 1']['Labs_Dicti']       = labels
    Results_Proposal['Table 1']['Labs_Dicti_Ticks'] = Tik_labs
    

    pickle.dump(Results_Proposal, open('Results\\Proposal\\Results_Proposal_' + model + '.pkl', 'wb'))
     
