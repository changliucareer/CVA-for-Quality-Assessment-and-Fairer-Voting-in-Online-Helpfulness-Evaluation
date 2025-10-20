import os
import sys
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################
from toolFunctions import format_time, writeIntoLog, savePlot, writeIntoResult, saveModel
import time
import datetime
import glob
import multiprocessing as mp
import math
from collections import defaultdict
import pickle
import csv
import pandas as pd
import operator
import copy
from itertools import groupby
import re
import psutil
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import norm, bernoulli
import numpy as np
import torch, gc
from tqdm import tqdm
from statistics import mean
from sklearn.model_selection import train_test_split
import sklearn
from CustomizedNN import LRNN_1layer_interaction, LRNN_1layer_interaction_bias, LRNN_1layer_interaction_withoutRankTerm, LRNN_1layer_interaction_bias_withoutRankTerm
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.preprocessing import normalize
import random
from collections import Counter
import gc

#################################################################
def myNLL (y_hat, y):
    y_hat = y_hat.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    l = []
    for i,yi in enumerate(y):
        yi_hat = y_hat[i]
        nll = - (yi*np.log(yi_hat) + (1-yi)*np.log(1-yi_hat))
        l.append(nll)
    return l

class MyIterableDataset(IterableDataset):
    def __init__(self, full_data_files,oneside, testingDataIndexListInFullData, trainOrtest):
        self.dataFiles = full_data_files
        self.oneside = oneside
        self.testingDataIndexListInFullData = testingDataIndexListInFullData
        self.trainOrtest = trainOrtest
    
    def process_data(self, dataFiles, oneside, testingDataIndexListInFullData, trainOrtest):
        baseIndex = 0
        for f in dataFiles:
            with open(f, 'rb') as inputFile:
                if len(testingDataIndexListInFullData)>0: # when having sorted testing data
                    dataset = pickle.load( inputFile)
                    X = dataset[0].todense() # convert sparse matrix to np.array
                    y = dataset[1]
                    
                    # tailor the columns for different parametrization.
                    # the first 4 columns of X are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio, IPW_rank)
                    if oneside:
                        X = X[:,2:] # for one side parametrization, removed the first two columns
                    else: # two sides
                        X = np.concatenate((X[:,:2] , X[:,3:] ), axis=1) # for two sides parametrization, removed the third column

                    for i in range(len(y)):
                        universalIndex = baseIndex + i
                        if trainOrtest == 'train': # to train
                            if universalIndex not in testingDataIndexListInFullData:
                                yield X[i,:], y[i]
                        elif trainOrtest == 'test': # to test
                            if universalIndex in testingDataIndexListInFullData:
                                yield X[i,:], y[i]
                        else: # need full data
                            yield X[i,:], y[i]
                    
                    baseIndex += len(y)
                
                else: # sampled data
                    questions_Data_part = pickle.load( inputFile)
                    for qid, tup in questions_Data_part.items():
                        X = tup[0].todense() # convert sparse matrix to np.array
                        y = tup[1]

                        cutIndex = int(len(y)*0.9) # for each question, its first 90% samples as training samples, the rest as testing samples
                        
                        # tailor the columns for different parametrization.
                        # the first 4 columns of X are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio, IPW_rank)
                        if oneside:
                            X = X[:,2:] # for one side parametrization, removed the first two columns
                        else: # two sides
                            X = np.concatenate((X[:,:2] , X[:,3:] ), axis=1) # for two sides parametrization, removed the third column

                        for i in range(len(y)):
                            if trainOrtest == 'train': # to train
                                if i < cutIndex:
                                    yield X[i,:], y[i]
                            elif trainOrtest == 'test': # to test
                                if i >= cutIndex:
                                    yield X[i,:], y[i]
                            else: # need full data
                                yield X[i,:], y[i]
    
    def __iter__(self):
        return self.process_data(self.dataFiles, self.oneside, self.testingDataIndexListInFullData,self.trainOrtest)
    
def preprocessing (X,y,normalize=False,oneside=True):
    X = np.array(X)
    y = np.array(y, dtype=float)
    # If normalize option is enabled, apply standard scaler transformation.
    if normalize:
        if oneside:
            try:
                X[:,:2] = sklearn.preprocessing.normalize(X[:,:2], axis=0, norm='l2') # only normalized the first two features
            except Exception as e:
                print(e)
        else:
            X[:,:3] = sklearn.preprocessing.normalize(X[:,:3], axis=0, norm='l2') # only normalized the first three features

    # convert X and y to tensors
    X=torch.from_numpy(X.astype(np.float32))
    y=torch.from_numpy(y.astype(np.float32))

    # # y has to be in the form of a column tensor and not a row tensor.
    # y=y.view(y.shape[0],1)
    return X,y

def resultFormat(weights, bias, oneside, ori_questionCount):
    coefs = [] # community-level coefficients
    nus = [] # question-level 
    qs = [] # answer-level qualities
    text = f"bias:{bias}\n"

    for j, coef in enumerate(weights):
        if not oneside: # when do twosides
            if j == 0: # the first feature is pos_vote_ratio. print lambda
                text += f"lambda: {coef}\n"
                coefs.append(coef)
            elif j == 1: # the second feature is neg_vote_ratio, print mu
                text += f"mu: {coef}\n"
                coefs.append(coef)
            elif j == 2:
                text += f"beta: {coef}\n" # the third feature is inversed rank, print beta
                coefs.append(coef)
            elif j == 3:
                text += f"delta: {coef}\n" # the interaction term of rank and seen_pos_vote_ratio, print delta
                coefs.append(coef)
            # elif j == 4:
            #     text += f"epsilon: {coef}\n" # the img column coef, print epsilon
            #     coefs.append(coef)
            # elif j == 5:
            #     text += f"zeta: {coef}\n" # the code column coef, print zeta
            #     coefs.append(coef)
            elif j < ori_questionCount+4:
                text += f"nu_{j-4}: {coef}\n" # the 5th feature to the (questionCount+4)th feature are ralative length for each question, print nus
                nus.append(coef)
            else: # for rest of j
                text += f"q_{j-4-ori_questionCount}: {coef}\n" # the quality features start from j = questionCount+4
                if bias != None:
                    text += f"q_{j-4-ori_questionCount}+bias: {coef+bias}\n"
                qs.append(coef)
        
        else: # when do oneside
            if j == 0: # the first feature is seen_pos_vote_ratio for oneside training, or pos_vote_ratio for only_pvr. print lambda
                text += f"lambda: {coef}\n"
                coefs.append(coef)
            elif j == 1:
                text += f"beta: {coef}\n" # with rank term, the second feature is inversed rank, print beta
                coefs.append(coef)
            elif j == 2:
                text += f"delta: {coef}\n" # the interaction term of rank and seen_pos_vote_ratio, print delta
                coefs.append(coef)
            # elif j == 3:
            #     text += f"epsilon: {coef}\n" # the img column coef, print epsilon
            #     coefs.append(coef)
            # elif j == 4:
            #     text += f"zeta: {coef}\n" # the code column coef, print zeta
            #     coefs.append(coef)
            elif j < ori_questionCount+3:
                text += f"nu_{j-3}: {coef}\n" # the 4th feature to the (questionCount+3)th feature are ralative length for each question, print nus
                nus.append(coef)
            else: # for rest of j
                text += f"q_{j-3-ori_questionCount}: {coef}\n" # the quality features start from j = questionCount+3
                if bias != None:
                    text += f"q_{j-3-ori_questionCount}+bias: {coef+bias}\n"
                qs.append(coef)
    
    return text, coefs, nus, qs
    
###################################################################################################

def myTrain(commIndex, commName, commDir, ori_questionCount, full_data_files, testingDataIndexListInFullData, log_file_name, reg_alpha, try_reg_strengthList, total_sample_count, original_n_feature, initial_batch_size,sampled_comms):
    t0=time.time()

    #######################################################################
    ### training settings #######
    if commName in sampled_comms:
        result_file_name = f"temperalOrderTraining13_trainNewModel_interaction_fixedTau_regAlpha({reg_alpha})_forSampledQuestion_results.txt"
        trained_model_file_name = f'temperalOrderTraining13_trainNewModel_interaction_fixedTau_regAlpha({reg_alpha})_forSampledQuestion_model.sav'
        trained_withSKLEARN_model_file_name = f'temperalOrderTraining13_trainNewModel_interaction_fixedTau_withSKLEARN_regAlpha({reg_alpha})_forSampledQuestion_model.pkl'
        result_withSKLEARN_file_name = f"temperalOrderTraining13_trainNewModel_interaction_fixedTau_withSKLEARN_regAlpha({reg_alpha})_forSampledQuestion_results.txt"
        trained_withlbfgs_model_file_name = f'temperalOrderTraining13_trainNewModel_interaction_fixedTau_withlbfgs_regAlpha({reg_alpha})_forSampledQuestion_model.pkl'
        result_withlbfgs_file_name = f"temperalOrderTraining13_trainNewModel_interaction_fixedTau_withlbfgs_regAlpha({reg_alpha})_forSampledQuestion_results.txt"
        plot_file_name = f'temperalOrderTraining13_trainNewModel_interaction_fixedTau_regAlpha({reg_alpha})_forSampledQuestion_Losses.png'
    else:
        result_file_name = f"temperalOrderTraining13_trainNewModel_interaction_fixedTau_regAlpha({reg_alpha})_results.txt"
        trained_model_file_name = f'temperalOrderTraining13_trainNewModel_interaction_fixedTau_regAlpha({reg_alpha})_model.sav'
        trained_withSKLEARN_model_file_name = f'temperalOrderTraining13_trainNewModel_interaction_fixedTau_withSKLEARN_regAlpha({reg_alpha})_model.pkl'
        result_withSKLEARN_file_name = f"temperalOrderTraining13_trainNewModel_interaction_fixedTau_withSKLEARN_regAlpha({reg_alpha})_results.txt"
        trained_withlbfgs_model_file_name = f'temperalOrderTraining13_trainNewModel_interaction_fixedTau_withlbfgs_regAlpha({reg_alpha})_model.pkl'
        result_withlbfgs_file_name = f"temperalOrderTraining13_trainNewModel_interaction_fixedTau_withlbfgs_regAlpha({reg_alpha})_results.txt"
        plot_file_name = f'temperalOrderTraining13_trainNewModel_interaction_fixedTau_regAlpha({reg_alpha})_Losses.png'
    log_text = ""
    # choices of training settings
    normFlag = False
    regFlag = True   # if True, apply l2 regularization on all parameters; if False, don't apply regularization

    oneside = True   # if True, use one side parametrization; if False, use two side parametrization

    withBias = False # whether add bias term
   
    learnTau = False # if True, learn tau;  if learnTau = False, fix tau = 1
    positiveTau = False # only useful when learnTau = True, if postiveTau = False, don't constrain tau as positive

    # interactionType = 'D' # 'D' means interation term has (1+D)*R, others mean interaction term has (1/(1+D))*R
    interactionType = 'ReciprocalD'

    if oneside:
        tauColumnIndex = 1
    else:
        tauColumnIndex = 2
   
    # select model type
    if learnTau:
        if withBias:
            modeltype='1layer_bias_interaction' 
        else: # without bias
            modeltype='1layer_interaction'
    else:
        if withBias:
            modeltype='1layer_bias_interaction_withoutRankTerm' 
        else: # without bias
            modeltype='1layer_interaction_withoutRankTerm'
    
    opt_forMiniBatchTrain = 'sgd'
    learning_rate = 0.1
    max_iter = 1000   # this is the total number of epochs
    ############################################################################################

    if original_n_feature == None or total_sample_count == 0: # exception
        print(f"Exception for {commName}, original_n_feature:{original_n_feature}, total_sample_count:{total_sample_count}!!!!!!!")
        time.sleep(5)
        return
    
    # check data size, skip training if the numbe of samples is too small
    if total_sample_count<10:
        writeIntoLog(f"consists of {total_sample_count} samples which < 10.\n", commDir , log_file_name)
        print(f"{commName} consists of {total_sample_count} samples which < 10.\n")
        return
    
    # compute the number of features in data. 
    # The first 3 columns of origianl dataset are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio)
    if oneside:
        n_feature = original_n_feature - 2 + 1  # removed 2 columns for one side parametrization, and add 1 column of interaction column
    else:
        n_feature = original_n_feature - 1 + 1  # removed 1 columns for two sides parametrization, and add 1 column of interaction column
    print(f"{commName} has total sample count: {total_sample_count}, and number of features: {n_feature}")
    
    # Prepare training data
    # Build a Streaming DataLoader with customized iterable dataset
    iterable_training_dataset = MyIterableDataset(full_data_files,oneside, testingDataIndexListInFullData, 'train') 
    iterable_testing_dataset = MyIterableDataset(full_data_files,oneside, testingDataIndexListInFullData, 'test') 
    iterable_full_dataset = MyIterableDataset(full_data_files,oneside, testingDataIndexListInFullData, 'full') 
    # prepare data loader
    # use original full data size as batch_size
    my_training_dataloader = DataLoader(iterable_training_dataset, batch_size=total_sample_count) 
    my_testing_dataloader = DataLoader(iterable_testing_dataset, batch_size=total_sample_count) 
    my_full_dataloader = DataLoader(iterable_full_dataset, batch_size=total_sample_count) 
    ####################################################################################################
    # # load sklearn trained model and get accuracy
    # print(f"try to load trained model from SKLEARN for {commName}...")
    # sklearn_accuracy = None
    # final_directory = os.path.join(commDir, r'trained_model_folder')
    # try:
    #     with open(final_directory+'/'+ trained_withSKLEARN_model_file_name, 'rb') as f:
    #         lr = pickle.load(f)
    #     batchIndex =0
    #     for batch in my_full_dataloader:
    #         batchIndex +=1
    #         # print(f"comm {commName}, preparing batch {batchIndex}...")
    #         X = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
    #         y = batch[1]
    #         sklearn_accuracy = lr.score(X, y)
    #         print(f"SKLEARN model accuracy: {sklearn_accuracy} for {commName} with reg_alpha({reg_alpha}).")
        
    #     del batch
    #     X = None
    #     y = None
    #     lr= None
    # except:
    #     print(f"{commName} doesn't have trained model with SKLEARN and reg_alpha({reg_alpha}).")
    
    print(f"first, try to train {commName} using SKLEARN...")
    failFlagSklearn = True
    try:
        ### Cross validation ###################################################################################
        batchIndex = 0
        for batch in my_full_dataloader:
            batchIndex +=1
            print(f"comm {commName}, preparing batch {batchIndex}...")
            X = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
            y = batch[1]

            # add interaction column
            if interactionType == 'D':
                interaction_column = np.multiply(np.reciprocal(X[:,tauColumnIndex:tauColumnIndex+1]) , X[:,0:1])
            else: # interaction with reciprocal of D
                interaction_column = np.multiply(X[:,tauColumnIndex:tauColumnIndex+1] , X[:,0:1])
                
            X = np.concatenate((X[:,:tauColumnIndex+1], interaction_column, X[:,tauColumnIndex+1:]), axis=1)
            X = np.asmatrix(X)
            
            if reg_alpha == try_reg_strengthList[0]: # only do once CV for the first reg_alpha option
                print(f"cross validation for {commName}...")
                myCs = [1/ (2*reg_alpha) for reg_alpha in try_reg_strengthList]
                lrcv = LogisticRegressionCV(Cs= myCs, solver='lbfgs',penalty='l2', fit_intercept=withBias, max_iter=max_iter, cv = 5)  # 5-fold
                lrcv.fit(X,y)
                # shape of lrcv.scores_ is (5, 27)
                CV_scores = list(np.mean(lrcv.scores_[1], axis=0))
                CV_bestC = lrcv.C_[0]
                CV_bestC_index = myCs.index(CV_bestC)
                CV_best_reg_alpha = try_reg_strengthList[CV_bestC_index]
                # CV_best_reg_alpha = None
                # CV_scores = [None]*len(try_reg_strengthList)
            else:
                CV_best_reg_alpha = None
                CV_scores = None
            
        ##########################################################################################################
            # convert regularization strength reg_alpha to C
            myC = 1/ (2*reg_alpha) # when using penalty='l2' and solver='lbfgs'
            print(f"training SKLEARN lr model with reg_alpha {reg_alpha} for {commName}...")
            # set LogisticRegression model 
            lr = LogisticRegression(solver='lbfgs',penalty='l2', fit_intercept=withBias, C=myC, max_iter=max_iter)
        
            lr.fit(X, y)
            # save lr model
            final_directory = os.path.join(commDir, r'trained_model_folder')
            with open(final_directory+'/'+ trained_withSKLEARN_model_file_name,'wb') as f:
                pickle.dump(lr,f)
            print(f"for {commName} model_withSKLEARN saved.")
            
            # compute conformity
            try:
                probOddList = []
                y_pred = lr.predict_proba(X) # shape of y_pred is (sampleCount, 2), for each sample [prob of negVote, prob of posVote]
                # multiply each odd with probOddProduct
                for i,row in enumerate(X): # the shape of row is (1,n_feature)
                    if row[0,0] >= 0: # when n_pos >= n_neg
                        odd = y_pred[i][1]/(1-y_pred[i][1])
                    else: # when n_pos < n_neg
                        odd = (1-y_pred[i][1])/y_pred[i][1]
                    probOddList.append(odd)
                    
                # compute conformity
                conformity_sklearn = np.exp( np.sum(np.log(probOddList)) / total_sample_count )
            except:
                conformity_sklearn = None
            print(f"===> {commName} conformity when using reg_alpha {reg_alpha} is {conformity_sklearn}.") 

            # retrieve the learning result
            if withBias:
                bias_sklearn = lr.intercept_[0]
            else:
                bias_sklearn = None

            tau_sklearn = 1

            weights_sklearn = lr.coef_[0]
            
            result_text_sklearn, coefs_sklearn, nus_sklearn, qs_sklearn = resultFormat(weights_sklearn, bias_sklearn, oneside, ori_questionCount)

            text = f"SKLEARN training ==================================\n"
            text += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '\n'
            text += f"Train oneside:{oneside}, normalize:{normFlag}, regFlag:{regFlag}, reg_alpha:{reg_alpha}, withBias:{withBias}\n"
            text += f"dataset size: ({total_sample_count}, {n_feature})\n conformity_sklearn: {conformity_sklearn}\n"
            print(text)
            writeIntoResult(text + result_text_sklearn, result_withSKLEARN_file_name)
            failFlagSklearn  = False
    except:
        log_text += f"fail to fit {commName} with sklearn LR. \n"
        coefs_sklearn = None
        bias_sklearn = None
        nus_sklearn = None
        qs_sklearn = None
        tau_sklearn = None
        conformity_sklearn = None
        CV_best_reg_alpha = None
        CV_scores = [None]*len(try_reg_strengthList)
        failFlagSklearn = True

    ##################################################################################################
    
    ####################################################################################################
    failFlagLBFGS = True
    coefs_lbfgs = None
    bias_lbfgs = None
    qs_lbfgs = None
    nus_lbfgs = None
    conformity_lbfgs = None
    if failFlagSklearn:
        try:
            print("try to train using cuda and lbfgs...")
            # check gpu count
            cuda_count = torch.cuda.device_count()
            # assign one of gpu as device
            d = commIndex % cuda_count
            device = torch.device('cuda:'+str(d) if torch.cuda.is_available() else 'cpu')

            # Create a NN which equavalent to logistic regression
            input_dim = n_feature
            initial_tau = 1
            if oneside:
                tauColumnIndex = 1
            else:
                tauColumnIndex = 2

            print(f"comm {commName}, preparing model...")
            if modeltype =='1layer_interaction':
                model = LRNN_1layer_interaction(input_dim,initial_tau,tauColumnIndex, positiveTau, interactionType)
            elif modeltype =='1layer_bias_interaction':
                model = LRNN_1layer_interaction_bias(input_dim,initial_tau,tauColumnIndex, positiveTau, interactionType)
            elif modeltype == '1layer_interaction_withoutRankTerm':
                model = LRNN_1layer_interaction_withoutRankTerm(input_dim,tauColumnIndex, interactionType)
            elif modeltype == '1layer_bias_interaction_withoutRankTerm':
                model = LRNN_1layer_interaction_bias_withoutRankTerm(input_dim,tauColumnIndex, interactionType)
            else:
                sys.exit(f"invalid modeltype: {modeltype}")


            # preparing optimizer
            def getOptimizer(opt):
                if opt == 'lbfgs':
                    return torch.optim.LBFGS(model.parameters(), max_iter=max_iter, )
                elif opt == 'sgd':
                    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
                elif opt == 'adamW':
                    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
                elif opt == 'adam':
                    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

            
            opt_forWholeBatchTrain = 'lbfgs'
            optimizer_forWholeBatchTrain = getOptimizer(opt_forWholeBatchTrain)
            
            # using Binary Cross Entropy Loss
            loss_fn = torch.nn.BCELoss(reduction='mean')

            coefs_lbfgs = None
            bias_lbfgs = None
            qs_lbfgs = None
            nus_lbfgs = None
            conformity_lbfgs = None

            
            print(f"comm {commName}, Start to train NN model with LBFGS and batch_size {batch_size}... on device: {device}")
            my_full_dataloader = DataLoader(iterable_full_dataset, batch_size=batch_size)
            model.to(device)

            batchIndex = 0
            for batch in my_full_dataloader:
                batchIndex +=1
                print(f"comm {commName}, preparing batch {batchIndex} of batch_size {batch_size}...")
                X = torch.squeeze(batch[0]).float() # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
                y = batch[1].float() # change tensor data type from int to float32
                
                X = X.to(device)
                y = y.to(device)

                print(f"current X type :{X.type()}, y type:{y.type()}")

                # initialize working optimizer as the whole batch training 
                optimizer = optimizer_forWholeBatchTrain

                torch.autograd.set_detect_anomaly(True) # only for debug, or this will slow the training

                
                probOddList = [] # for conformity computation
                trainingLosses = []

                # lbfgs does auto-train till converge, lbfgs doesn't need more epochs
                # training #####################################################
                model.train() 

                # -------------------------------------------
                # When using L-BFGS optimization, you should use a closure to compute loss (error) during training. 
                # A Python closure is a programming mechanism where the closure function is defined inside another function. 
                # The closure has access to all the parameters and local variables of the outer container function. 
                # This allows the closure function to be passed by name, without parameters, to any statement within the container function. 
                # In the case of L-BFGS training, the loss closure computes the loss when the name of the loss function is passed to the optimizer step() method.
                def loss_closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    outputs = model(X)
                    sample_n = X.shape[0]
                    if outputs.shape[0] ==1:
                        loss = loss_fn(outputs[0], y)
                    else:
                        loss = loss_fn(torch.squeeze(outputs), y) # [m,1] -squeeze-> [m] 
            
                    # add l2 regularization
                    l2_lambda = torch.tensor(reg_alpha, dtype=torch.float32).to(device)
                    l2_reg = torch.tensor(0., dtype=torch.float32).to(device) # initialize regular term as zero
                    
                    # Regularize all parameters:
                    for param in model.parameters():
                        l2_reg += torch.square(torch.norm(param))

                    loss = loss + (l2_lambda/sample_n)* l2_reg # match with reduction = 'mean' in loss_fn

                    trainingLosses.append(loss.item()) 
                    
                    if loss.requires_grad:
                        loss.backward()
                    return loss
                # -------------------------------------------
                optimizer.step(loss_closure) # Updates parameters with the optimizer recursive times till converge for each epoch
                
            # save model
            saveModel(model,trained_withlbfgs_model_file_name)
            print(f"for {commName} model trained_withlbfgs saved.")

            # predict the y and compute probOdds for conformity #####################################################
            model.eval()   

            y_pred = model(X)
            X = X.detach().cpu()
            y = y.detach().cpu()
            y_pred = y_pred.detach().cpu()

            if len(y_pred)==1: # only one sample
                y_pred = y_pred.numpy()[0]
            else:
                y_pred = torch.squeeze(y_pred).numpy()
            
            # multiply each odd with probOddProduct
            for i,row in enumerate(X): # the shape of row is (1,n_feature)
                if row[0] >= 0: # when n_pos >= n_neg
                    odd = y_pred[i]/(1-y_pred[i])
                else: # when n_pos < n_neg
                    odd = (1-y_pred[i])/y_pred[i]
                probOddList.append(odd)

            # compute conformity
            conformity_lbfgs = np.exp( np.sum(np.log(probOddList)) / total_sample_count )
            print(f"===> {commName} lbfgs conformity is {conformity_lbfgs}.")

            # save results as log for model
            # log the training settings
            text = f"torch LBFGS=========================================\n"
            text += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '\n'
            text += f"Train oneside:{oneside}, normalize:{normFlag}, regFlag:{regFlag}, reg_alpha:{reg_alpha},\n"
            text += f"      opt_forWholeBatchTrain :{opt_forWholeBatchTrain}, withBias:{withBias}\n"
            text += f"      max_iter = {max_iter},  learning_rate:{learning_rate}\n"
            text += f"dataset size: ({total_sample_count}, {n_feature})\n\n"
            text += f"min training loss: {trainingLosses[-1]} converge epoch: {len(trainingLosses)}\n"
            text += f"Conformity: {conformity_lbfgs}\n"
            print(text)
            
            # output learned parameters of model with lowest test loss
            parm = defaultdict()
            for name, param in model.named_parameters():
                parm[name]=param.detach().cpu().numpy() 
            
            if withBias:
                bias_lbfgs = parm['linear.bias'][0]
            else:
                bias_lbfgs = None
            
            if learnTau: 
                if positiveTau:
                    tau_lbfgs = np.exp(parm['tau'].item())
                else:
                    tau_lbfgs = parm['tau'].item()
            else:
                tau_lbfgs = 1

            weights_lbfgs = parm['linear.weight'][0]
            
            result_text, coefs_lbfgs, nus_lbfgs, qs_lbfgs = resultFormat(weights_lbfgs, bias_lbfgs, oneside, ori_questionCount)
            writeIntoResult(text +  result_text, result_withlbfgs_file_name)
            
            elapsed = format_time(time.time() - t0)
            
            log_text += "Elapsed: {:}\n".format(elapsed)

            writeIntoLog(text + log_text, commDir , log_file_name)
            failFlagLBFGS = False

            # delete model and release memory of cuda
            model=model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as ee:
            print(f"tried lbfgs with batch_size {total_sample_count}, failed: {ee}")
            writeIntoLog(f"failed to train with cuda and LBFGS using batch_size {total_sample_count}", commDir, log_file_name)
            try:
                del batch
                del X
                del y
            except Exception as eee:
                print(f"{eee}")
            torch.cuda.empty_cache()
            failFlagLBFGS = True
    
    ####################################################################################################
    coefs= None
    bias= None
    nus= None
    qs= None
    tau = None
    conformity= None
    test_accuracy = None
    lowest_test_loss = None
    lowest_test_loss_of_epoch = None
    batch_size = total_sample_count

    failToTrainSGD = True

    if failFlagSklearn and failFlagLBFGS:
        print("try to train using cuda and SGD...")
        # check gpu count
        cuda_count = torch.cuda.device_count()
        # assign one of gpu as device
        d = (commIndex+1) % cuda_count
        d = 2 # only use cuda 0 for debugging
        device = torch.device('cuda:'+str(d) if torch.cuda.is_available() else 'cpu')
        print(f"comm {commName}, Start to train NN model... on device: {device}")

        # Create a NN which equavalent to logistic regression
        input_dim = n_feature
        initial_tau = 1
        
        torch.autograd.set_detect_anomaly(True) # only for debug, or this will slow the training

        train_losses = []
        test_losses = []
        lowest_test_loss_of_epoch = None
        lowest_test_loss = 1000000
        model_with_lowest_test_loss = None
        test_accuracy = None

        coefs= None
        bias= None
        nus= None
        qs= None
        tau = None
        conformity= None

        max_iter_SGD = 20
        batch_size = initial_batch_size

        failToTrainSGD = True
        while (failToTrainSGD):
            try:
                # prepare data loader
                print(f"comm {commName}, Start to train NN model with SGD and batch_size {batch_size}... on device: {device}")
                # prepare data loader
                my_training_dataloader = DataLoader(iterable_training_dataset, batch_size=batch_size) 
                my_testing_dataloader = DataLoader(iterable_testing_dataset, batch_size=batch_size) 
                my_full_dataloader = DataLoader(iterable_full_dataset, batch_size=batch_size) 

                print(f"comm {commName}, preparing model...")
                if modeltype =='1layer_interaction':
                    model = LRNN_1layer_interaction(input_dim,initial_tau,tauColumnIndex, positiveTau, interactionType)
                elif modeltype =='1layer_bias_interaction':
                    model = LRNN_1layer_interaction_bias(input_dim,initial_tau,tauColumnIndex, positiveTau, interactionType)
                elif modeltype == '1layer_interaction_withoutRankTerm':
                    model = LRNN_1layer_interaction_withoutRankTerm(input_dim,tauColumnIndex, interactionType)
                elif modeltype == '1layer_bias_interaction_withoutRankTerm':
                    model = LRNN_1layer_interaction_bias_withoutRankTerm(input_dim,tauColumnIndex, interactionType)
                else:
                    sys.exit(f"invalid modeltype: {modeltype}")

            
                # using Binary Cross Entropy Loss
                loss_fn = torch.nn.BCELoss(size_average=True, reduction='mean')

                # preparing optimizer
                def getOptimizer(opt):
                    if opt == 'lbfgs':
                        return torch.optim.LBFGS(model.parameters(), max_iter=max_iter, )
                    elif opt == 'sgd':
                        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
                    elif opt == 'adamW':
                        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
                    elif opt == 'adam':
                        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

                optimizer_forMiniBatchTrain = getOptimizer(opt_forMiniBatchTrain)

                # initialize working optimizer as the one for mini batch training 
                optimizer = optimizer_forMiniBatchTrain
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[290], gamma=0.1)

                model.to(device)

                for epoch in tqdm(range(int(max_iter_SGD)),desc=f'Training Epochs ({commName})'):
                    # training #####################################################
                    model.train() 

                    batches_train_losses = []

                    batchIndex = 0
                    for batch in my_training_dataloader:
                        optimizer.zero_grad()
                        batchIndex +=1
        
                        # prepare the data tensors
                        # print(f"comm {commName}, preparing batch {batchIndex}...")
                        X_batch = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
                        y_batch = batch[1]
                        X_batch,y_batch = preprocessing(X_batch,y_batch,normFlag)
                        
                        
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)      

                        outputs = model(X_batch)
                        sample_n = X_batch.size()[0]
                        
                        if outputs.shape[0] ==1:
                            loss = loss_fn(outputs[0], y_batch)
                        else:
                            loss = loss_fn(torch.squeeze(outputs), y_batch) # [m,1] -squeeze-> [m] 
                        
                        if regFlag:
                            # add l2 regularization
                            l2_lambda = torch.tensor(reg_alpha).to(device)
                            l2_reg = torch.tensor(0.).to(device)
                            
                            # Regularize all parameters:
                            for param in model.parameters():
                                l2_reg += torch.square(torch.norm(param))

                            loss = loss + (l2_lambda/sample_n)* l2_reg  

                        try: # possible RuntimeError: Function 'AddmmBackward0' returned nan values in its 2th output.
                            loss.backward()
                        except Exception as eee:
                            print(f"fail to train epoch:{epoch+1}, batch:{batchIndex} of {commName}, {eee}")
                            return
                        
                        optimizer.step()
                        
                        with torch.no_grad():
                            # Calculating the loss for the train dataset
                            batches_train_losses.append(loss.item())
                        
                        # clear memory of cuda
                        del batch
                        X_batch = X_batch.detach().cpu()
                        y_batch = y_batch.detach().cpu()     
                        outputs = outputs.detach().cpu()
                        del X_batch
                        del y_batch     
                        del outputs
                        gc.collect()
                        torch.cuda.empty_cache() # make sure to get rid of cached memory to actually free up the space after deleting
                    
                    # testing #####################################################
                    model.eval()
            
                    batches_test_losses = []
                    batches_test_correct = []
                    batches_test_sampleCount = []

                    batchIndex = 0
                    for batch in my_testing_dataloader:
                        batchIndex +=1
        
                        # prepare the data tensors
                        # print(f"comm {commName}, preparing batch {batchIndex}...")
                        X_batch = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
                        y_batch = batch[1]
                        X_batch,y_batch = preprocessing(X_batch,y_batch,normFlag)
                        
                        
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)      

                        outputs = model(X_batch)
                        sample_n = X_batch.size()[0]
                        
                        if outputs.shape[0] ==1:
                            loss = loss_fn(outputs[0], y_batch)
                        else:
                            loss = loss_fn(torch.squeeze(outputs), y_batch) # [m,1] -squeeze-> [m] 
                        
                        if regFlag:
                            # add l2 regularization
                            l2_lambda = torch.tensor(reg_alpha).to(device)
                            l2_reg = torch.tensor(0.).to(device)
                            
                            # Regularize all parameters:
                            for param in model.parameters():
                                l2_reg += torch.square(torch.norm(param))

                            loss = loss + (l2_lambda/sample_n)* l2_reg  
                
                        # Calculating the loss and accuracy for the test dataset
                        correct = np.sum(torch.squeeze(outputs.cpu()).round().detach().numpy() == y_batch.cpu().detach().numpy())
                        batches_test_correct.append(correct)
                        batches_test_sampleCount.append(sample_n)
                        batches_test_losses.append(loss.item())

                        # clear memory of cuda
                        del batch
                        X_batch = X_batch.detach().cpu()
                        y_batch = y_batch.detach().cpu()     
                        outputs = outputs.detach().cpu()
                        del X_batch
                        del y_batch     
                        del outputs
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    # print out current training stat every 10 epochs
                    if epoch%10 == 0:
                        print(f"comm {commName}, Train Iteration {epoch+1} -  avg train batch Loss: {round(mean(batches_train_losses),4)},  avg test batch Loss: {round(mean(batches_test_losses),4)}, avg test batch Accuracy: {round(correct/sample_n,4)}")
                    
                    train_losses.append(sum(batches_train_losses)) # sum up the losses of all batches
                    test_losses.append(sum(batches_test_losses)) # sum up the losses of all batches

                    if test_losses[-1] < lowest_test_loss:
                        lowest_test_loss= test_losses[-1]
                        lowest_test_loss_of_epoch = epoch+1
                        model_with_lowest_test_loss = copy.deepcopy(model)
                        test_accuracy = sum(batches_test_correct) / sum(batches_test_sampleCount)

                    scheduler.step() 

                # delete model and release memory of cuda
                model=model.cpu()
                del model
                gc.collect()
                torch.cuda.empty_cache()

                # retrain with full_data
                print(f"retrain with full_data for {commName} for 1 epoch...")
                retrain_optimizer = torch.optim.SGD(model_with_lowest_test_loss.parameters(), lr=learning_rate, momentum=0.9)
                for epoch in tqdm(range(int(1)),desc=f'Retraining Epochs ({commName})'):
                    # training #####################################################
                    model_with_lowest_test_loss.train() 

                    retrain_batches_train_losses = []

                    batchIndex = 0
                    for batch in my_full_dataloader:
                        retrain_optimizer.zero_grad()
                        batchIndex +=1
        
                        # prepare the data tensors
                        print(f"comm {commName}, retraining batch {batchIndex}...")
                        X_batch = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
                        y_batch = batch[1]
                        X_batch,y_batch = preprocessing(X_batch,y_batch,normFlag)
                        
                        
                        X_batch = X_batch.to(device)
                        y_batch = y_batch.to(device)      

                        outputs = model_with_lowest_test_loss(X_batch)
                        sample_n = X_batch.size()[0]
                        
                        if outputs.shape[0] ==1:
                            loss = loss_fn(outputs[0], y_batch)
                        else:
                            loss = loss_fn(torch.squeeze(outputs), y_batch) # [m,1] -squeeze-> [m] 
                        
                        if regFlag:
                            # add l2 regularization
                            l2_lambda = torch.tensor(reg_alpha).to(device)
                            l2_reg = torch.tensor(0.).to(device)
                            
                            # Regularize all parameters:
                            for param in model_with_lowest_test_loss.parameters():
                                l2_reg += torch.square(torch.norm(param))

                            loss = loss + (l2_lambda/sample_n)* l2_reg  

                        try: # possible RuntimeError: Function 'AddmmBackward0' returned nan values in its 2th output.
                            loss.backward()
                        except Exception as eee:
                            print(f"fail to retrain epoch:{epoch+1}, batch:{batchIndex} of {commName}, {eee}")
                            return
                        
                        retrain_optimizer.step()
                        
                        with torch.no_grad():
                            # Calculating the loss for the full dataset
                            retrain_batches_train_losses.append(loss.item())

                        # clear memory of cuda
                        del batch
                        X_batch = X_batch.detach().cpu()
                        y_batch = y_batch.detach().cpu()     
                        outputs = outputs.detach().cpu()
                        del X_batch
                        del y_batch     
                        del outputs
                        gc.collect()
                        torch.cuda.empty_cache()  

                    # compute retrain loss and accuracy
                    retrain_loss = sum(retrain_batches_train_losses)
                
                # save model
                saveModel(model_with_lowest_test_loss,trained_model_file_name)
                print(f"for {commName} model_with_lowest_test_loss saved, retrain_loss: {retrain_loss}")

                # predict the y and compute probOdds for conformity #####################################################
                model_with_lowest_test_loss.eval()   
                probOddList = [] # for conformity computation
                retrain_batches_eval_correct = []
                retrain_batches_eval_sampleCount = []
                my_full_dataloader = DataLoader(iterable_full_dataset, batch_size=batch_size) # reinitialize the dataloader to reiterate
                batchIndex = 0
                for batch in my_full_dataloader:
                    batchIndex +=1

                    # prepare the data tensors
                    # print(f"comm {commName}, preparing batch {batchIndex}...")
                    X_batch = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
                    y_batch = batch[1]
                    X_batch,y_batch = preprocessing(X_batch,y_batch,normFlag)
                    sample_n = X_batch.size()[0]
                    
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)      

                    y_pred = model_with_lowest_test_loss(X_batch)

                    X_batch = X_batch.detach().cpu()
                    y_batch = y_batch.detach().cpu()
                    y_pred = y_pred.detach().cpu()

                    if len(y_pred)==1: # only one sample
                        y_pred = y_pred.numpy()[0]
                    else:
                        y_pred = torch.squeeze(y_pred).numpy()
                    # multiply each odd with probOddProduct
                    for i,row in enumerate(X_batch): # the shape of row is (1,n_feature)
                        if row[0] >= 0: # when n_pos >= n_neg
                            odd = y_pred[i]/(1-y_pred[i])
                        else: # when n_pos < n_neg
                            odd = (1-y_pred[i])/y_pred[i]
                        probOddList.append(odd)

                    # Calculating the accuracy for the full dataset
                    correct = np.sum(y_pred.round() == y_batch.numpy())
                    retrain_batches_eval_correct.append(correct)
                    retrain_batches_eval_sampleCount.append(sample_n)
                    
                    # clear memory of cuda
                    del batch  
                    del X_batch
                    del y_batch     
                    del y_pred
                    gc.collect()
                    torch.cuda.empty_cache()        
                    
                # compute conformity
                conformity = np.exp( np.sum(np.log(probOddList)) / total_sample_count )
                # compute retrain accuracy
                retrain_accuracy = sum(retrain_batches_eval_correct) / sum(retrain_batches_eval_sampleCount)
                print(f"===> {commName} conformity is {conformity}, retrain accuracy is {retrain_accuracy}") 

                # save results as log for model
                # log the training settings
                text = f"torch SGD=========================================\n"
                text += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '\n'
                text += f"Train oneside:{oneside}, normalize:{normFlag}, regFlag:{regFlag}, reg_alpha:{reg_alpha},\n"
                text += f"      opt_forMiniBatchTrain :{opt_forMiniBatchTrain}, withBias:{withBias}\n"
                text += f"      max_iter = {max_iter},  learning_rate:{learning_rate}\n"
                text += f"dataset size: ({total_sample_count}, {n_feature})\n\n"
                text += f"trained with batch_size:{batch_size}\navg training loss: {mean(train_losses)} \navg testing loss:{mean(test_losses)},lowest_test_loss_of_epoch:{lowest_test_loss_of_epoch} lowest_test_loss:{lowest_test_loss}\ntest accuracy:{test_accuracy}\n"
                text += f"retrain loss: {retrain_loss}, retrain_accuracy: {retrain_accuracy}\n"
                text += f"Conformity: {conformity}\n"
                print(text)
                
                # output learned parameters of model with lowest test loss
                parm = defaultdict()
                for name, param in model_with_lowest_test_loss.named_parameters():
                    parm[name]=param.detach().cpu().numpy() 
                
                if withBias:
                    bias = parm['linear.bias'][0]
                else:
                    bias = None

                weights = parm['linear.weight'][0]
                
                result_text, coefs, nus, qs = resultFormat(weights, bias, oneside, ori_questionCount)
                writeIntoResult(text + f"\ntraining losses: {train_losses}\n"+ f"\ntesting losses: {test_losses}\n"+ f"\ntesting accuracy: {test_accuracy}\n" + result_text, result_file_name)
                
                elapsed = format_time(time.time() - t0)
                
                log_text += "Elapsed: {:}\n".format(elapsed)

                writeIntoLog(text + log_text, commDir , log_file_name)

                failToTrainSGD = False # send signal to stop try
                model_with_lowest_test_loss = model_with_lowest_test_loss.cpu()
                del model_with_lowest_test_loss
                gc.collect()
                torch.cuda.empty_cache()

                # visualize the losses
                plt.cla()
                plt.plot(range(len(train_losses)), train_losses, 'g-', label=f'trainning')
                plt.plot(range(len(test_losses)), test_losses, 'b-', label=f'testing\nlowest loss:{lowest_test_loss}\nreached at epoch {lowest_test_loss_of_epoch}')
                plt.xlabel("epoch")
                plt.ylabel("Loss")
                plt.legend(loc="upper right")
                savePlot(plt, plot_file_name)


            except Exception as ee:
                print(f"tried sgd, failed: {ee}")
                writeIntoLog(f"failed to train with cuda and sgd using batch_size {batch_size}", commDir, log_file_name)
                batch_size = int(batch_size/2) +1 # shrink the batch_size
                try:
                    del batch
                    del X_batch
                    del y_batch  
                    del model
                except Exception as eee:
                    print(f"{eee}")
                gc.collect()
                torch.cuda.empty_cache()   

                if batch_size < 10000:
                    break
    

    return total_sample_count, n_feature, coefs_sklearn, nus_sklearn, qs_sklearn,bias_sklearn, conformity_sklearn,coefs_lbfgs, nus_lbfgs, qs_lbfgs,bias_lbfgs, conformity_lbfgs, CV_best_reg_alpha, CV_scores
    # return total_sample_count, n_feature, coefs, nus, qs, conformity, lowest_test_loss, test_accuracy, retrain_loss, retrain_accuracy, train_losses,test_losses,lowest_test_loss_of_epoch, sklearn_accuracy, batch_size

def myFun(commIndex, commName, commDir, root_dir, splitted_comms,sampled_comms):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    log_file_name = "temperalOrderTraining13_newModel_interaction_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    if commName == 'stackoverflow': # using subcomm to represent stackoverflow
        subComms_data_folder = os.path.join(commDir, f'subCommunities_folder')
        ## Load all sub community direcotries 
        with open(subComms_data_folder+'/'+f'subCommName2subCommDir.dict', 'rb') as inputFile:
            subCommName2commDir = pickle.load( inputFile)
        subCommDir = subCommName2commDir['reactjs']
        subComm_intermediate_directory = os.path.join(subCommDir, r'intermediate_data_folder')

    
    if commName != 'stackoverflow':
        if commName in sampled_comms:
            with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList_forSampledQuestion.dict', 'rb') as inputFile:
                ori_Questions = pickle.load( inputFile)
                ori_questionCount = len(ori_Questions)
                ori_Questions.clear() # clear this to same memory
        else:
            with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
                ori_Questions = pickle.load( inputFile)
                ori_questionCount = len(ori_Questions)
                ori_Questions.clear() # clear this to same memory
    else: # using subcomm to represent stackoverflow
        subComm_QuestionsWithEventList_directory = subCommDir+'/'+f'QuestionsWithEventList_tag_reactjs.dict'
        with open(subComm_QuestionsWithEventList_directory, 'rb') as inputFile:
            ori_Questions = pickle.load( inputFile)
            ori_questionCount = len(ori_Questions)
            ori_Questions.clear() # clear this to same memory
    
    # get full data file list
    if commName != 'stackoverflow':
        if (commName in splitted_comms) and (commName not in sampled_comms):
            splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
            sortedBatchesData_directory = os.path.join(splitFolder_directory, r'Datasets_sorted_batches_folder')
            full_data_files = [ f.path for f in os.scandir(sortedBatchesData_directory) if f.path.split('/')[-1].startswith("whole_datasets_sorted_removeFirstRealVote_batch_") ]
            full_data_files.sort(key=lambda f: int(f.split('/')[-1].split('_')[-1].split('.')[0]))
        elif commName in sampled_comms:
            full_data_files = [intermediate_directory+f"/whole_datasets_forEachQuestion_removeFirstRealVote_forSampledQuestion.dict"]
        else:
            full_data_files = [intermediate_directory+f"/whole_datasets_sorted_removeFirstRealVote.dict"]
    else:
        # using subcomm to represent stackoverflow
        resultFileFolder = os.path.join(subComm_intermediate_directory, r'Datasets_sorted_batches_folder')
        full_data_files = [ f.path for f in os.scandir(resultFileFolder) if f.path.split('/')[-1].startswith("whole_datasets_sorted_removeFirstRealVote_batch_") ]
        full_data_files.sort(key=lambda f: int(f.split('/')[-1].split('_')[-1].split('.')[0]))

    
    # load testing data's sortingBase
    # get testing data raw index in full data
    if commName in sampled_comms:
        testingDataIndexListInFullData = []
    else:
        with open(intermediate_directory+f"/temperalOrderTraining6_outputs.dict", 'rb') as inputFile:
            qid2TestingSortingBaseList, _ = pickle.load( inputFile)
        sortingBaseListOfTestingData = []
        for qid, testingSortingBaseList in qid2TestingSortingBaseList.items():
            sortingBaseListOfTestingData.extend(testingSortingBaseList)

        with open(intermediate_directory+f"/sorted_qidAndSampleIndexAndSortingBase.dict", 'rb') as inputFile:
            sorted_qidAndSampleIndex = pickle.load( inputFile)

        testingDataIndexListInFullData = [i for i, tup in enumerate(sorted_qidAndSampleIndex) if tup[2] in sortingBaseListOfTestingData]
        

    # get total_sample_count and original_n_feature
    total_sample_count = 0
    original_n_feature = None
    if commName in sampled_comms:
        for f in full_data_files:
            with open(f, 'rb') as inputFile:
                questions_Data_part = pickle.load( inputFile)
                for qid, tup in questions_Data_part.items():
                    y = tup[1]                   
                    total_sample_count += len(y)

                    if original_n_feature == None and len(y) != 0:
                        X = tup[0].todense()
                        original_n_feature = X.shape[1] # the column number of X is the original number of features
                questions_Data_part.clear()
    else:
        for f in full_data_files:
            with open(f, 'rb') as inputFile:
                full_data = pickle.load( inputFile)
                total_sample_count += len(full_data[1])
                if original_n_feature == None:
                    original_n_feature = full_data[0].todense().shape[1]
                full_data = [] # clear to save memory



    # training 
    try_reg_strengthList = [0.1,0.2, 0.3,0.4, 0.5,0.6, 0.7,0.8,0.9,
                            1, 2, 3,4,5, 6, 7,8,9,
                            10,20, 30,40,50,60, 70,80,90,
                            100, 200, 300, 400, 500, 600, 700, 800, 900,
                            1000]
    
    initial_batch_size = total_sample_count
    for reg_alpha in try_reg_strengthList:

        # check whether already done this step, skip
        resultFiles = [intermediate_directory+f"/temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha})_return.dict", intermediate_directory+f"/temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha})_forSampledQuestion_return.dict"]
        if os.path.exists(resultFiles[0]):
            # target date
            target_date = datetime.datetime(2024, 12, 28)
            # file last modification time
            timestamp = os.path.getmtime(resultFiles[0])
            # convert timestamp into DateTime object
            datestamp = datetime.datetime.fromtimestamp(timestamp)
            print(f'{commName} Modified Date/Time:{datestamp}')
            if datestamp >= target_date: # the final result file exists
                print(f"{commName} has already done this step for reg_alpha({reg_alpha}).")
                continue
        elif os.path.exists(resultFiles[1]):
            # target date
            target_date = datetime.datetime(2024, 12, 28)
            # file last modification time
            timestamp = os.path.getmtime(resultFiles[1])
            # convert timestamp into DateTime object
            datestamp = datetime.datetime.fromtimestamp(timestamp)
            print(f'{commName} Modified Date/Time:{datestamp}')
            if datestamp >= target_date: # the final result file exists
                print(f"{commName} has already done this step for reg_alpha({reg_alpha}).")
                continue

        output = myTrain(commIndex, commName, commDir, ori_questionCount, full_data_files, testingDataIndexListInFullData, log_file_name, reg_alpha, try_reg_strengthList, total_sample_count, original_n_feature, initial_batch_size, sampled_comms)
        total_sample_count, n_feature, coefs_sklearn, nus_sklearn, qs_sklearn,bias_sklearn, conformity_sklearn,coefs_lbfgs, nus_lbfgs, qs_lbfgs,bias_lbfgs, conformity_lbfgs, CV_best_reg_alpha, cur_CV_scores = output
        # total_sample_count, n_feature, coefs, nus, qs, conformity, lowest_test_loss, test_accuracy, retrain_loss, retrain_accuracy, train_losses,test_losses,lowest_test_loss_of_epoch, sklearn_accuracy, batch_size = output
       
        # # initial batch_size for next reg_alpha
        # initial_batch_size = batch_size

        if cur_CV_scores != None:
            CV_scores = cur_CV_scores

        # learned qs analysis
        # q_mean, q_std = norm.fit(qs)
        q_sklearn_mean, q_sklearn_std = norm.fit(qs_sklearn)
        q_sklearn_within1 = 0
        q_sklearn_within2 = 0

        for q in qs_sklearn:
            if q<= 1 and q>= -1 :
                q_sklearn_within1 += 1
            if q<=2 and q>= -2:
                q_sklearn_within2 += 1

        q_sklearn_within1_percent = q_sklearn_within1 /len(qs_sklearn)
        q_sklearn_within2_percent = q_sklearn_within2 /len(qs_sklearn)
        
        if qs_lbfgs != None:
            q_lbfgs_mean, q_lbfgs_std = norm.fit(qs_lbfgs)
            q_lbfgs_within1 = 0
            q_lbfgs_within2 = 0

            for q in qs_lbfgs:
                if q<= 1 and q>= -1 :
                    q_lbfgs_within1 += 1
                if q<=2 and q>= -2:
                    q_lbfgs_within2 += 1

            q_lbfgs_within1_percent = q_lbfgs_within1 /len(qs_lbfgs)
            q_lbfgs_within2_percent = q_lbfgs_within2 /len(qs_lbfgs)
        else:
            q_lbfgs_mean= None
            q_lbfgs_std = None
            q_lbfgs_within1_percent = None
            q_lbfgs_within2_percent = None

        # if qs != None:
        #     q_mean, q_std = norm.fit(qs)
        #     q_within1 = 0
        #     q_within2 = 0

        #     for q in qs:
        #         if q<= 1 and q>= -1 :
        #             q_within1 += 1
        #         if q<=2 and q>= -2:
        #             q_within2 += 1

        #     q_within1_percent = q_within1 /len(qs)
        #     q_within2_percent = q_within2 /len(qs)
        # else:
        #     q_mean= None
        #     q_std = None
        #     q_within1_percent = None
        #     q_within2_percent = None

        # learned nus analysis
        nu_sklearn_mean, nu_sklearn_std = norm.fit(nus_sklearn)
        nu_sklearn_within1 = 0
        nu_sklearn_within2 = 0

        for nu in nus_sklearn:
            if nu<= 1 and nu>= -1 :
                nu_sklearn_within1 += 1
            if nu<=2 and nu>= -2:
                nu_sklearn_within2 += 1

        nu_sklearn_within1_percent = nu_sklearn_within1 /len(nus_sklearn)
        nu_sklearn_within2_percent = nu_sklearn_within2 /len(nus_sklearn)
        
        if nus_lbfgs != None:
            nu_lbfgs_mean, nu_lbfgs_std = norm.fit(nus_lbfgs)
            nu_lbfgs_within1 = 0
            nu_lbfgs_within2 = 0

            for nu in nus_lbfgs:
                if nu<= 1 and nu>= -1 :
                    nu_lbfgs_within1 += 1
                if nu<=2 and nu>= -2:
                    nu_lbfgs_within2 += 1

            nu_lbfgs_within1_percent = nu_lbfgs_within1 /len(nus_lbfgs)
            nu_lbfgs_within2_percent = nu_lbfgs_within2 /len(nus_lbfgs)
        else:
            nu_lbfgs_mean= None
            nu_lbfgs_std = None
            nu_lbfgs_within1_percent = None
            nu_lbfgs_within2_percent = None
            
        # if nus != None:
        #     nu_mean, nu_std = norm.fit(nus)
        #     nu_within1 = 0
        #     nu_within2 = 0

        #     for nu in nus:
        #         if nu<= 1 and nu>= -1 :
        #             nu_within1 += 1
        #         if nu<=2 and nu>= -2:
        #             nu_within2 += 1

        #     nu_within1_percent = nu_within1 /len(nus)
        #     nu_within2_percent = nu_within2 /len(nus)
        # else:
        #     nu_mean= None
        #     nu_std = None
        #     nu_within1_percent = None
        #     nu_within2_percent = None

        
        if (coefs_lbfgs != None) or (coefs_sklearn != None):
            return_trainSuccess_dict = defaultdict()
            return_trainSuccess_dict[commName] = {'dataShape':(total_sample_count, n_feature),
                                                #  'trainLosses':train_losses, 'testLosses':test_losses, 'epoch':lowest_test_loss_of_epoch, 'lowest_test_loss':lowest_test_loss, 'testAcc':test_accuracy,
                                                # 'coefs':coefs,'bias':bias,'qs':qs, 'conformity':conformity,
                                                'coefs_sklearn':coefs_sklearn, 'nus_sklearn':nus_sklearn, 'qs_sklearn':qs_sklearn,'bias_sklearn':bias_sklearn,
                                                'conformity_sklearn':conformity_sklearn,
                                                'coefs_lbfgs':coefs_lbfgs, 'nus_lbfgs':nus_lbfgs, 'qs_lbfgs':qs_lbfgs,'bias_lbfgs':bias_lbfgs,
                                                'conformity_lbfgs':conformity_lbfgs,
                                                'CV_best_reg_alpha':CV_best_reg_alpha, 'CV_scores':CV_scores
                                                }
            if (commName).strip("sampled_") in sampled_comms:
                # save return dict
                with open(intermediate_directory+f"/temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha})_forSampledQuestion_return.dict", 'wb') as outputFile:
                    pickle.dump(return_trainSuccess_dict, outputFile)
                    print( f"saved return_trainSuccess_dict for {reg_alpha} of {commName}_forSampledQuestion.")
            else:
                # save return dict
                with open(intermediate_directory+f"/temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha})_return.dict", 'wb') as outputFile:
                    pickle.dump(return_trainSuccess_dict, outputFile)
                    print( f"saved return_trainSuccess_dict for {reg_alpha} of {commName}.")
        
        # if (coefs != None):
        #     return_trainSuccess_dict = defaultdict()
        #     return_trainSuccess_dict[commName] = {'dataShape':(total_sample_count, n_feature),
        #                                          'trainLosses':train_losses, 'testLosses':test_losses, 'epoch':lowest_test_loss_of_epoch, 'lowest_test_loss':lowest_test_loss, 'testAcc':test_accuracy,'retrain_loss':retrain_loss, 'retrainAcc':retrain_accuracy,
        #                                         'coefs':coefs,'qs':qs,'nus':nus, 'conformity':conformity
        #                                         }
            
        #     # save return dict
        #     with open(intermediate_directory+f"/temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha})_SGD_return.dict", 'wb') as outputFile:
        #         pickle.dump(return_trainSuccess_dict, outputFile)
        #         print( f"saved SGD return_trainSuccess_dict for {reg_alpha} of {commName}.")

        # # save csv
        with open(root_dir +'/'+'allComm_temperalOrderTraining13_newModel_interaction_fixedTau_results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if commName == 'stackoverflow':
                commName = 'reactjs_SOF'
            elif commName in sampled_comms:
                commName = 'sampled_' + commName.replace('.stackexchange','')
            else:
                commName = commName.replace('.stackexchange','')
            

            if (coefs_sklearn != None) and (coefs_lbfgs != None):
                writer.writerow( [commName,total_sample_count, reg_alpha, coefs_sklearn[0], coefs_sklearn[1], coefs_sklearn[2],  q_sklearn_mean, q_sklearn_std, q_sklearn_within1_percent, q_sklearn_within2_percent,nu_sklearn_mean, nu_sklearn_std, nu_sklearn_within1_percent, nu_sklearn_within2_percent, conformity_sklearn, coefs_lbfgs[0],coefs_lbfgs[1], coefs_lbfgs[2], q_lbfgs_mean, q_lbfgs_std, q_lbfgs_within1_percent, q_lbfgs_within2_percent,nu_lbfgs_mean, nu_lbfgs_std, nu_lbfgs_within1_percent, nu_lbfgs_within2_percent, conformity_lbfgs, CV_scores[try_reg_strengthList.index(reg_alpha)] , CV_best_reg_alpha])
            elif (coefs_sklearn != None):
                writer.writerow( [commName,total_sample_count, reg_alpha, coefs_sklearn[0], coefs_sklearn[1], coefs_sklearn[2],  q_sklearn_mean, q_sklearn_std, q_sklearn_within1_percent, q_sklearn_within2_percent,nu_sklearn_mean, nu_sklearn_std, nu_sklearn_within1_percent, nu_sklearn_within2_percent, conformity_sklearn,coefs_lbfgs,coefs_lbfgs,coefs_lbfgs, q_lbfgs_mean, q_lbfgs_std, q_lbfgs_within1_percent, q_lbfgs_within2_percent,nu_lbfgs_mean, nu_lbfgs_std, nu_lbfgs_within1_percent, nu_lbfgs_within2_percent, conformity_lbfgs, CV_scores[try_reg_strengthList.index(reg_alpha)] , CV_best_reg_alpha])

        # # # save csv
        # with open(root_dir +'/'+'allComm_temperalOrderTraining13_newModel_interaction_fixedTau_SGD_results.csv', 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',',
        #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     if commName == 'stackoverflow':
        #         commName = 'reactjs_SOF'
        #     else:
        #         commName = commName.replace('.stackexchange','')
        #     if (coefs != None) : # write SGD results with sklearn accuracy
        #         writer.writerow( [commName,total_sample_count, n_feature, len(testingDataIndexListInFullData), reg_alpha, coefs[0], coefs[1], coefs[2], q_mean, q_std, q_within1_percent, q_within2_percent,nu_mean, nu_std, nu_within1_percent, nu_within2_percent, conformity, lowest_test_loss, test_accuracy, retrain_loss, retrain_accuracy, train_losses[lowest_test_loss_of_epoch-1], sklearn_accuracy, lowest_test_loss_of_epoch])
            

def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # # # save csv
    # with open(root_dir +'/'+'allComm_temperalOrderTraining13_newModel_interaction_fixedTau_results.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow( ["commName","totalSampleCount", "reg_strength", 
    #                       "coef_lambda_sklearn","coef_beta_sklearn","coef_delta_sklearn", "q_sklearn_mean", "q_sklearn_std", "q_sklearn within (-1~1) percentage", "q_sklearn within (-2~2) percentage","nu_sklearn_mean", "nu_sklearn_std", "nu_sklearn within (-1~1) percentage", "nu_sklearn within (-2~2) percentage", "conformity_sklearn",
    #                       "coef_lambda_lbfgs","coef_beta_lbfgs","coef_delta_lbfgs", "q_lbfgs_mean", "q_lbfgs_std", "q_lbfgs within (-1~1) percentage", "q_lbfgs within (-2~2) percentage","nu_lbfgs_mean", "nu_lbfgs_std", "nu_lbfgs within (-1~1) percentage", "nu_lbfgs within (-2~2) percentage", "conformity_lbfgs",
    #                       "CrossValidation_Score of current reg_alpha", "best reg_alpha by Cross Validation"])

    # # # save csv
    # with open(root_dir +'/'+'allComm_temperalOrderTraining13_newModel_interaction_fixedTau_SGD_results.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow( ["commName","totalSampleCount","featureCount","testingDataSampleCount", "reg_strength", 
    #                       "coef_lambda_SGD","coef_beta_SGD", "coef_delta_SGD", "q_SGD_mean", "q_SGD_std", "q_SGD within (-1~1) percentage", "q_SGD within (-2~2) percentage","nu_SGD_mean", "nu_SGD_std", "nu_SGD within (-1~1) percentage", "nu_SGD within (-2~2) percentage", "conformity_SGD","lowest_test_loss","test_accuracy","retrain_loss","retrain_accuracy","train_loss_of_lowest_test_loss_epoch","sklearn_accuracy","lowest_test_loss_epoch"
    #                      ])
        
    """
    try:
        # test on comm "coffee.stackexchange" to debug
        myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1], root_dir)
        # test on comm "datascience.stackexchange" to debug
        # myFun(301,commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1], root_dir)
        # test on comm "webapps.stackexchange" to debug
        # myFun(305,commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1], root_dir)
        # test on comm "travel.stackexchange" to debug
        # myFun(319,commDir_sizes_sortedlist[319][0], commDir_sizes_sortedlist[319][1], root_dir)
        # test on comm "askubuntu" to debug
        # myFun(319,commDir_sizes_sortedlist[356][0], commDir_sizes_sortedlist[356][1], root_dir)
    except Exception as e:
        print(e)
        sys.exit()
    """
     # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    # for sampled comms
    sampled_comms = ['academia.stackexchange','askubuntu',
                      'english.stackexchange','math.stackexchange','mathoverflow.net',
                      'meta.stackexchange','meta.stackoverflow','serverfault',
                      'softwareengineering.stackexchange','superuser','unix.stackexchange',
                      'worldbuilding.stackexchange','physics.stackexchange','electronics.stackexchange',
                      'codegolf.stackexchange','workplace.stackexchange']
    
    # selected_comms = ['3dprinting.stackexchange','latin.stackexchange','meta.askubuntu','lifehacks.stackexchange',
    #                   'cstheory.stackexchange','stackoverflow','unix.meta.stackexchange','politics.stackexchange', 'math.meta.stackexchange']

    # selected_comms = ['mathoverflow.net','askubuntu','philosophy.stackexchange','mathematica.stackexchange']

    # selected_comms = ['academia.stackexchange',
    #                   'english.stackexchange','math.stackexchange',
    #                   'meta.stackexchange','meta.stackoverflow','serverfault',
    #                   'softwareengineering.stackexchange','superuser','unix.stackexchange',
    #                   'worldbuilding.stackexchange','physics.stackexchange','electronics.stackexchange',
    #                   'codegolf.stackexchange','workplace.stackexchange']

    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    commIndex = 0
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        # if commName not in selected_comms: 
        #     print(f"{commName} was not selected,skip")
        #     continue

        try:
            p = mp.Process(target=myFun, args=(commIndex, commName,commDir, root_dir, splitted_comms,sampled_comms))
            p.start()
            commIndex += 1
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()
            return

        processes.append(p)
        if len(processes)==1:
            # make sure all p finish before main process finish
            for p in processes:
                p.join()
                finishedCount +=1
                print(f"finished {finishedCount} comm.")
            # clear processes
            processes = []
    
    # join the last batch of processes
    if len(processes)>0:
        # make sure all p finish before main process finish
        for p in processes:
            p.join()
            finishedCount +=1
            print(f"finished {finishedCount} comm.")

    
    
    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('temperalOrderTraining13 train new model with interaction Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
