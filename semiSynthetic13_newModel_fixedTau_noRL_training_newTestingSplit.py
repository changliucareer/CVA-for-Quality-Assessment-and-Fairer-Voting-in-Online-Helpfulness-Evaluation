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
import numpy as np
import torch
from tqdm import tqdm
from statistics import mean
from sklearn.model_selection import train_test_split
import sklearn
from CustomizedNN import LRNN_1layer, LRNN_1layer_bias, LRNN_1layer_bias_specify,LRNN_1layer_bias_withoutRankTerm,LRNN_1layer_withoutRankTerm
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.preprocessing import normalize
import random
from collections import Counter

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
    def __init__(self, full_data,oneside, testingDataIndexListInFullData, trainOrtest, ori_questionCount):
        self.data = full_data
        self.oneside = oneside
        self.testingDataIndexListInFullData = testingDataIndexListInFullData
        self.trainOrtest = trainOrtest
        self.ori_questionCount = ori_questionCount
    
    def process_data(self, dataset, oneside, testingDataIndexListInFullData, trainOrtest, ori_questionCount):
        X = dataset[0].todense() # convert sparse matrix to np.array
        y = dataset[1]
        
        # tailor the columns for different parametrization.
        # the first 3 columns of X are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio, IPW_rank)
        if oneside:
            X = np.concatenate((X[:,2:4], X[:,4+ori_questionCount:]), axis=1) # for one side parametrization, removed the first two columns and the Inversed rank term column
        else: # two sides
            X = np.concatenate((X[:,:2], X[:,3:4] , X[:,4+ori_questionCount:] ), axis=1) # for two sides parametrization, removed the third and fourth column

        for i in range(len(y)):
            if trainOrtest == 'train': # True to train
                if i not in testingDataIndexListInFullData:
                    yield X[i,:], y[i]
            elif trainOrtest == 'test': # False to test
                if i in testingDataIndexListInFullData:
                    yield X[i,:], y[i]
            else: # need full data
                yield X[i,:], y[i]
    
    def __iter__(self):
        return self.process_data(self.data, self.oneside, self.testingDataIndexListInFullData, self.trainOrtest,self.ori_questionCount)
    
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

def resultFormat(weights, bias, oneside):
    coefs = [] # community-level coefficients

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
                text += f"beta: {coef}\n" # with rank term, the second feature is inversed rank, print beta
                coefs.append(coef)
            else: # for rest of j
                text += f"q_{j-3}: {coef}\n" # the quality features start from j = questionCount+2
                if bias != None:
                    text += f"q_{j-3}+bias: {coef+bias}\n"
                qs.append(coef)
        
        else: # when do oneside
            if j == 0: # the first feature is seen_pos_vote_ratio for oneside training, or pos_vote_ratio for only_pvr. print lambda
                text += f"lambda: {coef}\n"
                coefs.append(coef)
            elif j == 1:
                text += f"beta: {coef}\n" # with rank term, the second feature is inversed rank, print beta
                coefs.append(coef)
            else: # for rest of j
                text += f"q_{j-2}: {coef}\n" # the quality features start from j = questionCount+1
                if bias != None:
                    text += f"q_{j-2}+bias: {coef+bias}\n"
                qs.append(coef)
    
    return text, coefs, qs
    
###################################################################################################

def myTrain(commIndex, commName, commDir, ori_questionCount, full_data, testingDataIndexListInFullData, log_file_name, roundIndex, variation, reg_alpha1, reg_alpha2):
    t0=time.time()
    log_text = ""

    #######################################################################
    ### training settings #######
    result_file_name = f"semiSynthetic13_newModel{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_results.txt"
    trained_model_file_name = f'semiSynthetic13_newModel{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_model.sav'
    trained_withSKLEARN_model_file_name = f'semiSynthetic13_newModel{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withSKLEARN_model.pkl'
    result_withSKLEARN_file_name = f"semiSynthetic13_newModel{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withSKLEARN_results.txt"
    trained_withLBFGS_model_file_name = f'semiSynthetic13_newModel{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withLBFGS_model.sav'
    result_withLBFGS_file_name = f"semiSynthetic13_newModel{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withLBFGS_results.txt"
    plot_file_name = f'semiSynthetic13_newModel{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_Losses.png'
    # choices of training settings
    normFlag = False
    regFlag = True   # if True, apply l2 regularization on all parameters; if False, don't apply regularization
    # reg_alpha = 0.5  # the regularization strength. only useful when reg_Flag = True
    reg_alpha = reg_alpha2
    oneside = True   # if True, use one side parametrization; if False, use two side parametrization

    withBias = False # whether add bias term
    learnTau = False  # if True, learn tau;  if learnTau = False, fix tau = 1
    positiveTau = False # only useful when learnTau = True, if postiveTau = False, don't constrain tau as positive
   
    # select model type
    if learnTau:
        if withBias:
            modeltype='1layer_bias' # learn tau
        else: # without bias
            modeltype= '1layer'
    else:
        if withBias:
            modeltype='1layer_bias_withoutRankTerm' # equivalent to set tau as 1
        else: # without bias
            modeltype='1layer_withoutRankTerm'
    
    learning_rate = 0.1
    max_iter = 300   # this is the total number of epochs
    ############################################################################################

    # get total sample count
    total_sample_count = len(full_data[1])
    original_n_feature = full_data[0].todense().shape[1]

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
    # The first 4 columns of origianl dataset are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio, IPW_rank)
    if oneside:
        n_feature = original_n_feature - 2 - ori_questionCount# removed 2 columns for one side parametrization, and removed the ori_questionsCount columns of relative length terms
    else:
        n_feature = original_n_feature - 1 - ori_questionCount# removed 1 columns for two sides parametrization, and removed the ori_questionsCount columns of relative length terms
    print(f"{commName} has total sample count: {total_sample_count}, and number of features: {n_feature}")
    

    ####################################################################################################
    print("try to train using SKLEARN...")
    from sklearn.linear_model import LogisticRegression
    X = full_data[0].todense() # convert sparse matrix to np.array
    y = full_data[1]
    
    # tailor the columns for different parametrization.
    # the first 4 columns of X are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio, IPW_rank)
    if oneside:
        X = np.concatenate((X[:,2:4], X[:,4+ori_questionCount:]), axis=1) # for one side parametrization, removed the first two columns 
    else: # two sides
        X = np.concatenate((X[:,:2], X[:,3:4] , X[:,4+ori_questionCount:] ), axis=1) # for two sides parametrization, removed the third column


    # lr = LogisticRegression(random_state=1, solver='liblinear',penalty='l2', fit_intercept=withBias, C=1)
    myC = 1/(2*reg_alpha2)
    lr = LogisticRegression(solver='lbfgs',penalty='l2', fit_intercept=withBias, C=myC, max_iter=max_iter)
    try:
        lr.fit(X, y)
        # save lr model
        final_directory = os.path.join(commDir, r'trained_model_folder')
        with open(final_directory+'/'+ trained_withSKLEARN_model_file_name,'wb') as f:
            pickle.dump(lr,f)
        print(f"for {commName} model_withSKLEARN saved.")
        
        # compute conformity
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
        print(f"===> {commName} conformity is {conformity_sklearn}.") 

        # retrieve the learning result
        if withBias:
            bias_sklearn = lr.intercept_[0]
        else:
            bias_sklearn = None

        weights_sklearn = lr.coef_[0]
        
        result_text_sklearn, coefs_sklearn, qs_sklearn = resultFormat(weights_sklearn, bias_sklearn, oneside)

        text = f"SKLEARN training ==================================\n"
        text += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '\n'
        text += f"Train oneside:{oneside}, normalize:{normFlag},priorRegAlpha:{reg_alpha1}, regFlag:{regFlag}, reg_alpha:{reg_alpha}, withBias:{withBias}\n"
        text += f"dataset size: ({total_sample_count}, {n_feature})\n conformity_sklearn: {conformity_sklearn}\n"
        print(text)
        writeIntoResult(text + result_text_sklearn, result_withSKLEARN_file_name)
    except:
        log_text += f"fail to fit with sklearn LR. \n"
        coefs_sklearn = None
        bias_sklearn = None
        nus_sklearn = None
        qs_sklearn = None

    ####################################################################################################
    print("try to train using cuda and LBFGS...")
    # check gpu count
    cuda_count = torch.cuda.device_count()
    # assign one of gpu as device
    d = (commIndex+1) % cuda_count
    device = torch.device('cuda:'+str(d) if torch.cuda.is_available() else 'cpu')
    print(f"comm {commName}, Start to train NN model with LBFGS... on device: {device}")

    # convert X and y to tensors
    X = np.array(X)
    y = np.array(y, dtype=float)
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.float32))
    
    X = X.to(device)
    y = y.to(device)

    # Create a NN which equavalent to logistic regression
    input_dim = n_feature

    print(f"comm {commName}, preparing model...")
    if modeltype == '1layer_withoutRankTerm':
        try:
            model = LRNN_1layer_withoutRankTerm(input_dim)
        except Exception as e:
            print("can't initialize model! "+e)
    else:
        sys.exit(f"invalid modeltype: {modeltype}")

    model.to(device)
    
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

    
    opt_forWholeBatchTrain = 'lbfgs'
    optimizer_forWholeBatchTrain = getOptimizer(opt_forWholeBatchTrain)

    # initialize working optimizer as the whole batch training 
    optimizer = optimizer_forWholeBatchTrain

    torch.autograd.set_detect_anomaly(True) # only for debug, or this will slow the training

    try:
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
            l2_lambda = torch.tensor(reg_alpha).to(device)
            l2_reg = torch.tensor(0.).to(device) # initialize regular term as zero
            
            # Regularize all parameters:
            for name, param in model.named_parameters():
                l2_reg += torch.square(torch.norm(param))

            loss = loss + (l2_lambda/sample_n)* l2_reg # match with reduction = 'mean' in loss_fn

            trainingLosses.append(loss.item()) 
            
            if loss.requires_grad:
                loss.backward()
            return loss
        # -------------------------------------------
        optimizer.step(loss_closure) # Updates parameters with the optimizer recursive times till converge for each epoch
        
        # save model
        saveModel(model,trained_withLBFGS_model_file_name)
        print(f"for {commName} model trained_withlbfgs saved.")

        # predict the y and compute probOdds for conformity #####################################################
        model.eval()   

        y_pred = model(X)
        if len(y)==1: # only one sample
            y_pred = y_pred.cpu().detach().numpy()[0]
        else:
            y_pred = torch.squeeze(y_pred.cpu()).detach().numpy()
        # multiply each odd with probOddProduct
        for i,row in enumerate(X):
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
        text += f"Train oneside:{oneside}, normalize:{normFlag},PriorRegAlpha:{reg_alpha1}, regFlag:{regFlag}, reg_alpha:{reg_alpha},\n"
        text += f"      opt_forWholeBatchTrain :{opt_forWholeBatchTrain}, withBias:{withBias}\n"
        text += f"      max_iter = {max_iter},  learning_rate:{learning_rate}\n"
        text += f"dataset size: ({total_sample_count}, {n_feature})\n\n"
        text += f"min training loss: {trainingLosses[-1]} converge epoch: {len(trainingLosses)}\n"
        text += f"Conformity: {conformity_lbfgs}\n"
        print(text)
        
        # output learned parameters of model with lowest test loss
        parm = defaultdict()
        for name, param in model.named_parameters():
            parm[name]=param.cpu().detach().numpy() 
        
        if withBias:
            bias_lbfgs = parm['linear.bias'][0]
        else:
            bias_lbfgs = None

        weights_lbfgs = parm['linear.weight'][0]
        
        result_text, coefs_lbfgs, qs_lbfgs = resultFormat(weights_lbfgs, bias_lbfgs, oneside)
        writeIntoResult(text +  result_text, result_withLBFGS_file_name)
        
        elapsed = format_time(time.time() - t0)
        
        log_text += "Elapsed: {:}\n".format(elapsed)

        writeIntoLog(text + log_text, commDir , log_file_name)

    except Exception as ee:
        print(f"tried lbfgs, failed: {ee}")
        writeIntoLog(f"failed to train with cuda and LBFGS", commDir, log_file_name)
        coefs_lbfgs= None
        bias_lbfgs= None
        nus_lbfgs= None
        qs_lbfgs= None
        conformity_lbfgs= None
    
    ####################################################################################################
    """
    print("try to train using cuda and SGD...")
    opt_forMiniBatchTrain = 'sgd'
    # check gpu count
    cuda_count = torch.cuda.device_count()
    # assign one of gpu as device
    d = (commIndex +1) % cuda_count
    device = torch.device('cuda:'+str(d) if torch.cuda.is_available() else 'cpu')
    print(f"comm {commName}, Start to train NN model... on device: {device}")

    # Create a NN which equavalent to logistic regression
    input_dim = n_feature
    initial_tau = 1
    if oneside:
        tauColumnIndex = 1
    else:
        tauColumnIndex = 2

    print(f"comm {commName}, preparing model...")
    if modeltype =='1layer_bias':
        model = LRNN_1layer_bias(input_dim,initial_tau,tauColumnIndex)
    elif modeltype == '1layer_withoutRankTerm':
        model = LRNN_1layer_withoutRankTerm(input_dim)
    elif modeltype == '1layer':
        model = LRNN_1layer(input_dim,initial_tau,tauColumnIndex, positiveTau)
    else:
        sys.exit(f"invalid modeltype: {modeltype}")

    model.to(device)
    
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

    torch.autograd.set_detect_anomaly(True) # only for debug, or this will slow the training

    train_losses = []
    test_losses = []
    lowest_test_loss_of_epoch = None
    lowest_test_loss = 1000000
    model_with_lowest_test_loss = None
    test_accuracy = None


    try:
        optimizer = optimizer_forMiniBatchTrain
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[290], gamma=0.1)

        # Build a Streaming DataLoader with customized iterable dataset
        iterable_training_dataset = MyIterableDataset(full_data,oneside, testingDataIndexListInFullData, 'train', ori_questionCount) 
        iterable_testing_dataset = MyIterableDataset(full_data,oneside, testingDataIndexListInFullData, 'test', ori_questionCount) 
        iterable_full_dataset = MyIterableDataset(full_data,oneside, testingDataIndexListInFullData, 'full', ori_questionCount) 
        # prepare data loader
        batch_size = 2000000
        my_training_dataloader = DataLoader(iterable_training_dataset, batch_size=batch_size) 
        my_testing_dataloader = DataLoader(iterable_testing_dataset, batch_size=batch_size) 
        my_full_dataloader = DataLoader(iterable_full_dataset, batch_size=batch_size) 

        probOddList = [] # for conformity computation

        for epoch in tqdm(range(int(max_iter)),desc='Training Epochs'):
            # training #####################################################
            model.train() 

            batches_train_losses = []

            batchIndex = 0
            for batch in my_training_dataloader:
                optimizer.zero_grad()
                batchIndex +=1
 
                # prepare the data tensors
                # print(f"comm {commName}, preparing batch {batchIndex}...")
                X_batch_train = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
                y_batch_train = batch[1]
                X_batch_train,y_batch_train = preprocessing(X_batch_train,y_batch_train,normFlag)
                
                
                X_batch_train = X_batch_train.to(device)
                y_batch_train = y_batch_train.to(device)      

                outputs = model(X_batch_train)
                sample_n = X_batch_train.size()[0]
                
                if outputs.shape[0] ==1:
                    loss = loss_fn(outputs[0], y_batch_train)
                else:
                    loss = loss_fn(torch.squeeze(outputs), y_batch_train) # [m,1] -squeeze-> [m] 
                
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
                X_batch_test = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
                y_batch_test = batch[1]
                X_batch_test,y_batch_test = preprocessing(X_batch_test,y_batch_test,normFlag)

                if len(X_batch_test.shape) == 1 : # 1D shape when only 1 sample, need to reshape to (1,n_feature)
                    X_batch_test = X_batch_test.reshape((1,-1))
                
                
                X_batch_test = X_batch_test.to(device)
                y_batch_test = y_batch_test.to(device)      

                outputs = model(X_batch_test)
                sample_n = X_batch_test.size()[0]
                
                if outputs.shape[0] ==1:
                    loss = loss_fn(outputs[0], y_batch_test)
                else:
                    loss = loss_fn(torch.squeeze(outputs), y_batch_test) # [m,1] -squeeze-> [m] 
                
                if regFlag:
                    # add l2 regularization
                    l2_lambda = torch.tensor(reg_alpha).to(device)
                    l2_reg = torch.tensor(0.).to(device)
                    
                    # Regularize all parameters:
                    for param in model.parameters():
                        l2_reg += torch.square(torch.norm(param))

                    loss = loss + (l2_lambda/sample_n)* l2_reg  
        
                # Calculating the loss and accuracy for the test dataset
                correct = np.sum(torch.squeeze(outputs.cpu()).round().detach().numpy() == y_batch_test.cpu().detach().numpy())
                batches_test_correct.append(correct)
                batches_test_sampleCount.append(sample_n)
                batches_test_losses.append(loss.item())

                # clear the gpu
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
        
        # save model
        saveModel(model_with_lowest_test_loss,trained_model_file_name)
        print(f"for {commName} model_with_lowest_test_loss saved.")

        # predict the y and compute probOdds for conformity #####################################################
        model_with_lowest_test_loss.eval()   
        batchIndex = 0
        for batch in my_full_dataloader:
            batchIndex +=1

            # prepare the data tensors
            # print(f"comm {commName}, preparing batch {batchIndex}...")
            X_batch_train = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
            y_batch_train = batch[1]
            X_batch_train,y_batch_train = preprocessing(X_batch_train,y_batch_train,normFlag)
            
            X_batch_train = X_batch_train.to(device)
            y_batch_train = y_batch_train.to(device)      

            y_pred = model_with_lowest_test_loss(X_batch_train)
            if len(y_batch_train)==1: # only one sample
                y_pred = y_pred.cpu().detach().numpy()[0]
            else:
                y_pred = torch.squeeze(y_pred.cpu()).detach().numpy()
            # multiply each odd with probOddProduct
            for i,row in enumerate(X_batch_train):
                if row[0] >= 0: # when n_pos >= n_neg
                    odd = y_pred[i]/(1-y_pred[i])
                else: # when n_pos < n_neg
                    odd = (1-y_pred[i])/y_pred[i]
                probOddList.append(odd)

        # compute conformity
        conformity = np.exp( np.sum(np.log(probOddList)) / total_sample_count )
        print(f"===> {commName} conformity is {conformity}.") 

        print(f"start to save results... for {commName}")

        # save results as log for model
        # log the training settings
        text = f"=========================================\n"
        text += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '\n'
        text += f"Train oneside:{oneside}, normalize:{normFlag},PriorRegAlpha:{reg_alpha1}, regFlag:{regFlag}, reg_alpha:{reg_alpha},\n"
        text += f"      learnTau:{learnTau}, opt_forMiniBatchTrain :{opt_forMiniBatchTrain}, withBias:{withBias}, positiveTau:{positiveTau}\n"
        text += f"      max_iter = {max_iter},  learning_rate:{learning_rate}\n"
        text += f"dataset size: ({total_sample_count}, {n_feature})\n\n"
        text += f"trained with batch_size:{batch_size}\navg training loss: {mean(train_losses)} \navg testing loss:{mean(test_losses)},lowest_test_loss_of_epoch:{lowest_test_loss_of_epoch} lowest_test_loss:{lowest_test_loss}\ntest accuracy:{test_accuracy}\n"
        text += f"Conformity: {conformity}\n"
        print(text)
        
        # output learned parameters of model with lowest test loss
        parm = defaultdict()
        for name, param in model_with_lowest_test_loss.named_parameters():
            parm[name]=param.cpu().detach().numpy() 
        
        if withBias:
            bias = parm['linear.bias'][0]
        else:
            bias = None
        
        if learnTau: 
            if positiveTau:
                tau = np.exp(parm['tau'].item())
            else:
                tau = parm['tau'].item()
        else:
            tau = 1

        weights = parm['linear.weight'][0]
        
        result_text, coefs, qs = resultFormat(weights, bias, oneside)
        writeIntoResult(text + f"\ntraining losses: {train_losses}\n"+ f"\ntesting losses: {test_losses}\n"+ f"\ntesting accuracy: {test_accuracy}\n" + result_text, result_file_name)
        
        elapsed = format_time(time.time() - t0)
        
        log_text = "Elapsed: {:}\n".format(elapsed)

        writeIntoLog(text + log_text, commDir , log_file_name)

        # visualize the losses
        plt.cla()
        plt.plot(range(len(train_losses)), train_losses, 'g-', label=f'trainning')
        plt.plot(range(len(test_losses)), test_losses, 'b-', label=f'testing\nlowest loss:{lowest_test_loss}\nreached at epoch {lowest_test_loss_of_epoch}')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        savePlot(plt, plot_file_name)

        return total_sample_count, n_feature, train_losses, test_losses, lowest_test_loss_of_epoch, lowest_test_loss, test_accuracy, coefs, bias, qs, conformity, coefs_sklearn, bias_sklearn, qs_sklearn, conformity_sklearn, coefs_lbfgs, bias_lbfgs, qs_lbfgs, conformity_lbfgs


    except Exception as ee:
        print(f"tried sgd, failed: {ee}")
        return None
    """
    print(f"not trying sgd, ")
    train_losses = None
    test_losses= None
    lowest_test_loss_of_epoch= None
    lowest_test_loss= None
    test_accuracy= None
    coefs= None
    bias= None
    nus= None
    qs= None
    conformity= None
    
    return total_sample_count, n_feature, train_losses, test_losses, lowest_test_loss_of_epoch, lowest_test_loss, test_accuracy, coefs, bias, qs, conformity, coefs_sklearn, bias_sklearn, qs_sklearn, conformity_sklearn, coefs_lbfgs, bias_lbfgs, qs_lbfgs, conformity_lbfgs


def myFun(commIndex, commName, commDir,root_dir, roundIndex, variation, selected_reg_strengthList, try_reg_strengthList):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    log_file_name = "semiSynthetic13_newModelTraining_newTestingSplit_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # # check whether already done this step, skip
    # resultFiles = [f'semiSynthetic13_newModel{variation}_round{roundIndex}_training_return.dict']
    # resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
    # if os.path.exists(resultFiles[0]):
    #     # target date
    #     target_date = datetime.datetime(2023, 8, 27)
    #     # file last modification time
    #     timestamp = os.path.getmtime(resultFiles[0])
    #     # convert timestamp into DateTime object
    #     datestamp = datetime.datetime.fromtimestamp(timestamp)
    #     print(f'{commName} Modified Date/Time:{datestamp}')
    #     if datestamp >= target_date:
    #         print(f"{commName} has already done this step.")
    #         return

    # train using the data and priors from selected reg_alpha options
    for reg_alpha1 in selected_reg_strengthList:

        with open(intermediate_directory+'/'+f'simulated_data_byCVP{variation}_round{roundIndex}_regAlpha({reg_alpha1}).dict', 'rb') as inputFile:
            loadedFile = pickle.load( inputFile)
        simulatedQuestions = loadedFile[0]
        ori_questionCount = len(simulatedQuestions)
        simulatedQuestions.clear() # clear this to same memory
        
        # load full data
        try:
            with open(intermediate_directory+f"/semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_datasets_sorted_removeFirstRealVote.dict", 'rb') as inputFile:
                full_data = pickle.load( inputFile)
        except:
            print(f"{commName} hasn't done temperalOrderSorting yet, skip")
            return

        # load testing data's sortingBase
        # get testing data raw index in full data
        with open(intermediate_directory+f"/semiSynthetic8{variation}_round{roundIndex}_regAlpha({reg_alpha1})_outputs.dict", 'rb') as inputFile:
            qid2TestingSortingBaseList, _ = pickle.load( inputFile)
        sortingBaseListOfTestingData = []
        for qid, testingSortingBaseList in qid2TestingSortingBaseList.items():
            sortingBaseListOfTestingData.extend(testingSortingBaseList)

        with open(intermediate_directory+f"/semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha1})_sorted_qidAndSampleIndexAndSortingBase.dict", 'rb') as inputFile:
            sorted_qidAndSampleIndex = pickle.load( inputFile)

        testingDataIndexListInFullData = [i for i, tup in enumerate(sorted_qidAndSampleIndex) if tup[2] in sortingBaseListOfTestingData]
        
        # training using different reg_alpha2 options
        for reg_alpha2 in try_reg_strengthList:
            # training 
            output = myTrain(commIndex, commName, commDir, ori_questionCount, full_data, testingDataIndexListInFullData, log_file_name, roundIndex, variation, reg_alpha1, reg_alpha2)
            total_sample_count, n_feature, train_losses, test_losses, lowest_test_loss_of_epoch, lowest_test_loss, test_accuracy, coefs, bias, qs, conformity, coefs_sklearn, bias_sklearn, qs_sklearn, conformity_sklearn, coefs_lbfgs, bias_lbfgs, qs_lbfgs, conformity_lbfgs = output
            
            return_trainSuccess_dict = defaultdict()
            return_trainSuccess_dict[commName] = {'dataShape':(total_sample_count, n_feature), 'trainLosses':train_losses, 'testLosses':test_losses, 'epoch':lowest_test_loss_of_epoch, 'lowest_test_loss':lowest_test_loss, 'testAcc':test_accuracy,
                                                'coefs':coefs,'bias':bias, 'qs':qs, 'conformity':conformity,
                                                    'coefs_sklearn':coefs_sklearn, 'bias_sklearn':bias_sklearn, 'qs_sklearn':qs_sklearn,
                                                    'conformity_sklearn':conformity_sklearn,
                                                    'coefs_lbfgs':coefs_lbfgs, 'bias_lbfgs':bias_lbfgs, 'qs_lbfgs':qs_lbfgs,
                                                    'conformity_lbfgs':conformity_lbfgs}
            
            # save return dict
            with open(intermediate_directory+f"/semiSynthetic13_newModel{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict", 'wb') as outputFile:
                pickle.dump(return_trainSuccess_dict, outputFile)
                print( f"saved return_trainSuccess_dict_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2}) of {commName}.")

            # # save csv
            with open(root_dir +'/'+f'allComm_semiSynthetic13_newModel{variation}_round{roundIndex}_training_results.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                if coefs != None:
                    writer.writerow( [commName,total_sample_count, reg_alpha1, reg_alpha2, coefs_sklearn[0], coefs_lbfgs[0],coefs[0], mean(qs_sklearn), mean(qs_lbfgs),mean(qs)])
                else:
                    writer.writerow( [commName,total_sample_count, reg_alpha1, reg_alpha2, coefs_sklearn[0], coefs_lbfgs[0],coefs, mean(qs_sklearn), mean(qs_lbfgs),qs])


def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # roundIndex = 1
    # roundIndex = 2  # one question one answer
    # roundIndex = 3  # one question one answer, 100 times vote count per answer
    # roundIndex = 4  # one question one answer, 1000 times vote count per answer, q_std = 1
    variation = '_fixedTau_noRL'

    # roundIndex = 9 # one question one answer, 1000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    # roundIndex = 10 # one question one answer, 5000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    # roundIndex = 11 # one question one answer, 10000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    #roundIndex = 12 # one question one answer, 50000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    
    # roundIndex = 13 # one question one answer, 1000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    # roundIndex = 14 # one question one answer, 5000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    # roundIndex = 15 # one question one answer, 10000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    
    # roundIndex = 16 # one question multiple answer, 10000 events, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]

    roundIndex = 17 # multiple question multiple answer, amplified 1000 times of original total event count, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]

    selected_reg_strengthList = [300, 500, 700]

    try_reg_strengthList = [0.001, 0.003, 0.005,0.007, 0.01, 0.03, 0.05, 0.07, 0.1,0.3, 0.5,0.7, 1, 3, 5,7,10,30, 50,70,100, 300, 500, 700, 1000, 3000, 5000]


    # # save csv
    with open(root_dir +'/'+f'allComm_semiSynthetic13_newModel{variation}_round{roundIndex}_training_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","totalSampleCount", "reg_alpha1","reg_alpha2", "coefs_sklearnLR", "coefs_torchLBFGS","coefs_torchSGD", "q_sklearnLR_mean", "q_torchLBFGS_mean","q_torchSGD_mean"])


    
    try:
        # test on comm "3dprinting.stackexchange" to debug
        myFun(0,commDir_sizes_sortedlist[227][0], commDir_sizes_sortedlist[227][1], root_dir, roundIndex, variation, selected_reg_strengthList, try_reg_strengthList)
        # test on comm "latin.stackexchange" to debug
        myFun(1,commDir_sizes_sortedlist[229][0], commDir_sizes_sortedlist[229][1], root_dir, roundIndex, variation, selected_reg_strengthList, try_reg_strengthList)
        # test on comm "lifehacks.stackexchange" to debug
        # myFun(2,commDir_sizes_sortedlist[233][0], commDir_sizes_sortedlist[233][1], root_dir, roundIndex, variation, selected_reg_strengthList, try_reg_strengthList)
        # test on comm "askubuntu.stackexchange" to debug
        # myFun(3,commDir_sizes_sortedlist[231][0], commDir_sizes_sortedlist[231][1], root_dir, roundIndex, variation, selected_reg_strengthList, try_reg_strengthList)
    except Exception as e:
        print(e)
        sys.exit()
    """
     # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    selected_comms = ['3dprinting.stackexchange','latin.stackexchange','meta.askubuntu','lifehacks.stackexchange']
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    commIndex = 0
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        if commName not in selected_comms: # skip non selected communities
            print(f"{commName} was not selected,skip")
            continue

        # if commName in splitted_comms: # skip splitted big communities
        #     print(f"{commName} was splitted,skip")
        #     continue

        try:
            p = mp.Process(target=myFun, args=(commIndex, commName,commDir,root_dir, roundIndex, variation, selected_reg_strengthList, try_reg_strengthList))
            p.start()
            commIndex += 1
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()
            return

        processes.append(p)
        if len(processes)==4:
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
    
    """
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('semiSynthetic13_newModelTraining_newTestingSplit Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
