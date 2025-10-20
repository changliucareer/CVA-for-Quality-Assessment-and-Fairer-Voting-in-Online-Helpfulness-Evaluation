import os
import sys
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################
from toolFunctions import format_time, writeIntoLog, writeIntoResult,saveModel,savePlot
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
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#### Tool Functions #######################################################
def resultFormat(weights, bias, ori_questionCount):
    coefs = [] # community-level coefficients
    nus = [] # question-level 
    qs = [] # answer-level qualities
    text = f"bias:{bias}\n"

    for j, coef in enumerate(weights):
        # when do onesides
        if j == 0: # the first feature is seen_pos_vote_ratio. print lambda
            text += f"lambda: {coef}\n"
            coefs.append(coef)

        elif j < ori_questionCount+1:
            text += f"nu_{j-1}: {coef}\n" # the 4th feature to the (questionCount+3)th feature are ralative length for each question, print nus
            nus.append(coef)
        else: # for rest of j
            text += f"q_{j-1-ori_questionCount}: {coef}\n" # the quality features start from j = questionCount+3
            if bias != None:
                text += f"q_{j-1-ori_questionCount}+bias: {coef+bias}\n"
            qs.append(coef)
    
    return text, coefs, nus, qs


#################################################################

def preprocessing (X,y,normalize=False,oneside=True):
    X = np.array(X)
    y = [int(yy) for yy in y]
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

#################################################################
class MyIterableDataset(IterableDataset):
    def __init__(self, questions_Data, oneside):
        self.data = questions_Data
        self.oneside = oneside
    
    def process_data(self, questions_Data, oneside):
        for qid, tup in questions_Data.items():
            X = tup[0].todense() # convert sparse matrix to np.array
            y = tup[1]

            if len(y) <= 10: # less than 10 samples for this question, skip
                continue

            cutIndex = int(len(y)*0.7) # for each question, its first 70% samples as training samples, the rest as testing samples
            
            # tailor the columns for different parametrization.
            # the first 3 columns of X are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio)
            if oneside:
                cur_X_train = np.concatenate((X[:cutIndex,2:3], X[:cutIndex,4:]), axis=1) # for one side parametrization, removed the first two columns
                cur_y_train = y[:cutIndex]
                cur_X_test = np.concatenate((X[cutIndex:,2:3], X[cutIndex:,4:]), axis=1) # for one side parametrization, removed the first two columns
                cur_y_test = y[cutIndex:]
            else: # two sides
                cur_X_train = np.concatenate((X[:cutIndex,:2] , X[:cutIndex,4:] ), axis=1) # for two sides parametrization, removed the third and fourth column
                cur_y_train = y[:cutIndex]
                cur_X_test = np.concatenate((X[cutIndex:,:2] , X[cutIndex:,4:] ), axis=1) # for two sides parametrization, removed the third and fourth column
                cur_y_test = y[cutIndex:]

            yield cur_X_train,cur_y_train,cur_X_test,cur_y_test
    
    def __iter__(self):
        return self.process_data(self.data, self.oneside)

def myTrain(commIndex, questions_Data,n_feature, opt_forMiniBatchTrain, learning_rate, max_iter,commName,normFlag, regFlag, reg_alpha, modeltype):

    # check gpu count
    cuda_count = torch.cuda.device_count()
    # assign one of gpu as device
    d = commIndex % cuda_count
    device = torch.device('cuda:'+str(d) if torch.cuda.is_available() else 'cpu')
    print(f"comm {commName}, Start to train NN model... on device: {device}")
    # if 'askubuntu' in commName:
    #     print("debug")

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
    loss_fn = torch.nn.BCELoss(weight=None, size_average=None, reduction='mean')

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

    torch.autograd.set_detect_anomaly(True)

    train_losses = []
    test_losses = []
    lowest_test_loss_of_epoch = None
    lowest_test_loss = 1000000
    model_with_lowest_test_loss = None

    try:
        optimizer = optimizer_forMiniBatchTrain
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,280], gamma=0.1)

        # Build a Streaming DataLoader with customized iterable dataset
        oneside = True
        iterable_dataset = MyIterableDataset(questions_Data,oneside)
        # prepare data loader
        batch_size = 1 # get one question's all samples each time. Can't change to other number!
        mydataloader = DataLoader(iterable_dataset, batch_size=batch_size) 

        for epoch in tqdm(range(int(max_iter)),desc='Training Epochs'):
            model.train() 

            batches_train_losses = []
            batches_test_losses = []
            batches_train_accuracies = []
            batches_test_accuracies = [] 

            batchIndex = 0
            for batch in mydataloader:
                optimizer.zero_grad()
                batchIndex +=1
 
                # prepare the data tensors
                # print(f"comm {commName}, preparing batch {batchIndex}...")
                X_batch_train = batch[0][0] # reshape from [ 1, n_sample, n_feature] to [n_sample, n_feature]
                y_batch_train = batch[1]
                X_batch_train,y_batch_train = preprocessing(X_batch_train,y_batch_train,normFlag)
                X_batch_test = batch[2][0]
                y_batch_test = batch[3]
                X_batch_test,y_batch_test = preprocessing(X_batch_test,y_batch_test,normFlag)
                
                X_batch_train = X_batch_train.to(device)
                y_batch_train = y_batch_train.to(device)      
                X_batch_test = X_batch_test.to(device)
                y_batch_test = y_batch_test.to(device)   

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
                    # Calculating the loss and accuracy for the train dataset
                    total = y_batch_train.size(0)
                    outputs = model(X_batch_train)
                    correct = np.sum(torch.squeeze(outputs.cpu()).round().detach().numpy() == y_batch_train.cpu().detach().numpy())
                    accuracy = correct / total
                    batches_train_accuracies.append(accuracy)
                    batches_train_losses.append(loss.item())

                    # Calculating the loss and accuracy for the test dataset
                    total = y_batch_test.size(0)
                    outputs = model(X_batch_test)
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
                        loss = loss + (l2_lambda/total)* l2_reg
                    correct = np.sum(torch.squeeze(outputs.cpu()).round().detach().numpy() == y_batch_test.cpu().detach().numpy())
                    accuracy = correct / total
                    batches_test_accuracies.append(accuracy)
                    batches_test_losses.append(loss.item())
                
                # clear the gpu
                torch.cuda.empty_cache()
            
            # print out current training stat every 10 epochs
            if epoch%10 == 0:
                print(f"comm {commName}, Train Iteration {epoch+1} -  avg train batch Loss: {mean(batches_train_losses)}, avg train batch Accuracy: {mean(batches_train_accuracies)}\n-  avg test batch Loss: {mean(batches_test_losses)}, avg test batch Accuracy: {mean(batches_test_accuracies)}")
            
            train_losses.append(sum(batches_train_losses)) # sum up the losses of all batches
            test_losses.append(sum(batches_test_losses)) # sum up the losses of all batches

            if test_losses[-1] < lowest_test_loss:
                lowest_test_loss= test_losses[-1]
                lowest_test_loss_of_epoch = epoch+1
                model_with_lowest_test_loss = copy.deepcopy(model)

            scheduler.step()
        
        
        ### Retraining with full data ##################################
        # model = copy.deepcopy(model_with_lowest_test_loss)
        model = LRNN_1layer_withoutRankTerm(input_dim) # restart from random state
        model.to(device)
        # initialize working optimizer as the one for mini batch training 
        optimizer = getOptimizer(opt_forMiniBatchTrain)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,280], gamma=0.1)

        # for test accuracy with all testing data
        total_correct = 0
        total_sample = 0 
        # prepare test dataset again
        retrain_iterable_dataset = MyIterableDataset(questions_Data, oneside)
        batch_size = 1 # get one question's all samples each time. Can't change to other number!
        mydataloader_retrain = DataLoader(retrain_iterable_dataset, batch_size=batch_size)

        for epoch in tqdm(range(lowest_test_loss_of_epoch),desc='ReTraining Epochs'):
            model.train() 
            batchIndex = 0
            for batch_retrain in mydataloader_retrain:
                optimizer.zero_grad()
                batchIndex +=1
                # prepare the data tensors 
                X_batch_train = batch_retrain[0][0]
                y_batch_train = batch_retrain[1]
                X_batch_test = batch_retrain[2][0]
                y_batch_test = batch_retrain[3]
                X_batch_full = np.concatenate((X_batch_train , X_batch_test ), axis=0)
                y_batch_full = y_batch_train + y_batch_test 
                X_batch_full,y_batch_full = preprocessing(X_batch_full,y_batch_full,normFlag)    
                X_batch_full = X_batch_full.to(device)
                y_batch_full = y_batch_full.to(device)    

                outputs = model(X_batch_full)
                sample_n = X_batch_full.size()[0]
                
                if outputs.shape[0] ==1:
                    loss = loss_fn(outputs[0], y_batch_full)
                else:
                    loss = loss_fn(torch.squeeze(outputs), y_batch_full) # [m,1] -squeeze-> [m] 
                
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

        
        # calculate test accuracy with all testing data
        with torch.no_grad():
            for batch_test in mydataloader_retrain:
                # prepare the data tensors 
                X_batch_test = batch_test[2][0]
                y_batch_test = batch_test[3]
                X_batch_test,y_batch_test = preprocessing(X_batch_test,y_batch_test,normFlag)    
                X_batch_test = X_batch_test.to(device)
                y_batch_test = y_batch_test.to(device)   
        
                outputs = model(X_batch_test) # use the model trained with all data
                sample_n = X_batch_test.size()[0]
                total_sample += sample_n
                correct = np.sum(torch.squeeze(outputs.cpu()).round().detach().numpy() == y_batch_test.cpu().detach().numpy())
                total_correct += correct
            test_accuracy = total_correct / total_sample     
        
        # Return avg training accuracy and model.
        return train_losses, test_losses, test_accuracy, model_with_lowest_test_loss,lowest_test_loss_of_epoch,lowest_test_loss, batch_size, model

    except Exception as ee:
        print(f"tried sgd, failed: {ee}")
        return None

class MyIterableDataset_forConformity(IterableDataset):
    def __init__(self, questions_Data):
        self.data = questions_Data
    
    def process_data(self, questions_Data):
        # extract data batch for each time step
        # get X_train_batches and y_train_batches

        max_votesCountOfComm = list(questions_Data.values())[0][2]
        timeStep = 0
        
        while timeStep < max_votesCountOfComm:
            X_train_curStep = []
            y_train_curStep = []

            for qid, tup in questions_Data.items():
                if isinstance(tup[0],np.ndarray):
                    X = tup[0]
                else:
                    X = tup[0].toarray() # convert sparse matrix to np.array
                y = tup[1]

                if len(y) <= 10: # less than 10 samples for this question, skip
                    continue

                n_feature = len(X[0])

                if isinstance(X_train_curStep,np.ndarray):
                    try:
                        if timeStep < X.shape[0]:
                            X_train_curStep = np.vstack((X_train_curStep,X[timeStep,:]))
                            y_train_curStep.extend([y[timeStep]])
                    except Exception as e:
                        print(e)
                else:
                    if timeStep < X.shape[0]:
                        X_train_curStep = X[timeStep,:].reshape((1,n_feature))
                        y_train_curStep.extend([y[timeStep]])
        
            timeStep += 1
            yield X_train_curStep, y_train_curStep
    
    def __iter__(self):
        return self.process_data(self.data)




def myTrain_forConformity(commIndex, questions_Data,n_feature, opt_forMiniBatchTrain, learning_rate, max_iter,commName,normFlag, regFlag, reg_alpha, modeltype):

    # check gpu count
    cuda_count = torch.cuda.device_count()
    # assign one of gpu as device
    d = commIndex % cuda_count
    device = torch.device('cuda:'+str(d) if torch.cuda.is_available() else 'cpu')
    print(f"comm {commName}, Start to train NN model... on device: {device}")
    # if 'askubuntu' in commName:
    #     print("debug")

    # Create a NN which equavalent to logistic regression
    input_dim = n_feature

    print(f"comm {commName}, preparing model...")
    if modeltype == '1layer_bias_withoutRankTerm':
        try:
            model = LRNN_1layer_bias_withoutRankTerm(input_dim)
        except Exception as e:
            print("can't initialize model! "+e)
    else:
        sys.exit(f"invalid modeltype: {modeltype}")

    model.to(device)
    
    # using Binary Cross Entropy Loss
    loss_fn = torch.nn.BCELoss(weight=None, size_average=None, reduction='mean')

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

    torch.autograd.set_detect_anomaly(True)

    # prob odds list for conformation computation
    probOddList = []
    totalRealSamplesCount = 0

    try:
        optimizer = optimizer_forMiniBatchTrain
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,280], gamma=0.1)

        # Build a Streaming DataLoader with customized iterable dataset
        iterable_dataset = MyIterableDataset_forConformity(questions_Data)
        # prepare data loader
        batch_size = 1 # get one batch for just one step, can't change to other number
        mydataloader = DataLoader(iterable_dataset, batch_size=batch_size) 

        timeSteps_losses = []
        timeSteps_accuracies = []

        timeStep = 0
        model_with_lowest_loss_of_previous_step = None
        
        for batch in mydataloader:
            timeStep +=1

            # prepare the data tensors
            # print(f"comm {commName}, preparing batch {batchIndex}...")
            X = batch[0][0] # reshape from [ 1, n_sample, n_feature] to [n_sample, n_feature]
            y = batch[1]
            totalRealSamplesCount += len(y)

            # tailor the columns for different parametrization.
            # the first 4 columns of X are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio, IPWrank)
            X_timeStep = np.concatenate((X[:,2:3] , X[:,4:] ), axis=1) # for one side parametrization, removed the first, second and fourth columns
            y_timeStep = y
            
            X_timeStep,y_timeStep = preprocessing(X_timeStep,y_timeStep,normFlag)
            
            X_timeStep = X_timeStep.to(device)
            y_timeStep = y_timeStep.to(device)   

            timeStep_losses = []
            timeStep_accuracies = []
            lowest_loss = 1000000
            lowest_loss_of_epoch = 0
            accuracy_with_lowest_loss =0
            model_with_lowest_loss = None   

            for epoch in tqdm(range(int(max_iter)),desc='Training Epochs'):

                if epoch ==0 and model_with_lowest_loss_of_previous_step != None: # at the first epoch of each step, initialize the model as the model with the lowest loss of last step
                    model = model_with_lowest_loss_of_previous_step

                model.train() 
                optimizer.zero_grad()
                outputs = model(X_timeStep)
                sample_n = X_timeStep.size()[0]
                
                if outputs.shape[0] ==1:
                    loss = loss_fn(outputs[0], y_timeStep)
                else:
                    loss = loss_fn(torch.squeeze(outputs), y_timeStep) # [m,1] -squeeze-> [m] 
                
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
                    print(f"fail to train time step {timeStep} epoch:{epoch+1} of {commName}, {eee}")
                    return
                
                optimizer.step()
                
                with torch.no_grad():
                    # Calculating the loss and accuracy for the train dataset
                    total = y_timeStep.size(0)
                    outputs = model(X_timeStep)
                    correct = np.sum(torch.squeeze(outputs.cpu()).round().detach().numpy() == y_timeStep.cpu().detach().numpy())
                    accuracy = correct / total
                    timeStep_accuracies.append(accuracy)
                    timeStep_losses.append(loss.item())

                    if timeStep_losses[-1] < lowest_loss:
                        lowest_loss= timeStep_losses[-1]
                        lowest_loss_of_epoch = epoch+1
                        accuracy_with_lowest_loss = accuracy
                        model_with_lowest_loss = copy.deepcopy(model)
                
                # clear the gpu
                torch.cuda.empty_cache()

                scheduler.step()
            
                # print out current training stat every 10 epochs
                if epoch%10 == 0:
                    print(f"comm {commName}, Train time step {timeStep} Iteration {epoch+1} -  train Loss: {loss.item()}, train Accuracy: {accuracy}")

            timeSteps_losses.append(lowest_loss)
            timeSteps_accuracies.append(accuracy_with_lowest_loss)
            model_with_lowest_loss_of_previous_step = model_with_lowest_loss
            
            # use the model with the lowest loss to compute the conformity
            # predict probabilities of votes for current time step batch
            with torch.no_grad():
                y_pred = model_with_lowest_loss(X_timeStep)
                if len(y_timeStep)==1: # only one sample
                    y_pred = y_pred.cpu().detach().numpy()[0]
                else:
                    y_pred = torch.squeeze(y_pred.cpu()).detach().numpy()
                # multiply each odd with probOddProduct
                for i,row in enumerate(X):
                    if row[0] >= row[1]: # for pvr >= nvr
                        odd = y_pred[i]/(1-y_pred[i])
                    else: # # for nvr > pvr
                        odd = (1-y_pred[i])/y_pred[i]
                    probOddList.append(odd)

        # compute conformity
        conformity = np.exp( np.sum(np.log(probOddList)) / totalRealSamplesCount )
        print(f"===> {commName} conformity is {conformity}.")    

        
        # Return avg training accuracy and model.
        return timeSteps_accuracies[-1], timeSteps_losses, conformity, model_with_lowest_loss
    
    except Exception as ee:
        print(f"tried sgd, failed: {ee}")
        return None


def myFun(commIndex, commName, commDir, return_trainSuccess_dict, root_dir, forConformity):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # # check whether already done this step, skip
    # result_directory = os.path.join(commDir, r'result_folder')
    # resultFiles = ['CVP2_votingStage_forConformity_results.txt']
    # resultFiles = [result_directory+'/'+f for f in resultFiles]
    # # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    # if os.path.exists(resultFiles[0]):
    #     print(f"{commName} has already done this step.")
    #     return

    # load intermediate_data files
    final_directory = os.path.join(commDir, r'intermediate_data_folder')
    
    print(f"loading data... for {commName}")
    with open(final_directory+'/'+'semiSynthetic_whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'rb') as inputFile:
        try:
            questions_Data = pickle.load( inputFile)
        except Exception as e:
            print(f"for {commName} error when load the Questions data: {e}")
            return
    questionCount = len(questions_Data) # this not equal to the origianl question count when generating nus
    # to get the origianl question count when generating nus, load eventlist file
    with open(final_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        ori_Questions = pickle.load( inputFile)
    ori_questionCount = len(ori_Questions)
    ori_Questions.clear() # clear this to same memory
    print(f"{commName} has {questionCount} questions in training data, but has {ori_questionCount} questions corresponding to nus.")

    ############################################################################################

    # choices of training settings
    normFlag = False
    regFlag = True   # if True, apply l2 regularization on all parameters; if False, don't apply regularization
    reg_alpha = 0.5  # the regularization strength. only useful when reg_Flag = True

    oneside = True   # if True, use one side parametrization; if False, use two side parametrization

    withBias = False # whether add bias term
    
    # select model type
    if withBias:
        modeltype='1layer_bias_withoutRankTerm' # equivalent to set tau as 1
    else: # without bias
        modeltype='1layer_withoutRankTerm'
    
    opt_forMiniBatchTrain = 'sgd'
    learning_rate = 0.1
    max_iter = 300   # this is the total number of epochs

    if forConformity:
        log_file_name = 'semiSynthetic_CVP2_votingStage_oneside_forConformity_Log.txt'
        result_file_name = "semiSynthetic_CVP2_votingStage_oneside_forConformity_results.txt"
        trained_model_file_name = 'semiSynthetic_CVP2_votingStage_oneside_forConformity_model.sav'
    else:
        log_file_name = 'semiSynthetic_CVP2_votingStage_oneside_Log.txt'
        result_file_name = "semiSynthetic_CVP2_votingStage_oneside_results.txt"
        result_withFullData_file_name =  "semiSynthetic_CVP2_votingStage_oneside_results_withFullData.txt"
        trained_model_file_name = 'semiSynthetic_CVP2_votingStage_oneside_model.sav'
        trained_model_withFullData_file_name = 'semiSynthetic_CVP2_votingStage_oneside_model_withFullData.sav'
    ############################################################################################

    # get total sample count
    total_sample_count = 0
    original_n_feature = None
    for qid, tup in questions_Data.items():
        X = tup[0].todense()
        y = tup[1]
        
        total_sample_count += len(y)

        if original_n_feature == None and len(y) != 0:
            original_n_feature = X.shape[1] # the column number of X is the original number of features

    if original_n_feature == None or total_sample_count == 0: # exception
        print(f"Exception for {commName}, original_n_feature:{original_n_feature}, total_sample_count:{total_sample_count}!!!!!!!")
        time.sleep(5)
        return
    
    # check data size, skip training if the number of samples is too small
    if total_sample_count<10:
        writeIntoLog(f"consists of {total_sample_count} samples which < 10.\n" + log_text, commDir , log_file_name)
        print(f"{commName} consists of {total_sample_count} samples which < 10.\n")
        return
    
    # compute the number of features in data. 
    # The first 4 columns of origianl dataset are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio, IPWrank)
    n_feature = original_n_feature - 3 # removed the first, second and fourth columns for one side parametrization
    assert n_feature >0
    print(f"{commName} has total sample count: {total_sample_count}, and number of features: {n_feature}")

    # my NN model
    # training with myNN...
    if forConformity:
        outputs = myTrain_forConformity(commIndex, questions_Data,n_feature, opt_forMiniBatchTrain, learning_rate, max_iter,commName,normFlag, regFlag, reg_alpha, modeltype)
    else:
        outputs = myTrain(commIndex, questions_Data,n_feature, opt_forMiniBatchTrain, learning_rate, max_iter,commName,normFlag, regFlag, reg_alpha, modeltype)

    if outputs != None:
        if forConformity:
            train_accuracy, timeSteps_losses, conformity, model = outputs
        else:
            train_losses, test_losses, test_accuracy, model_with_lowest_test_loss,lowest_test_loss_of_epoch,lowest_test_loss, batch_size, model_trainedWithFullData = outputs
    
        print(f"start to save results... for {commName}")

        # save results as log for model
        # log the training settings
        text = f"=========================================\n"
        text += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '\n'
        text += f"Train one side, normalize:{normFlag}, regFlag:{regFlag}, reg_alpha:{reg_alpha},\n"
        text += f"      opt_forMiniBatchTrain :{opt_forMiniBatchTrain}\n"
        text += f"      max_iter = {max_iter},  learning_rate:{learning_rate}\n"
        text += f"dataset size: ({total_sample_count}, {n_feature})\n\n"
        # text += f"avg training loss: {mean(timeSteps_losses)} \ntrain accuracy:{train_accuracy}\n"
        if forConformity:
            text += f"conformity : {conformity}\n"
        print(text)

        # save model
        saveModel(model_with_lowest_test_loss,trained_model_file_name)
        print(f"for {commName} model_with_lowest_test_loss saved.")
        
        # output learned parameters
        parm = defaultdict()
        for name, param in model_with_lowest_test_loss.named_parameters():
            parm[name]=param.cpu().detach().numpy() 
        
        if withBias:
            bias = parm['linear.bias'][0]
        else:
            bias = None

        weights = parm['linear.weight'][0]
        
        result_text, coefs, nus, qs = resultFormat(weights, bias, ori_questionCount)
        writeIntoResult(text + result_text, result_file_name)

        # double check the number of qs
        with open(final_directory+'/'+'semiSynthetic_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
            total_answersWithVotes_indice = pickle.load( inputFile)
        assert len(total_answersWithVotes_indice) == len(qs)

        # save model trained with full data
        saveModel(model_trainedWithFullData,trained_model_withFullData_file_name)
        print(f"for {commName} model_trainedWithFullData saved.")
        
        # output learned parameters of model trained with full data
        parm = defaultdict()
        for name, param in model_trainedWithFullData.named_parameters():
            parm[name]=param.cpu().detach().numpy() 
        
        if withBias:
            bias_withFullData = parm['linear.bias'][0]
        else:
            bias_withFullData = None

        weights_withFullData = parm['linear.weight'][0]
        
        result_text_withFullData, coefs_withFullData, nus_withFullData, qs_withFullData = resultFormat(weights_withFullData, bias_withFullData, ori_questionCount)
        writeIntoResult(text + result_text_withFullData, result_withFullData_file_name)

        elapsed = format_time(time.time() - t0)
        
        log_text = "Elapsed: {:}\n".format(elapsed)
        current_directory = os.getcwd()
        writeIntoLog(text + log_text, current_directory , log_file_name)

        # visualize the losses
        plt.cla()
        if forConformity:
            plt.plot(range(len(timeSteps_losses)), timeSteps_losses, 'g-', label=f'trainning')
            plt.xlabel("time step")
            plotFileName = "semiSynthetic_CVP2_votingStage_oneside_forConformity_Losses.png"
        else:
            plt.plot(range(len(test_losses)), test_losses, 'g-', label=f'testing')
            plt.xlabel("epoch")
            plotFileName = "semiSynthetic_CVP2_votingStage_oneside_Losses.png"
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        savePlot(plt, plotFileName)

        if forConformity:
            return_trainSuccess_dict[commName] = {'dataShape':(total_sample_count, n_feature), 'conformity':conformity,
                                           'coefs':coefs,'bias':bias,'nus':nus, 'qs':qs}
        else:
            return_trainSuccess_dict[commName] = {'dataShape':(total_sample_count, n_feature), 'conformity':None,
                                           'coefs':coefs,'bias':bias,'nus':nus, 'qs':qs,
                                           'coefs_withFullData':coefs_withFullData,'bias_withFullData':bias_withFullData,'nus_withFullData':nus_withFullData, 'qs_withFullData':qs_withFullData}
        
        # save current return 
        if forConformity:
            resultLogFileName = "semiSynthetic_CVP2_votingStage_oneside_forConformity_result_trainSuccess_Log.txt"
            resultDictFileName = 'semiSynthetic_CVP2_votingStage_oneside_forConformity_return_trainSuccess.dict'
        else:
            resultLogFileName = "semiSynthetic_CVP2_votingStage_oneside_result_trainSuccess_Log.txt"
            resultDictFileName = 'semiSynthetic_CVP2_votingStage_oneside_return_trainSuccess.dict'
        
        log_text = ""
        commIndex = 0
        return_trainSuccess_normalDict = defaultdict()
        for commName, d in return_trainSuccess_dict.items():
            commIndex +=1
            log_text += f"{commIndex} {commName} data_shape {d['dataShape']} conformity {d['conformity']} coefs {d['coefs']} bias {d['bias']}\n"
            return_trainSuccess_normalDict[commName] = {'dataShape':d['dataShape'], 'conformity':d['conformity'], 'coefs':d['coefs'], 'bias':d['bias'], 'nus':d['nus'],'qs':d['qs'],
                                                        'coefs_withFullData':d['coefs_withFullData'], 'bias_withFullData':d['bias_withFullData'], 'nus_withFullData':d['nus_withFullData'],'qs_withFullData':d['qs_withFullData']}
        log_text += f"{len(return_trainSuccess_dict)} communitites were successfully trained\n"
        writeIntoLog(log_text, root_dir, resultLogFileName)
        os.chdir(root_dir) # go back to root directory
        with open(resultDictFileName, 'wb') as outputFile:
            pickle.dump(return_trainSuccess_normalDict, outputFile)
            print(f"saved results of {commName}, current length of return_trainSuccess_normalDict is {len(return_trainSuccess_normalDict)}")

    else: # training outputs == None
        print(f"{commName} failed to train!!!!!!!!!!!!!!")
        time.sleep(5)
        sys.exit(f"{commName} failed to train!!!!!!!!!!!!!!")

#    print('training1_allTimeSteps_oneside Done.    Elapsed: {:}.\n'.format(elapsed))
    

def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    forConformity = False

    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    return_trainSuccess_dict = manager.dict() # to save the used train mode (wholebatch or minibatch) of each community

    
    try:
        # test on comm "coffee.stackexchange" to debug
        myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1], return_trainSuccess_dict, root_dir, forConformity)
        # test on comm "datascience.stackexchange" to debug
        # myFun(301,commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1], return_trainSuccess_dict, root_dir)
        # test on comm "webapps.stackexchange" to debug
        # myFun(305,commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1], return_trainSuccess_dict, root_dir)
        # test on comm "travel.stackexchange" to debug
        # myFun(319,commDir_sizes_sortedlist[319][0], commDir_sizes_sortedlist[319][1], return_trainSuccess_dict, root_dir)
        # test on comm "english.stackexchange" to debug
        # myFun(349,commDir_sizes_sortedlist[349][0], commDir_sizes_sortedlist[349][1], return_trainSuccess_dict, root_dir)
    except Exception as e:
        print(e)
        sys.exit()
    """
    
     # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    commIndex =0
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        if commName in splitted_comms: # skip splitted big communities
            print(f"{commName} was splitted. skip")
            continue

        try:
            # p = mp.Process(target=myFun, args=(commIndex, commName,commDir, return_trainSuccess_dict, root_dir))
            p = mp.Process(target=myFun, args=(commIndex, commName,commDir, return_trainSuccess_dict, root_dir, forConformity))
            p.start()
            commIndex +=1
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
    if forConformity:
        resultLogFileName = "semiSynthetic_CVP2_votingStage_oneside_forConformity_result_trainSuccess_Log.txt"
        resultDictFileName = 'semiSynthetic_CVP2_votingStage_oneside_forConformity_return_trainSuccess.dict'
    else:
        resultLogFileName = "semiSynthetic_CVP2_votingStage_oneside_result_trainSuccess_Log.txt"
        resultDictFileName = 'semiSynthetic_CVP2_votingStage_oneside_return_trainSuccess.dict'

    # save return_trainSuccess_dict
    os.chdir(root_dir) # go back to root directory

    # convert and save the last return_trainSuccess
    log_text = ""
    commIndex = 0
    return_trainSuccess_normalDict = defaultdict()
    for commName, d in return_trainSuccess_dict.items():
        commIndex +=1
        log_text += f"{commIndex} {commName} data_shape {d['dataShape']} conformity {d['conformity']} coefs {d['coefs']} bias {d['bias']}\n"
        return_trainSuccess_normalDict[commName] = {'dataShape':d['dataShape'], 'conformity':d['conformity'], 'coefs':d['coefs'], 'bias':d['bias'], 'nus':d['nus'],'qs':d['qs'],
                                                    'coefs_withFullData':d['coefs_withFullData'], 'bias_withFullData':d['bias_withFullData'], 'nus_withFullData':d['nus_withFullData'],'qs_withFullData':d['qs_withFullData']}
    
    log_text += f"{len(return_trainSuccess_dict)} communitites were successfully trained\n"
    writeIntoLog(log_text, root_dir, resultLogFileName)
    os.chdir(root_dir) # go back to root directory
    with open(resultDictFileName, 'wb') as outputFile:
        pickle.dump(return_trainSuccess_normalDict, outputFile)
        print(f"saved return_trainSuccess_normalDict for {len(return_trainSuccess_normalDict)} comms")

    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('semiSynthetic2_CVP2_votingStage_oneside training NEW Done completely.    Elapsed: {:}.\n'.format(elapsed))
    

if __name__ == "__main__":
  
    # calling main function
    main()
