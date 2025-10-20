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
from CustomizedNN import LRNN_1layer_withoutRankTerm
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
    def __init__(self, full_data,oneside, testingDataIndexListInFullData, trainOrtest):
        self.data = full_data
        self.oneside = oneside
        self.testingDataIndexListInFullData = testingDataIndexListInFullData
        self.trainOrtest = trainOrtest
    
    def process_data(self, dataset, oneside, testingDataIndexListInFullData, trainOrtest):
        X = dataset[0].todense() # convert sparse matrix to np.array
        y = dataset[1]
        
        # tailor the columns for different parametrization.
        # the first 3 columns of X are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio)
        if oneside:
            X = np.concatenate((X[:,2:3], X[:,4:]), axis=1) # for one side parametrization, removed the first two columns and the Inversed rank term column
        else: # two sides
            X = np.concatenate((X[:,:2] , X[:,4:] ), axis=1) # for two sides parametrization, removed the third and fourth column

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
        return self.process_data(self.data, self.oneside, self.testingDataIndexListInFullData, self.trainOrtest)
    
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
            elif j < ori_questionCount+2:
                text += f"nu_{j-2}: {coef}\n" # the 3th feature to the (questionCount+2)th feature are ralative length for each question, print nus
                nus.append(coef)
            else: # for rest of j
                text += f"q_{j-2-ori_questionCount}: {coef}\n" # the quality features start from j = questionCount+2
                if bias != None:
                    text += f"q_{j-2-ori_questionCount}+bias: {coef+bias}\n"
                qs.append(coef)
        
        else: # when do oneside
            if j == 0: # the first feature is seen_pos_vote_ratio for oneside training, or pos_vote_ratio for only_pvr. print lambda
                text += f"lambda: {coef}\n"
                coefs.append(coef)
            elif j < ori_questionCount+1:
                text += f"nu_{j-1}: {coef}\n" # the 2th feature to the (questionCount+1)th feature are ralative length for each question, print nus
                nus.append(coef)
            else: # for rest of j
                text += f"q_{j-1-ori_questionCount}: {coef}\n" # the quality features start from j = questionCount+1
                if bias != None:
                    text += f"q_{j-1-ori_questionCount}+bias: {coef+bias}\n"
                qs.append(coef)
    
    return text, coefs, nus, qs
    
###################################################################################################

def myTrain(commIndex, commName, commDir, ori_questionCount, full_data, testingDataIndexListInFullData, log_file_name):
    t0=time.time()

    #######################################################################
    ### training settings #######
    result_file_name = f"temperalOrderTraining8_trainCVP_results.txt"
    trained_model_file_name = f'temperalOrderTraining8_trainCVP_model.sav'
    trained_withSKLEARN_model_file_name = f'temperalOrderTraining8_trainCVP_withSKLEARN_model.pkl'
    result_withSKLEARN_file_name = f"temperalOrderTraining8_trainCVP_withSKLEARN_results.txt"
    plot_file_name = f'temperalOrderTraining8_trainCVP_Losses.png'
    log_text = ""
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
    # The first 3 columns of origianl dataset are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio)
    if oneside:
        n_feature = original_n_feature - 3 # removed 3 columns for one side parametrization
    else:
        n_feature = original_n_feature - 2 # removed 2 columns for two sides parametrization
    print(f"{commName} has total sample count: {total_sample_count}, and number of features: {n_feature}")
    

    ####################################################################################################
    print("try to train using SKLEARN...")
    from sklearn.linear_model import LogisticRegression
    X = full_data[0].todense() # convert sparse matrix to np.array
    y = full_data[1]
    
    # tailor the columns for different parametrization.
    # the first 4 columns of X are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio, IPW_rank)
    if oneside:
        X = np.concatenate((X[:,2:3], X[:,4:]), axis=1) # for one side parametrization, removed the first two columns and the Inversed rank term column
    else: # two sides
        X = np.concatenate((X[:,:2] , X[:,4:] ), axis=1) # for two sides parametrization, removed the third and fourth column   


    # lr = LogisticRegression(random_state=1, solver='liblinear',penalty='l2', fit_intercept=withBias, C=1)
    lr = LogisticRegression(solver='lbfgs',penalty='l2', fit_intercept=withBias, C=1, max_iter=max_iter)
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
        
        result_text_sklearn, coefs_sklearn, nus_sklearn, qs_sklearn = resultFormat(weights_sklearn, bias_sklearn, oneside, ori_questionCount)

        text = f"SKLEARN training ==================================\n"
        text += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '\n'
        text += f"Train oneside:{oneside}, normalize:{normFlag}, regFlag:{regFlag}, reg_alpha:{reg_alpha}, withBias:{withBias}\n"
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
    print("try to train using cuda and SGD...")
    # check gpu count
    cuda_count = torch.cuda.device_count()
    # assign one of gpu as device
    d = commIndex % cuda_count
    device = torch.device('cuda:'+str(d) if torch.cuda.is_available() else 'cpu')
    print(f"comm {commName}, Start to train NN model... on device: {device}")

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
        iterable_training_dataset = MyIterableDataset(full_data,oneside, testingDataIndexListInFullData, 'train') 
        iterable_testing_dataset = MyIterableDataset(full_data,oneside, testingDataIndexListInFullData, 'test') 
        iterable_full_dataset = MyIterableDataset(full_data,oneside, testingDataIndexListInFullData, 'full') 
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

        # save results as log for model
        # log the training settings
        text = f"=========================================\n"
        text += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '\n'
        text += f"Train oneside:{oneside}, normalize:{normFlag}, regFlag:{regFlag}, reg_alpha:{reg_alpha},\n"
        text += f"      opt_forMiniBatchTrain :{opt_forMiniBatchTrain}, withBias:{withBias}\n"
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

        weights = parm['linear.weight'][0]
        
        result_text, coefs, nus, qs = resultFormat(weights, bias, oneside, ori_questionCount)
        writeIntoResult(text + f"\ntraining losses: {train_losses}\n"+ f"\ntesting losses: {test_losses}\n"+ f"\ntesting accuracy: {test_accuracy}\n" + result_text, result_file_name)
        
        elapsed = format_time(time.time() - t0)
        
        log_text += "Elapsed: {:}\n".format(elapsed)

        writeIntoLog(text + log_text, commDir , log_file_name)

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
        writeIntoLog(f"failed to train with cuda and SGD")
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
    
    return total_sample_count, n_feature, train_losses, test_losses, lowest_test_loss_of_epoch, lowest_test_loss, test_accuracy, coefs, bias, nus, qs, conformity, coefs_sklearn, bias_sklearn, nus_sklearn, qs_sklearn, conformity_sklearn


def myFun(commIndex, commName, commDir, root_dir):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    log_file_name = "temperalOrderTraining3_CVP_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # check whether already done this step, skip
    # result_directory = os.path.join(commDir, r'result_folder')
    # resultFiles = ['training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_results.txt']
    # resultFiles = [result_directory+'/'+f for f in resultFiles]
    # # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    # if os.path.exists(resultFiles[0]):
    #     print(f"{commName} has already done this step.")
    #     return

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        ori_Questions = pickle.load( inputFile)
    ori_questionCount = len(ori_Questions)
    ori_Questions.clear() # clear this to same memory
    
    # load full data
    try:
        with open(intermediate_directory+f"/whole_datasets_sorted_removeFirstRealVote.dict", 'rb') as inputFile:
            full_data = pickle.load( inputFile)
    except:
        print(f"{commName} hasn't done temperalOrderSorting yet, skip")
        return

    # load testing data's sortingBase
    # get testing data raw index in full data
    with open(intermediate_directory+f"/temperalOrderTraining6_outputs.dict", 'rb') as inputFile:
        qid2TestingSortingBaseList, _ = pickle.load( inputFile)
    sortingBaseListOfTestingData = []
    for qid, testingSortingBaseList in qid2TestingSortingBaseList.items():
        sortingBaseListOfTestingData.extend(testingSortingBaseList)

    with open(intermediate_directory+f"/sorted_qidAndSampleIndexAndSortingBase.dict", 'rb') as inputFile:
        sorted_qidAndSampleIndex = pickle.load( inputFile)

    testingDataIndexListInFullData = [i for i, tup in enumerate(sorted_qidAndSampleIndex) if tup[2] in sortingBaseListOfTestingData]
    

    # training 
    output = myTrain(commIndex, commName, commDir, ori_questionCount, full_data, testingDataIndexListInFullData, log_file_name)
    total_sample_count, n_feature, train_losses, test_losses, lowest_test_loss_of_epoch, lowest_test_loss, test_accuracy, coefs, bias, nus, qs, conformity, coefs_sklearn, bias_sklearn, nus_sklearn, qs_sklearn, conformity_sklearn = output
    
    if (coefs != None) or (coefs_sklearn != None):
        return_trainSuccess_dict = defaultdict()
        return_trainSuccess_dict[commName] = {'dataShape':(total_sample_count, n_feature), 'trainLosses':train_losses, 'testLosses':test_losses, 'epoch':lowest_test_loss_of_epoch, 'lowest_test_loss':lowest_test_loss, 'testAcc':test_accuracy,
                                            'coefs':coefs,'bias':bias,'nus':nus, 'qs':qs, 'conformity':conformity,
                                            'coefs_sklearn':coefs_sklearn, 'bias_sklearn':bias_sklearn, 'nus_sklearn':nus_sklearn, 'qs_sklearn':qs_sklearn,
                                            'conformity_sklearn':conformity_sklearn}
        
        # save return dict
        with open(intermediate_directory+f"/temperalOrderTraining8_CVP_return.dict", 'wb') as outputFile:
            pickle.dump(return_trainSuccess_dict, outputFile)
            print( f"saved return_trainSuccess_dict of {commName}.")

    # # save csv
    with open(root_dir +'/'+'allComm_temperalOrderTraining8_CVP_results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( [commName,total_sample_count, coefs_sklearn[0],coefs[0], conformity_sklearn, conformity])


def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # # save csv
    with open(root_dir +'/'+'allComm_temperalOrderTraining8_CVP_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","totalSampleCount", "coefs_sklearnLR","coefs_newModel", "conformity_sklearnLR","conformity_newModel"])

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
    except Exception as e:
        print(e)
        sys.exit()
    """
     # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        if commName in splitted_comms: # skip splitted big communities
            print(f"{commName} was splitted,skip")
            continue

        try:
            p = mp.Process(target=myFun, args=(commIndex, commName,commDir, root_dir))
            p.start()
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
    
    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('temperalOrderTraining3 train CVP  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
