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
from CustomizedNN import LRNN_1layer, LRNN_1layer_bias, LRNN_1layer_bias_specify,LRNN_1layer_bias_withoutRankTerm
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
    def __init__(self, batchFileDirList, oneside):
        self.fileList = batchFileDirList
        self.oneside = oneside
    
    def process_data(self, batchFileDirList, oneside):
        for i, subDir in enumerate(batchFileDirList):
            with open(subDir, 'rb') as inputFile:
                percentData= pickle.load( inputFile)
            X = percentData[0].todense() # convert sparse matrix to np.array
            y = percentData[1]
            
            # tailor the columns for different parametrization.
            # the first 3 columns of X are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio)
            if oneside:
                X = X[:,2:] # for one side parametrization, removed the first two columns
            else: # two sides
                X = np.concatenate((X[:,:2] , X[:,3:] ), axis=1) # for two sides parametrization, removed the third column

            for i in range(len(y)):
                yield X[i,:], y[i]
    
    def __iter__(self):
        return self.process_data(self.fileList, self.oneside)
    
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

def resultFormat(weights, bias, tau, oneside, learnTau, ori_questionCount):
    coefs = [] # community-level coefficients
    nus = [] # question-level 
    qs = [] # answer-level qualities
    text = f"bias:{bias}\n"

    if learnTau: 
        text += f"tau:{tau}\n"

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
            elif j < ori_questionCount+3:
                text += f"nu_{j-3}: {coef}\n" # the 4th feature to the (questionCount+3)th feature are ralative length for each question, print nus
                nus.append(coef)
            else: # for rest of j
                text += f"q_{j-3-ori_questionCount}: {coef}\n" # the quality features start from j = questionCount+3
                if bias != None:
                    text += f"q_{j-3-ori_questionCount}+bias: {coef+bias}\n"
                qs.append(coef)
        
        else: # when do oneside
            if j == 0: # the first feature is seen_pos_vote_ratio for oneside training, or pos_vote_ratio for only_pvr. print lambda
                text += f"lambda: {coef}\n"
                coefs.append(coef)
            elif j == 1:
                text += f"beta: {coef}\n" # with rank term, the second feature is inversed rank, print beta
                coefs.append(coef)
            elif j < ori_questionCount+2:
                text += f"nu_{j-2}: {coef}\n" # the 3th feature to the (questionCount+2)th feature are ralative length for each question, print nus
                nus.append(coef)
            else: # for rest of j
                text += f"q_{j-2-ori_questionCount}: {coef}\n" # the quality features start from j = questionCount+2
                if bias != None:
                    text += f"q_{j-2-ori_questionCount}+bias: {coef+bias}\n"
                qs.append(coef)
    
    return text, coefs, nus, qs
    
###################################################################################################

def myTrain(commIndex, baseModelDir, trainingPercentage,trainingPercentageBatchFileDirList,testingPercentage,testingPercentageBatchFileDirList, commName, commDir, ori_questionCount, log_file_name):
    t0=time.time()

    #######################################################################
    ### training settings #######
    result_file_name = f"predictionAnalysis2_negLogLikelyhoods_newModel_posTau_trainWith{trainingPercentage}_testWith{testingPercentage}_results.txt"
    trained_model_file_name = f'predictionAnalysis2_negLogLikelyhoods_newModel_posTau_trainWith{trainingPercentage}_model.sav'
    plot_file_name = f'predictionAnalysis2_negLogLikelyhoods_newModel_posTau_trainWith{trainingPercentage}_testWith{testingPercentage}_Losses.png'
    # choices of training settings
    normFlag = False
    regFlag = True   # if True, apply l2 regularization on all parameters; if False, don't apply regularization
    reg_alpha = 0.5  # the regularization strength. only useful when reg_Flag = True
    oneside = True   # if True, use one side parametrization; if False, use two side parametrization

    withBias = False # whether add bias term
    learnTau = True  # if True, learn tau;  if learnTau = False, fix tau = 1
    positiveTau = True # only useful when learnTau = True, if postiveTau = False, don't constrain tau as positive
   
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
    
    opt_forMiniBatchTrain = 'sgd'
    learning_rate = 0.1
    max_iter = 300   # this is the total number of epochs
    ############################################################################################

    # get total sample count
    total_sample_count = 0
    for i, subDir in enumerate(trainingPercentageBatchFileDirList):
        batch = i+1
        print(f"scan batch {batch} of percent {trainingPercentage} data of {commName}...")
        batchFileDir = subDir
        with open(batchFileDir, 'rb') as inputFile:
            percentData= pickle.load( inputFile)
            total_sample_count += percentData[0].shape[0]
    
    original_n_feature = percentData[0].todense().shape[1]
    percentData=tuple() # clear

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
        n_feature = original_n_feature - 2 # removed 2 columns for one side parametrization
    else:
        n_feature = original_n_feature - 1 # removed 1 columns for two sides parametrization
    print(f"{commName} has total sample count: {total_sample_count}, and number of features: {n_feature}")
    

    ####################################################################################################
    print("start to train...")
    # check gpu count
    cuda_count = torch.cuda.device_count()
    # assign one of gpu as device
    d = commIndex % cuda_count
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
    elif modeltype == '1layer_bias_withoutRankTerm':
        model = LRNN_1layer_bias_withoutRankTerm(input_dim)
    elif modeltype == '1layer':
        model = LRNN_1layer(input_dim,initial_tau,tauColumnIndex, positiveTau)
    else:
        sys.exit(f"invalid modeltype: {modeltype}")

    # load model
    if baseModelDir != None:
        model.load_state_dict(torch.load(baseModelDir))

    model.to(device)
    
    # using Binary Cross Entropy Loss
    loss_fn = torch.nn.BCELoss(size_average=True, reduction='mean')
    loss_nll =  torch.nn.BCELoss(reduction='none')

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
    avg_negloglikelyhood_with_lowest_test_loss = None
    test_accuracy = None


    try:
        optimizer = optimizer_forMiniBatchTrain
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,280], gamma=0.1)

        # Build a Streaming DataLoader with customized iterable dataset
        iterable_training_dataset = MyIterableDataset(trainingPercentageBatchFileDirList,oneside)
        iterable_testing_dataset = MyIterableDataset(testingPercentageBatchFileDirList,oneside)
        # prepare data loader
        batch_size = 1000
        my_training_dataloader = DataLoader(iterable_training_dataset, batch_size=batch_size) 
        my_testing_dataloader = DataLoader(iterable_testing_dataset, batch_size=batch_size) 

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
            test_negloglikelyhoods = []

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

                if outputs.shape[0] ==1:
                    negloglikelyhoods = loss_nll(outputs[0], y_batch_test)
                    # my_negloglikelyhoods = myNLL(outputs[0], y_batch_test) # only for sanity check
                else:
                    negloglikelyhoods = loss_nll(torch.squeeze(outputs), y_batch_test) # [m,1] -squeeze-> [m]
                    # my_negloglikelyhoods = myNLL(torch.squeeze(outputs), y_batch_test) # only for sanity check
                negloglikelyhoods = negloglikelyhoods.cpu().detach().numpy()
        
                # Calculating the loss and accuracy for the test dataset
                correct = np.sum(torch.squeeze(outputs.cpu()).round().detach().numpy() == y_batch_test.cpu().detach().numpy())
                batches_test_correct.append(correct)
                batches_test_sampleCount.append(sample_n)
                batches_test_losses.append(loss.item())
                test_negloglikelyhoods.extend(negloglikelyhoods)
                
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
                avg_negloglikelyhood_with_lowest_test_loss = mean(test_negloglikelyhoods)
                test_accuracy = sum(batches_test_correct) / sum(batches_test_sampleCount)

            scheduler.step()   
        
        # save model
        saveModel(model_with_lowest_test_loss,trained_model_file_name)
        print(f"for {commName} model saved.")

        print(f"start to save results... for {commName}")

        # save results as log for model
        # log the training settings
        text = f"=========================================\n"
        text += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '\n'
        text += f"Train oneside:{oneside}, normalize:{normFlag}, regFlag:{regFlag}, reg_alpha:{reg_alpha},\n"
        text += f"      learnTau:{learnTau}, opt_forMiniBatchTrain :{opt_forMiniBatchTrain}, withBias:{withBias}, positiveTau:{positiveTau}\n"
        text += f"      max_iter = {max_iter},  learning_rate:{learning_rate}\n"
        text += f"dataset size: ({total_sample_count}, {n_feature})\n\n"
        text += f"trained with batch_size:{batch_size}\navg training loss: {mean(train_losses)} \navg testing loss:{mean(test_losses)},lowest_test_loss_of_epoch:{lowest_test_loss_of_epoch} lowest_test_loss:{lowest_test_loss}\ntest accuracy:{test_accuracy}\n"
        text += f"training with the first {trainingPercentage} percent and test with the {testingPercentage} percent \n"
        print(text)
        
        # output learned parameters
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
        
        result_text, coefs, nus, qs = resultFormat(weights, bias, tau, oneside, learnTau, ori_questionCount)
        writeIntoResult(text + f"\ntraining losses: {train_losses}\n"+ f"\ntesting losses: {test_losses}\n"+ f"\ntesting accuracy: {test_accuracy}\n" + result_text, result_file_name)

        elapsed = format_time(time.time() - t0)
        
        log_text = "Elapsed: {:}\n".format(elapsed)
        current_directory = os.getcwd()
        writeIntoLog(text + log_text, current_directory , log_file_name)

        # visualize the losses
        plt.cla()
        plt.plot(range(len(train_losses)), train_losses, 'g-', label=f'trainning')
        plt.plot(range(len(test_losses)), test_losses, 'b-', label=f'testing\nlowest loss:{lowest_test_loss}\nreached at epoch {lowest_test_loss_of_epoch}')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        savePlot(plt, plot_file_name)

        return trainingPercentage, testingPercentage, avg_negloglikelyhood_with_lowest_test_loss, test_accuracy


    except Exception as ee:
        print(f"tried sgd, failed: {ee}")
        return None


def myFun(commIndex, commName, commDir):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    log_file_name = "predictionAnalysis2_negLogLikelyhoods_newModel_forSplittedFiles_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    
    # check whether already done this step, skip
    resultFiles = ['predictionAnalysis2_newModel_return.dict']
    resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
    # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    if os.path.exists(resultFiles[0]):
        print(f"{commName} has already done this step.")
        return
    

    if commName == 'stackoverflow':
        splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
        split_QuestionsWithEventList_files_directory = os.path.join(splitFolder_directory, r'QuestionsPartsWithEventList')

        # to get the origianl question count when generating nus, load eventlist file
        ori_questionCount = 0
        partFiles = [ f.path for f in os.scandir(split_QuestionsWithEventList_files_directory) if f.path.startswith("QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList_part_") ]
        for i, partDir in enumerate(partFiles):
            print(f"scanning part {i+1} eventlist file of {commName} for original question count...")
            # get question count of each part
            with open(partDir, 'rb') as inputFile:
                Questions_part = pickle.load( inputFile)
            ori_questionCount += len(Questions_part)
            Questions_part.clear() # clear this to same memory
    else:
        with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
            ori_Questions = pickle.load( inputFile)
        ori_questionCount = len(ori_Questions)
        ori_Questions.clear() # clear this to same memory

    # mkdir to keep predictionAnalysis data
    predictionAnalysis_data_folder = os.path.join(intermediate_directory, r'predictionAnalysis_data_folder')
    if not os.path.exists(predictionAnalysis_data_folder): # don't have dataset splitted intermediate data folder, skip
        print(f"{commName} has no  predictionAnalysis_data_folder.")
        return
    
    model_folder = os.path.join(commDir, r'trained_model_folder')

    # training part by part
    all_outputs = []
    percentageList = [25,50,75,100]
    for i in range(3):
        trainingPercentage = percentageList[i]
        testingPercentage = percentageList[i+1]

        trainingPercentageBatchFileDirList = [ f.path for f in os.scandir(predictionAnalysis_data_folder) if f.path.split('/')[-1].startswith(f"percent{trainingPercentage}PureData_batch_")]
        trainingPercentageBatchFileDirList.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
        if len(trainingPercentageBatchFileDirList)==0:
            print(f"{commName} has not done the pure data ready.")
            return

        testingPercentageBatchFileDirList = [ f.path for f in os.scandir(predictionAnalysis_data_folder) if f.path.split('/')[-1].startswith(f"percent{testingPercentage}PureData_batch_")]
        testingPercentageBatchFileDirList.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
        
        baseModelDir = None
        if i>0:
            baseModelDir = model_folder + f'/predictionAnalysis2_negLogLikelyhoods_newModel_posTau_trainWith{percentageList[i-1]}_model.sav'
        output = myTrain(commIndex, baseModelDir, trainingPercentage,trainingPercentageBatchFileDirList,testingPercentage,testingPercentageBatchFileDirList, commName, commDir, ori_questionCount, log_file_name)
        all_outputs.append(output)
    
    return_dict = defaultdict()
    log_text = ""
    for tup in all_outputs:
        trainingPercentage, testingPercentage, avg_negloglikelyhood_with_lowest_test_loss, test_accuracy = tup
        return_dict[(trainingPercentage,testingPercentage)] = {'avg_negloglikelyhood': avg_negloglikelyhood_with_lowest_test_loss, 'test_accuracy':test_accuracy}
        log_text += f"training with {trainingPercentage} percent, testing with {testingPercentage}: avg_negloglikelyhood={avg_negloglikelyhood_with_lowest_test_loss}, test_accuracy={test_accuracy}\n"
    
    writeIntoLog(log_text, commDir, log_file_name)
    
    # save return dict
    with open(intermediate_directory+f"/predictionAnalysis2_newModel_return.dict", 'wb') as outputFile:
        pickle.dump(return_dict, outputFile)
        print( f"saved return_dict of {commName}:\n{return_dict}")


def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    """
    try:
        # test on comm "coffee.stackexchange" to debug
        myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
        # test on comm "datascience.stackexchange" to debug
        # myFun(301,commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
        # test on comm "webapps.stackexchange" to debug
        # myFun(305,commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
        # test on comm "travel.stackexchange" to debug
        # myFun(319,commDir_sizes_sortedlist[319][0], commDir_sizes_sortedlist[319][1])
    except Exception as e:
        print(e)
        sys.exit()
    """
     # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    undone_comms = ['softwareengineering.stackexchange','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    commIndex = 0
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        if commName not in splitted_comms: # skip un-splitted small communities
            print(f"{commName} was not splitted,skip")
            continue
        
        if commName =='stackoverflow':
            print(f"skip {commName}")
            continue

        if commName not in undone_comms: # skip done communities
            print(f"{commName} was done,skip")
            continue

        try:
            p = mp.Process(target=myFun, args=(commIndex, commName,commDir))
            p.start()
            commIndex += 1
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()
            return

        processes.append(p)
        if len(processes)==10:
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
    print('prediction analysis 2 NLL new model training  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
