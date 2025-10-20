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
from CustomizedNN import LRNN_1layer, LRNN_1layer_bias, LRNN_1layer_bias_specify,LRNN_1layer_bias_withoutRankTerm
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.preprocessing import normalize
import random
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#### Tool Functions #######################################################
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
                cur_X_train = X[:cutIndex,2:] # for one side parametrization, removed the first two columns
                cur_y_train = y[:cutIndex]
                cur_X_test = X[cutIndex:,2:] # for one side parametrization, removed the first two columns
                cur_y_test = y[cutIndex:]
            else: # two sides
                cur_X_train = np.concatenate((X[:cutIndex,:2] , X[:cutIndex,3:] ), axis=1) # for two sides parametrization, removed the third column
                cur_y_train = y[:cutIndex]
                cur_X_test = np.concatenate((X[cutIndex:,:2] , X[cutIndex:,3:] ), axis=1) # for two sides parametrization, removed the third column
                cur_y_test = y[cutIndex:]

            yield cur_X_train,cur_y_train,cur_X_test,cur_y_test
    
    def __iter__(self):
        return self.process_data(self.data, self.oneside)




def myTrain(commIndex, questions_Data,n_feature, opt_forMiniBatchTrain, learning_rate, max_iter,commName,normFlag, regFlag, reg_alpha, modeltype, oneside, positiveTau):

    # check gpu count
    cuda_count = torch.cuda.device_count()
    # assign one of gpu as device
    d = (commIndex+1) % cuda_count
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

    torch.autograd.set_detect_anomaly(True) # only for debug, or this will slow the training

    train_losses = []
    test_losses = []
    lowest_test_loss_of_epoch = None
    lowest_test_loss = 1000000
    model_with_lowest_test_loss = None

    try:
        optimizer = optimizer_forMiniBatchTrain
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,280], gamma=0.1)

        # Build a Streaming DataLoader with customized iterable dataset
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
        model = LRNN_1layer(input_dim,initial_tau,tauColumnIndex, positiveTau) # restart from random state
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
            total_correct = 0
            total_sample = 0 
            for batch_test in mydataloader_retrain:
                # prepare the data tensors 
                X_batch_test = batch_test[2][0]
                y_batch_test = batch_test[3]
                X_batch_test,y_batch_test = preprocessing(X_batch_test,y_batch_test,normFlag)    
                X_batch_test = X_batch_test.to(device)
                y_batch_test = y_batch_test.to(device)   
        
                outputs = model_with_lowest_test_loss(X_batch_test) # use the model with the lowest test loss
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


def myFun(commIndex, commName, commDir, return_trainSuccess_dict, root_dir):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

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
    log_file_name = 'semiSynthetic_training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_Log.txt'
    result_file_name = "semiSynthetic_training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_posTau_results.txt"
    result_withFullData_file_name = "semiSynthetic_training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_posTau_results_withFullData.txt"
    trained_model_file_name = 'semiSynthetic_training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_posTau_model.sav'
    trained_model_withFullData_file_name = 'semiSynthetic_training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_posTau_model_withFullData.sav'
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
    
    # check data size, skip training if the numbe of samples is too small
    if total_sample_count<10:
        writeIntoLog(f"consists of {total_sample_count} samples which < 10.\n" + log_text, commDir , log_file_name)
        print(f"{commName} consists of {total_sample_count} samples which < 10.\n")
        return
    
    # compute the number of features in data. 
    # The first 3 columns of origianl dataset are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio)
    if oneside:
        n_feature = original_n_feature - 2 # removed 2 columns for one side parametrization
    else:
        n_feature = original_n_feature - 1 # removed 1 columns for two sides parametrization
    print(f"{commName} has total sample count: {total_sample_count}, and number of features: {n_feature}")

    # my NN model
    # training with myNN...
    outputs = myTrain(commIndex, questions_Data,n_feature, opt_forMiniBatchTrain, learning_rate, max_iter,commName,normFlag, regFlag, reg_alpha, modeltype, oneside, positiveTau)

    if outputs != None:
        train_losses, test_losses, test_accuracy, model_with_lowest_test_loss,lowest_test_loss_of_epoch,lowest_test_loss, batch_size, model_trainedWithFullData = outputs
    
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
        
        if learnTau: 
            if positiveTau:
                tau = np.exp(parm['tau'].item())
            else:
                tau = parm['tau'].item()
        else:
            tau = 1

        weights = parm['linear.weight'][0]
        
        result_text, coefs, nus, qs = resultFormat(weights, bias, tau, oneside, learnTau, ori_questionCount)
        writeIntoResult(text + f"\ntraining losses: {train_losses}\n"+ f"\ntesting losses: {test_losses}\n" + result_text, result_file_name)

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

        if learnTau: 
            if positiveTau:
                tau_withFullData = np.exp(parm['tau'].item())
            else:
                tau_withFullData = parm['tau'].item()
        else:
            tau_withFullData = 1

        weights_withFullData = parm['linear.weight'][0]
        
        result_text_withFullData, coefs_withFullData, nus_withFullData, qs_withFullData = resultFormat(weights_withFullData, bias_withFullData, tau_withFullData, oneside, learnTau, ori_questionCount)
        writeIntoResult(text + result_text_withFullData, result_withFullData_file_name)

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
        savePlot(plt, "semiSynthetic_training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting__posTau_Losses.png")

        return_trainSuccess_dict[commName] = {'dataShape':(total_sample_count, n_feature), 'testAcc':test_accuracy, 'testLoss':lowest_test_loss, 'epoch':lowest_test_loss_of_epoch,
                                           'coefs':coefs,'bias':bias,'tau':tau, 'nus':nus, 'qs':qs,
                                           'coefs_withFullData':coefs_withFullData,'bias_withFullData':bias_withFullData, 'tau_withFullData':tau_withFullData,'nus_withFullData':nus_withFullData, 'qs_withFullData':qs_withFullData}
        
        # save current return 
        log_text = ""
        commIndex = 0
        return_trainSuccess_normalDict = defaultdict()
        for commName, d in return_trainSuccess_dict.items():
            commIndex +=1
            log_text += f"{commIndex} {commName} data_shape {d['dataShape']} test_acc {d['testAcc']} test_loss {d['testLoss']} epoch {d['epoch']} coefs {d['coefs']} bias {d['bias']} tau {d['tau']}\n"
            return_trainSuccess_normalDict[commName] = {'dataShape':d['dataShape'], 'testAcc':d['testAcc'], 'testLoss':d['testLoss'], 'epoch':d['epoch'], 
                                                        'coefs':d['coefs'], 'bias':d['bias'], 'tau':d['tau'], 'nus':d['nus'],'qs':d['qs'],
                                                        'coefs_withFullData':d['coefs_withFullData'], 'bias_withFullData':d['bias_withFullData'], 'tau_withFullData':d['tau_withFullData'], 'nus_withFullData':d['nus_withFullData'],'qs_withFullData':d['qs_withFullData']}
        log_text += f"{len(return_trainSuccess_dict)} communitites were successfully trained\n"
        writeIntoLog(log_text, root_dir, "semiSynthetic_training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_result_trainSuccess_posTau_nonSplitted_Log.txt")
        os.chdir(root_dir) # go back to root directory
        with open('semiSynthetic_training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_return_trainSuccess_posTau_nonSplitted.dict', 'wb') as outputFile:
            pickle.dump(return_trainSuccess_normalDict, outputFile)

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

    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    return_trainSuccess_dict = manager.dict() # to save the used train mode (wholebatch or minibatch) of each community

    
    try:
        # test on comm "coffee.stackexchange" to debug
        myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1], return_trainSuccess_dict, root_dir)
        # test on comm "datascience.stackexchange" to debug
        # myFun(301,commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1], return_trainSuccess_dict, root_dir)
        # test on comm "webapps.stackexchange" to debug
        # myFun(305,commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1], return_trainSuccess_dict, root_dir)
        # test on comm "travel.stackexchange" to debug
        # myFun(319,commDir_sizes_sortedlist[319][0], commDir_sizes_sortedlist[319][1], return_trainSuccess_dict, root_dir)
    except Exception as e:
        print(e)
        sys.exit()
    """
     # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    commIndex = 0
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        if commName in splitted_comms: # skip splitted big communities
            print(f"{commName} was splitted. skip")
            continue

        try:
            p = mp.Process(target=myFun, args=(commIndex, commName,commDir, return_trainSuccess_dict, root_dir))
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
    # save return_trainSuccess_dict
    os.chdir(root_dir) # go back to root directory

    # convert and save the last return_trainSuccess
    log_text = ""
    commIndex = 0
    return_trainSuccess_normalDict = defaultdict()
    for commName, d in return_trainSuccess_dict.items():
        commIndex +=1
        log_text += f"{commIndex} {commName} data_shape {d['dataShape']} test_acc {d['testAcc']} test_loss {d['testLoss']} epoch {d['epoch']} coefs {d['coefs']} bias {d['bias']} tau {d['tau']}\n"
        return_trainSuccess_normalDict[commName] = {'dataShape':d['dataShape'], 'testAcc':d['testAcc'], 'testLoss':d['testLoss'], 'epoch':d['epoch'], 
                                                    'coefs':d['coefs'], 'bias':d['bias'], 'tau':d['tau'], 'nus':d['nus'],'qs':d['qs'],
                                                    'coefs_withFullData':d['coefs_withFullData'], 'bias_withFullData':d['bias_withFullData'], 'tau_withFullData':d['tau_withFullData'], 'nus_withFullData':d['nus_withFullData'],'qs_withFullData':d['qs_withFullData']}
    log_text += f"{len(return_trainSuccess_dict)} communitites were successfully trained\n"
    writeIntoLog(log_text, root_dir, "semiSynthetic_training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_result_trainSuccess_posTau_nonSplitted_Log.txt")
    os.chdir(root_dir) # go back to root directory
    with open('semiSynthetic_training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_return_trainSuccess_posTau_nonSplitted.dict', 'wb') as outputFile:
        pickle.dump(return_trainSuccess_normalDict, outputFile)

    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('semiSynthetic_train1_allTimeSteps (one side)  removed first real vote and temperal testing  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
