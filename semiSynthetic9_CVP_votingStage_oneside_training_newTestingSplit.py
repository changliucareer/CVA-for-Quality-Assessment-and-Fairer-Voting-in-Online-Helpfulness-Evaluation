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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.preprocessing import normalize
import random
from collections import Counter
from sklearn.model_selection import KFold
from scipy.special import kl_div
from scipy import stats 

def get_zScore(rankList):
    rankList = list(rankList)
    mean = np.mean(rankList)
    # standard deviation
    std = np.std(rankList, dtype=np.float64)
    zScores = []
    for r in rankList:
        if std==0:
            zScores.append(0)
        else:
            zScores.append(-(r-mean)/std) # flip the sign of z, since top ranked z is smaller and lower ranked z is bigger
    return zScores

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
                        X = np.concatenate((X[:,2:3], X[:,4:]), axis=1) # for one side parametrization, removed the first two columns and the Inversed rank term column
                    else: # two sides
                        X = np.concatenate((X[:,:2] , X[:,4:] ), axis=1) # for two sides parametrization, removed the third and fourth column

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
                
                else: # when having un-sorted data
                    questions_Data_part = pickle.load( inputFile)
                    for qid, tup in questions_Data_part.items():
                        X = tup[0].todense() # convert sparse matrix to np.array
                        y = tup[1]

                        cutIndex = int(len(y)*0.9) # for each question, its first 90% samples as training samples, the rest as testing samples
                        
                        # tailor the columns for different parametrization.
                        # the first 4 columns of X are (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio, IPW_rank)
                        if oneside:
                            X = np.concatenate((X[:,2:3], X[:,4:]), axis=1) # for one side parametrization, removed the first two columns and the Inversed rank term column
                        else: # two sides
                            X = np.concatenate((X[:,:2] , X[:,4:] ), axis=1) # for two sides parametrization, removed the third and fourth column

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
def myLogisticRegressionCV(Cs, solver,penalty, fit_intercept, max_iter, cv, commName, full_data, oneside, ori_questionCount, aid2prior_q, total_answersWithVotes_ids, qid2involvedAnswerList):
    # split training and testing dataset
    kf = KFold(n_splits=cv, shuffle=True, random_state=1)

    myC2Scores = defaultdict()

    for i, (train_indexList, test_indexList) in enumerate(kf.split(full_data[0].todense())):
        testingDataIndexListInFullData = test_indexList
        # prepare data
        iterable_train_dataset = MyIterableDataset(full_data,oneside, testingDataIndexListInFullData, 'train') 
        iterable_test_dataset = MyIterableDataset(full_data,oneside, testingDataIndexListInFullData, 'test') 
        # prepare data loader
        batch_size = full_data[0].shape[0] # original full data size
        my_train_dataloader = DataLoader(iterable_train_dataset, batch_size=batch_size) 
        my_test_dataloader = DataLoader(iterable_test_dataset, batch_size=batch_size) 

        # start cross validation training
        print(f"start cross validation training for {commName}, {i+1}th split...")
        # training dataset
        batchIndex = 0
        for batch in my_train_dataloader:
            batchIndex +=1
            if batchIndex > 1: # more than 1 batch, exception
                print(f"{commName} has more than one batch trainging data, exception!")
                return
            X = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
            y = batch[1]

        # testing dataset
        batchIndex = 0
        for batch in my_test_dataloader:
            batchIndex +=1
            if batchIndex > 1: # more than 1 batch, exception
                print(f"{commName} has more than one batch testing data, exception!")
                return
            X_test = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
            y_test = batch[1]

        # train with different C
        for myC in Cs:
            print(f"train cross validation for {commName}, with myC={myC} for {i+1}th split...")
            # set LogisticRegression model 
            lr = LogisticRegression(solver=solver,penalty=penalty, fit_intercept=fit_intercept, C=myC, max_iter=max_iter)
            lr.fit(X, y)

            # test with testing data to get accuracy score
            cur_test_accuracy = lr.score(X_test, y_test)
            cur_test_error = 1 - cur_test_accuracy

            # get learned quality estimation score
            # retrieve the learning result
            if fit_intercept: # withBias
                bias = lr.intercept_[0]
            else:
                bias = None

            weights = lr.coef_[0]
            
            result_text, coefs,nus, qs = resultFormat(weights, bias, oneside, ori_questionCount)
            assert len(qs) == len(total_answersWithVotes_ids)

            aid2learned_q = defaultdict()
            for i, tup in enumerate(total_answersWithVotes_ids):
                qid, aid = tup
                learned_q = qs[i]
                aid2learned_q[aid] = learned_q

            # convert q to rank_z_score
            aid2prior_q_rankZscore = defaultdict()
            aid2learned_q_rankZscore  = defaultdict()

            for qid, involved_answerList in qid2involvedAnswerList.items():
                # get prior q rank z score
                involved_aid2prior_q = {aid : aid2prior_q[aid] for aid in involved_answerList}
                involved_aid2rankBasedOnPriorQ = defaultdict()
                for i, kv in enumerate(sorted(involved_aid2prior_q.items(), key=lambda kv: (kv[1],kv[0]),reverse=True)) :
                    aid = kv[0]
                    involved_aid2rankBasedOnPriorQ[aid] = i+1 # rank
                
                zScores = get_zScore(involved_aid2rankBasedOnPriorQ.values())
                for i,aid in enumerate(involved_aid2rankBasedOnPriorQ.keys()):
                    aid2prior_q_rankZscore[aid] = zScores[i]

                # get learned q rank z score
                involved_aid2learned_q = {aid : aid2learned_q[aid] for aid in involved_answerList}
                involved_aid2rankBasedOnlearnedQ = defaultdict()
                for i, kv in enumerate(sorted(involved_aid2learned_q.items(), key=lambda kv: (kv[1],kv[0]),reverse=True)) :
                    aid = kv[0]
                    involved_aid2rankBasedOnlearnedQ[aid] = i+1 # rank
                
                zScores = get_zScore(involved_aid2rankBasedOnlearnedQ.values())
                for i,aid in enumerate(involved_aid2rankBasedOnlearnedQ.keys()):
                    aid2learned_q_rankZscore[aid] = zScores[i]

            combineAll = []
            for aid, learned_q_rankZscore in aid2learned_q_rankZscore.items():
                combineAll.append((aid2prior_q_rankZscore[aid], learned_q_rankZscore))

            KL_learnedQToPriorQ_rankZscore = sum([0 if math.isinf(kl) else kl for kl in kl_div([t[1] for t in combineAll], [t[0] for t in combineAll])])

            if myC in myC2Scores.keys():
                myC2Scores[myC]['error'].append(cur_test_error)
                myC2Scores[myC]['KL'].append(KL_learnedQToPriorQ_rankZscore)
            else:
                myC2Scores[myC] = {'error':[cur_test_error], 'KL':[KL_learnedQToPriorQ_rankZscore]}

    myC2avgScores = defaultdict()
    for myC, scores in myC2Scores.items():
        myC2avgScores[myC] = (mean(scores['KL']), mean(scores['error']))

    return myC2avgScores


###################################################################################################

def myTrain(commIndex, commName, commDir, ori_questionCount, full_data_files, testingDataIndexListInFullData, log_file_name, roundIndex, variation, reg_alpha1, reg_alpha2, try_reg_strengthList, CVPgenerated, aid2prior_q, total_answersWithVotes_ids, qid2involvedAnswerList, total_sample_count, original_n_feature,sampled_comms, Interaction, Quadratic):
    t0=time.time()

    #######################################################################
    ### training settings #######
    if Interaction: # new model+Interaction generated
        result_file_name = f"semiSynthetic9_CVP{variation}_newModelInteractionGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_results.txt"
        trained_model_file_name = f'semiSynthetic9_CVP{variation}_newModelInteractionGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_model.sav'
        trained_withSKLEARN_model_file_name = f'semiSynthetic9_CVP{variation}_newModelInteractionGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withSKLEARN_model.pkl'
        result_withSKLEARN_file_name = f"semiSynthetic9_CVP{variation}_newModelInteractionGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withSKLEARN_results.txt"
        trained_withLBFGS_model_file_name = f'semiSynthetic9_CVP{variation}_newModelInteractionGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withLBFGS_model.sav'
        result_withLBFGS_file_name = f"semiSynthetic9_CVP{variation}_newModelInteractionGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withLBFGS_results.txt"
        plot_file_name = f'semiSynthetic9_CVP{variation}_newModelInteractionGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_Losses.png'
    elif Quadratic: # new model+Quadratic generated
        result_file_name = f"semiSynthetic9_CVP{variation}_newModelQuadraticGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_results.txt"
        trained_model_file_name = f'semiSynthetic9_CVP{variation}_newModelQuadraticGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_model.sav'
        trained_withSKLEARN_model_file_name = f'semiSynthetic9_CVP{variation}_newModelQuadraticGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withSKLEARN_model.pkl'
        result_withSKLEARN_file_name = f"semiSynthetic9_CVP{variation}_newModelQuadraticGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withSKLEARN_results.txt"
        trained_withLBFGS_model_file_name = f'semiSynthetic9_CVP{variation}_newModelQuadraticGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withLBFGS_model.sav'
        result_withLBFGS_file_name = f"semiSynthetic9_CVP{variation}_newModelQuadraticGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withLBFGS_results.txt"
        plot_file_name = f'semiSynthetic9_CVP{variation}_newModelQuadraticGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_Losses.png'
    
    elif CVPgenerated:
        result_file_name = f"semiSynthetic9_CVP{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_results.txt"
        trained_model_file_name = f'semiSynthetic9_CVP{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_model.sav'
        trained_withSKLEARN_model_file_name = f'semiSynthetic9_CVP{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withSKLEARN_model.pkl'
        result_withSKLEARN_file_name = f"semiSynthetic9_CVP{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withSKLEARN_results.txt"
        trained_withLBFGS_model_file_name = f'semiSynthetic9_CVP{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withLBFGS_model.sav'
        result_withLBFGS_file_name = f"semiSynthetic9_CVP{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withLBFGS_results.txt"
        plot_file_name = f'semiSynthetic9_CVP{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_Losses.png'
    else: # new model generated
        result_file_name = f"semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_results.txt"
        trained_model_file_name = f'semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_model.sav'
        trained_withSKLEARN_model_file_name = f'semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withSKLEARN_model.pkl'
        result_withSKLEARN_file_name = f"semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withSKLEARN_results.txt"
        trained_withLBFGS_model_file_name = f'semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withLBFGS_model.sav'
        result_withLBFGS_file_name = f"semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_withLBFGS_results.txt"
        plot_file_name = f'semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_Losses.png'
    
    log_text = ""
    # choices of training settings
    normFlag = False
    regFlag = True   # if True, apply l2 regularization on all parameters; if False, don't apply regularization
    reg_alpha = reg_alpha2
    oneside = True   # if True, use one side parametrization; if False, use two side parametrization

    withBias = False # whether add bias term
   
    # select model type
    if withBias:
        modeltype='1layer_bias_withoutRankTerm' # equivalent to set tau as 1
    else: # without bias
        modeltype='1layer_withoutRankTerm'
    
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
        n_feature = original_n_feature - 3 # removed 3 columns for one side parametrization
    else:
        n_feature = original_n_feature - 2 # removed 2 columns for two sides parametrization
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
    print(f"first, try to train {commName} using SKLEARN...")
    failFlagSklearn = True
    try:
        batchIndex = 0
        for batch in my_full_dataloader:
            batchIndex +=1
            print(f"comm {commName}, preparing batch {batchIndex}...")
            X = torch.squeeze(batch[0]) # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
            y = batch[1]
    
            ### Cross validation ###################################################################################
                
            # if reg_alpha2 == try_reg_strengthList[0]: # only do once CV for the first reg_alpha option
            #     print(f"doing cross validation with sklearn for {commName} reg_alpha1:{reg_alpha1}...")
            #     myCs = [1/ (2*reg_alpha) for reg_alpha in try_reg_strengthList]
            #     lrcv = LogisticRegressionCV(Cs= myCs, solver='lbfgs',penalty='l2', fit_intercept=withBias, max_iter=max_iter, cv = 5)  # 5-fold
            #     lrcv.fit(np.asarray(X),y)
            #     # shape of lrcv.scores_ is (5, 27)
            #     CV_scores = list(np.mean(lrcv.scores_[1], axis=0))
            #     CV_bestC = lrcv.C_[0]
            #     CV_bestC_index = myCs.index(CV_bestC)
            #     CV_best_reg_alpha = try_reg_strengthList[CV_bestC_index]
            # else:
            #     CV_best_reg_alpha = None
            #     CV_scores = None
                
            # own cross validation
            if reg_alpha2 == try_reg_strengthList[0]: # only do once CV for the first reg_alpha option
                # print(f"doing cross validation with sklearn for {commName} reg_alpha1:{reg_alpha1}...")
                # myCs = [1/ (2*reg_alpha) for reg_alpha in try_reg_strengthList]
                # myC2avgScores = myLogisticRegressionCV(Cs= myCs, solver='lbfgs',penalty='l2', fit_intercept=withBias, max_iter=max_iter, cv = 5, commName = commName, full_data=full_data, oneside=oneside, ori_questionCount=ori_questionCount, aid2prior_q = aid2prior_q, total_answersWithVotes_ids = total_answersWithVotes_ids, qid2involvedAnswerList= qid2involvedAnswerList)
                # sortedCsAndScores = sorted(myC2avgScores.items(), key=lambda kv: (kv[1][0],kv[1][1]))  # smaller KL is better, smaller error is better
                # CV_bestC = sortedCsAndScores[0][0]
                # CV_bestC_index = myCs.index(CV_bestC)
                # CV_best_reg_alpha = try_reg_strengthList[CV_bestC_index]
                # CV_scores = [(tup[0],(1-tup[1])) for myC, tup in myC2avgScores.items()] # convert error rate back to accuracy
                CV_best_reg_alpha = None
                CV_scores = [(None,None)] * len(try_reg_strengthList)
            else:
                CV_best_reg_alpha = None
                CV_scores = None
            
            ##########################################################################################################
            # convert regularization strength reg_alpha to C
            print(f"training sklearn lr using current reg_alpha pair ({reg_alpha1},{reg_alpha2}) for {commName}...")
            myC = 1/ (2*reg_alpha) # when using penalty='l2' and solver='lbfgs'
            lr = LogisticRegression(solver='lbfgs',penalty='l2', fit_intercept=withBias, C=myC, max_iter=max_iter)
            
            lr.fit(X, y)
            # save lr model
            final_directory = os.path.join(commDir, r'trained_model_folder')
            with open(final_directory+'/'+ trained_withSKLEARN_model_file_name,'wb') as f:
                pickle.dump(lr,f)
            print(f"for {commName} reg_alpha pair ({reg_alpha1},{reg_alpha2}) model_withSKLEARN saved.")
            try:
                # compute conformity
                probOddList = []
                y_pred = lr.predict_proba(X) # shape of y_pred is (sampleCount, 2), for each sample [prob of negVote, prob of posVote]
                # multiply each odd with probOddProduct
                for i,row in enumerate(X): # the shape of row is (1,n_feature)
                    if row[0] >= 0: # when n_pos >= n_neg
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

            weights_sklearn = lr.coef_[0]
            
            result_text_sklearn, coefs_sklearn,nus_sklearn, qs_sklearn = resultFormat(weights_sklearn, bias_sklearn, oneside, ori_questionCount)

            text = f"SKLEARN training ==================================\n"
            text += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '\n'
            text += f"Train oneside:{oneside}, normalize:{normFlag},priorRegAlpha:{reg_alpha1}, regFlag:{regFlag}, reg_alpha:{reg_alpha}, withBias:{withBias}\n"
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
        conformity_sklearn = None
        CV_best_reg_alpha = None
        CV_scores = [(None,None)] * len(try_reg_strengthList)
        failFlagSklearn = True

    
    ####################################################################################################
    failFlagLBFGS = True
    coefs_lbfgs = None
    bias_lbfgs = None
    qs_lbfgs = None
    nus_lbfgs = None
    conformity_lbfgs = None
    if failFlagSklearn:
        print(f"try to train using cuda and LBFGS reg_alpha pair ({reg_alpha1},{reg_alpha2}) for {commName}...")
        try:
            # check gpu count
            cuda_count = torch.cuda.device_count()
            # assign one of gpu as device
            d = (commIndex) % cuda_count
            device = torch.device('cuda:'+str(d) if torch.cuda.is_available() else 'cpu')
            print(f"comm {commName}, Start to train NN model with LBFGS... on device: {device}")

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
            loss_fn = torch.nn.BCELoss(size_average=True, reduction='mean')

            print(f"comm {commName}, Start to train NN model with LBFGS and batch_size {total_sample_count}... on device: {device}")
            my_full_dataloader = DataLoader(iterable_full_dataset, batch_size=total_sample_count)
            model.to(device)

            batchIndex = 0
            for batch in my_full_dataloader:
                batchIndex +=1
                X = torch.squeeze(batch[0]).float() # reshape from [n_sample, 1, n_feature] to [n_sample, n_feature]
                y = batch[1].float() # change tensor data type from int to float32
                
                X = X.to(device)
                y = y.to(device)

                # print(f"current X type :{X.type()}, y type:{y.type()}")

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
                    l2_lambda = torch.tensor(reg_alpha).to(device)
                    l2_reg = torch.tensor(0.).to(device) # initialize regular term as zero
                    
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
                saveModel(model,trained_withLBFGS_model_file_name)
                print(f"for {commName} reg_alpha pair ({reg_alpha1},{reg_alpha2}) model trained_withlbfgs saved.")

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
                
                result_text, coefs_lbfgs, nus_lbfgs, qs_lbfgs = resultFormat(weights_lbfgs, bias_lbfgs, oneside, ori_questionCount)
                writeIntoResult(text +  result_text, result_withLBFGS_file_name)
                
                elapsed = format_time(time.time() - t0)
                
                log_text += "Elapsed: {:}\n".format(elapsed)

                writeIntoLog(text + log_text, commDir , log_file_name)
                failFlagLBFGS = False

                # delete model and release memory of cuda
                model=model.cpu()
                del model
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
    """
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
    
    return total_sample_count, n_feature, train_losses, test_losses, lowest_test_loss_of_epoch, lowest_test_loss, test_accuracy, coefs, bias, nus, qs, conformity, coefs_sklearn, bias_sklearn, nus_sklearn, qs_sklearn, conformity_sklearn, coefs_lbfgs, bias_lbfgs, nus_lbfgs, qs_lbfgs, conformity_lbfgs, CV_best_reg_alpha, CV_scores


def myFun(commIndex, commName, commDir, root_dir, roundIndex, variation, selected_reg_strengthList, try_reg_strengthList, CVPgenerated,sampled_comms, Interaction, Quadratic):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    log_file_name = "semiSynthetic9_CVP_votingStage_oneside_training_newTestingSplit_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())
    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # train using the data and priors from selected reg_alpha options
    for reg_alpha1 in selected_reg_strengthList:
        if Interaction:
            with open(intermediate_directory+'/'+f'simulated_data_byNewModelInteraction{variation}_round{roundIndex}_regAlpha({reg_alpha1}).dict', 'rb') as inputFile:
                loadedFile = pickle.load( inputFile)
        elif Quadratic:
            with open(intermediate_directory+'/'+f'simulated_data_byNewModelQuadratic{variation}_round{roundIndex}_regAlpha({reg_alpha1}).dict', 'rb') as inputFile:
                loadedFile = pickle.load( inputFile)
        elif CVPgenerated:
            with open(intermediate_directory+'/'+f'simulated_data_byCVP{variation}_round{roundIndex}_regAlpha({reg_alpha1}).dict', 'rb') as inputFile:
                loadedFile = pickle.load( inputFile)
        else: # new model generated
            with open(intermediate_directory+'/'+f'simulated_data_byNewModel{variation}_round{roundIndex}_regAlpha({reg_alpha1}).dict', 'rb') as inputFile:
                loadedFile = pickle.load( inputFile)

        simulatedQuestions = loadedFile[0]
        ori_questionCount = len(simulatedQuestions)
        
        # get full data file list
        if Interaction: # new model+Interaction generated
            full_data_files = [intermediate_directory+f"/semiSynthetic_newModelInteraction_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_datasets_sorted_removeFirstRealVote.dict"]
            if not os.path.exists(full_data_files[0]):
                print(f"{commName} hasn't done temperalOrderSorting yet, load un-sorted dataset")
                full_data_files = [intermediate_directory+f"/semiSynthetic_newModelInteraction_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_datasets_forEachQuestion_removeFirstRealVote.dict"]
        elif Quadratic: # new model+Quadratic generated
            full_data_files = [intermediate_directory+f"/semiSynthetic_newModelQuadratic_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_datasets_sorted_removeFirstRealVote.dict"]
            if not os.path.exists(full_data_files[0]):
                print(f"{commName} hasn't done temperalOrderSorting yet, load un-sorted dataset")
                full_data_files = [intermediate_directory+f"/semiSynthetic_newModelQuadratic_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_datasets_forEachQuestion_removeFirstRealVote.dict"]

        elif CVPgenerated:
            # with open(intermediate_directory+f"/semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_datasets_sorted_removeFirstRealVote.dict", 'rb') as inputFile:
            #     full_data = pickle.load( inputFile)
            full_data_files = [intermediate_directory+f"/semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_datasets_sorted_removeFirstRealVote.dict"]

        else: # new model generated
            # with open(intermediate_directory+f"/semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_datasets_sorted_removeFirstRealVote.dict", 'rb') as inputFile:
            #     full_data = pickle.load( inputFile)
            full_data_files = [intermediate_directory+f"/semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_datasets_sorted_removeFirstRealVote.dict"]
            if not os.path.exists(full_data_files[0]):
                print(f"{commName} hasn't done temperalOrderSorting yet, load un-sorted dataset")
                full_data_files = [intermediate_directory+f"/semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_datasets_forEachQuestion_removeFirstRealVote.dict"]

        # get total_sample_count and original_n_feature
        total_sample_count = 0
        original_n_feature = None
        if 'forEachQuestion' in full_data_files[0]: # un-sorted dataset
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
        else: # sorted dataset
            for f in full_data_files:
                with open(f, 'rb') as inputFile:
                    full_data = pickle.load( inputFile)
                    total_sample_count += len(full_data[1])
                    if original_n_feature == None:
                        original_n_feature = full_data[0].todense().shape[1]
                    full_data = [] # clear to save memory


        # load testing data's sortingBase
        # get testing data raw index in full data
        try: # when having sorted testing data
            if Interaction:
                with open(intermediate_directory+f"/semiSynthetic8_newModelInteraction{variation}_round{roundIndex}_regAlpha({reg_alpha1})_outputs.dict", 'rb') as inputFile:
                    qid2TestingSortingBaseList, _ = pickle.load( inputFile)
            elif Quadratic:
                with open(intermediate_directory+f"/semiSynthetic8_newModelQuadratic{variation}_round{roundIndex}_regAlpha({reg_alpha1})_outputs.dict", 'rb') as inputFile:
                    qid2TestingSortingBaseList, _ = pickle.load( inputFile)
            elif CVPgenerated:
                with open(intermediate_directory+f"/semiSynthetic8{variation}_round{roundIndex}_regAlpha({reg_alpha1})_outputs.dict", 'rb') as inputFile:
                    qid2TestingSortingBaseList, _ = pickle.load( inputFile)
            else: # new model generated
                with open(intermediate_directory+f"/semiSynthetic8_newModel{variation}_round{roundIndex}_regAlpha({reg_alpha1})_outputs.dict", 'rb') as inputFile:
                    qid2TestingSortingBaseList, _ = pickle.load( inputFile)

            sortingBaseListOfTestingData = []
            for qid, testingSortingBaseList in qid2TestingSortingBaseList.items():
                sortingBaseListOfTestingData.extend(testingSortingBaseList)

            if Interaction:
                with open(intermediate_directory+f"/semiSynthetic_newModelInteraction_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_sorted_qidAndSampleIndexAndSortingBase.dict", 'rb') as inputFile:
                    sorted_qidAndSampleIndex = pickle.load( inputFile)
            elif Quadratic:
                with open(intermediate_directory+f"/semiSynthetic_newModelQuadratic_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_sorted_qidAndSampleIndexAndSortingBase.dict", 'rb') as inputFile:
                    sorted_qidAndSampleIndex = pickle.load( inputFile)
            elif CVPgenerated:
                with open(intermediate_directory+f"/semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha1})_sorted_qidAndSampleIndexAndSortingBase.dict", 'rb') as inputFile:
                    sorted_qidAndSampleIndex = pickle.load( inputFile)
            else:
                with open(intermediate_directory+f"/semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_sorted_qidAndSampleIndexAndSortingBase.dict", 'rb') as inputFile:
                    sorted_qidAndSampleIndex = pickle.load( inputFile)

            testingDataIndexListInFullData = [i for i, tup in enumerate(sorted_qidAndSampleIndex) if tup[2] in sortingBaseListOfTestingData]
        except: # has no sorted testing data
            testingDataIndexListInFullData = []

        # get prior qs
        # if roundIndex in [16, 17]:
        #     with open(intermediate_directory+'/'+f'simulated_data_byCVP{variation}_round{roundIndex}_regAlpha({reg_alpha1}).dict', 'rb') as inputFile:
        #         loadedFile = pickle.load( inputFile)
        # elif roundIndex in [19,20, 21]:
        #     with open(intermediate_directory+'/'+f'simulated_data_byCVP_round{roundIndex}_regAlpha({reg_alpha1}).dict', 'rb') as inputFile:
        #         loadedFile = pickle.load( inputFile)
        # else:
        #     with open(intermediate_directory+'/'+f'simulated_data_byCVP{variation}_round{roundIndex}.dict', 'rb') as inputFile:
        #         loadedFile = pickle.load( inputFile)
        # simulatedQuestions = loadedFile[0]
        
        # a dict map aid to prior_q
        aid2prior_q = defaultdict()
        for qid, d in simulatedQuestions.items():
            answerQualityList = d['answerQualityList']
            answerList = d['answerList']
            assert len(answerList) == len(answerQualityList)
            for ai, tup in enumerate(answerList):
                aid = tup[0]
                aid2prior_q[aid] = answerQualityList[ai]
        
        # a dict map aid to learned_q
        if Interaction: # new model+Interaction generated
            if roundIndex in [2,3,4]:
                with open(intermediate_directory+'/'+f'semiSynthetic_newModelInteraction_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
                    total_answersWithVotes_indice = pickle.load( inputFile)
        elif Quadratic: # new model+Quadratic generated
            if roundIndex in [2,3,4]:
                with open(intermediate_directory+'/'+f'semiSynthetic_newModelQuadratic_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
                    total_answersWithVotes_indice = pickle.load( inputFile)
        elif CVPgenerated:
            if roundIndex in [16,17,19,20, 21]:
                with open(intermediate_directory+'/'+f'semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
                    total_answersWithVotes_indice = pickle.load( inputFile)
            else:
                with open(intermediate_directory+'/'+f'semiSynthetic{variation}_round{roundIndex}_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
                    total_answersWithVotes_indice = pickle.load( inputFile)
        else: # new model generated
            if roundIndex in [2,3,4]:
                with open(intermediate_directory+'/'+f'semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
                    total_answersWithVotes_indice = pickle.load( inputFile)

        total_answersWithVotes_ids = []
        answer2parentQ = defaultdict()

        for i,(qid, ai) in enumerate(total_answersWithVotes_indice):
            answerList = simulatedQuestions[qid]['answerList']
            aid = answerList[ai][0]
            total_answersWithVotes_ids.append((qid,aid))
            if aid not in answer2parentQ.keys():
                answer2parentQ[aid] = qid

        # get qid2involvedAnswerList 
        qid2involvedAnswerList = defaultdict()
        for qid, d in simulatedQuestions.items():
            answerList = [tup[0] for tup in d['answerList']]
            involved_answerList = set(answerList).intersection(set(answer2parentQ.keys()))
            if len(involved_answerList) >0:
                qid2involvedAnswerList[qid] = involved_answerList

        simulatedQuestions.clear() # clear this to same memory
        loadedFile = None # clear to save memory
        
        # training using different reg_alpha2 options
        CV_scores = None
        for reg_alpha2 in try_reg_strengthList:
            
            # check whether already done this step, skip
            if Interaction: # new model+Interaction generated
                resultFiles = [f'semiSynthetic9_CVP{variation}_newModelInteractionGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict']
            elif Quadratic: # new model+Quadratic generated
                resultFiles = [f'semiSynthetic9_CVP{variation}_newModelQuadraticGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict']
            elif CVPgenerated:
                resultFiles = [f'semiSynthetic9_CVP{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict']
            else: # new model generated
                resultFiles = [f'semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict']
            resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
            if os.path.exists(resultFiles[0]):
                # target date
                target_date = datetime.datetime(2025, 3, 27)
                # file last modification time
                timestamp = os.path.getmtime(resultFiles[0])
                # convert timestamp into DateTime object
                datestamp = datetime.datetime.fromtimestamp(timestamp)
                print(f'{commName} Modified Date/Time:{datestamp}')
                if datestamp >= target_date:
                    print(f"{commName} ({reg_alpha1}, {reg_alpha2}) has already done this step.")
                    continue

            # training 
            output = myTrain(commIndex, commName, commDir, ori_questionCount, full_data_files, testingDataIndexListInFullData, log_file_name, roundIndex, variation, reg_alpha1, reg_alpha2, try_reg_strengthList, CVPgenerated, aid2prior_q, total_answersWithVotes_ids, qid2involvedAnswerList, total_sample_count, original_n_feature,sampled_comms, Interaction, Quadratic)
            total_sample_count, n_feature, train_losses, test_losses, lowest_test_loss_of_epoch, lowest_test_loss, test_accuracy, coefs, bias, nus, qs, conformity, coefs_sklearn, bias_sklearn, nus_sklearn, qs_sklearn, conformity_sklearn, coefs_lbfgs, bias_lbfgs, nus_lbfgs, qs_lbfgs, conformity_lbfgs, CV_best_reg_alpha, cur_CV_scores = output
            
            if cur_CV_scores != None:
                CV_scores = cur_CV_scores
            if CV_scores == None:
                CV_scores =[(None,None)] * len(try_reg_strengthList)

            if (coefs_lbfgs != None) or (coefs_sklearn != None):
                return_trainSuccess_dict = defaultdict()
                return_trainSuccess_dict[commName] = {'dataShape':(total_sample_count, n_feature), 'trainLosses':train_losses, 'testLosses':test_losses, 'epoch':lowest_test_loss_of_epoch, 'lowest_test_loss':lowest_test_loss, 'testAcc':test_accuracy,
                                                    'coefs':coefs,'bias':bias,'nus':nus, 'qs':qs, 'conformity':conformity,
                                                    'coefs_sklearn':coefs_sklearn, 'nus_sklearn':nus_sklearn, 'qs_sklearn':qs_sklearn,
                                                    'conformity_sklearn':conformity_sklearn,
                                                    'coefs_lbfgs':coefs_lbfgs, 'nus_lbfgs':nus_lbfgs, 'qs_lbfgs':qs_lbfgs,
                                                    'conformity_lbfgs':conformity_lbfgs,
                                                    'CV_best_reg_alpha':CV_best_reg_alpha, 'CV_scores':CV_scores}
                
                # save return dict
                if Interaction: # new model+Interaction generated
                    with open(intermediate_directory+f"/semiSynthetic9_CVP{variation}_newModelInteractionGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict", 'wb') as outputFile:
                        pickle.dump(return_trainSuccess_dict, outputFile)
                elif Quadratic: # new model+Quadratic generated
                    with open(intermediate_directory+f"/semiSynthetic9_CVP{variation}_newModelQuadraticGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict", 'wb') as outputFile:
                        pickle.dump(return_trainSuccess_dict, outputFile)
                elif CVPgenerated:
                    with open(intermediate_directory+f"/semiSynthetic9_CVP{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict", 'wb') as outputFile:
                        pickle.dump(return_trainSuccess_dict, outputFile)
                else: # new model generated
                    with open(intermediate_directory+f"/semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict", 'wb') as outputFile:
                        pickle.dump(return_trainSuccess_dict, outputFile)
                    print( f"saved return_trainSuccess_dict of {commName}.")

            # # save csv
            if Interaction: # new model+Interaction generated
                with open(root_dir +'/'+f'allComm_semiSynthetic9_CVP{variation}_newModelInteractionGenerated_round{roundIndex}_votingStage_training_results.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    if (coefs_sklearn != None) and (coefs_lbfgs != None):
                        writer.writerow( [commName,total_sample_count, reg_alpha1, reg_alpha2, coefs_sklearn[0], coefs_lbfgs[0],coefs, mean(nus_sklearn), mean(nus_lbfgs),nus, mean(qs_sklearn), mean(qs_lbfgs),qs, CV_scores[try_reg_strengthList.index(reg_alpha2)][0],CV_scores[try_reg_strengthList.index(reg_alpha2)][1], CV_best_reg_alpha])
                    elif coefs_sklearn !=None:
                        writer.writerow( [commName,total_sample_count, reg_alpha1, reg_alpha2, coefs_sklearn[0], coefs_lbfgs,coefs, mean(nus_sklearn), nus_lbfgs,nus, mean(qs_sklearn), qs_lbfgs,qs, CV_scores[try_reg_strengthList.index(reg_alpha2)][0],CV_scores[try_reg_strengthList.index(reg_alpha2)][1], CV_best_reg_alpha])
            elif Quadratic: # new model+Quadratic generated
                with open(root_dir +'/'+f'allComm_semiSynthetic9_CVP{variation}_newModelQuadraticGenerated_round{roundIndex}_votingStage_training_results.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    if (coefs_sklearn != None) and (coefs_lbfgs != None):
                        writer.writerow( [commName,total_sample_count, reg_alpha1, reg_alpha2, coefs_sklearn[0], coefs_lbfgs[0],coefs, mean(nus_sklearn), mean(nus_lbfgs),nus, mean(qs_sklearn), mean(qs_lbfgs),qs, CV_scores[try_reg_strengthList.index(reg_alpha2)][0],CV_scores[try_reg_strengthList.index(reg_alpha2)][1], CV_best_reg_alpha])
                    elif coefs_sklearn !=None:
                        writer.writerow( [commName,total_sample_count, reg_alpha1, reg_alpha2, coefs_sklearn[0], coefs_lbfgs,coefs, mean(nus_sklearn), nus_lbfgs,nus, mean(qs_sklearn), qs_lbfgs,qs, CV_scores[try_reg_strengthList.index(reg_alpha2)][0],CV_scores[try_reg_strengthList.index(reg_alpha2)][1], CV_best_reg_alpha])
            elif CVPgenerated:
                with open(root_dir +'/'+f'allComm_semiSynthetic9_CVP{variation}_round{roundIndex}_votingStage_training_results.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    if (coefs_sklearn != None) and (coefs_lbfgs != None):
                        writer.writerow( [commName,total_sample_count, reg_alpha1, reg_alpha2, coefs_sklearn[0], coefs_lbfgs[0],coefs, mean(nus_sklearn), mean(nus_lbfgs),nus, mean(qs_sklearn), mean(qs_lbfgs),qs, CV_scores[try_reg_strengthList.index(reg_alpha2)][0],CV_scores[try_reg_strengthList.index(reg_alpha2)][1], CV_best_reg_alpha])
                    elif coefs_sklearn !=None:
                        writer.writerow( [commName,total_sample_count, reg_alpha1, reg_alpha2, coefs_sklearn[0], coefs_lbfgs,coefs, mean(nus_sklearn), nus_lbfgs,nus, mean(qs_sklearn), qs_lbfgs,qs, CV_scores[try_reg_strengthList.index(reg_alpha2)][0],CV_scores[try_reg_strengthList.index(reg_alpha2)][1], CV_best_reg_alpha])
            else: # new model generated
                with open(root_dir +'/'+f'allComm_semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_votingStage_training_results.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    if (coefs_sklearn != None) and (coefs_lbfgs != None):
                        writer.writerow( [commName,total_sample_count, reg_alpha1, reg_alpha2, coefs_sklearn[0], coefs_lbfgs[0],coefs, mean(nus_sklearn), mean(nus_lbfgs),nus, mean(qs_sklearn), mean(qs_lbfgs),qs, CV_scores[try_reg_strengthList.index(reg_alpha2)][0],CV_scores[try_reg_strengthList.index(reg_alpha2)][1], CV_best_reg_alpha])
                    elif coefs_sklearn !=None:
                        writer.writerow( [commName,total_sample_count, reg_alpha1, reg_alpha2, coefs_sklearn[0], coefs_lbfgs,coefs, mean(nus_sklearn), nus_lbfgs,nus, mean(qs_sklearn), qs_lbfgs,qs, CV_scores[try_reg_strengthList.index(reg_alpha2)][0],CV_scores[try_reg_strengthList.index(reg_alpha2)][1], CV_best_reg_alpha])

def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    ### using CVP to generate semi-synthetic dataset 
    # CVPgenerated = True
        # Interaction = False
    # Quadratic = False
    # # roundIndex = 19 ## multiple question multiple answer, original total event count, fix tau = 1, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    # roundIndex = 20 ## multiple question multiple answer, original total event count, learn tau, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    # if roundIndex in [19, 20]:
    #     variation = ''
    # selected_reg_strengthList = [500, 700]
    
    # roundIndex = 21 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # commName2selected_reg_strengthList = {'cstheory.stackexchange':[800, 900, 1000],
    #                                       'stackoverflow':[1000],
    #                                       'unix.meta.stackexchange':[60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
    #                                       'politics.stackexchange':[900,1000],
    #                                       '3dprinting.stackexchange':[30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700],
    #                                       'latin.stackexchange':[50, 60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
    #                                       'meta.askubuntu':[70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
    #                                       'lifehacks.stackexchange':[400, 500, 600, 700, 800,900,1000]}
    # variation = ''

    # ### using new Model to generate semi-synthetic dataset 
    # CVPgenerated = False
    # Interaction = False
    # Quadratic = False
    # # roundIndex = 1 ## multiple question multiple answer, original total event count, fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # # roundIndex = 2 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # # roundIndex = 3 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and DOUBLED beta (for different regularization strength) selected_reg_strengthList of each comm
    # roundIndex = 4 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and TRIPLED beta (for different regularization strength) selected_reg_strengthList of each comm

    ### using new Model+ Interaction or + Quadratic rank term to generate semi-synthetic dataset statistics
    CVPgenerated = False
    Interaction = False
    Quadratic = True
    roundIndex = 2 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # roundIndex = 3 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and DOUBLED beta (for different regularization strength) selected_reg_strengthList of each comm
    # roundIndex = 4 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and TRIPLED beta (for different regularization strength) selected_reg_strengthList of each comm

    if Interaction:
        commName2selected_reg_strengthList = {
                                          'cstheory.stackexchange':[700, 800, 900, 1000],
                                          'stackoverflow':[900,1000],
                                          'politics.stackexchange':[200, 300, 400, 500, 600, 700, 800,900,1000],
                                          'math.meta.stackexchange':[100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                          'mathoverflow.net':[400,500],
                                          'askubuntu':[400,500],
                                          'philosophy.stackexchange':[50, 60, 70,80,90,100, 200, 300, 400, 500],
                                          'codegolf.meta.stackexchange':[200, 300, 400, 500, 600, 700, 800, 900,1000]
                                          }
    elif Quadratic:
        commName2selected_reg_strengthList = {
                                          'cstheory.stackexchange':[400, 500, 600, 700],
                                          'stackoverflow':[900,1000],
                                          'politics.stackexchange':[50,60,70,80,90,100],
                                          'math.meta.stackexchange':[70, 90, 100],
                                          'mathoverflow.net':[400,500],
                                          'askubuntu':[500,600],
                                          'philosophy.stackexchange':[3,4,5,6,7,8,9,10,20],
                                          'codegolf.meta.stackexchange':[100, 200, 300, 400, 500, 600, 700, 800, 900,1000]
                                          }
    else: # CVP or NewModel generated
        commName2selected_reg_strengthList = {
                                            #   '3dprinting.stackexchange':[20, 30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700],
                                            #   'latin.stackexchange':[40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                            #   'meta.askubuntu':[20, 30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                            #   'lifehacks.stackexchange':[300, 400, 500, 600, 700, 800, 900],
                                            #   'cstheory.stackexchange':[700, 800, 900, 1000],
                                            #   'stackoverflow':[1000],
                                            #   'unix.meta.stackexchange':[30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                            #   'politics.stackexchange':[200, 300, 400, 500, 600, 700, 800,900,1000],
                                            #   'math.meta.stackexchange':[30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000]
                                            #   'mathoverflow.net':[500,600],
                                            #   'mathematica.stackexchange':[80,90,100],
                                            #   'askubuntu':[300,400,500,600],
                                            #   'philosophy.stackexchange':[60, 70,80,90,100, 200, 300, 400, 500, 600],
                                                'codegolf.meta.stackexchange':[100, 200, 300, 400, 500, 600, 700, 800, 900,1000]
                                            }
    variation = '_fixedTau'

    try_reg_strengthList = [0.1,0.2, 0.3,0.4, 0.5,0.6, 0.7,0.8,0.9,
                            1, 2, 3,4,5, 6, 7,8,9,
                            10,20, 30,40,50,60, 70,80,90,
                            100, 200, 300, 400, 500, 600, 700, 800, 900,
                            1000]
    
    # for sampled comms
    sampled_comms = ['academia.stackexchange','askubuntu',
                      'english.stackexchange','math.stackexchange','mathoverflow.net',
                      'meta.stackexchange','meta.stackoverflow','serverfault',
                      'softwareengineering.stackexchange','superuser','unix.stackexchange',
                      'worldbuilding.stackexchange','physics.stackexchange','electronics.stackexchange',
                      'codegolf.stackexchange','workplace.stackexchange']
    """
    # # save csv
    if Interaction:
        with open(root_dir +'/'+f'allComm_semiSynthetic9_CVP{variation}_newModelInteractionGenerated_round{roundIndex}_votingStage_training_results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow( ["commName","totalSampleCount", "reg_alpha1","reg_alpha2", "lambda_sklearn", "lambda_torchLBFGS","lambda_torchSGD", "nu_sklearn_mean", "nu_torchLBFGS_mean","nu_torchSGD_mean","q_sklearn_mean", "q_torchLBFGS_mean","q_torchSGD_mean", "cross validation learnedQ rankZscore to priorQ rankZscore average KL score", "cross validation voting prediction average accuracy score", "best cross validation reg_alpha"])
   
    elif Quadratic:
        with open(root_dir +'/'+f'allComm_semiSynthetic9_CVP{variation}_newModelQuadraticGenerated_round{roundIndex}_votingStage_training_results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow( ["commName","totalSampleCount", "reg_alpha1","reg_alpha2", "lambda_sklearn", "lambda_torchLBFGS","lambda_torchSGD", "nu_sklearn_mean", "nu_torchLBFGS_mean","nu_torchSGD_mean","q_sklearn_mean", "q_torchLBFGS_mean","q_torchSGD_mean", "cross validation learnedQ rankZscore to priorQ rankZscore average KL score", "cross validation voting prediction average accuracy score", "best cross validation reg_alpha"])
   
    elif CVPgenerated:
        with open(root_dir +'/'+f'allComm_semiSynthetic9_CVP{variation}_round{roundIndex}_votingStage_training_results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow( ["commName","totalSampleCount", "reg_alpha1","reg_alpha2", "lambda_sklearn", "lambda_torchLBFGS","lambda_torchSGD", "nu_sklearn_mean", "nu_torchLBFGS_mean","nu_torchSGD_mean","q_sklearn_mean", "q_torchLBFGS_mean","q_torchSGD_mean", "cross validation learnedQ rankZscore to priorQ rankZscore average KL score", "cross validation voting prediction average accuracy score", "best cross validation reg_alpha"])
    else: # new model generated
        with open(root_dir +'/'+f'allComm_semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_votingStage_training_results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow( ["commName","totalSampleCount", "reg_alpha1","reg_alpha2", "lambda_sklearn", "lambda_torchLBFGS","lambda_torchSGD", "nu_sklearn_mean", "nu_torchLBFGS_mean","nu_torchSGD_mean","q_sklearn_mean", "q_torchLBFGS_mean","q_torchSGD_mean", "cross validation learnedQ rankZscore to priorQ rankZscore average KL score", "cross validation voting prediction average accuracy score","best cross validation reg_alpha2"])

    
    try:
        # # test on comm "3dprinting.stackexchange" to debug
        # myFun(0,commDir_sizes_sortedlist[227][0], commDir_sizes_sortedlist[227][1], root_dir, roundIndex, variation, commName2selected_reg_strengthList[commDir_sizes_sortedlist[227][0]], try_reg_strengthList, CVPgenerated)
        # # test on comm "latin.stackexchange" to debug
        # myFun(1,commDir_sizes_sortedlist[229][0], commDir_sizes_sortedlist[229][1], root_dir, roundIndex, variation, commName2selected_reg_strengthList[commDir_sizes_sortedlist[229][0]], try_reg_strengthList, CVPgenerated)
        # # test on comm "meta.askubuntu" to debug
        # myFun(3,commDir_sizes_sortedlist[231][0], commDir_sizes_sortedlist[231][1], root_dir, roundIndex, variation, commName2selected_reg_strengthList[commDir_sizes_sortedlist[231][0]], try_reg_strengthList, CVPgenerated)
        # # test on comm "lifehacks.stackexchange" to debug
        # myFun(2,commDir_sizes_sortedlist[233][0], commDir_sizes_sortedlist[233][1], root_dir, roundIndex, variation, commName2selected_reg_strengthList[commDir_sizes_sortedlist[233][0]], try_reg_strengthList, CVPgenerated)
        # test on comm "cstheory.stackexchange" to debug
        # myFun(4,commDir_sizes_sortedlist[256][0], commDir_sizes_sortedlist[256][1], root_dir, roundIndex, variation, commName2selected_reg_strengthList[commDir_sizes_sortedlist[256][0]], try_reg_strengthList, CVPgenerated)
        # test on comm "stackoverflow" to debug
        # myFun(5,commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1], root_dir, roundIndex, variation, commName2selected_reg_strengthList[commDir_sizes_sortedlist[359][0]], try_reg_strengthList, CVPgenerated)
        # test on comm "unix.meta.stackexchange" to debug
        # myFun(6,commDir_sizes_sortedlist[173][0], commDir_sizes_sortedlist[173][1], root_dir, roundIndex, variation, commName2selected_reg_strengthList[commDir_sizes_sortedlist[173][0]], try_reg_strengthList, CVPgenerated)
        # # test on comm "politics.stackexchange" to debug
        # myFun(7, commDir_sizes_sortedlist[283][0], commDir_sizes_sortedlist[283][1], root_dir, roundIndex, variation, commName2selected_reg_strengthList[commDir_sizes_sortedlist[283][0]], try_reg_strengthList, CVPgenerated)
    
    except Exception as e:
        print(e)
        sys.exit()
    """
    
     # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    selected_comms = [
                    "cstheory.stackexchange", 
                     "stackoverflow", 
                     "politics.stackexchange",
                     "codegolf.meta.stackexchange",
                     "math.meta.stackexchange",
                     "mathoverflow.net",
                     "askubuntu",
                     "philosophy.stackexchange",
                     ]

    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        if commName not in selected_comms: # skip non selected communities
            print(f"{commName} was not selected,skip")
            continue

        # if commName in splitted_comms: # skip splitted big communities
        #     print(f"{commName} was splitted,skip")
        #     continue

        selected_reg_strengthList = commName2selected_reg_strengthList[commName]

        try:
            p = mp.Process(target=myFun, args=(commIndex, commName,commDir, root_dir, roundIndex, variation, selected_reg_strengthList, try_reg_strengthList, CVPgenerated,sampled_comms, Interaction, Quadratic))
            p.start()
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
    print('semiSynthetic9_CVP_votingStage_training  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
