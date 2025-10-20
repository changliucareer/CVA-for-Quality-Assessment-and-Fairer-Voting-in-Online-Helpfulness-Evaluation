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
# from scipy.sparse import csr_matrix, lil_matrix
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
import math
import scipy.stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
from scipy import stats
import json

def curveFitOLS(X,y):
    z = np.polyfit(X,y, deg=1, full=True)
    p = np.poly1d(z[0])
    return z,p

def robust_HuberT(X,y):
    # robust regression 
    X = pd.DataFrame(X)
    y = pd.Series(y)
    # The predictor variables should include a constant term.
    X = sm.add_constant(X)
    # Fit the robust regression model
    Huber_robust_model = sm.RLM(y, X, M=sm.robust.norms.HuberT()) # The tuning constant for Huber’s t function. The default value is 1.345.
    Huber_robust_results = Huber_robust_model.fit()
    Huber_robust_param = Huber_robust_results.params[0]
    # pred_y = Huber_robust_model.predict(Huber_robust_results.params,X)
    pred_y = Huber_robust_results.fittedvalues
    Huber_robust_resids = y - pred_y
    Huber_robust_resid = sum([r*r for r in Huber_robust_resids])
    return Huber_robust_param,pred_y,Huber_robust_resid

def robust_TheilSen(X,y):
    X = np.array(X).reshape(len(X),1)
    y = np.array(y)
    estimator = TheilSenRegressor(random_state=42)
    reg = estimator.fit(X, y)
    slope = reg.coef_[0]
    pred_y = reg.predict(X)
    resid = sum([r*r for r in (y-pred_y)])
    return slope,pred_y,resid

def robust_RANSAC(X,y):
    X = np.array(X).reshape(len(X),1)
    y = np.array(y)
    estimator = RANSACRegressor(random_state=42, loss='squared_error')
    reg = estimator.fit(X, y)
    slope = reg.estimator_.coef_[0]
    pred_y = reg.predict(X)
    resid = sum([r*r for r in (y-pred_y)])
    inlier_count = sum([1 for m in reg.inlier_mask_ if m])
    inlier_mask = reg.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    return slope,pred_y,resid, inlier_count, inlier_mask, outlier_mask

def myNorm (l):
    new_l = []
    max_l = max(l)
    min_l = min(l)
    mean_l = mean(l)
    for i in l:
        new_l.append((i-mean_l) / (max_l - min_l))
    return new_l

def get_zScore_new(rankList, maxAnswerCount):
    mean = np.mean(list(range(1,maxAnswerCount+1)))
    # standard deviation
    std = np.std(list(range(1,maxAnswerCount+1)), dtype=np.float64)
    zScores = []
    for r in rankList:
        if std==0:
            zScores.append(0)
        else:
            zScores.append(-(r-mean)/std) # flip the sign of z, since top ranked z is smaller and lower ranked z is bigger
    return zScores

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
    
def get_uniScore(rankList):
    uniScores = []
    return uniScores

def myGrouping(combineAll, sortingColumn):
    combineAll.sort(key=lambda t:t[sortingColumn]) # sorted by zScore
    group_count = 400
    # group_count = len(combineAll) # equivalent to not grouping

    percentages = [x / group_count for x in range(1, group_count+1)]
    zScoreIntervals = [scipy.stats.norm.ppf(p) for p in percentages]

    group_combineAll = []

    startIndex = 0
    for zintv in zScoreIntervals:
        cur_group = []
        for i in range(startIndex, len(combineAll)):
            cur_tup = combineAll[i]
            cur_zScore = cur_tup[sortingColumn]
            if cur_zScore <= zintv:
                cur_group.append(cur_tup)
            else:
                startIndex = i
                break
        if len(cur_group)>0: 
            column_wise_mean = np.mean(cur_group,axis=0)
            group_combineAll.append(tuple(column_wise_mean))
    
    # print(f"group z score sample count {len(group_combineAll)} sorted by column {sortingColumn}")
    return group_combineAll

def residulToDiagonalLine(x, y):
    squredResiduals = []
    for i in range(len(y)):
        squredResiduals.append((x[i]-y[i])**2)
    return sum(squredResiduals)

#######################################################################################################
# plot with all data point
def myPlot(commName, combineAll, plotFileName, reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP):
    topAnswersFlag = ('topAnswers' in plotFileName)
    overEstimatedFlag = ('overEstimated' in plotFileName)
    underEstimatedFlag = ('underEstimated' in plotFileName)
    disagreeAnswersFlag = ('_disagreeAnswers' in plotFileName)
    agreeAnswersFlag = ('_agreeAnswers' in plotFileName)
    
    subTitleTail = 'all answers'
    if topAnswersFlag:
        subTitleTail = 'topAnswers'
    if overEstimatedFlag:
        subTitleTail = 'overEstimatedAnswers'
    if underEstimatedFlag:
        subTitleTail = 'underEstimatedAnswers'
    if disagreeAnswersFlag:
        subTitleTail = 'disagreeAnswers'
    if agreeAnswersFlag:
        subTitleTail = 'agreeAnswers'

    sentimentFlag = ('forSentiment' in plotFileName)
    helpfulFlag = ('forHelpful' in plotFileName)

    yAxisName = None
    if sentimentFlag:
        yAxisName = 'sentiment rankZscore'
    if helpfulFlag:
        yAxisName = 'helpfulness rankZscore'

    plt.cla()
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=3.0)

    group_combineAll_basedOnVoteDiff = myGrouping(combineAll, sortingColumn=1)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiff]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnVoteDiff]
    axs[0, 0].scatter(group_voteDiff_rankZscores, group_priorQ_rankZscores,s = 2)

    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_voteDiff_rankZscores = [t[1] for t in combineAll]
    # axs[0, 0].scatter(all_voteDiff_rankZscores, all_priorQ_rankZscores,s = 2)
    
    axs[0, 0].set_xlabel('vote diff rankZscore', fontsize = 8)
    axs[0, 0].set_ylabel(yAxisName, fontsize = 8)
    axs[0, 0].set_xlim(-2,2)
    axs[0, 0].set_ylim(-2,2)

    # OLS fit
    z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls = residulToDiagonalLine(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1, 
                  label=f'slope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    
    axs[0,0].legend(loc="best", fontsize = 6)


    group_combineAll_basedOnCVP = myGrouping(combineAll, sortingColumn=2)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVP]
    group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnCVP]
    axs[0, 1].scatter(group_CVPsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll]
    # axs[0, 1].scatter(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2)
    
    axs[0, 1].set_xlabel('CVP sklearn q rankZscore', fontsize = 8)
    axs[0, 1].set_ylabel(yAxisName, fontsize = 8)
    axs[0, 1].set_xlim(-2,2)
    axs[0, 1].set_ylim(-2,2)

    # OLS fit
    z1,p1 = curveFitOLS(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls = residulToDiagonalLine(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    axs[0,1].plot(all_CVPsklearnQ_rankZscores,p1(all_CVPsklearnQ_rankZscores),"r-", linewidth=1, 
                  label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    axs[0,1].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    
    axs[0,1].legend(loc="best", fontsize = 6)

    group_combineAll_basedOnNewModel = myGrouping(combineAll, sortingColumn=3)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnNewModel]
    group_newModelsklearnQ_rankZscores = [t[3] for t in group_combineAll_basedOnNewModel]
    axs[1, 0].scatter(group_newModelsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2)
    
    # ### using all raw points to fit
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_newModelsklearnQ_rankZscores = [t[3] for t in combineAll]
    # # axs[1, 0].scatter(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2)

    # ### using grouped points to fit
    # all_priorQ_rankZscores = group_priorQ_rankZscores
    # all_newModelsklearnQ_rankZscores = group_newModelsklearnQ_rankZscores
    
    axs[1, 0].set_xlabel('new model sklearn q rankZscore', fontsize = 8)
    axs[1, 0].set_ylabel(yAxisName, fontsize = 8)
    axs[1, 0].set_xlim(-2,2)
    axs[1, 0].set_ylim(-2,2)

    # OLS fit
    z3,p3 = curveFitOLS(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls = residulToDiagonalLine(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    axs[1,0].plot(all_newModelsklearnQ_rankZscores,p3(all_newModelsklearnQ_rankZscores),"r-", linewidth=1, 
                  label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    axs[1,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)

    
    axs[1,0].legend(loc="best", fontsize = 6)


    group_combineAll_basedOnNewModelInteraction = myGrouping(combineAll, sortingColumn=4)
    group_priorQ_rankZscores = [t[0] for t in  group_combineAll_basedOnNewModelInteraction]
    group_newModelInteractionsklearnQ_rankZscores = [t[4] for t in  group_combineAll_basedOnNewModelInteraction]
    axs[1, 1].scatter(group_newModelInteractionsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2)
    
    # ### using all raw points to fit
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_newModelInteractionsklearnQ_rankZscores = [t[4] for t in combineAll]
    # # axs[1, 1].scatter(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2)

    # ### using grouped points to fit
    # all_priorQ_rankZscores = group_priorQ_rankZscores
    # all_newModelInteractionsklearnQ_rankZscores = group_newModelInteractionsklearnQ_rankZscores
    
    axs[1, 1].set_xlabel('newModelInteraction sklearn q rankZscore', fontsize = 8)
    axs[1, 1].set_ylabel(yAxisName, fontsize = 8)
    axs[1, 1].set_xlim(-2,2)
    axs[1, 1].set_ylim(-2,2)

    # OLS fit
    z4,p4 = curveFitOLS(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls = residulToDiagonalLine(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores)
    axs[1,1].plot(all_newModelInteractionsklearnQ_rankZscores,p4(all_newModelInteractionsklearnQ_rankZscores),"r-", linewidth=1, 
                  label=f'slope={round(z4[0][0],4)}\nresidual={round(z3[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    axs[1,1].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    

    axs[1,1].legend(loc="best", fontsize = 6)

    fig.suptitle(f"{commName.replace('.stackexchange','')}\nnewModelInteractionReg({reg_alpha_NewModelInteraction}) newModelReg({reg_alpha_NewModel}) CVPReg({reg_alpha_CVP})\n({subTitleTail} : {len(combineAll)})", fontsize = 6)

    savePlot(fig, plotFileName)

    tup = (z0,z1,z3,z4 )
    return tup

#####################################################################################################################
def extractFun(commIndex, commName,commDir, roundIndex, variation, commName2bestRegAlphas, sampled_comms, commName2selected_reg_strengthList):
    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())
    print(f"processing {commName}")

    try_reg_strengthList = [0.1,0.2, 0.3,0.4, 0.5,0.6, 0.7,0.8,0.9,
                            1, 2, 3,4,5, 6, 7,8,9,
                            10,20, 30,40,50,60, 70,80,90,
                            100, 200, 300, 400, 500, 600, 700, 800, 900,
                            1000]

    # load intermediate_data filesdd
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    if commName == 'stackoverflow': # using subcomm to represent stackoverflow
        subComms_data_folder = os.path.join(commDir, f'subCommunities_folder')
        ## Load all sub community direcotries 
        with open(subComms_data_folder+'/'+f'subCommName2subCommDir.dict', 'rb') as inputFile:
            subCommName2commDir = pickle.load( inputFile)
        subCommDir = subCommName2commDir['reactjs']
        subComm_intermediate_directory = os.path.join(subCommDir, r'intermediate_data_folder')

    # prior coefs for each reg_strength
    CVP_testAccuracyAndRegList = []
    newModel_testAccuracyAndRegList = []
    newModelInteraction_testAccuracyAndRegList = []

    reg_alpha = 0.1
    if variation == "_fixedTau":
        if roundIndex in [1,2]:

            if commName in sampled_comms:
                try:
                    with open(intermediate_directory+'/'+f"temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha})_forSampledQuestion_return.dict", 'rb') as inputFile:
                        return_trainSuccess_dict_newModelInteraction = pickle.load( inputFile)
                except:
                    with open(intermediate_directory+'/'+f"temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha})_return.dict", 'rb') as inputFile:
                        return_trainSuccess_dict_newModelInteraction = pickle.load( inputFile)
            else:
                with open(intermediate_directory+'/'+f"temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha})_return.dict", 'rb') as inputFile:
                    return_trainSuccess_dict_newModelInteraction = pickle.load( inputFile)
            
            newModelInteraction_testAccuracyList = return_trainSuccess_dict_newModelInteraction[list(return_trainSuccess_dict_newModelInteraction.keys())[0]]['CV_scores']
            for i, reg in enumerate(try_reg_strengthList):
                newModelInteraction_testAccuracyAndRegList.append((newModelInteraction_testAccuracyList[i], reg))
            
            if commName in sampled_comms:
                with open(intermediate_directory+'/'+f"temperalOrderTraining12_newModel_fixedTau_regAlpha({reg_alpha})_forSampledQuestion_return.dict", 'rb') as inputFile:
                    return_trainSuccess_dict_newModel = pickle.load( inputFile)
            else:
                with open(intermediate_directory+'/'+f"temperalOrderTraining12_newModel_fixedTau_regAlpha({reg_alpha})_return.dict", 'rb') as inputFile:
                    return_trainSuccess_dict_newModel = pickle.load( inputFile)
            
            newModel_testAccuracyList = return_trainSuccess_dict_newModel[list(return_trainSuccess_dict_newModel.keys())[0]]['CV_scores']
            for i, reg in enumerate(try_reg_strengthList):
                newModel_testAccuracyAndRegList.append((newModel_testAccuracyList[i], reg))

            if commName in sampled_comms:
                try:
                    with open(intermediate_directory+'/'+f"temperalOrderTraining11_CVP_regAlpha({reg_alpha})_forSampledQuestion_return.dict", 'rb') as inputFile:
                        return_trainSuccess_dict_CVP = pickle.load( inputFile)
                except:
                    with open(intermediate_directory+'/'+f"temperalOrderTraining11_CVP_regAlpha({reg_alpha})_return.dict", 'rb') as inputFile:
                        return_trainSuccess_dict_CVP = pickle.load( inputFile)
            else:
                with open(intermediate_directory+'/'+f"temperalOrderTraining11_CVP_regAlpha({reg_alpha})_return.dict", 'rb') as inputFile:
                    return_trainSuccess_dict_CVP = pickle.load( inputFile)
            
            CVP_testAccuracyList = return_trainSuccess_dict_CVP[list(return_trainSuccess_dict_CVP.keys())[0]]['CV_scores']
            for i, reg in enumerate(try_reg_strengthList):
                CVP_testAccuracyAndRegList.append((CVP_testAccuracyList[i], reg))


    if commName in commName2selected_reg_strengthList.keys():
            selected_reg_strengthList = commName2selected_reg_strengthList[commName]
            bestRegAlpha_newModelInteraction = selected_reg_strengthList[0]
            bestRegAlpha_newModel = selected_reg_strengthList[1]
            bestRegAlpha_CVP = selected_reg_strengthList[2]
    else:
        # get the best reg_alpha for each model
        bestRegAlpha_CVP = sorted(CVP_testAccuracyAndRegList, key=lambda x:x[0],reverse=True)[0][1]
        bestRegAlpha_newModel = sorted(newModel_testAccuracyAndRegList, key=lambda x:x[0],reverse=True)[0][1]
        bestRegAlpha_newModelInteraction = sorted(newModelInteraction_testAccuracyAndRegList, key=lambda x:x[0],reverse=True)[0][1]
    
    sampleCount = return_trainSuccess_dict_newModel[list(return_trainSuccess_dict_newModelInteraction.keys())[0]]['dataShape'][0]

    commName2bestRegAlphas[commName] = [bestRegAlpha_newModelInteraction, bestRegAlpha_newModel, bestRegAlpha_CVP, sampleCount]
    return


#####################################################################################################################
def myFun(commName, commDir, rootDir, roundIndex, variation, reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP, sampled_comms):
   
    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())
    print(f"processing {commName}")


    # load intermediate_data filesdd
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    if commName == 'stackoverflow': # using subcomm to represent stackoverflow
        subComms_data_folder = os.path.join(commDir, f'subCommunities_folder')
        ## Load all sub community direcotries 
        with open(subComms_data_folder+'/'+f'subCommName2subCommDir.dict', 'rb') as inputFile:
            subCommName2commDir = pickle.load( inputFile)
        subCommDir = subCommName2commDir['reactjs']
        subComm_intermediate_directory = os.path.join(subCommDir, r'intermediate_data_folder')

    # prior coefs
    if variation == "_fixedTau":
        if roundIndex in [1,2]:
            if reg_alpha_NewModelInteraction != None:
                if commName in sampled_comms:
                    try:
                        with open(intermediate_directory+'/'+f"temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha_NewModelInteraction})_forSampledQuestion_return.dict", 'rb') as inputFile:
                            return_trainSuccess_dict_newModelInteraction = pickle.load( inputFile)
                    except:
                        with open(intermediate_directory+'/'+f"temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha_NewModelInteraction})_return.dict", 'rb') as inputFile:
                            return_trainSuccess_dict_newModelInteraction = pickle.load( inputFile)
                else:
                    with open(intermediate_directory+'/'+f"temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha_NewModelInteraction})_return.dict", 'rb') as inputFile:
                        return_trainSuccess_dict_newModelInteraction = pickle.load( inputFile)
            else:
                return_trainSuccess_dict_newModelInteraction = None
            
            if reg_alpha_NewModel != None:
                if commName in sampled_comms:
                    try:
                        with open(intermediate_directory+'/'+f"temperalOrderTraining12_newModel_fixedTau_regAlpha({reg_alpha_NewModel})_forSampledQuestion_return.dict", 'rb') as inputFile:
                            return_trainSuccess_dict_newModel = pickle.load( inputFile)
                    except:
                        with open(intermediate_directory+'/'+f"temperalOrderTraining12_newModel_fixedTau_regAlpha({reg_alpha_NewModel})_return.dict", 'rb') as inputFile:
                            return_trainSuccess_dict_newModel = pickle.load( inputFile)
                else:
                    with open(intermediate_directory+'/'+f"temperalOrderTraining12_newModel_fixedTau_regAlpha({reg_alpha_NewModel})_return.dict", 'rb') as inputFile:
                        return_trainSuccess_dict_newModel = pickle.load( inputFile)
            else:
                return_trainSuccess_dict_newModel = None
            
            if reg_alpha_CVP != None:
                if commName in sampled_comms:
                    try:
                        with open(intermediate_directory+'/'+f"temperalOrderTraining11_CVP_regAlpha({reg_alpha_CVP})_forSampledQuestion_return.dict", 'rb') as inputFile:
                            return_trainSuccess_dict_CVP = pickle.load( inputFile)
                    except:
                        with open(intermediate_directory+'/'+f"temperalOrderTraining11_CVP_regAlpha({reg_alpha_CVP})_return.dict", 'rb') as inputFile:
                            return_trainSuccess_dict_CVP = pickle.load( inputFile)
                else:
                    with open(intermediate_directory+'/'+f"temperalOrderTraining11_CVP_regAlpha({reg_alpha_CVP})_return.dict", 'rb') as inputFile:
                        return_trainSuccess_dict_CVP = pickle.load( inputFile)
            else:
                return_trainSuccess_dict_CVP = None
    
    ############################################################################################
        
    print(f"loading total_answersWithVotes_indice... for {commName}")
    if commName != 'stackoverflow':
        try:
            if commName in sampled_comms:
                with open(intermediate_directory+'/'+'whole_answersWithVotes_indice_removeFirstRealVote_forSampledQuestion.dict', 'rb') as inputFile:
                    total_answersWithVotes_indice = pickle.load( inputFile)
            else:
                with open(intermediate_directory+'/'+'whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
                    total_answersWithVotes_indice = pickle.load( inputFile)
        except Exception as e:
            print(f"for {commName} error when load the total_answersWithVotes_indice: {e}")
            return
    else: # using subcomm to represent stackoverflow
        try:
            with open(subComm_intermediate_directory+'/'+f'whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
                total_answersWithVotes_indice = pickle.load( inputFile)
        except Exception as e:
            print(f"for {commName} error when load the total_answersWithVotes_indice: {e}")
            return

    
    if commName != 'stackoverflow':
        if commName in sampled_comms:
            with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList_forSampledQuestion.dict', 'rb') as inputFile:
                Questions = pickle.load( inputFile)
        else:
            with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
                Questions = pickle.load( inputFile)
    else: # using subcomm to represent stackoverflow
        subComm_QuestionsWithEventList_directory = subCommDir+'/'+f'QuestionsWithEventList_tag_reactjs.dict'
        with open(subComm_QuestionsWithEventList_directory, 'rb') as inputFile:
            Questions = pickle.load( inputFile)

    # # load userIds for comment filtering
    # try:
    #     with open(intermediate_directory+'/'+'postId2UserId.dict', 'rb') as inputFile:
    #         postId2UserId = pickle.load( inputFile)
    # except:
    #     print(f"no postId2UserId for {commName}")
    #     return
    # try:
    #     with open(intermediate_directory+'/'+'postId2commentIdAndCreationTimeAndUserId_tillCurChunk.dict', 'rb') as inputFile:
    #         post2commentIdAndCreationTimeAndUserId = pickle.load( inputFile)
    # except:
    #     print(f"no post2commentIdAndCreationTimeAndUserId for {commName}")
    #     return
    ############################################################################################
    
    # get max answer count of one question
    maxAnswerCount = 0
    for qid, content in Questions.items():
        answerCount = len(content['filtered_answerList'])
        if answerCount > maxAnswerCount:
            maxAnswerCount = answerCount
    

    # get learned qs (without bias)
    try:
        if len(return_trainSuccess_dict_newModelInteraction)==1:
            simplifiedCommName = list(return_trainSuccess_dict_newModelInteraction.keys())[0]
        learned_qs_interaction = return_trainSuccess_dict_newModelInteraction[simplifiedCommName]['qs_sklearn']  # after retrain with full data
        bias_newModelInteraction = return_trainSuccess_dict_newModelInteraction[simplifiedCommName]['bias_sklearn']
    except:
        learned_qs_interaction = return_trainSuccess_dict_newModelInteraction[commName]['qs_sklearn']  # after retrain with full data
        bias_newModelInteraction = return_trainSuccess_dict_newModelInteraction[commName]['bias_sklearn']
    if bias_newModelInteraction != None:
        learned_qs_interaction = [q + bias_newModelInteraction for q in learned_qs_interaction] # update qs by adding bias
    
    try:
        if len(return_trainSuccess_dict_newModel)==1:
            simplifiedCommName = list(return_trainSuccess_dict_newModel.keys())[0]
        learned_qs = return_trainSuccess_dict_newModel[simplifiedCommName]['qs_sklearn']  # after retrain with full data
    except:
        learned_qs = return_trainSuccess_dict_newModel[commName]['qs_sklearn']  # after retrain with full data

    try:
        if len(return_trainSuccess_dict_CVP)==1:
            simplifiedCommName = list(return_trainSuccess_dict_CVP.keys())[0]
        learned_qs_CVP = return_trainSuccess_dict_CVP[simplifiedCommName]['qs_sklearn']  # after retrain with full data
    except:
        learned_qs_CVP = return_trainSuccess_dict_CVP[commName]['qs_sklearn']  # after retrain with full data
    
    assert len(learned_qs)==len(total_answersWithVotes_indice)

    # extract answer ids 
    total_aids = set()
    total_answersWithVotes_ids = []
    for i,(qid, ai) in enumerate(total_answersWithVotes_indice):
        filtered_answerList = Questions[qid]['filtered_answerList']
        aid = filtered_answerList[ai]
        total_aids.add(aid)
        total_answersWithVotes_ids.append((qid,aid))
    
    # get corresponding sentiment and corresponding vote differences
    print(f"loading GPT sentiment scores ... for {commName}")
    promptType = 1
    my_model = "gpt-4o"

    if commName != 'stackoverflow':
        with open(commDir+ f'/promptJson_folder/prompt_template_{promptType}.json') as json_file:
            prompts_Dict = json.load(json_file)

        with open(commDir+ f'/GPTresponse_folder/qid2resDict_prompt{promptType}_{my_model}.json') as json_file:
            qid2resDict = json.load(json_file)
    else: # using subcomm reactjs to replace SOF
        with open(commDir+ f'/promptJson_folder/prompt_template_{promptType}_reactjs.json') as json_file:
            prompts_Dict = json.load(json_file)

        with open(commDir+ f'/GPTresponse_folder/qid2resDict_prompt{promptType}_{my_model}_reactjs.json') as json_file:
            qid2resDict = json.load(json_file)

    print(f"load qid2resDict ({len(qid2resDict)}) for {commName}")

    print(f"extracting aspact2sentimentScores...")
    aid2sentimentScore = defaultdict()
    aid2helpfulScore = defaultdict()

    aid2quality_CVP = defaultdict()
    aid2quality_newModel = defaultdict()
    aid2quality_newModelInteration = defaultdict()

    for i,tup in enumerate(total_answersWithVotes_ids):
        qid, aid = tup
        
        if str(qid) in qid2resDict.keys():
            aid2results = qid2resDict[str(qid)]
            if str(aid) in aid2results.keys():
                cur_res = aid2results[str(aid)]
                sentimentScores = cur_res['sentiments']
                commentsAboutOtherByGPT= cur_res['commentsAboutOther']
                helpfulScore = cur_res['helpfulScore']
                if helpfulScore == None:
                    continue

                commentSerialListStartedWithAt = prompts_Dict[str(qid)][str(aid)]['commentSerialListStartedWithAt']

                # try:
                #     owerUserId = int(postId2UserId[aid]['userId'])
                # except:
                #     try:
                #         owerUserId = int(postId2UserId[aid]['lastEditorUserId'])
                #     except:
                #         owerUserId = None
                # commentUserIds = post2commentIdAndCreationTimeAndUserId[aid]['userIds']
                # assert len(commentUserIds) == len(sentimentScores)
                
                commentIndexToRemove = []
                # find out which comment to be ignored

                # if ignore the comments that written by owerUser
                # for j, uid in enumerate(commentUserIds):
                #     try:
                #         uid = int(uid)
                #     except: # uid is nan, skip
                #         uid = None
                #         continue
                    # filter out the answer author's comments
                    # if uid == owerUserId:
                    #     commentIndexToRemove.append(j)
                
                # if ignore the comments that identified by GPT as about other comments
                # for sn in commentsAboutOtherByGPT: # serial number of comments, need to convert to index
                #     commentIndexToRemove.append(sn-1)

                # if ignore the comments that identified by being started with @ 
                # for sn in commentSerialListStartedWithAt: # serial number of comments, need to convert to index
                #     commentIndexToRemove.append(sn-1)

                filtered_sentimentScores = [s for ii, s in enumerate(sentimentScores) if ii not in commentIndexToRemove]

                if len(filtered_sentimentScores)==0: # filtered none comment, skip this answer
                    continue 
                else:
                    avg_sentScore = mean(filtered_sentimentScores)

                aid2sentimentScore[aid] = avg_sentScore
                aid2helpfulScore[aid] = helpfulScore

                aid2quality_CVP[aid] = learned_qs_CVP[i]
                aid2quality_newModel[aid] = learned_qs[i]
                aid2quality_newModelInteration[aid] = learned_qs_interaction[i]


    # reconstruct the rankings 
    print(f"reconstructing the rankings...")
    aid2sentiment_rankZscore = defaultdict()
    aid2helpfulScore_rankZscore = defaultdict()
    aid2VoteDiff_rankZscore = defaultdict()
    aid2CVP_sklearn_q_rankZscore  = defaultdict()
    aid2newModel_sklearn_q_rankZscore  = defaultdict()
    aid2newModelInteraction_sklearn_q_rankZscore  = defaultdict()

    aid2sentiment_rank = defaultdict()
    aid2helpfulScore_rank = defaultdict()
    aid2VoteDiff_rank = defaultdict()
    aid2CVP_sklearn_q_rank  = defaultdict()
    aid2newModel_sklearn_q_rank  = defaultdict()
    aid2newModelInteraction_sklearn_q_rank = defaultdict()

    topAnswersIdList = []
    topThreshold = 5

    underEstimatedAnswersIdList_forSentiment = []
    overEstimatedAnswersIdList_forSentiment = []
    underEstimatedAnswersIdList_forHelpful = []
    overEstimatedAnswersIdList_forHelpful = []
    changedThreshold = 3

    disagreeAnswersIdList = [] # those CVP learned q rank is different from new model learned q rank
    disagreeGaps = []  # store the corresponding differences of above

    for qid, d in Questions.items():
        answerList =  d['answerList']
        involved_answerList = set(answerList).intersection(set(aid2quality_newModel.keys()))
        if len(involved_answerList) == 0: # none answer in the learned qs, skip this question
            continue

        # get voteDiff of each answer
        involved_aid2voteDiff = defaultdict()
        involved_ai2voteList = defaultdict()
        eventList = d['eventList']
        for e in eventList:
            eventType = e['et']
            if eventType != 'v': # skip all event that is not a vote
                continue
            
            vote = e['v']
            if vote == 1:
                cur_vote = 1
            else: # vote == 0 is negative vote
                cur_vote = -1

            ai = e['ai']
            if answerList[ai] not in involved_answerList: # skip the events of non-involved answer
                continue

            if ai in involved_ai2voteList.keys():
                involved_ai2voteList[ai].append(cur_vote)
            else:
                involved_ai2voteList[ai] = [cur_vote]
        
        if len(involved_ai2voteList)==0: # skip when no involved answer having votes
            continue 

        for ai, voteList in involved_ai2voteList.items():
            involved_aid2voteDiff[answerList[ai]] = sum(voteList)
        
        # get rankZscore based on voteDiff
        involved_aid2rankBasedOnVoteDiff = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2voteDiff.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) : # sort by the vote score and then by the answer index
            aid = kv[0]
            involved_aid2rankBasedOnVoteDiff[aid] = i+1 # rank
            if i < topThreshold: # filter out the top answers
                    topAnswersIdList.append(aid)
        
        aid2VoteDiff_rank.update(involved_aid2rankBasedOnVoteDiff)

        zScores = get_zScore(involved_aid2rankBasedOnVoteDiff.values())
        for i,aid in enumerate(involved_aid2rankBasedOnVoteDiff.keys()):
            aid2VoteDiff_rankZscore[aid] = zScores[i]
        
        # get ranks based on sentiment GPT
        involved_aid2sentimentScore = {aid : aid2sentimentScore[aid] for aid in involved_aid2rankBasedOnVoteDiff.keys()}
        involved_aid2rankBasedOnSentimentScore = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2sentimentScore.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
            aid = kv[0]
            involved_aid2rankBasedOnSentimentScore[aid] = i+1 # rank

            # get over estimated answers and under estimated answers
            change = involved_aid2rankBasedOnSentimentScore[aid] - involved_aid2rankBasedOnVoteDiff[aid] # pos: over estimated, neg: under estimated
            if change >= changedThreshold:
                overEstimatedAnswersIdList_forSentiment.append(aid)
            elif change <= -changedThreshold:
                underEstimatedAnswersIdList_forSentiment.append(aid)
        
        aid2sentiment_rank.update(involved_aid2rankBasedOnSentimentScore)

        zScores = get_zScore(involved_aid2rankBasedOnSentimentScore.values())
        for i,aid in enumerate(involved_aid2rankBasedOnSentimentScore.keys()):
            aid2sentiment_rankZscore[aid] = zScores[i]

        # get ranks based on helpful score of GPT
        involved_aid2helpfulScore = {aid : aid2helpfulScore[aid] for aid in involved_aid2rankBasedOnVoteDiff.keys()}
        involved_aid2rankBasedOnHelpfulScore = defaultdict()
        try:
            for i, kv in enumerate(sorted(involved_aid2helpfulScore.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
                aid = kv[0]
                involved_aid2rankBasedOnHelpfulScore[aid] = i+1 # rank

                # get over estimated answers and under estimated answers
                change = involved_aid2rankBasedOnHelpfulScore[aid] - involved_aid2rankBasedOnVoteDiff[aid] # pos: over estimated, neg: under estimated
                if change >= changedThreshold:
                    overEstimatedAnswersIdList_forHelpful.append(aid)
                elif change <= -changedThreshold:
                    underEstimatedAnswersIdList_forHelpful.append(aid)
        except Exception as e:
            print(e)
            print(f"debug")
            return
        
        aid2helpfulScore_rank.update(involved_aid2rankBasedOnHelpfulScore)

        zScores = get_zScore(involved_aid2rankBasedOnHelpfulScore.values())
        for i,aid in enumerate(involved_aid2rankBasedOnHelpfulScore.keys()):
            aid2helpfulScore_rankZscore[aid] = zScores[i]
        
        # get ranks based on CVP_sklearn_q
        involved_aid2CVP_sklearn_q = {aid : aid2quality_CVP[aid] for aid in involved_aid2rankBasedOnVoteDiff.keys()}
        involved_aid2rankBasedOnCVPsklearnQ = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2CVP_sklearn_q.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
            aid = kv[0]
            involved_aid2rankBasedOnCVPsklearnQ[aid] = i+1 # rank
        
        aid2CVP_sklearn_q_rank.update(involved_aid2rankBasedOnCVPsklearnQ)

        zScores = get_zScore(involved_aid2rankBasedOnCVPsklearnQ.values())
        for i,aid in enumerate(involved_aid2rankBasedOnCVPsklearnQ.keys()):
            aid2CVP_sklearn_q_rankZscore[aid] = zScores[i]

        
        # get ranks based on newModel_q_sklearn
        involved_aid2newModel_sklearn_q = {aid : aid2quality_newModel[aid] for aid in involved_aid2rankBasedOnVoteDiff.keys()}
        involved_aid2rankBasedOnNewModelsklearnQ = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2newModel_sklearn_q.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
            aid = kv[0]
            involved_aid2rankBasedOnNewModelsklearnQ[aid] = i+1 # rank

            # get disagree answers 
            gap = involved_aid2rankBasedOnNewModelsklearnQ[aid] - involved_aid2rankBasedOnCVPsklearnQ[aid] # pos: new model lower rank, neg: CVP lower rank
            if gap != 0:
                disagreeAnswersIdList.append(aid)
                disagreeGaps.append(gap)
        
        aid2newModel_sklearn_q_rank.update(involved_aid2rankBasedOnNewModelsklearnQ)

        zScores = get_zScore(involved_aid2rankBasedOnNewModelsklearnQ.values())
        for i,aid in enumerate(involved_aid2rankBasedOnNewModelsklearnQ.keys()):
            aid2newModel_sklearn_q_rankZscore[aid] = zScores[i]
        

        # get ranks based on newModelInteraction_q_sklearn
        involved_aid2newModelInteraction_sklearn_q = {aid : aid2quality_newModelInteration[aid] for aid in involved_aid2rankBasedOnVoteDiff.keys()}
        involved_aid2rankBasedOnNewModelInteractionsklearnQ = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2newModelInteraction_sklearn_q.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
            aid = kv[0]
            involved_aid2rankBasedOnNewModelInteractionsklearnQ[aid] = i+1 # rank
        
        aid2newModelInteraction_sklearn_q_rank.update(involved_aid2rankBasedOnNewModelInteractionsklearnQ)

        zScores = get_zScore(involved_aid2rankBasedOnNewModelInteractionsklearnQ.values())
        for i,aid in enumerate(involved_aid2rankBasedOnNewModelInteractionsklearnQ.keys()):
            aid2newModelInteraction_sklearn_q_rankZscore[aid] = zScores[i]

    Questions.clear()

    # save intermediate outputs
    with open(intermediate_directory+f"/temperalOrderTraining15_verifyingQualities_outputs_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).dict", 'wb') as outputFile:
        pickle.dump((aid2sentiment_rank,
                    aid2helpfulScore_rank,
                    aid2VoteDiff_rank,
                    aid2CVP_sklearn_q_rank,
                    aid2newModel_sklearn_q_rank,
                    aid2newModelInteraction_sklearn_q_rank,
                    aid2sentiment_rankZscore,
                    aid2helpfulScore_rankZscore,
                    aid2VoteDiff_rankZscore,
                    aid2CVP_sklearn_q_rankZscore,
                    aid2newModel_sklearn_q_rankZscore,
                    aid2newModelInteraction_sklearn_q_rankZscore,
                    disagreeAnswersIdList, disagreeGaps, total_answersWithVotes_ids, 
                    topAnswersIdList,
                    underEstimatedAnswersIdList_forSentiment,
                    overEstimatedAnswersIdList_forSentiment,
                    underEstimatedAnswersIdList_forHelpful,
                    overEstimatedAnswersIdList_forHelpful,
                    disagreeAnswersIdList,
                    aid2quality_CVP,
                    aid2quality_newModel,
                    aid2quality_newModelInteration), outputFile)
        print( f"saved intermediate outputs for {commName}.")
    
    """
    combineAll_forSentiment = []
    combineAll_topAnswers_forSentiment = []
    combineAll_disagreeAnswers_forSentiment = []
    combineAll_agreeAnswers_forSentiment = []
    combineAll_overEstimatedAnswers_forSentiment = []
    combineAll_underEstimatedAnswers_forSentiment = []

    for aid in aid2sentiment_rankZscore.keys():
        try:
            tup = (aid2sentiment_rankZscore[aid],aid2VoteDiff_rankZscore[aid],aid2CVP_sklearn_q_rankZscore[aid],aid2newModel_sklearn_q_rankZscore[aid],aid2newModelInteraction_sklearn_q_rankZscore[aid])
            combineAll_forSentiment.append(tup)
            if aid in topAnswersIdList:
                combineAll_topAnswers_forSentiment.append(tup)
            if aid in disagreeAnswersIdList:
                combineAll_disagreeAnswers_forSentiment.append(tup)
            else: # agree answers 
                combineAll_agreeAnswers_forSentiment.append(tup)
            if aid in overEstimatedAnswersIdList_forSentiment:
                combineAll_overEstimatedAnswers_forSentiment.append(tup)
            if aid in underEstimatedAnswersIdList_forSentiment:
                combineAll_underEstimatedAnswers_forSentiment.append(tup)
        except Exception as e:
            print(e)

    combineAll_forHelpful = []
    combineAll_topAnswers_forHelpful = []
    combineAll_disagreeAnswers_forHelpful = []
    combineAll_agreeAnswers_forHelpful = []
    combineAll_overEstimatedAnswers_forHelpful = []
    combineAll_underEstimatedAnswers_forHelpful = []

    for aid in aid2helpfulScore_rankZscore.keys():
        try:
            tup = (aid2helpfulScore_rankZscore[aid],aid2VoteDiff_rankZscore[aid],aid2CVP_sklearn_q_rankZscore[aid],aid2newModel_sklearn_q_rankZscore[aid],aid2newModelInteraction_sklearn_q_rankZscore[aid])
            combineAll_forHelpful.append(tup)
            if aid in topAnswersIdList:
                combineAll_topAnswers_forHelpful.append(tup)
            if aid in disagreeAnswersIdList:
                combineAll_disagreeAnswers_forHelpful.append(tup)
            else: # agree answers
                combineAll_agreeAnswers_forHelpful.append(tup)
            if aid in overEstimatedAnswersIdList_forHelpful:
                combineAll_overEstimatedAnswers_forHelpful.append(tup)
            if aid in underEstimatedAnswersIdList_forHelpful:
                combineAll_underEstimatedAnswers_forHelpful.append(tup)
        except Exception as e:
            print(e)

    ### for using sentiment scores as ground truth
    plotFileName = f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forSentiment.png" # group by interval , 1layer grouping on x_axis
    # plotFileName = f"temperalOrderTraining15_(ignoreByGPT)_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).png" # group by interval , 1layer grouping on x_axis
    # plotFileName = f"temperalOrderTraining15_(ignoreStartedWithAt)_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).png" # group by interval , 1layer grouping on x_axis
    tup = myPlot(commName, combineAll_forSentiment, plotFileName, reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)
    z0,z1,z3,z4 = tup

    # plot for top answers
    myPlot(commName, combineAll_topAnswers_forSentiment, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_topAnswers_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forSentiment.png", reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)
    # plot for disagree answers
    myPlot(commName, combineAll_disagreeAnswers_forSentiment, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_disagreeAnswers_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forSentiment.png", reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)
    # plot for agree answers
    myPlot(commName, combineAll_agreeAnswers_forSentiment, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_agreeAnswers_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forSentiment.png", reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)
    # plot for changedMost answers
    myPlot(commName, combineAll_overEstimatedAnswers_forSentiment, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_overEstimatedAnswers_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forSentiment.png", reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)
    myPlot(commName, combineAll_underEstimatedAnswers_forSentiment, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_underEstimatedAnswers_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forSentiment.png", reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)

    ### for using helpful scores as ground truth
    plotFileName = f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forHelpful.png" # group by interval , 1layer grouping on x_axis
    # plotFileName = f"temperalOrderTraining15_(ignoreByGPT)_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).png" # group by interval , 1layer grouping on x_axis
    # plotFileName = f"temperalOrderTraining15_(ignoreStartedWithAt)_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).png" # group by interval , 1layer grouping on x_axis
    tup = myPlot(commName, combineAll_forHelpful, plotFileName, reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)
    z0,z1,z3,z4 = tup

    # plot for top answers
    myPlot(commName, combineAll_topAnswers_forHelpful, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_topAnswers_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forHelpful.png", reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)
    # plot for disagree answers
    myPlot(commName, combineAll_disagreeAnswers_forHelpful, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_disagreeAnswers_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forHelpful.png", reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)
     # plot for agree answers
    myPlot(commName, combineAll_agreeAnswers_forHelpful, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_agreeAnswers_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forHelpful.png", reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)
    # plot for changedMost answers
    myPlot(commName, combineAll_overEstimatedAnswers_forHelpful, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_overEstimatedAnswers_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forHelpful.png", reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)
    myPlot(commName, combineAll_underEstimatedAnswers_forHelpful, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_underEstimatedAnswers_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_forHelpful.png", reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP)


    # # # save disagree aids into csv
    # with open(rootDir +'/'+'allComm_temperalOrderTraining15_newModel_and_CVP_disagree_answers.csv', 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     for i in range(len(disagreeAnswersIdList)):
    #         aid = disagreeAnswersIdList[i]
    #         gap = disagreeGaps[i]
    #         writer.writerow( [commName, aid, gap])
    """
def main():

    t0=time.time()
    rootDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    # # # save disagree aids into csv
    # with open('allComm_temperalOrderTraining15_newModel_and_CVP_disagree_answers.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow( ["commName","aid", "gap"])

    # roundIndex = 1 ## multiple question multiple answer, original total event count, fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    roundIndex = 2 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    variation = '_fixedTau'
    
    # for sampled comms
    sampled_comms = ['academia.stackexchange','askubuntu',
                      'english.stackexchange','math.stackexchange','mathoverflow.net',
                      'meta.stackexchange','meta.stackoverflow','serverfault',
                      'softwareengineering.stackexchange','superuser','unix.stackexchange',
                      'worldbuilding.stackexchange','physics.stackexchange','electronics.stackexchange',
                      'codegolf.stackexchange','workplace.stackexchange']
    
    commName2selected_reg_strengthList = {
                                        'cstheory.stackexchange':(400,500,500),
                                          'unix.meta.stackexchange':(300,300,300),
                                          'stackoverflow':(0.1,0.1,0.1),
                                          'politics.stackexchange':(0.2,0.1,0.2),
                                        #   '3dprinting.stackexchange':(40,20,80),
                                        #   'latin.stackexchange':(0.3,0.3,0.3),
                                        #   'meta.askubuntu':(700,700,500),
                                        #   'lifehacks.stackexchange':(0.2,0.2,600)
                                          'math.meta.stackexchange':(0.4,0.4,0.2),
                                        'mathoverflow.net':(600,600,0.1),
                                            'mathematica.stackexchange':(90,80,100),
                                            'askubuntu':(0.1,600,0.1),
                                            'philosophy.stackexchange':(700,700,600),        
                                        'codegolf.meta.stackexchange':(200,100,90),    
                                          }
    
    """
    # extract commName2selected_reg_strengthList
    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    commName2bestRegAlphas = manager.dict() # to save the best regularizer pair for each community
    # run on all communities 
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        try:
            p = mp.Process(target=extractFun, args=(commIndex, commName,commDir, roundIndex, variation, commName2bestRegAlphas, sampled_comms, commName2selected_reg_strengthList))
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

    # convert to normal dict
    commName2selected_reg_strengthList = defaultdict()
    for commName, d in commName2bestRegAlphas.items():
        commName2selected_reg_strengthList[commName] = d
    # save return_trainResult_dict
    os.chdir(rootDir) # go back to root directory
    with open(f'allComm_bestRegAlphas_fixedTau.dict', 'wb') as outputFile:
        pickle.dump(commName2selected_reg_strengthList, outputFile)
        print(f"saved allComm_bestRegAlphas_fixedTau for {len(commName2selected_reg_strengthList)} comms.")

    """
    # load commName2selected_reg_strengthList
    with open(f'allComm_bestRegAlphas_fixedTau.dict', 'rb') as inputFile:
        commName2selected_reg_strengthList = pickle.load( inputFile)
    
    # debug_comms = ['math.stackexchange']
    # prepare args
    argsList = []
    for commName, tup in commName2selected_reg_strengthList.items():
        # if commName not in debug_comms:
        #     continue

        reg_alpha_newModelInteraction = tup[0]
        reg_alpha_newModel = tup[1]
        reg_alpha_CVP = tup[2]
        for comm in commDir_sizes_sortedlist:
            if comm[0] == commName:
                commDir = comm[1]
                break
        argsList.append((commName, commDir, rootDir, roundIndex, variation, reg_alpha_newModelInteraction, reg_alpha_newModel, reg_alpha_CVP, sampled_comms))


    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for myargs in argsList:
 
        try:
            p = mp.Process(target=myFun, args=myargs)
            p.start()
        except Exception as e:
            print(e)
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
    
    # Report progress.
    elapsed = format_time(time.time() - t0)
    print('verify qualities  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
