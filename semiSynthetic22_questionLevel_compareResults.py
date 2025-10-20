import os
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################
from toolFunctions import format_time, writeIntoLog, writeIntoResult,saveModel,savePlot
import time
import datetime
import glob
import multiprocessing as mp
from matplotlib import cm
import math
from collections import defaultdict
import pickle
import csv
import pandas as pd
import operator
from itertools import groupby
import re
import psutil
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import sys
from statistics import mean
from scipy.stats import norm, bernoulli, pearsonr
import numpy as np
from scipy.optimize import fsolve
from scipy.special import psi # digamma
import matplotlib.pyplot as plt
import scipy
from scipy.special import kl_div
import random
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

def curveFitOLS(X,y):
    z = np.polyfit(X,y, deg=1, full=True)
    p = np.poly1d(z[0])
    return z,p

def mySolver (answerCount, eventCount ):
    def myFunction(theta):
        F = np.empty((1))
        F[0] = theta * (psi(theta + eventCount) - psi(theta)) - answerCount
        return F

    zGuess = np.array([1])
    z = fsolve(myFunction,zGuess)
    return z[0]


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

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

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]
##########################################################################
# plot with all data point
def myPlot(commName, group_combineAll_basedOnPriorQ, combineAll, plotFileName):
    plt.cla()
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=3.0)
    
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnPriorQ]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnPriorQ]
    axs[0, 0].scatter(group_voteDiff_rankZscores, group_priorQ_rankZscores,s = 2)

    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_voteDiff_rankZscores = [t[1] for t in combineAll]
    # axs[0, 0].scatter(all_voteDiff_rankZscores, all_priorQ_rankZscores,s = 2)
    
    axs[0, 0].set_xlabel('vote diff rankZscore', fontsize = 8)
    axs[0, 0].set_ylabel('prior q rankZscore', fontsize = 8)

    # OLS fit
    z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"r-", linewidth=1, label=f'slope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}')
    axs[0,0].legend(loc="best", fontsize = 6)

    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnPriorQ]
    group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnPriorQ]
    axs[0, 1].scatter(group_CVPsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll]
    # axs[0, 1].scatter(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2)
    
    axs[0, 1].set_xlabel('CVP sklearn q rankZscore', fontsize = 8)
    axs[0, 1].set_ylabel('prior q rankZscore', fontsize = 8)

    # OLS fit
    z1,p1 = curveFitOLS(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    axs[0,1].plot(all_CVPsklearnQ_rankZscores,p1(all_CVPsklearnQ_rankZscores),"r-", linewidth=1, label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}')
    axs[0,1].legend(loc="best", fontsize = 6)

    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnPriorQ]
    group_newModelsklearnQ_rankZscores = [t[5] for t in group_combineAll_basedOnPriorQ]
    # group_combineAll_basedOnPriorQ = myGridGrouping(combineAll, sortingColumn=0, xColumn=5)
    # group_priorQ_rankZscores = group_combineAll_basedOnPriorQ[0]
    # group_newModelsklearnQ_rankZscores = group_combineAll_basedOnPriorQ[1]
    axs[1, 0].scatter(group_newModelsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2)
    
    # ### using all raw points to fit
    # all_priorQ_rankZscores = [t[0] for t in combineAll]
    # all_newModelsklearnQ_rankZscores = [t[5] for t in combineAll]
    # # axs[1, 0].scatter(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2)

    ### using grouped points to fit
    all_priorQ_rankZscores = group_priorQ_rankZscores
    all_newModelsklearnQ_rankZscores = group_newModelsklearnQ_rankZscores
    
    axs[1, 0].set_xlabel('new model sklearn q rankZscore', fontsize = 8)
    axs[1, 0].set_ylabel('prior q rankZscore', fontsize = 8)

    # OLS fit
    z3,p3 = curveFitOLS(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    axs[1,0].plot(all_newModelsklearnQ_rankZscores,p3(all_newModelsklearnQ_rankZscores),"r-", linewidth=1, label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}')
    axs[1,0].legend(loc="best", fontsize = 6)

    group_priorQ_rankZscores = [t[0] for t in  group_combineAll_basedOnPriorQ]
    group_newModelInteractionsklearnQ_rankZscores = [t[7] for t in  group_combineAll_basedOnPriorQ]
    # group_combineAll_basedOnPriorQ = myGridGrouping(combineAll, sortingColumn=0, xColumn=7)
    # group_priorQ_rankZscores = group_combineAll_basedOnPriorQ[0]
    # group_newModelInteractionsklearnQ_rankZscores = group_combineAll_basedOnPriorQ[1]
    axs[1, 1].scatter(group_newModelInteractionsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2)
    
    # ### using all raw points to fit
    # all_priorQ_rankZscores = [t[0] for t in combineAll]
    # all_newModelInteractionsklearnQ_rankZscores = [t[7] for t in combineAll]
    # # axs[1, 1].scatter(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2)

    ### using grouped points to fit
    all_priorQ_rankZscores = group_priorQ_rankZscores
    all_newModelInteractionsklearnQ_rankZscores = group_newModelInteractionsklearnQ_rankZscores
    
    axs[1, 1].set_xlabel('newModelInteraction sklearn q rankZscore', fontsize = 8)
    axs[1, 1].set_ylabel('prior q rankZscore', fontsize = 8)

    # OLS fit
    z4,p4 = curveFitOLS(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores)
    axs[1,1].plot(all_newModelInteractionsklearnQ_rankZscores,p4(all_newModelInteractionsklearnQ_rankZscores),"r-", linewidth=1, label=f'slope={round(z4[0][0],4)}\nresidual={round(z3[1][0],4)}')
    axs[1,1].legend(loc="best", fontsize = 6)


    fig.suptitle(f"{commName.replace('.stackexchange','')} (all answers)")
    savePlot(fig, plotFileName)
    return z0,z1,z3,z4
##########################################################################
# plot fitting lines per question
def questionLevelPlot(commName, combineAll, plotFileName, compareType, given_plotQids=None, given_myColorDict=None):
    qid2combingAll = defaultdict()
    for t in combineAll:
        qid = t[-1]
        if qid in qid2combingAll.keys():
            qid2combingAll[qid].append(t)
        else:
            qid2combingAll[qid]=[t]
    
    if compareType == 'rankZscore':
        tupIndex_prior = 0
        tupIndex_voteDiff = 1
        tupIndex_CVP = 2
        tupIndex_newModel = 5
        tupIndex_newModelInteraction = 7
    elif compareType == 'rank':
        tupIndex_prior = 8
        tupIndex_voteDiff = 9
        tupIndex_CVP = 10
        tupIndex_newModel = 13
        tupIndex_newModelInteraction = 15

    answerCountThreshold = 10

    selectedQids = []
    for qid, combineAll in qid2combingAll.items():
        if len(combineAll) == answerCountThreshold:
            selectedQids.append(qid)


    voteDiffSlopes = []
    CVPSlopes = []
    newModelSlopes = []
    newModelInteractionSlopes = []

    maxPlotTimes = 4 # the number of randomly selected questions
    if given_plotQids == None:
        plotQids = random.sample(selectedQids, maxPlotTimes)
    else:
        plotQids = given_plotQids

    if given_myColorDict == None:
        n_colors = maxPlotTimes
        colours = cm.rainbow(np.linspace(0, 1, n_colors))
        myColorDict = dict([(qid,colours[i]) for i, qid in enumerate(plotQids)])
    else:
        myColorDict = given_myColorDict

    plt.cla()
    fig, axs = plt.subplots(maxPlotTimes, 4)
    fig.tight_layout(pad=1.0)
    fig.set_figwidth(16)
    fig.set_figheight(12)
    fs = 8

    for i,qid in enumerate(plotQids):
        combineAll = qid2combingAll[qid]
    
        cur_color = myColorDict[qid]

        axs[i, 0].set_xlabel(f'vote diff {compareType}', fontsize = fs)
        axs[i, 0].set_ylabel(f'prior q {compareType}', fontsize = fs)
        axs[i, 1].set_xlabel(f'CVP sklearn q {compareType}', fontsize = fs)
        axs[i, 1].set_ylabel(f'prior q {compareType}', fontsize = fs)
        axs[i, 2].set_xlabel(f'new model sklearn q {compareType}', fontsize = fs)
        axs[i, 2].set_ylabel(f'prior q {compareType}', fontsize = fs)
        axs[i, 3].set_xlabel(f'newModelInteraction sklearn q {compareType}', fontsize = fs)
        axs[i, 3].set_ylabel(f'prior q {compareType}', fontsize = fs)

        ### using all raw points
        all_priorQ_rankZscores = [t[tupIndex_prior] for t in combineAll]
        all_voteDiff_rankZscores = [t[tupIndex_voteDiff] for t in combineAll]
        axs[i, 0].scatter(all_voteDiff_rankZscores, all_priorQ_rankZscores,s = 2, color=cur_color)

        # compute Pearson's Product Moment Correlation Coefficient 
        pearsonCor_voteDiff = stats.pearsonr(all_priorQ_rankZscores, all_voteDiff_rankZscores,alternative='two-sided')[0]
        # compute Spearman's Rank Correlation Coefficient
        spearmanCor_voteDiff = stats.spearmanr(all_priorQ_rankZscores, all_voteDiff_rankZscores,alternative='two-sided')[0]

        # OLS fit
        z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
        axs[i,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"-", linewidth=0.5, color=cur_color, label=f'question:{qid}\nslope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}')
        voteDiffSlopes.append(z0[0][0])
        axs[i,0].legend(loc="best", fontsize = fs)
        

        ### using all raw points
        all_priorQ_rankZscores = [t[tupIndex_prior] for t in combineAll]
        all_CVPsklearnQ_rankZscores = [t[tupIndex_CVP] for t in combineAll]
        axs[i, 1].scatter(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2, color=cur_color)

        # compute Pearson's Product Moment Correlation Coefficient 
        pearsonCor_CVP = stats.pearsonr(all_priorQ_rankZscores, all_CVPsklearnQ_rankZscores,alternative='two-sided')[0]
        # compute Spearman's Rank Correlation Coefficient
        spearmanCor_CVP = stats.spearmanr(all_priorQ_rankZscores, all_CVPsklearnQ_rankZscores,alternative='two-sided')[0]

        # OLS fit
        z1,p1 = curveFitOLS(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
        axs[i,1].plot(all_CVPsklearnQ_rankZscores,p1(all_CVPsklearnQ_rankZscores),"-", linewidth=0.5, color=cur_color, label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}')
        CVPSlopes.append(z1[0][0])
        axs[i,1].legend(loc="best", fontsize = fs)

        
        # ### using all raw points to fit
        all_priorQ_rankZscores = [t[tupIndex_prior] for t in combineAll]
        all_newModelsklearnQ_rankZscores = [t[tupIndex_newModel] for t in combineAll]
        axs[i, 2].scatter(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2, color=cur_color)

        # compute Pearson's Product Moment Correlation Coefficient 
        pearsonCor_newModel = stats.pearsonr(all_priorQ_rankZscores, all_newModelsklearnQ_rankZscores,alternative='two-sided')[0]
        # compute Spearman's Rank Correlation Coefficient
        spearmanCor_newModel = stats.spearmanr(all_priorQ_rankZscores, all_newModelsklearnQ_rankZscores,alternative='two-sided')[0]

        # OLS fit
        z3,p3 = curveFitOLS(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
        axs[i,2].plot(all_newModelsklearnQ_rankZscores,p3(all_newModelsklearnQ_rankZscores),"-", linewidth=0.5, color=cur_color, label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}')
        newModelSlopes.append(z3[0][0])
        axs[i,2].legend(loc="best", fontsize = fs)

        # ### using all raw points to fit
        all_priorQ_rankZscores = [t[tupIndex_prior] for t in combineAll]
        all_newModelInteractionsklearnQ_rankZscores = [t[tupIndex_newModelInteraction] for t in combineAll]
        axs[i, 3].scatter(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2, color=cur_color)

        # compute Pearson's Product Moment Correlation Coefficient 
        pearsonCor_newModelInteraction = stats.pearsonr(all_priorQ_rankZscores, all_newModelInteractionsklearnQ_rankZscores,alternative='two-sided')[0]
        # compute Spearman's Rank Correlation Coefficient
        spearmanCor_newModelInteraction = stats.spearmanr(all_priorQ_rankZscores, all_newModelInteractionsklearnQ_rankZscores,alternative='two-sided')[0]

        # OLS fit
        z4,p4 = curveFitOLS(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores)
        axs[i,3].plot(all_newModelInteractionsklearnQ_rankZscores,p4(all_newModelInteractionsklearnQ_rankZscores),"-", linewidth=0.5, color=cur_color, label=f'slope={round(z4[0][0],4)}\nresidual={round(z4[1][0],4)}')
        newModelInteractionSlopes.append(z4[0][0])
        axs[i,3].legend(loc="best", fontsize = fs)

    fig.suptitle(f"{commName.replace('.stackexchange','')} (questions with {answerCountThreshold} answers)")
    savePlot(fig, plotFileName)
    return plotQids, given_myColorDict
    
##########################################################################

def myFun(commName, commDir, rootDir, roundIndex, variation, reg_alpha1, reg_alpha2):

    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "semiSynthetic22_compareResults_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

        
    # prior coefs
    if variation == "":
        if roundIndex in [19,20,21]:
            with open(intermediate_directory+'/'+f"temperalOrderTraining11_CVP_regAlpha({reg_alpha1})_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_CVP = pickle.load( inputFile)
        else:
            # load CVP one-side temperal training result
            with open(intermediate_directory+'/'+'temperalOrderTraining8_CVP_return.dict', 'rb') as inputFile:
                return_trainSuccess_dict_CVP = pickle.load( inputFile)
    elif variation == "_fixedTau_noRL":
        if roundIndex in [8, 9,10,11,12, 13,14,15, 16, 17]:
            with open(intermediate_directory+'/'+f"temperalOrderTraining10_CVP_fixedTau_noRL_regAlpha({reg_alpha1})_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_CVP = pickle.load( inputFile)
        else:
            with open(intermediate_directory+'/'+'temperalOrderTraining9_CVP_fixedTau_noRL_return.dict', 'rb') as inputFile:
                return_trainSuccess_dict_CVP = pickle.load( inputFile)

    print(f"return train success dict CVP loaded. length {len(return_trainSuccess_dict_CVP)}")

    # learned coefs by CVP
    if roundIndex in [16,17]:
        with open(intermediate_directory+'/'+f'semiSynthetic12_CVP{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict', 'rb') as inputFile:
            semiSynthetic_return_trainSuccess_dict_CVP = pickle.load( inputFile)
    elif roundIndex in [19,20, 21]:
        with open(intermediate_directory+'/'+f'semiSynthetic9_CVP{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict', 'rb') as inputFile:
            semiSynthetic_return_trainSuccess_dict_CVP = pickle.load( inputFile)
    else:
        with open(intermediate_directory+'/'+f'semiSynthetic12_CVP{variation}_round{roundIndex}_training_return.dict', 'rb') as inputFile:
            semiSynthetic_return_trainSuccess_dict_CVP = pickle.load( inputFile)
    print(f"semiSynthetic_return train success dict CVP loaded. ")

    # learned coefs by new model
    if roundIndex in [16,17]:
        with open(intermediate_directory+'/'+f'semiSynthetic13_newModel{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict', 'rb') as inputFile:
            semiSynthetic_return_trainSuccess_dict_newModel = pickle.load( inputFile)
    if roundIndex in [19,20, 21]:
        with open(intermediate_directory+f"/semiSynthetic10_newModelTraining{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_posTau_return.dict", 'rb') as inputFile:
            semiSynthetic_return_trainSuccess_dict_newModel = pickle.load( inputFile)
    else:
        with open(intermediate_directory+f"/semiSynthetic13_newModel{variation}_round{roundIndex}_training_return.dict", 'rb') as inputFile:
            semiSynthetic_return_trainSuccess_dict_newModel = pickle.load( inputFile)
    print(f"semiSynthetic_return train success dict new Model loaded. ")

    # learned coefs by new model with interaction
    if roundIndex in [21]:
        with open(intermediate_directory+f"/semiSynthetic21_newModelInteractionTraining{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_return.dict", 'rb') as inputFile:
            semiSynthetic_return_trainSuccess_dict_newModelInteraction = pickle.load( inputFile)
    print(f"semiSynthetic_return train success dict new Model with interaction loaded. ")

    # get prior coefs
    try:
        if len(return_trainSuccess_dict_CVP)==1:
            simplifiedCommName = list(return_trainSuccess_dict_CVP.keys())[0]
        prior_coefs = return_trainSuccess_dict_CVP[simplifiedCommName]['coefs_sklearn']
        prior_lamb = prior_coefs[0] # for one side training
        prior_nus = return_trainSuccess_dict_CVP[simplifiedCommName]['nus_sklearn']
    except:
        print(f"No CVP voting stage training results for {commName}")
        return

    # get learned coefs by training CVP with SKLEARN
    try:
        semiSynthetic_coefs_CVP_sklearn = semiSynthetic_return_trainSuccess_dict_CVP[commName]['coefs_sklearn']
        semiSynthetic_lamb_CVP_sklearn = semiSynthetic_coefs_CVP_sklearn[0] # for one side training
        semiSynthetic_qs_CVP_sklearn = semiSynthetic_return_trainSuccess_dict_CVP[commName]['qs_sklearn']
        semiSynthetic_nus_CVP_sklearn = semiSynthetic_return_trainSuccess_dict_CVP[commName]['nus_sklearn']
    except:
        print(f"No semiSynthetic CVP voting stage training with SKLEARN results for {commName}")
        return
    
    # get learned coefs by training CVP with torchLBFGS
    try:
        semiSynthetic_coefs_CVP_torchLBFGS = semiSynthetic_return_trainSuccess_dict_CVP[commName]['coefs_lbfgs']
        semiSynthetic_lamb_CVP_torchLBFGS = semiSynthetic_coefs_CVP_torchLBFGS[0] # for one side training
        semiSynthetic_qs_CVP_torchLBFGS = semiSynthetic_return_trainSuccess_dict_CVP[commName]['qs_lbfgs']
        semiSynthetic_nus_CVP_torchLBFGS = semiSynthetic_return_trainSuccess_dict_CVP[commName]['nus_lbfgs']
    except:
        print(f"No semiSynthetic CVP voting stage training with torch SGD results for {commName}")
        return
    
    # get learned coefs by training newModel with SKLEARN
    try:
        semiSynthetic_coefs_newModel_sklearn = semiSynthetic_return_trainSuccess_dict_newModel[commName]['coefs_sklearn']
        semiSynthetic_lamb_newModel_sklearn = semiSynthetic_coefs_newModel_sklearn[0] # for one side training
        semiSynthetic_beta_newModel_sklearn = semiSynthetic_coefs_newModel_sklearn[1] # for one side training
        semiSynthetic_qs_newModel_sklearn = semiSynthetic_return_trainSuccess_dict_newModel[commName]['qs_sklearn']
        semiSynthetic_nus_newModel_sklearn = semiSynthetic_return_trainSuccess_dict_newModel[commName]['nus_sklearn']
        semiSynthetic_tau_newModel_sklearn = 1
    except:
        print(f"No semiSynthetic new model training results for {commName}")
        return

    # get learned coefs by training newModel with torchLBFGS
    try:
        semiSynthetic_coefs_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['coefs_lbfgs']
        if semiSynthetic_coefs_newModel == None:
            semiSynthetic_lamb_newModel = None
        else:
            semiSynthetic_lamb_newModel = semiSynthetic_coefs_newModel[0] # for one side training
        semiSynthetic_qs_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['qs_lbfgs']
        semiSynthetic_nus_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['nus_lbfgs']
        if roundIndex in [20]:
            semiSynthetic_tau_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['tau_lbfgs']
        else:
            semiSynthetic_tau_newModel = 1
    except:
        print(f"No semiSynthetic new model training results for {commName}")
        return

    # get learned coefs by training newModelInteraction with SKLEARN
    try:
        semiSynthetic_coefs_newModelInteraction_sklearn = semiSynthetic_return_trainSuccess_dict_newModelInteraction[commName]['coefs_sklearn']
        semiSynthetic_lamb_newModelInteraction_sklearn = semiSynthetic_coefs_newModelInteraction_sklearn[0] # for one side training
        semiSynthetic_beta_newModelInteraction_sklearn = semiSynthetic_coefs_newModelInteraction_sklearn[1] # for one side training
        semiSynthetic_qs_newModelInteraction_sklearn = semiSynthetic_return_trainSuccess_dict_newModelInteraction[commName]['qs_sklearn']
        semiSynthetic_bias_newModelInteraction_sklearn = semiSynthetic_return_trainSuccess_dict_newModelInteraction[commName]['bias_sklearn']
        if semiSynthetic_bias_newModelInteraction_sklearn != None:
            semiSynthetic_qs_newModelInteraction_sklearn = [q + semiSynthetic_bias_newModelInteraction_sklearn for q in semiSynthetic_qs_newModelInteraction_sklearn] # update qs by adding bias
        semiSynthetic_nus_newModelInteraction_sklearn = semiSynthetic_return_trainSuccess_dict_newModelInteraction[commName]['nus_sklearn']
        semiSynthetic_tau_newModelInteraction_sklearn = 1
    except:
        print(f"No semiSynthetic new model training results for {commName}")
        return
    
    # get learned coefs by training newModelInteraction with torchLBFGS
    try:
        semiSynthetic_coefs_newModelInteraction = semiSynthetic_return_trainSuccess_dict_newModelInteraction[commName]['coefs_lbfgs']
        if semiSynthetic_coefs_newModelInteraction == None:
            semiSynthetic_lamb_newModelInteraction = None
            semiSynthetic_beta_newModelInteraction = None
        else:
            semiSynthetic_lamb_newModelInteraction = semiSynthetic_coefs_newModelInteraction[0] # for one side training
            semiSynthetic_beta_newModelInteraction = semiSynthetic_coefs_newModelInteraction[1] # for one side training
        semiSynthetic_qs_newModelInteraction = semiSynthetic_return_trainSuccess_dict_newModelInteraction[commName]['qs_lbfgs']
        semiSynthetic_bias_newModelInteraction = semiSynthetic_return_trainSuccess_dict_newModelInteraction[commName]['bias_lbfgs']
        if semiSynthetic_bias_newModelInteraction != None:
            semiSynthetic_qs_newModelInteraction = [q + semiSynthetic_bias_newModelInteraction for q in semiSynthetic_qs_newModelInteraction] # update qs by adding bias
        semiSynthetic_nus_newModelInteraction = semiSynthetic_return_trainSuccess_dict_newModelInteraction[commName]['nus_lbfgs']
        semiSynthetic_tau_newModelInteraction = 1
    except:
        print(f"No semiSynthetic new model training results for {commName}")
        return
    
    # get prior tau
    if roundIndex in [20]:
        result_directory = os.path.join(commDir, r'result_folder')
        with open(result_directory+'/'+ 'CVP1_selectionPhaseTrainingResults.dict', 'rb')  as inputFile:
            CVP_selectionPhaseResults= pickle.load( inputFile)
            learned_tau, tau_record, ll_record, convergeFlag, convergeIter = CVP_selectionPhaseResults

        prior_tau = learned_tau
    else:
        prior_tau = 1
    
    # get learned tau by CVP
    if roundIndex in [20]:
        result_directory = os.path.join(commDir, r'result_folder')
        with open(result_directory+'/'+ f'semiSynthetic_CVP{variation}_round{roundIndex}_regAlpha({reg_alpha1})_selectionPhaseTrainingResults.dict', 'rb')  as inputFile:
            CVP_selectionPhaseResults= pickle.load( inputFile)
            learned_tau, tau_record, ll_record, convergeFlag, convergeIter = CVP_selectionPhaseResults
        
        semiSynthetic_tau_CVP = learned_tau 
    else:
        semiSynthetic_tau_CVP = 1


    # get prior qs
    if roundIndex in [16, 17]:
        with open(intermediate_directory+'/'+f'simulated_data_byCVP{variation}_round{roundIndex}_regAlpha({reg_alpha1}).dict', 'rb') as inputFile:
            loadedFile = pickle.load( inputFile)
    elif roundIndex in [19,20, 21]:
        with open(intermediate_directory+'/'+f'simulated_data_byCVP_round{roundIndex}_regAlpha({reg_alpha1}).dict', 'rb') as inputFile:
            loadedFile = pickle.load( inputFile)
    else:
        with open(intermediate_directory+'/'+f'simulated_data_byCVP{variation}_round{roundIndex}.dict', 'rb') as inputFile:
            loadedFile = pickle.load( inputFile)
    simulatedQuestions = loadedFile[0]
    generated_neg_vote_count = loadedFile[4]
    generated_pos_vote_count = loadedFile[5]
    generated_vote_count = generated_neg_vote_count  +  generated_pos_vote_count
    generated_answer_count = loadedFile[6]
    generated_event_count = generated_vote_count + generated_answer_count
    originalQidList = list(simulatedQuestions.keys())

    # a dict mayp qid to prior nu
    qid2prior_nu = defaultdict()
    qid2CVP_sklearn_nu = defaultdict()
    qid2CVP_torchLBFGS_nu = defaultdict()
    qid2newModel_nu = defaultdict()
    qid2newModel_sklearn_nu = defaultdict()
    qid2newModelInteraction_nu = defaultdict()
    qid2newModelInteraction_sklearn_nu = defaultdict()

    for i, qid in enumerate(originalQidList):
        qid2prior_nu[qid] = prior_nus[i]
        qid2CVP_sklearn_nu[qid] = semiSynthetic_nus_CVP_sklearn[i]
        if semiSynthetic_nus_CVP_torchLBFGS != None:
            qid2CVP_torchLBFGS_nu[qid] = semiSynthetic_nus_CVP_torchLBFGS[i]
        else:
            qid2CVP_torchLBFGS_nu[qid] = 0
        if semiSynthetic_nus_newModel != None:
            qid2newModel_nu[qid] = semiSynthetic_nus_newModel[i]
        else:
            qid2newModel_nu[qid] = 0
        qid2newModel_sklearn_nu[qid] = semiSynthetic_nus_newModel_sklearn[i]
        if semiSynthetic_nus_newModelInteraction != None:
            qid2newModelInteraction_nu[qid] = semiSynthetic_nus_newModelInteraction[i]
        else:
            qid2newModelInteraction_nu[qid] = 0
        qid2newModelInteraction_sklearn_nu[qid] = semiSynthetic_nus_newModelInteraction_sklearn[i]
    
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
    if roundIndex in [16,17,19,20, 21]:
        with open(intermediate_directory+'/'+f'semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha1})_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
            total_answersWithVotes_indice = pickle.load( inputFile)
    else:
        with open(intermediate_directory+'/'+f'semiSynthetic{variation}_round{roundIndex}_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
            total_answersWithVotes_indice = pickle.load( inputFile)
    
    total_answersWithVotes_ids = []
    answer2parentQ = defaultdict()

    for i,(qid, ai) in enumerate(total_answersWithVotes_indice):
        answerList = simulatedQuestions[qid]['answerList']
        aid = answerList[ai][0]
        total_answersWithVotes_ids.append((qid,aid))
        if aid not in answer2parentQ.keys():
            answer2parentQ[aid] = qid
    
    aid2CVP_sklearn_q = defaultdict()
    aid2CVP_torchLBFGS_q = defaultdict()
    aid2newModel_q = defaultdict()
    aid2newModel_sklearn_q = defaultdict()
    aid2newModelInteraction_q = defaultdict()
    aid2newModelInteraction_sklearn_q = defaultdict()
    for i, tup in enumerate(total_answersWithVotes_ids):
        qid, aid = tup
        q_CVP_sklearn = semiSynthetic_qs_CVP_sklearn[i]
        if semiSynthetic_qs_CVP_torchLBFGS != None:
            q_CVP_torchLBFGS = semiSynthetic_qs_CVP_torchLBFGS[i]
        else:
            q_CVP_torchLBFGS = 0
        if semiSynthetic_qs_newModel != None:
            q_newModel = semiSynthetic_qs_newModel[i]
        else:
            q_newModel = 0
        q_newModel_sklearn = semiSynthetic_qs_newModel_sklearn[i]
        if semiSynthetic_qs_newModelInteraction != None:
            q_newModelInteraction = semiSynthetic_qs_newModelInteraction[i]
        else:
            q_newModelInteraction = 0
        q_newModelInteraction_sklearn = semiSynthetic_qs_newModelInteraction_sklearn[i]
        aid2CVP_sklearn_q[aid] = q_CVP_sklearn
        aid2CVP_torchLBFGS_q[aid] = q_CVP_torchLBFGS
        aid2newModel_q[aid] = q_newModel
        aid2newModel_sklearn_q[aid] = q_newModel_sklearn
        aid2newModelInteraction_q[aid] = q_newModelInteraction
        aid2newModelInteraction_sklearn_q[aid] = q_newModelInteraction_sklearn
    
    # compute z-scores
    if roundIndex in [2,3,4]:
        """
        prior_qs = []
        semiSynthetic_qs_CVP_sklearn = []
        semiSynthetic_qs_CVP_torchLBFGS = []
        semiSynthetic_qs_newModel = []
        for aid in aid2CVP_sklearn_q.keys():
            prior_qs.append(aid2prior_q[aid])
            semiSynthetic_qs_CVP_sklearn.append(aid2CVP_sklearn_q[aid])
            semiSynthetic_qs_CVP_torchLBFGS.append(aid2CVP_torchLBFGS_q[aid])
            semiSynthetic_qs_newModel.append(aid2newModel_q[aid])


        # compute Pearson's Product Moment Correlation Coefficient 
        correlation_CVP_sklearn_to_prior = pearsonr(prior_qs, semiSynthetic_qs_CVP_sklearn,alternative='two-sided').statistic
        correlation_CVP_sklearn_to_prior_pvalue = pearsonr(prior_qs, semiSynthetic_qs_CVP_sklearn,alternative='two-sided').pvalue
        correlation_CVP_torchLBFGS_to_prior = pearsonr(prior_qs, semiSynthetic_qs_CVP_torchLBFGS,alternative='two-sided').statistic
        correlation_CVP_torchLBFGS_to_prior_pvalue = pearsonr(prior_qs, semiSynthetic_qs_CVP_torchLBFGS,alternative='two-sided').pvalue
        correlation_newModel_to_prior = pearsonr(prior_qs, semiSynthetic_qs_newModel,alternative='two-sided').statistic
        correlation_newModel_to_prior_pvalue = pearsonr(prior_qs, semiSynthetic_qs_newModel,alternative='two-sided').pvalue
        """

        prior_q = aid2prior_q[0]
        semiSynthetic_q_CVP_sklearn = aid2CVP_sklearn_q[0]
        semiSynthetic_q_CVP_torchLBFGS = aid2CVP_torchLBFGS_q[0]
        semiSynthetic_q_newModel = aid2newModel_q[0]

    
    else: 
        # compare ranking order based on different qs and voteDiff
            
        # # load voteDiff before the last vote of each answer for torchLBFGS training
        # with open(intermediate_directory+f"/semiSynthetic8{variation}_round{roundIndex}_outputs.dict", 'rb') as inputFile:
        #     _, qid2aid2voteDiffBeforeLastTestingVote = pickle.load( inputFile)

        aid2Prior_q_rankZscore = defaultdict()
        aid2VoteDiff_rankZscore = defaultdict()
        aid2CVP_sklearn_q_rankZscore  = defaultdict()
        aid2CVP_torchLBFGS_q_rankZscore  = defaultdict()
        aid2newModel_q_rankZscore  = defaultdict()
        aid2newModel_sklearn_q_rankZscore  = defaultdict()
        aid2newModelInteraction_q_rankZscore  = defaultdict()
        aid2newModelInteraction_sklearn_q_rankZscore  = defaultdict()

        aid2Prior_q_rank = defaultdict()
        aid2VoteDiff_rank = defaultdict()
        aid2CVP_sklearn_q_rank  = defaultdict()
        aid2CVP_torchLBFGS_q_rank  = defaultdict()
        aid2newModel_q_rank  = defaultdict()
        aid2newModel_sklearn_q_rank  = defaultdict()
        aid2newModelInteraction_q_rank  = defaultdict()
        aid2newModelInteraction_sklearn_q_rank  = defaultdict()

        top3AnswersIdList = []
        topThreshold = 3

        for qid, d in simulatedQuestions.items():
            answerList = [tup[0] for tup in d['answerList']]
            involved_answerList = set(answerList).intersection(set(aid2CVP_sklearn_q.keys()))
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
            
            for ai, voteList in involved_ai2voteList.items():
                involved_aid2voteDiff[answerList[ai]] = sum(voteList)
            
            # get rankZscore based on voteDiff
            involved_aid2rankBasedOnVoteDiff = defaultdict()
            for i, kv in enumerate(sorted(involved_aid2voteDiff.items(), key=lambda kv: (kv[1],kv[0]),reverse=True)) : # sort by the vote score and then by the answer index
                aid = kv[0]
                involved_aid2rankBasedOnVoteDiff[aid] = i+1 # rank
                if i < topThreshold:
                    top3AnswersIdList.append(aid)
            
            aid2VoteDiff_rank.update(involved_aid2rankBasedOnVoteDiff)

            zScores = get_zScore(involved_aid2rankBasedOnVoteDiff.values())
            for i,aid in enumerate(involved_aid2rankBasedOnVoteDiff.keys()):
                aid2VoteDiff_rankZscore[aid] = zScores[i]
            
            # # get voteDiff before the last vote of each answer
            # aid2voteDiffBeforeLastTestingVote = qid2aid2voteDiffBeforeLastTestingVote[qid]
            # involved_aid2voteDiffBeforeLastTestingVote = {aid : aid2voteDiffBeforeLastTestingVote[aid] for aid in involved_answerList}

            # # get ranks based on voteDiffBeforeLastVote
            # involved_aid2rankBasedOnVoteDiffBeforeLastVote = defaultdict()
            # for i, kv in enumerate(sorted(involved_aid2voteDiffBeforeLastTestingVote.items(), key=lambda kv: kv[1],reverse=True)) :
            #     aid = kv[0]
            #     involved_aid2rankBasedOnVoteDiffBeforeLastVote[aid] = i+1 # rank
            
            # zScores = get_zScore(involved_aid2rankBasedOnVoteDiffBeforeLastVote.values())
            # for i,aid in enumerate(involved_aid2rankBasedOnVoteDiffBeforeLastVote.keys()):
            #     aid2VoteDiffBeforeLastVote_rankZscore[aid] = zScores[i]

            # get ranks based on prior_q
            involved_aid2prior_q = {aid : aid2prior_q[aid] for aid in involved_answerList}
            involved_aid2rankBasedOnPriorQ = defaultdict()
            for i, kv in enumerate(sorted(involved_aid2prior_q.items(), key=lambda kv: (kv[1],kv[0]),reverse=True)) :
                aid = kv[0]
                involved_aid2rankBasedOnPriorQ[aid] = i+1 # rank

            aid2Prior_q_rank.update(involved_aid2rankBasedOnPriorQ)
            
            zScores = get_zScore(involved_aid2rankBasedOnPriorQ.values())
            for i,aid in enumerate(involved_aid2rankBasedOnPriorQ.keys()):
                aid2Prior_q_rankZscore[aid] = zScores[i]
            
            # get ranks based on CVP_sklearn_q
            involved_aid2CVP_sklearn_q = {aid : aid2CVP_sklearn_q[aid] for aid in involved_answerList}
            involved_aid2rankBasedOnCVPsklearnQ = defaultdict()
            for i, kv in enumerate(sorted(involved_aid2CVP_sklearn_q.items(), key=lambda kv: (kv[1],kv[0]),reverse=True)) :
                aid = kv[0]
                involved_aid2rankBasedOnCVPsklearnQ[aid] = i+1 # rank
            
            aid2CVP_sklearn_q_rank.update(involved_aid2rankBasedOnCVPsklearnQ)
            
            zScores = get_zScore(involved_aid2rankBasedOnCVPsklearnQ.values())
            for i,aid in enumerate(involved_aid2rankBasedOnCVPsklearnQ.keys()):
                aid2CVP_sklearn_q_rankZscore[aid] = zScores[i]

            # get ranks based on CVP_torchLBFGS_q
            involved_aid2CVP_torchLBFGS_q = {aid : aid2CVP_torchLBFGS_q[aid] for aid in involved_answerList}
            involved_aid2rankBasedOnCVPtorchQ = defaultdict()
            for i, kv in enumerate(sorted(involved_aid2CVP_torchLBFGS_q.items(), key=lambda kv: (kv[1],kv[0]),reverse=True)) :
                aid = kv[0]
                involved_aid2rankBasedOnCVPtorchQ[aid] = i+1 # rank
            
            aid2CVP_torchLBFGS_q_rank.update(involved_aid2rankBasedOnCVPtorchQ)
            
            zScores = get_zScore(involved_aid2rankBasedOnCVPtorchQ.values())
            for i,aid in enumerate(involved_aid2rankBasedOnCVPtorchQ.keys()):
                aid2CVP_torchLBFGS_q_rankZscore[aid] = zScores[i]
            
            # get ranks based on newModel_q
            involved_aid2newModel_q = {aid : aid2newModel_q[aid] for aid in involved_answerList}
            involved_aid2rankBasedOnNewModelQ = defaultdict()
            for i, kv in enumerate(sorted(involved_aid2newModel_q.items(), key=lambda kv: (kv[1],kv[0]),reverse=True)) :
                aid = kv[0]
                involved_aid2rankBasedOnNewModelQ[aid] = i+1 # rank
            
            aid2newModel_q_rank.update(involved_aid2rankBasedOnNewModelQ)
            
            zScores = get_zScore(involved_aid2rankBasedOnNewModelQ.values())
            for i,aid in enumerate(involved_aid2rankBasedOnNewModelQ.keys()):
                aid2newModel_q_rankZscore[aid] = zScores[i]
            
            
            # get ranks based on newModel_q_sklearn
            involved_aid2newModel_sklearn_q = {aid : aid2newModel_sklearn_q[aid] for aid in involved_answerList}
            involved_aid2rankBasedOnNewModelsklearnQ = defaultdict()
            for i, kv in enumerate(sorted(involved_aid2newModel_sklearn_q.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
                aid = kv[0]
                involved_aid2rankBasedOnNewModelsklearnQ[aid] = i+1 # rank
            
            aid2newModel_sklearn_q_rank.update(involved_aid2rankBasedOnNewModelsklearnQ)
            
            zScores = get_zScore(involved_aid2rankBasedOnNewModelsklearnQ.values())
            for i,aid in enumerate(involved_aid2rankBasedOnNewModelsklearnQ.keys()):
                aid2newModel_sklearn_q_rankZscore[aid] = zScores[i]
            
            # get ranks based on newModelInteraction_q
            involved_aid2newModelInteraction_q = {aid : aid2newModelInteraction_q[aid] for aid in involved_answerList}
            involved_aid2rankBasedOnNewModelInteractionQ = defaultdict()
            for i, kv in enumerate(sorted(involved_aid2newModelInteraction_q.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
                aid = kv[0]
                involved_aid2rankBasedOnNewModelInteractionQ[aid] = i+1 # rank
            
            aid2newModelInteraction_q_rank.update(involved_aid2rankBasedOnNewModelInteractionQ)
            
            zScores = get_zScore(involved_aid2rankBasedOnNewModelInteractionQ.values())
            for i,aid in enumerate(involved_aid2rankBasedOnNewModelInteractionQ.keys()):
                aid2newModelInteraction_q_rankZscore[aid] = zScores[i]

            # get ranks based on newModelInteraction_q_sklearn
            involved_aid2newModelInteraction_sklearn_q = {aid : aid2newModelInteraction_sklearn_q[aid] for aid in involved_answerList}
            involved_aid2rankBasedOnNewModelInteractionsklearnQ = defaultdict()
            for i, kv in enumerate(sorted(involved_aid2newModelInteraction_sklearn_q.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
                aid = kv[0]
                involved_aid2rankBasedOnNewModelInteractionsklearnQ[aid] = i+1 # rank
            
            aid2newModelInteraction_sklearn_q_rank.update(involved_aid2rankBasedOnNewModelInteractionsklearnQ)
            
            zScores = get_zScore(involved_aid2rankBasedOnNewModelInteractionsklearnQ.values())
            for i,aid in enumerate(involved_aid2rankBasedOnNewModelInteractionsklearnQ.keys()):
                aid2newModelInteraction_sklearn_q_rankZscore[aid] = zScores[i]
        
        # save results
        with open(intermediate_directory+f"/semiSynthetic22{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_outputs.dict", 'wb') as outputFile:
            pickle.dump((aid2Prior_q_rankZscore,
                         aid2VoteDiff_rankZscore,
                         aid2CVP_sklearn_q_rankZscore,
                         aid2CVP_torchLBFGS_q_rankZscore,
                         aid2newModel_q_rankZscore,
                         aid2newModel_sklearn_q_rankZscore,
                         aid2newModelInteraction_q_rankZscore,
                         aid2newModelInteraction_sklearn_q_rankZscore,
                         aid2Prior_q_rank,
                         aid2VoteDiff_rank,
                         aid2CVP_sklearn_q_rank,
                         aid2CVP_torchLBFGS_q_rank,
                         aid2newModel_q_rank,
                         aid2newModel_sklearn_q_rank,
                         aid2newModelInteraction_q_rank,
                         aid2newModelInteraction_sklearn_q_rank), outputFile)
            print( f"saved aid2zScore_outputs of {commName}")
        

        # prepare for plot
        # prior_q_rankZscores = []
        # voteDiff_rankZscores = []
        # CVP_sklearn_q_rankZscores  = []
        # voteDiffBeforeLastVote_rankZscores = []
        # CVP_torchLBFGS_q_rankZscores  = []
        # newModel_q_rankZscores  = []
        combineAll = [] # a list of tuple corresponding to each answer (prior_q_rankZscore, voteDff_rankZscore, CVP_sklearn_q_rankZscore, voteDiffBeforeLastVote_rankZscore, CVP_torchLBFGS_q_rankZscore, newModel_q_rankZscore)
        combineAll_top3Answers = []

        for aid in aid2Prior_q_rankZscore.keys():
            tup = (aid2Prior_q_rankZscore[aid],
                   aid2VoteDiff_rankZscore[aid],
                   aid2CVP_sklearn_q_rankZscore[aid],
                   aid2CVP_torchLBFGS_q_rankZscore[aid],
                   aid2newModel_q_rankZscore[aid],
                   aid2newModel_sklearn_q_rankZscore[aid],
                   aid2newModelInteraction_q_rankZscore[aid],
                   aid2newModelInteraction_sklearn_q_rankZscore[aid],
                   aid2Prior_q_rank[aid],
                   aid2VoteDiff_rank[aid],
                   aid2CVP_sklearn_q_rank[aid],
                   aid2CVP_torchLBFGS_q_rank[aid],
                   aid2newModel_q_rank[aid],
                   aid2newModel_sklearn_q_rank[aid],
                   aid2newModelInteraction_q_rank[aid],
                   aid2newModelInteraction_sklearn_q_rank[aid],
                   answer2parentQ[aid])
            combineAll.append(tup)
            if aid in top3AnswersIdList:
                combineAll_top3Answers.append(tup)

        # group
        # group points by intervals based on voteDiff zscores
        group_combineAll_basedOnPriorQ = myGrouping(combineAll, sortingColumn=0)
        # group_combineAll_basedOnVoteDiff = myGrouping(combineAll, sortingColumn=1)
        # group_combineAll_basedOnCVPsklearnQ = myGrouping(combineAll, sortingColumn=2)
        # group_combineAll_basedOnCVPtorchQ = myGrouping(combineAll, sortingColumn=3)
        # group_combineAll_basedOnNewModelQ = myGrouping(combineAll, sortingColumn=4)

        # plotFileName = f"semiSynthetic22{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2}).png"
        # z0,z1,z3,z4 = myPlot(commName, group_combineAll_basedOnPriorQ, combineAll, plotFileName)

        # question-level plot
        questionLevelPlotFileNameForRankZscoreComparison = f"semiSynthetic22{variation}_questionLevelRankZscoreComparison_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2}).png"
        plotQids,given_myColorDict = questionLevelPlot(commName, combineAll,  questionLevelPlotFileNameForRankZscoreComparison, 'rankZscore')
        questionLevelPlotFileNameForRankComparison = f"semiSynthetic22{variation}_questionLevelRankComparison_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2}).png"
        plotQids = questionLevelPlot(commName, combineAll,  questionLevelPlotFileNameForRankComparison, 'rank', plotQids,given_myColorDict)

        #####################################################################################################################
        """
        # plot with top 3 answers data point
        group_combineAll_basedOnPriorQ = myGrouping(combineAll_top3Answers, sortingColumn=0)

        plt.cla()
        fig, axs = plt.subplots(2, 2)
        fig.tight_layout(pad=3.0)
        
        group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnPriorQ]
        group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnPriorQ]
        axs[0, 0].scatter(group_voteDiff_rankZscores, group_priorQ_rankZscores,s = 2)

        ### using all raw points
        all_priorQ_rankZscores = [t[0] for t in combineAll_top3Answers]
        all_voteDiff_rankZscores = [t[1] for t in combineAll_top3Answers]
        # axs[0, 0].scatter(all_voteDiff_rankZscores, all_priorQ_rankZscores,s = 2)
        
        axs[0, 0].set_xlabel('vote diff rankZscore', fontsize = 8)
        axs[0, 0].set_ylabel('prior q rankZscore', fontsize = 8)

        # OLS fit
        z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
        axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"r-", linewidth=1, label=f'slope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}')
        axs[0,0].legend(loc="best", fontsize = 6)

        group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnPriorQ]
        group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnPriorQ]
        axs[0, 1].scatter(group_CVPsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2)
        ### using all raw points
        all_priorQ_rankZscores = [t[0] for t in combineAll_top3Answers]
        all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll_top3Answers]
        # axs[0, 1].scatter(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2)
        
        axs[0, 1].set_xlabel('CVP sklearn q rankZscore', fontsize = 8)
        axs[0, 1].set_ylabel('prior q rankZscore', fontsize = 8)

        # OLS fit
        z1,p1 = curveFitOLS(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
        axs[0,1].plot(all_CVPsklearnQ_rankZscores,p1(all_CVPsklearnQ_rankZscores),"r-", linewidth=1, label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}')
        axs[0,1].legend(loc="best", fontsize = 6)

        # group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiffBeforeLastVote]
        # group_voteDiffBeforeLastVote_rankZscores = [t[3] for t in group_combineAll_basedOnVoteDiffBeforeLastVote]
        # axs[1, 0].scatter(group_voteDiffBeforeLastVote_rankZscores, group_priorQ_rankZscores,s = 2)
        # axs[1, 0].set_xlabel(f'vote diff before last \ntesting vote rankZscore', fontsize = 8)
        # axs[1, 0].set_ylabel('prior q rankZscore', fontsize = 8)
        # ### using all raw points
        # all_priorQ_rankZscores = [t[0] for t in combineAll]
        # all_voteDiffBeforeLastVote_rankZscores = [t[3] for t in combineAll]
        # # OLS fit
        # z2,p2 = curveFitOLS(all_voteDiffBeforeLastVote_rankZscores, all_priorQ_rankZscores)
        # axs[1,0].plot(all_voteDiffBeforeLastVote_rankZscores,p2(all_voteDiffBeforeLastVote_rankZscores),"r-", linewidth=1, label=f'slope={round(z2[0][0],4)}\nresidual={round(z2[1][0],4)}')
        # axs[1,0].legend(loc="best", fontsize = 6)

        group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnPriorQ]
        group_CVPtorchQ_rankZscores = [t[3] for t in group_combineAll_basedOnPriorQ]
        axs[1, 0].scatter(group_CVPtorchQ_rankZscores, group_priorQ_rankZscores,s = 2)
        ### using all raw points
        all_priorQ_rankZscores = [t[0] for t in combineAll_top3Answers]
        all_CVPtorchQ_rankZscores = [t[3] for t in combineAll_top3Answers]
        # axs[1, 0].scatter(all_CVPtorchQ_rankZscores, all_priorQ_rankZscores,s = 2)
        
        axs[1, 0].set_xlabel('CVP torchLBFGS q rankZscore', fontsize = 8)
        axs[1, 0].set_ylabel('prior q rankZscore', fontsize = 8)

        # OLS fit
        z3,p3 = curveFitOLS(all_CVPtorchQ_rankZscores, all_priorQ_rankZscores)
        axs[1,0].plot(all_CVPtorchQ_rankZscores,p3(all_CVPtorchQ_rankZscores),"r-", linewidth=1, label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}')
        axs[1,0].legend(loc="best", fontsize = 6)

        group_priorQ_rankZscores = [t[0] for t in  group_combineAll_basedOnPriorQ]
        group_newModelQ_rankZscores = [t[4] for t in  group_combineAll_basedOnPriorQ]
        axs[1, 1].scatter(group_newModelQ_rankZscores, group_priorQ_rankZscores,s = 2)
        ### using all raw points
        all_priorQ_rankZscores = [t[0] for t in combineAll_top3Answers]
        all_newModelQ_rankZscores = [t[4] for t in combineAll_top3Answers]
        # axs[1, 1].scatter(all_newModelQ_rankZscores, all_priorQ_rankZscores,s = 2)
        
        axs[1, 1].set_xlabel('newModel q rankZscore', fontsize = 8)
        axs[1, 1].set_ylabel('prior q rankZscore', fontsize = 8)

        # OLS fit
        z4,p4 = curveFitOLS(all_newModelQ_rankZscores, all_priorQ_rankZscores)
        axs[1,1].plot(all_newModelQ_rankZscores,p4(all_newModelQ_rankZscores),"r-", linewidth=1, label=f'slope={round(z4[0][0],4)}\nresidual={round(z3[1][0],4)}')
        axs[1,1].legend(loc="best", fontsize = 6)

        fig.suptitle(f"{commName.replace('.stackexchange','')} (top 3 answers)")
        savePlot(fig, f"semiSynthetic22{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_top3Answers.png")
        """

    


def main():
    rootDir = os.getcwd()
    t0=time.time()
    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # roundIndex = 1
    # roundIndex = 2  # one question one answer
    # roundIndex = 3  # one question one answer, 100 times vote count per answer
    # roundIndex = 4  # one question one answer, 1000 times vote count per answer, q_std = 1

    # variation = '_fixedTau_noRL'

    # # roundIndex = 16 # one question multiple answer, 10000 events, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]

    # roundIndex = 17 # multiple question multiple answer, amplified 10 times of original total event count, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]

    # # roundIndex = 18 # multiple question multiple answer, amplified 10 times of original total event count, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    # if roundIndex in [18]:
    #     variation = '_noRL'


    # roundIndex = 19 ## multiple question multiple answer, original total event count, fix tau = 1, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    # roundIndex = 20 ## multiple question multiple answer, original total event count, learn tau, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    # if roundIndex in [19, 20]:
    #     variation = ''

    # selected_reg_strengthList = [500, 700]

    # for roundIndex 19
    # selected_reg_strengthPairs = [("3dprinting.stackexchange", 500, 0.5),
    #                               ("3dprinting.stackexchange", 500, 100),
    #                               ("3dprinting.stackexchange", 700, 0.3),
    #                               ("latin.stackexchange", 500, 0.5),
    #                               ("latin.stackexchange", 500, 100),
    #                               ("meta.askubuntu", 500, 0.3),
    #                               ("meta.askubuntu", 500, 0.5),
    #                               ("meta.askubuntu", 500, 300),
    #                               ("meta.askubuntu", 700, 0.3),
    #                               ("meta.askubuntu", 700, 300),
    #                               ("lifehacks.stackexchange", 500, 0.3),
    #                               ("lifehacks.stackexchange", 500, 100),
    #                               ("lifehacks.stackexchange", 700, 0.3),
    #                               ("lifehacks.stackexchange", 700, 300),]
    
    # for roundIndex 20
    # selected_reg_strengthPairs = [("3dprinting.stackexchange", 500, 0.5),
    #                               ("latin.stackexchange", 500, 0.3),
    #                               ("meta.askubuntu", 500, 0.3),
    #                               ("meta.askubuntu", 500, 300),
    #                               ("meta.askubuntu", 700, 0.3),
    #                               ("meta.askubuntu", 700, 0.5),
    #                               ("meta.askubuntu", 700, 300),
    #                               ("lifehacks.stackexchange", 500, 0.3),
    #                               ("lifehacks.stackexchange", 500, 0.5),
    #                               ("lifehacks.stackexchange", 500, 50),
    #                               ("lifehacks.stackexchange", 500, 70),
    #                               ("lifehacks.stackexchange", 500, 100),
    #                               ("lifehacks.stackexchange", 700, 0.3),
    #                               ("lifehacks.stackexchange", 700, 300),]

    roundIndex = 21 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    commName2selected_reg_strengthList = {'cstheory.stackexchange':[800, 900, 1000],
                                          'stackoverflow':[1000],
                                          'unix.meta.stackexchange':[60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                          'politics.stackexchange':[900,1000]}
    variation = ''
    try_reg_strengthList = [0.1,0.2, 0.3,0.4, 0.5,0.6, 0.7,0.8,0.9,
                            1, 2, 3,4,5, 6, 7,8,9,
                            10,20, 30,40,50,60, 70,80,90,
                            100, 200, 300, 400, 500, 600, 700, 800, 900,
                            1000]
    
    selected_reg_strengthPairs = []
    for commName, selected_reg_strengthList in commName2selected_reg_strengthList.items():
        for reg_1 in selected_reg_strengthList:
            for reg_2 in try_reg_strengthList:
                selected_reg_strengthPairs.append((commName, reg_1, reg_2))
    
    
    # prepare args
    argsList = []
    for tup in selected_reg_strengthPairs:
        commName, reg_alpha1, reg_alpha2 = tup
        for comm in commDir_sizes_sortedlist:
            if comm[0] == commName:
                commDir = comm[1]
                break
        argsList.append((commName, commDir, rootDir, roundIndex, variation, reg_alpha1, reg_alpha2))


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
    print('semiSynthetic22_compareResults Done completely.    Elapsed: {:}.\n'.format(elapsed))
    
      
if __name__ == "__main__":
  
    # calling main function
    main()


##########################################################################