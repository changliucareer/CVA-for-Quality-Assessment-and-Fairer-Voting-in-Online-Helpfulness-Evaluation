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
from scipy.optimize import minimize
from scipy.optimize import curve_fit

def simplifiedpowerlawFuncForMLE(b,c):
    return 1/(x**b+1) + c

def MLERegressionOfSimplifiedPowerLaw (params):
    yhat = simplifiedpowerlawFuncForMLE(params[0],params[1])

    # compute PDF of observed values normally distributed around mean(yhat)
    # with a standard deviation of sd
    negLL = -np.sum(stats.norm.logpdf(y,loc=yhat,scale=params[2]))
    return negLL

def getEndFitRank (values):
    for i in range(len(values)):
        if i==0:
            continue
        else:
            if values[i]>values[i-1]:
                return i
            elif values[i-1] - values[i] <= 0.001:
                return i
    return len(values)

#######################################################################################################
def myPlot(cur_aidAndQidAndLastVoteAndAttributes, chosen_aid, rank2probV_newModel, rank2probV_newModel_givenEvenVotes, voteDiffRank, newModelRank, originalRank, commName, roundIndex, variation, reg_alpha_NewModel, cur_voteDiff, rl):
    curAid = cur_aidAndQidAndLastVoteAndAttributes['aid']
    curQid = cur_aidAndQidAndLastVoteAndAttributes['qid']

    originalRank = cur_aidAndQidAndLastVoteAndAttributes['lastAttributes'][1]
    
    # only plot the beginning
    global x,y
    ori_x=[i+1 for i in range(len(rank2probV_newModel))]

    y_realVoteRatio = [probV for rank, probV in rank2probV_newModel.items()]
    y_evenVoteRatio = [probV for rank, probV in rank2probV_newModel_givenEvenVotes.items()]

    endFit_realVoteRatio = getEndFitRank(y_realVoteRatio) # dynamically fitting
    endFit_evenVoteRatio = getEndFitRank(y_evenVoteRatio) # dynamically fitting

    # simplified PowerLaw using MLE 
    # let’s start with some random coefficient guesses and optimize
    initParams = np.array([1,1,1])
    
    # fit the y_realVoteRatio
    x = ori_x[:endFit_realVoteRatio]
    y = y_realVoteRatio[:endFit_realVoteRatio]
    spl_results = minimize(MLERegressionOfSimplifiedPowerLaw, initParams, method = 'Nelder-Mead', options={'disp': True})
    if spl_results.success:
        spl_estParams_forRealVoteRatio = spl_results.x
    else:
        print(f"for {commName} answer {curAid} failed to fit the y_realVoteRatio.")
        return None
    # fit the y_evenVoteRatio
    x = ori_x[:endFit_evenVoteRatio]
    y = y_evenVoteRatio[:endFit_evenVoteRatio]
    spl_results = minimize(MLERegressionOfSimplifiedPowerLaw, initParams, method = 'Nelder-Mead', options={'disp': True})
    if spl_results.success:
        spl_estParams_forEvenVoteRatio = spl_results.x
    else:
        print(f"for {commName} answer {curAid} failed to fit the y_evenVoteRatio.")
        return None

    # plot the bars and fitted curve for the chosen_aid
    if curAid == chosen_aid:
        Fontsize=40
        fig1, axes1 = plt.subplots(ncols=1, nrows=1)
        plt.grid(color='grey', linestyle='--', linewidth=0.2, alpha = 0.5)
        fig1.set_size_inches(20, 10)
        axes1.set_xlabel('rank',fontsize=Fontsize)
        axes1.set_ylabel('Vote Probability',fontsize=Fontsize)
        axes1.set_title(f"{commName}\nQuestion {curQid}, Answer {curAid}\n original at Rank {originalRank}, real relative length {round(rl,4)}, real vote difference {cur_voteDiff}",fontsize=20)
        min_y = min(min(y_realVoteRatio), min(y_evenVoteRatio))

        if round(min_y,1) > min_y:
            min_y = round(min_y,1)-0.1
        else:
            min_y = round(min_y,1)

        # plot double bars
        width = 0.3
        bottom = np.zeros(len(ori_x))

        weight_counts = {
            "realVoteRatio": np.array(y_realVoteRatio),
            "evenVoteRatio": np.array(y_evenVoteRatio),
        }
        mycolors = {
            "realVoteRatio": 'limegreen',
            "evenVoteRatio": 'skyblue',
        }
        myalphas = {
            "realVoteRatio": 0.5,
            "evenVoteRatio": 0.5,
        }
        myMoves = {
            "realVoteRatio": -width/2,
            "evenVoteRatio": +width/2,
        }

        for mylabel, weight_count in weight_counts.items():
            cur_x = [xx + myMoves[mylabel] for xx in ori_x]
            axes1.bar(cur_x, weight_count, width, label=mylabel, bottom=bottom, color=mycolors[mylabel],alpha = myalphas[mylabel])
            # bottom += weight_count # for stacked bar plot
            
            x =np.arange(ori_x[0], ori_x[-1]+1, 0.2)
            if mylabel == "realVoteRatio":
                cur_y_realVoteRatio = simplifiedpowerlawFuncForMLE(spl_estParams_forRealVoteRatio[0],spl_estParams_forRealVoteRatio[1])
                axes1.plot(x, cur_y_realVoteRatio, '-', color = mycolors[mylabel])
                # Annotate the end of the line
                axes1.text(x[-1], cur_y_realVoteRatio[-1], f'b_realVoteRatio={round(spl_estParams_forRealVoteRatio[0],2)}', fontsize=20, horizontalalignment='right')
            elif mylabel == "evenVoteRatio":
                cur_y_evenVoteRatio = simplifiedpowerlawFuncForMLE(spl_estParams_forEvenVoteRatio[0],spl_estParams_forEvenVoteRatio[1])
                axes1.plot(x, cur_y_evenVoteRatio, '-', color = mycolors[mylabel])
                # Annotate the end of the line
                axes1.text(x[-1], cur_y_evenVoteRatio[-1], f'b_evenVoteRatio={round(spl_estParams_forEvenVoteRatio[0],2)}', fontsize=20, horizontalalignment='right')

            
        xtickStep = 5
        xticks = list(np.arange(0, len(ori_x)+1, xtickStep))
        xticks[0] = 1
        axes1.set_xticks(xticks)
        
        # # add numbers on each bar
        # for rect in rects:
        #     height = rect.get_height()
        #     axes1.text(rect.get_x() + rect.get_width()/2., 1.05*height,
        #             '%5.3f' % height,\
        #             ha='center', va='bottom')
        # axes1.set_ylim(0, 1)

        axes1.set_ylim(min_y, 1)
        
        axes1.legend(fontsize=20, frameon=False)
        plt.setp(axes1.get_xticklabels(),fontsize=20)
        plt.setp(axes1.get_yticklabels(),fontsize=20)

        axes1.spines['top'].set_visible(False)
        axes1.spines['right'].set_visible(False)
        
        # save plot
        savePlot(fig1, f'counterfacutual_questions_newModelGenereated{variation}_round{roundIndex}_newModelRegAlpha({reg_alpha_NewModel})_chosenAid.pdf')
    
    return spl_estParams_forRealVoteRatio, spl_estParams_forEvenVoteRatio


def myNewPlot(rank2probV_newModel_posMood_avg,rank2probV_newModel_negMood_avg, rank2probV_newModel_givenEvenVotes_avg, commName, roundIndex, variation, reg_alpha_NewModel):
    # only plot the beginning
    global x,y
    ori_x=[i+1 for i in range(len(rank2probV_newModel_posMood_avg))]

    y_realVoteRatio_posMood = [probV for rank, probV in rank2probV_newModel_posMood_avg.items()]
    y_realVoteRatio_negMood = [probV for rank, probV in rank2probV_newModel_negMood_avg.items()]
    y_evenVoteRatio = [probV for rank, probV in rank2probV_newModel_givenEvenVotes_avg.items()]

    endFit_realVoteRatio_posMood = getEndFitRank(y_realVoteRatio_posMood) # dynamically fitting
    endFit_realVoteRatio_negMood = getEndFitRank(y_realVoteRatio_negMood) # dynamically fitting
    endFit_evenVoteRatio = getEndFitRank(y_evenVoteRatio) # dynamically fitting
    # endFit_realVoteRatio_posMood = len(ori_x) # fitting all
    # endFit_realVoteRatio_negMood = len(ori_x) # fitting all
    # endFit_evenVoteRatio = len(ori_x) # fitting all
    print(f"{commName} fitted to {endFit_realVoteRatio_posMood} for realVoteRatio_posMood,{endFit_realVoteRatio_negMood} for realVoteRatio_negMood, {endFit_evenVoteRatio} for evenVoteRatio.")

    # simplified PowerLaw using MLE 
    # let’s start with some random coefficient guesses and optimize
    initParams = np.array([1,1,1])
    
    # fit the y_realVoteRatio_posMood
    x = ori_x[:endFit_realVoteRatio_posMood]
    y = y_realVoteRatio_posMood[:endFit_realVoteRatio_posMood]
    spl_results = minimize(MLERegressionOfSimplifiedPowerLaw, initParams, method = 'Nelder-Mead', options={'disp': True})
    if spl_results.success:
        spl_estParams_forRealVoteRatio_posMood = spl_results.x
    else:
        print(f"for {commName} failed to fit the y_realVoteRatio_posMood.")
        return None
    
    # fit the y_realVoteRatio_negMood
    x = ori_x[:endFit_realVoteRatio_negMood]
    y = y_realVoteRatio_negMood[:endFit_realVoteRatio_negMood]
    spl_results = minimize(MLERegressionOfSimplifiedPowerLaw, initParams, method = 'Nelder-Mead', options={'disp': True})
    if spl_results.success:
        spl_estParams_forRealVoteRatio_negMood = spl_results.x
    else:
        print(f"for {commName} failed to fit the y_realVoteRatio_negMood.")
        return None

    # fit the y_evenVoteRatio
    x = ori_x[:endFit_evenVoteRatio]
    y = y_evenVoteRatio[:endFit_evenVoteRatio]
    spl_results = minimize(MLERegressionOfSimplifiedPowerLaw, initParams, method = 'Nelder-Mead', options={'disp': True})
    if spl_results.success:
        spl_estParams_forEvenVoteRatio = spl_results.x
    else:
        print(f"for {commName}failed to fit the y_evenVoteRatio.")
        return None

    # plot the bars and fitted curve for the avg of all answers
    Fontsize=40
    fig1, axes1 = plt.subplots(ncols=1, nrows=1)
    plt.grid(color='grey', linestyle='--', linewidth=0.2, alpha = 0.5)
    fig1.set_size_inches(20, 10)
    axes1.set_xlabel('rank',fontsize=Fontsize)
    axes1.set_ylabel('Vote Probability',fontsize=Fontsize)
    axes1.set_title(f"{commName.replace('.stackexchange','')}",fontsize=20)
    min_y = min(min(y_realVoteRatio_negMood), min(y_evenVoteRatio))

    if round(min_y,1) > min_y:
        min_y = round(min_y,1)-0.1
    else:
        min_y = round(min_y,1)

    # plot double bars
    width = 0.3
    bottom = np.zeros(len(ori_x))

    weight_counts = {
        "realVoteRatio_posMood": np.array(y_realVoteRatio_posMood),
        "evenVoteRatio": np.array(y_evenVoteRatio),
        "realVoteRatio_negMood": np.array(y_realVoteRatio_negMood),
    }
    mycolors = {
        "realVoteRatio_posMood": 'lightcoral',
        "realVoteRatio_negMood": 'limegreen',
        "evenVoteRatio": 'skyblue',
    }
    myalphas = {
        # "realVoteRatio_posMood": 0.5,
        # "realVoteRatio_negMood": 0.5,
        # "evenVoteRatio": 0.5,
        "realVoteRatio_posMood": 1,
        "realVoteRatio_negMood": 1,
        "evenVoteRatio": 1,
    }
    myMoves = {
        # "realVoteRatio_posMood": -width/2,
        # "realVoteRatio_negMood": -width/2,
        # "evenVoteRatio": +width/2,
        "realVoteRatio_posMood": 0,
        "realVoteRatio_negMood": 0,
        "evenVoteRatio": 0,
    }

    for mylabel, weight_count in weight_counts.items():
        cur_x = [xx + myMoves[mylabel] for xx in ori_x]
        axes1.bar(cur_x, weight_count, width, label=mylabel, bottom=bottom, color=mycolors[mylabel],alpha = myalphas[mylabel])
        # bottom += weight_count # for stacked bar plot
        
        x =np.arange(ori_x[0], ori_x[-1]+1, 0.2)
        if mylabel == "realVoteRatio_posMood":
            cur_y_realVoteRatio_posMood = simplifiedpowerlawFuncForMLE(spl_estParams_forRealVoteRatio_posMood[0],spl_estParams_forRealVoteRatio_posMood[1])
            axes1.plot(x, cur_y_realVoteRatio_posMood, '-', color = mycolors[mylabel])
            # Annotate the end of the line
            axes1.text(x[-1], cur_y_realVoteRatio_posMood[-1], f'b_realVoteRatio_posMood={round(spl_estParams_forRealVoteRatio_posMood[0],2)}', fontsize=20, horizontalalignment='right')
        
        elif mylabel == "evenVoteRatio":
            cur_y_evenVoteRatio = simplifiedpowerlawFuncForMLE(spl_estParams_forEvenVoteRatio[0],spl_estParams_forEvenVoteRatio[1])
            axes1.plot(x, cur_y_evenVoteRatio, '-', color = mycolors[mylabel])
            # Annotate the end of the line
            axes1.text(x[-1], cur_y_evenVoteRatio[-1], f'b_evenVoteRatio={round(spl_estParams_forEvenVoteRatio[0],2)}', fontsize=20, horizontalalignment='right')

        elif mylabel == "realVoteRatio_negMood":
            cur_y_realVoteRatio_negMood = simplifiedpowerlawFuncForMLE(spl_estParams_forRealVoteRatio_negMood[0],spl_estParams_forRealVoteRatio_negMood[1])
            axes1.plot(x, cur_y_realVoteRatio_negMood, '-', color = mycolors[mylabel])
            # Annotate the end of the line
            if cur_y_realVoteRatio_negMood[-1] <= min_y:
                text_y = min_y + 0.02
            else:
                text_y = cur_y_realVoteRatio_negMood[-1]
            axes1.text(x[-1], text_y, f'b_realVoteRatio_negMood={round(spl_estParams_forRealVoteRatio_negMood[0],2)}', fontsize=20, horizontalalignment='right')

    xtickStep = 5
    xticks = list(np.arange(0, len(ori_x)+1, xtickStep))
    xticks[0] = 1
    axes1.set_xticks(xticks)
    
    # # add numbers on each bar
    # for rect in rects:
    #     height = rect.get_height()
    #     axes1.text(rect.get_x() + rect.get_width()/2., 1.05*height,
    #             '%5.3f' % height,\
    #             ha='center', va='bottom')
    # axes1.set_ylim(0, 1)

    axes1.set_ylim(min_y, 1)
    
    axes1.legend(fontsize=20, frameon=False)
    plt.setp(axes1.get_xticklabels(),fontsize=20)
    plt.setp(axes1.get_yticklabels(),fontsize=20)

    axes1.spines['top'].set_visible(False)
    axes1.spines['right'].set_visible(False)
    
    # save plot
    savePlot(fig1, f'counterfacutual_questions_newModelGenereated{variation}_round{roundIndex}_newModelRegAlpha({reg_alpha_NewModel}).pdf')

    return spl_estParams_forEvenVoteRatio
#####################################################################################################################
def processAnswer(cur_aidAndQidAndLastVoteAndAttributes, chosen_aid, avgAnswerCount, aid2quality_newModel, aid2newModel_sklearn_q_rank, aid2helpfulScore_rank, aid2VoteDiff_rank, coefs, qid2nus_newModel, commName, roundIndex, variation, reg_alpha_NewModel):
    # compute probV for the chosen answer at different rank
    rank2probV_newModel = defaultdict()
    cur_AnswerCount = cur_aidAndQidAndLastVoteAndAttributes['lastAttributes'][3]
    curAid = cur_aidAndQidAndLastVoteAndAttributes['aid']
    curQid = cur_aidAndQidAndLastVoteAndAttributes['qid']
    q = aid2quality_newModel[curAid]
    nu = qid2nus_newModel[curQid]
    seen_pvr = cur_aidAndQidAndLastVoteAndAttributes['lastAttributes'][0]
    cur_voteDiff = cur_aidAndQidAndLastVoteAndAttributes['lastAttributes'][4] - cur_aidAndQidAndLastVoteAndAttributes['lastAttributes'][5] 
    rl = cur_aidAndQidAndLastVoteAndAttributes['lastAttributes'][2]
    lamb = coefs[0]
    beta = coefs[1]

    newModelRank = aid2newModel_sklearn_q_rank[curAid]
    helpfulRank = aid2helpfulScore_rank[curAid]
    voteDiffRank = aid2VoteDiff_rank[curAid]
    originalRank = cur_aidAndQidAndLastVoteAndAttributes['lastAttributes'][1]

    maxRank = 20
    # maxRank = avgAnswerCount
    for r in range(1, maxRank+1):
        cur_IPW = (1/(1+r))
        z = q + lamb*seen_pvr + beta*cur_IPW + nu*rl
        logit = 1.0 / (1 + np.exp(-z))
        rank2probV_newModel[r] = logit
    
    # compute probV for the chosen answer at different rank by fixing seen_pvr = 0.5  (n_pos == n_neg)
    rank2probV_newModel_givenEvenVotes = defaultdict()
    for r in range(1, maxRank+1):
        cur_IPW = (1/(1+r))
        z = q + lamb*0.5 + beta*cur_IPW + nu*rl
        logit = 1.0 / (1 + np.exp(-z))
        rank2probV_newModel_givenEvenVotes[r] = logit

    # # difference between ProbVs of real vote ratio and even vote ratio on rank 1
    # probV_realVoteRatio = rank2probV_newModel[1]
    # probV_evenVoteRatio = rank2probV_newModel_givenEvenVotes[1]
    # probVdiff = abs(probV_realVoteRatio - probV_evenVoteRatio)
    
    # plot
    if curAid == chosen_aid:
        myPlot(cur_aidAndQidAndLastVoteAndAttributes, chosen_aid, rank2probV_newModel, rank2probV_newModel_givenEvenVotes, voteDiffRank, newModelRank, originalRank, commName, roundIndex, variation, reg_alpha_NewModel, cur_voteDiff, rl)
    # output_tuple = myPlot(cur_aidAndQidAndLastVoteAndAttributes, chosen_aid, rank2probV_newModel, rank2probV_newModel_givenEvenVotes, voteDiffRank, newModelRank, originalRank, commName, roundIndex, variation, reg_alpha_NewModel, cur_voteDiff, rl)
    # if output_tuple != None:
    #     spl_estParams_forRealVoteRatio, spl_estParams_forEvenVoteRatio = output_tuple
    #     return spl_estParams_forEvenVoteRatio, probVdiff
    # else:
    #     return None

    return rank2probV_newModel, rank2probV_newModel_givenEvenVotes


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
    # load intermediate outputs
    with open(intermediate_directory+f"/temperalOrderTraining15_verifyingQualities_outputs_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).dict", 'rb') as inputFile:
        tup = pickle.load( inputFile)
        aid2sentiment_rank = tup[0]
        aid2helpfulScore_rank = tup[1]
        aid2VoteDiff_rank = tup[2]
        aid2CVP_sklearn_q_rank = tup[3]
        aid2newModel_sklearn_q_rank = tup[4]
        aid2newModelInteraction_sklearn_q_rank = tup[5]
        aid2sentiment_rankZscore = tup[6]
        aid2helpfulScore_rankZscore = tup[7]
        aid2VoteDiff_rankZscore = tup[8]
        aid2CVP_sklearn_q_rankZscore = tup[9]
        aid2newModel_sklearn_q_rankZscore = tup[10]
        aid2newModelInteraction_sklearn_q_rankZscore = tup[11]
        disagreeAnswersIdList = tup[12]
        disagreeGaps = tup[13]
        total_answersWithVotes_ids = tup[14]
        print( f"loaded intermediate outputs for {commName}.")

      
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

    ############################################################################################
    qidList = list(Questions.keys())

    # get learned qs (without bias)
    try:
        if len(return_trainSuccess_dict_newModelInteraction)==1:
            simplifiedCommName = list(return_trainSuccess_dict_newModelInteraction.keys())[0]
        coefs_newModelInteraction = return_trainSuccess_dict_newModelInteraction[simplifiedCommName]['coefs_sklearn']  
        nus_newModelInteraction = return_trainSuccess_dict_newModelInteraction[simplifiedCommName]['nus_sklearn']
        learned_qs_interaction = return_trainSuccess_dict_newModelInteraction[simplifiedCommName]['qs_sklearn']  
        bias_newModelInteraction = return_trainSuccess_dict_newModelInteraction[simplifiedCommName]['bias_sklearn']
    except:
        coefs_newModelInteraction = return_trainSuccess_dict_newModelInteraction[commName]['coefs_sklearn']
        nus_newModelInteraction = return_trainSuccess_dict_newModelInteraction[commName]['nus_sklearn']
        learned_qs_interaction = return_trainSuccess_dict_newModelInteraction[commName]['qs_sklearn']  
        bias_newModelInteraction = return_trainSuccess_dict_newModelInteraction[commName]['bias_sklearn']
    if bias_newModelInteraction != None:
        learned_qs_interaction = [q + bias_newModelInteraction for q in learned_qs_interaction] # update qs by adding bias
    
    try:
        if len(return_trainSuccess_dict_newModel)==1:
            simplifiedCommName = list(return_trainSuccess_dict_newModel.keys())[0]
        coefs = return_trainSuccess_dict_newModel[simplifiedCommName]['coefs_sklearn']
        nus = return_trainSuccess_dict_newModel[simplifiedCommName]['nus_sklearn']
        learned_qs = return_trainSuccess_dict_newModel[simplifiedCommName]['qs_sklearn']  
    except:
        coefs = return_trainSuccess_dict_newModel[commName]['coefs_sklearn']
        nus = return_trainSuccess_dict_newModel[commName]['nus_sklearn']
        learned_qs = return_trainSuccess_dict_newModel[commName]['qs_sklearn']  

    try:
        if len(return_trainSuccess_dict_CVP)==1:
            simplifiedCommName = list(return_trainSuccess_dict_CVP.keys())[0]
        coefs_CVP = return_trainSuccess_dict_CVP[simplifiedCommName]['coefs_sklearn']
        nus_CVP = return_trainSuccess_dict_CVP[simplifiedCommName]['nus_sklearn']
        learned_qs_CVP = return_trainSuccess_dict_CVP[simplifiedCommName]['qs_sklearn']  
    except:
        coefs_CVP = return_trainSuccess_dict_CVP[commName]['coefs_sklearn']
        nus_CVP = return_trainSuccess_dict_CVP[commName]['nus_sklearn']
        learned_qs_CVP = return_trainSuccess_dict_CVP[commName]['qs_sklearn']  
    
    assert len(learned_qs)==len(total_answersWithVotes_indice)
    assert len(qidList)==len(nus)

    # get qid2nus
    qid2nus_CVP = dict()
    qid2nus_newModel = dict()
    qid2nus_newModelInteraction = dict()
    for i, qid in enumerate(qidList):
        qid2nus_CVP[qid] = nus_CVP[i]
        qid2nus_newModel[qid] = nus[i]
        qid2nus_newModelInteraction[qid] = nus_newModelInteraction[i]

    # extract answer ids 
    total_aids = set()
    total_answersWithVotes_ids = []
    for i,(qid, ai) in enumerate(total_answersWithVotes_indice):
        filtered_answerList = Questions[qid]['filtered_answerList']
        aid = filtered_answerList[ai]
        total_aids.add(aid)
        total_answersWithVotes_ids.append((qid,aid))
    
    aid2quality_CVP = defaultdict()
    aid2quality_newModel = defaultdict()
    aid2quality_newModelInteration = defaultdict()
    for i,tup in enumerate(total_answersWithVotes_ids):
        qid, aid = tup
        aid2quality_CVP[aid] = learned_qs_CVP[i]
        aid2quality_newModel[aid] = learned_qs[i]
        aid2quality_newModelInteration[aid] = learned_qs_interaction[i]

    
    # get qid 2 involved aids with their last vote and attributes
    qid2involved_aid2LastVoteAndAttributes = defaultdict()

    for qid, d in Questions.items():
        answerList =  d['answerList']
        involved_answerList = set(answerList).intersection(set(aid2quality_newModel.keys()))
        if len(involved_answerList) == 0: # none answer in the learned qs, skip this question
            continue

        # get voteDiff of each answer
        voteCount = 0
        involved_aid2LastVote = defaultdict()
        involved_aid2LastAttributes = defaultdict()
        
        involved_ai2voteList = defaultdict()
        involved_ai2attributesList = defaultdict()
        eventList = d['eventList']
        for e in eventList:
            eventType = e['et']
            if eventType != 'v': # skip all event that is not a vote
                continue
            voteCount += 1
            vote = e['v']
            if vote == 1:
                cur_vote = 1
            else: # vote == 0 is negative vote
                cur_vote = -1

            ai = e['ai']
            if answerList[ai] not in involved_answerList: # skip the events of non-involved answer
                continue
            
            ranksOfAnswersBeforeT = e['ranks']
            cur_rank = ranksOfAnswersBeforeT[ai]
            attributes = (e['seen_pvr'], cur_rank, e['rl'], e['J'], e['n_pos'],e['n_neg'])

            if ai in involved_ai2voteList.keys():
                involved_ai2voteList[ai].append(cur_vote)
                involved_ai2attributesList[ai].append(attributes)
            else:
                involved_ai2voteList[ai] = [cur_vote]
                involved_ai2attributesList[ai] = [attributes]
        
        if len(involved_ai2voteList)==0: # skip when no involved answer having votes
            continue 

        for ai, voteList in involved_ai2voteList.items():
            involved_aid2LastVote[answerList[ai]] = voteList[-1]
            involved_aid2LastAttributes[answerList[ai]] = involved_ai2attributesList[ai][-1]
        
        qid2involved_aid2LastVoteAndAttributes[qid] = (involved_aid2LastVote, involved_aid2LastAttributes, voteCount)
        
    Questions.clear()
    
    # when usning new model
    # choose one answer from one question
    chosen_aidAndQidAndLastVoteAndAttributes = {'qid':None,'aid':None,'lastVote':None,'lastAttributes':None}

    for qid, tup in qid2involved_aid2LastVoteAndAttributes.items():
        involved_aid2LastVote, involved_aid2LastAttributes, voteCount = tup
        
        if len(involved_aid2LastVote)<3: # skip
            continue

        for aid, lastVote in involved_aid2LastVote.items():
            lastAttributes = involved_aid2LastAttributes[aid]
            realVoteDiff = lastAttributes[4] - lastAttributes[5]
            if realVoteDiff < 10:
                continue
            
            if (lastVote==1 and lastAttributes[0] > 0.5) and (lastAttributes[1]>2 and lastAttributes[3]>3): # pos votes > neg votes, rank > 2, cur_answerCount >3
                if (aid in aid2newModel_sklearn_q_rank.keys()) and (aid in aid2helpfulScore_rank.keys()):
                    newModelRank = aid2newModel_sklearn_q_rank[aid]
                    helpfulRank = aid2helpfulScore_rank[aid]
                    voteDiffRank = aid2VoteDiff_rank[aid]

                    if (voteDiffRank > helpfulRank) and (newModelRank == helpfulRank):  # choose an underestimated answer
                        chosen_aidAndQidAndLastVoteAndAttributes['qid'] = qid
                        chosen_aidAndQidAndLastVoteAndAttributes['aid'] = aid
                        chosen_aidAndQidAndLastVoteAndAttributes['lastVote'] = lastVote
                        chosen_aidAndQidAndLastVoteAndAttributes['lastAttributes'] = lastAttributes
                        break

        if chosen_aidAndQidAndLastVoteAndAttributes['qid'] != None:
            break
    
    if chosen_aidAndQidAndLastVoteAndAttributes['aid'] == None: # found nothing
        print(f"for {commName} found nothing.")
        return

    # save intermediate outputs
    with open(intermediate_directory+f"/temperalOrderTraining18_outputs_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).dict", 'wb') as outputFile:
        pickle.dump((qid2involved_aid2LastVoteAndAttributes, chosen_aidAndQidAndLastVoteAndAttributes), outputFile)
        print( f"saved intermediate outputs for {commName}.")
    
    
    # load intermediate outputs
    with open(intermediate_directory+f"/temperalOrderTraining18_outputs_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).dict", 'rb') as inputFile: 
        tup = pickle.load( inputFile)
        qid2involved_aid2LastVoteAndAttributes = tup[0]
        chosen_aidAndQidAndLastVoteAndAttributes = tup[1]
        print( f"loaded intermediate outputs for {commName}.")

    totalQuestionCount = len(qid2involved_aid2LastVoteAndAttributes)
    totalAnswerCount = sum([len(v[0]) for k,v in qid2involved_aid2LastVoteAndAttributes.items()])
    totalVoteCount = sum([v[2] for k,v in qid2involved_aid2LastVoteAndAttributes.items()])

    # get average answerCount
    answerCountList = []
    for qid, tup in qid2involved_aid2LastVoteAndAttributes.items():
        involved_aid2LastVote, involved_aid2LastAttributes, voteCount = tup
        # cur_answerCount = 0
        # for aid, lastVote in involved_aid2LastVote.items():
        #     if (aid in aid2newModel_sklearn_q_rank.keys()) and (aid in aid2helpfulScore_rank.keys()):
        #         cur_answerCount += 1
        # answerCountList.append(cur_answerCount)
        answerCountList.append(len(involved_aid2LastVote))
    avgAnswerCount = int(mean(answerCountList))

    chosen_aid = chosen_aidAndQidAndLastVoteAndAttributes['aid']
    # fixedRealVoteDiff = 1 # only consider the answers with real vote diff = 1
    # fixedRealRL = 1 # only consider the answers with real RL = 1

    # process every chosen answer
    rank2probV_newModel_forAllAnswers_posMood = defaultdict()
    rank2probV_newModel_forAllAnswers_negMood = defaultdict()
    rank2probV_newModel_givenEvenVotes_forAllAnswers = defaultdict()

    # aid2QidAndEstimatedBs = defaultdict()
    for qid, tup in qid2involved_aid2LastVoteAndAttributes.items():
        involved_aid2LastVote, involved_aid2LastAttributes, voteCount = tup

        for aid, lastVote in involved_aid2LastVote.items():
            lastAttributes = involved_aid2LastAttributes[aid]
            realVoteDiff = lastAttributes[4] - lastAttributes[5]
            
            if (aid in aid2newModel_sklearn_q_rank.keys()) and (aid in aid2helpfulScore_rank.keys()):
                newModelRank = aid2newModel_sklearn_q_rank[aid]
                helpfulRank = aid2helpfulScore_rank[aid]
                voteDiffRank = aid2VoteDiff_rank[aid]

                cur_aidAndQidAndLastVoteAndAttributes = {'qid':qid,'aid':aid,'lastVote':lastVote,'lastAttributes':lastAttributes}
                cur_voteDiff = cur_aidAndQidAndLastVoteAndAttributes['lastAttributes'][4] - cur_aidAndQidAndLastVoteAndAttributes['lastAttributes'][5] 
                # if cur_voteDiff != fixedRealVoteDiff:
                #     continue
                # rl = cur_aidAndQidAndLastVoteAndAttributes['lastAttributes'][2]
                # if rl > fixedRealRL*1.01 or rl < fixedRealRL*0.99:
                #     continue

                output_tuple = processAnswer(cur_aidAndQidAndLastVoteAndAttributes, chosen_aid, avgAnswerCount, aid2quality_newModel, aid2newModel_sklearn_q_rank, aid2helpfulScore_rank, aid2VoteDiff_rank, coefs, qid2nus_newModel, commName, roundIndex, variation, reg_alpha_NewModel)
                # if output_tuple != None:
                #     spl_estParams_forEvenVoteRatio, probVdiff = output_tuple
                # else:
                #     continue
                # b_forEvenVoteRatio = spl_estParams_forEvenVoteRatio[0]
                # aid2QidAndEstimatedBs[aid] = (qid, b_forEvenVoteRatio, probVdiff)

                rank2probV_newModel, rank2probV_newModel_givenEvenVotes = output_tuple

                if cur_voteDiff > 0: # in pos mood
                    for r, probV in rank2probV_newModel.items():
                        if r in rank2probV_newModel_forAllAnswers_posMood.keys():
                            rank2probV_newModel_forAllAnswers_posMood[r].append(probV)
                        else:
                            rank2probV_newModel_forAllAnswers_posMood[r] = [probV]
                        if r in rank2probV_newModel_givenEvenVotes_forAllAnswers.keys():
                            rank2probV_newModel_givenEvenVotes_forAllAnswers[r].append(rank2probV_newModel_givenEvenVotes[r])
                        else:   
                            rank2probV_newModel_givenEvenVotes_forAllAnswers[r] = [rank2probV_newModel_givenEvenVotes[r]]
                elif cur_voteDiff < 0: # in neg mood
                    for r, probV in rank2probV_newModel.items():
                        if r in rank2probV_newModel_forAllAnswers_negMood.keys():
                            rank2probV_newModel_forAllAnswers_negMood[r].append(probV)
                        else:
                            rank2probV_newModel_forAllAnswers_negMood[r] = [probV]
                        if r in rank2probV_newModel_givenEvenVotes_forAllAnswers.keys():
                            rank2probV_newModel_givenEvenVotes_forAllAnswers[r].append(rank2probV_newModel_givenEvenVotes[r])
                        else:   
                            rank2probV_newModel_givenEvenVotes_forAllAnswers[r] = [rank2probV_newModel_givenEvenVotes[r]]
    
    # get avg rank2probVs
    rank2probV_newModel_posMood_avg = defaultdict()
    rank2probV_newModel_negMood_avg = defaultdict()
    rank2probV_newModel_givenEvenVotes_avg = defaultdict()

    probVDiffList_posMood = []
    probVDiffList_negMood = []
    
    for r, probVs in rank2probV_newModel_forAllAnswers_posMood.items():
        rank2probV_newModel_posMood_avg[r] = mean(probVs)
        rank2probV_newModel_negMood_avg[r] = mean(rank2probV_newModel_forAllAnswers_negMood[r])
        rank2probV_newModel_givenEvenVotes_avg[r] = mean(rank2probV_newModel_givenEvenVotes_forAllAnswers[r])

        probV_realVoteRatio_posMood = rank2probV_newModel_posMood_avg[r]
        probV_realVoteRatio_negMood = rank2probV_newModel_negMood_avg[r]
        probV_evenVoteRatio = rank2probV_newModel_givenEvenVotes_avg[r]
        
        assert probV_realVoteRatio_posMood > probV_evenVoteRatio
        assert probV_realVoteRatio_negMood < probV_evenVoteRatio

        probVdiff_posMood = probV_realVoteRatio_posMood - probV_evenVoteRatio
        probVdiff_negMood = probV_evenVoteRatio - probV_realVoteRatio_negMood
        probVDiffList_posMood.append(probVdiff_posMood)
        probVDiffList_negMood.append(probVdiff_negMood)
    
    probVdiff_posMood_avg = mean(probVDiffList_posMood)
    probVdiff_negMood_avg = mean(probVDiffList_negMood)
    

    # # get probV diff between real vote ratio and even vote ratio at rank 1
    # probV_realVoteRatio_posMood = rank2probV_newModel_posMood_avg[1]
    # probV_realVoteRatio_negMood = rank2probV_newModel_negMood_avg[1]
    # probV_evenVoteRatio = rank2probV_newModel_givenEvenVotes_avg[1]
    
    # assert probV_realVoteRatio_posMood > probV_evenVoteRatio
    # assert probV_realVoteRatio_negMood < probV_evenVoteRatio
    
    # probVdiff_posMood = probV_realVoteRatio_posMood - probV_evenVoteRatio
    # probVdiff_negMood = probV_evenVoteRatio - probV_realVoteRatio_negMood

    
    # plot
    spl_estParams_forEvenVoteRatio = myNewPlot(rank2probV_newModel_posMood_avg,rank2probV_newModel_negMood_avg, rank2probV_newModel_givenEvenVotes_avg, commName, roundIndex, variation, reg_alpha_NewModel)
    b_forEvenVoteRatio = spl_estParams_forEvenVoteRatio[0]
    
    # # integragate all estimatedBs
    # allBs_forEvenVoteRatio = []
    # allProbVDiffs = []
    # for aid, tup in aid2QidAndEstimatedBs.items():
    #     qid, b_forEvenVoteRatio, probVdiff = tup
    #     allBs_forEvenVoteRatio.append(b_forEvenVoteRatio)
    #     allProbVDiffs.append(probVdiff)
    
    # avgB_forEvenVoteRatio = mean(allBs_forEvenVoteRatio)
    # avgProbVDiff = mean(allProbVDiffs)

    # save intermediate outputs
    with open(intermediate_directory+f"/temperalOrderTraining18_estimatedBs_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).dict", 'wb') as outputFile:
        # pickle.dump((aid2QidAndEstimatedBs, avgB_forEvenVoteRatio, avgProbVDiff), outputFile)
        pickle.dump((avgAnswerCount, b_forEvenVoteRatio, probVdiff_posMood_avg, probVdiff_negMood_avg), outputFile)
        print( f"saved estimated b and probVdiff for {commName}.")


    # get comment counts
    if commName == 'stackoverflow':
            # in case using subComm reactjs
            with open(intermediate_directory+'/'+'filtered_aid2qidAndComments_tillCurChunk_reactjs.json') as json_file:
                filtered_aid2qidAndComments = json.load(json_file)
    else:
        with open(intermediate_directory+'/'+'filtered_aid2qidAndComments_tillCurChunk.json') as json_file:
            filtered_aid2qidAndComments = json.load(json_file)
    totalCommentCount = 0
    for aid, d in filtered_aid2qidAndComments.items():
        totalCommentCount += len(d['comments'])

    # save into csv
    with open(rootDir + f'/allComm_temperalOrderTraining18_avgBAsTrendiness_avgDeltaBAsConformity.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( [commName,totalQuestionCount, totalAnswerCount, totalVoteCount ,totalCommentCount, avgAnswerCount, b_forEvenVoteRatio, probVdiff_posMood_avg, probVdiff_negMood_avg])   
    

def main():

    t0=time.time()
    rootDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    # # save avgBs and avgDeltaBs into csv
    with open('allComm_temperalOrderTraining18_avgBAsTrendiness_avgDeltaBAsConformity.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","total question count","total answer count","total vote count","total comment count","average answer count per question","estimated b by curvefitting using even vote ratio", "probV difference between real vote ratio in pos mood and even vote ratio", "probV difference between real vote ratio in neg mood and even vote ratio"])


    # roundIndex = 1 ## multiple question multiple answer, original total event count, fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    roundIndex = 2 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    
    commName2selected_reg_strengthList = {
                                        # 'cstheory.stackexchange':(400,500,500),
                                        #   'unix.meta.stackexchange':(300,300,300),
                                        #   'stackoverflow':(0.1,0.1,0.1),
                                        #   'politics.stackexchange':(0.2,0.1,0.2),
                                        #   '3dprinting.stackexchange':(40,20,80),
                                        #   'latin.stackexchange':(0.3,0.3,0.3),
                                        #   'meta.askubuntu':(700,700,500),
                                        #   'lifehacks.stackexchange':(0.2,0.2,600)
                                        #   'math.meta.stackexchange':(0.4,0.4,0.2),
                                        # 'mathoverflow.net':(600,600,0.1),
                                        #     'mathematica.stackexchange':(90,80,100),
                                        #     'askubuntu':(0.1,600,0.1),
                                        #     'philosophy.stackexchange':(700,700,600),
                                            # 'codegolf.meta.stackexchange':(200,100,90), 
                                          }
    variation = '_fixedTau'
    
    # for sampled comms
    sampled_comms = ['academia.stackexchange','askubuntu',
                      'english.stackexchange','math.stackexchange','mathoverflow.net',
                      'meta.stackexchange','meta.stackoverflow','serverfault',
                      'softwareengineering.stackexchange','superuser','unix.stackexchange',
                      'worldbuilding.stackexchange','physics.stackexchange','electronics.stackexchange',
                      'codegolf.stackexchange','workplace.stackexchange']
    
    # load commName2selected_reg_strengthList (extracted from temperalOrderTraining15_verifyingQualities_GPT.py)
    with open(f'allComm_bestRegAlphas_fixedTau.dict', 'rb') as inputFile:
        commName2selected_reg_strengthList = pickle.load( inputFile)

    # prepare args
    argsList = []
    for commName, tup in commName2selected_reg_strengthList.items():
        reg_alpha_newModelInteraction, reg_alpha_newModel, reg_alpha_CVP, sampleCount = tup
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
        if len(processes)==5:
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
