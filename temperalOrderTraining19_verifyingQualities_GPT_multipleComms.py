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
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
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
    group_combineAll = []

    ### group by fixed number of answers
    group_count = 150
    group_size = int(len(combineAll)/group_count) # the number of answers in each group
    
    startIndex = 0
    for i in range(startIndex, len(combineAll), group_size):
        cur_group = combineAll[i:group_size+i]
        if len(cur_group)>0: 
            # groupCount = len(cur_group)
            column_wise_mean = np.mean(cur_group,axis=0)
            group_combineAll.append(tuple(list(column_wise_mean) + [group_size])) # force as the same group size
        startIndex = i

    # ### group by fixed range of z score
    # group_count = 150

    # zScoreIntervals = []
    # startZscore = -2
    # endZscore = 2
    # step = (endZscore-startZscore)/group_count
    # cur_zScore = startZscore + step
    # while cur_zScore <= endZscore:
    #     zScoreIntervals.append(cur_zScore)
    #     cur_zScore += step
    
    # if zScoreIntervals[-1] < endZscore:
    #     zScoreIntervals.append(endZscore)

    # startIndex = 0
    # for zintv in zScoreIntervals:
    #     cur_group = []
    #     for i in range(startIndex, len(combineAll)):
    #         cur_tup = combineAll[i]
    #         cur_zScore = cur_tup[sortingColumn]
    #         if cur_zScore <= zintv:
    #             cur_group.append(cur_tup)
    #         else:
    #             startIndex = i
    #             break
    #     if len(cur_group)>0: 
    #         groupCount = len(cur_group)
    #         column_wise_mean = np.mean(cur_group,axis=0)
    #         group_combineAll.append(tuple(list(column_wise_mean) + [groupCount]))

    # ### group by interval
    # group_count = 150
    # # group_count = len(combineAll) # equivalent to not grouping

    # percentages = [x / group_count for x in range(1, group_count+1)]
    # zScoreIntervals = [scipy.stats.norm.ppf(p) for p in percentages]

    # startIndex = 0
    # for zintv in zScoreIntervals:
    #     cur_group = []
    #     for i in range(startIndex, len(combineAll)):
    #         cur_tup = combineAll[i]
    #         cur_zScore = cur_tup[sortingColumn]
    #         if cur_zScore <= zintv:
    #             cur_group.append(cur_tup)
    #         else:
    #             startIndex = i
    #             break
    #     if len(cur_group)>0: 
    #         groupCount = len(cur_group)
    #         column_wise_mean = np.mean(cur_group,axis=0)
    #         group_combineAll.append(tuple(list(column_wise_mean) + [groupCount]))
    
    # print(f"group z score sample count {len(group_combineAll)} sorted by column {sortingColumn}")
    return group_combineAll

def myGrouping_new(combineAll, sortingColumn, underEstimatedAnswersIdList, overEstimatedAnswersIdList, disagreeAnswersIdList, targetType):
    combineAll.sort(key=lambda t:t[sortingColumn]) # sorted by zScore
    group_combineAll = []

    # ### group by fixed number of answers
    # group_count = 80
    # group_size = int(len(combineAll)/group_count) # the number of answers in each group
    # # filter out the non-disagree answers
    # if targetType != 'all':
    #     combineAll_filtered = [t for t in combineAll if t[-1] in disagreeAnswersIdList]
    # else: # all answers
    #     combineAll_filtered = combineAll
    
    # startIndex = 0
    # for i in range(startIndex, len(combineAll_filtered), group_size):
    #     cur_group = combineAll_filtered[i:group_size+i]
    #     if len(cur_group)>0: 
    #         underGroup = [t for t in cur_group if t[-1] in underEstimatedAnswersIdList]
    #         overGroup = [t for t in cur_group if t[-1] in overEstimatedAnswersIdList]
    #         normalGroup = [t for t in cur_group if t[-1] not in underEstimatedAnswersIdList and t[-1] not in overEstimatedAnswersIdList]
    #         if len(underGroup) > 0:
    #             column_wise_mean_underGroup = np.mean(underGroup,axis=0)
    #             group_combineAll.append(tuple(list(column_wise_mean_underGroup) + ['under']))
    #         if len(overGroup) > 0:
    #             column_wise_mean_overGroup = np.mean(overGroup,axis=0)
    #             group_combineAll.append(tuple(list(column_wise_mean_overGroup)+ ['over']))
    #         if len(normalGroup) > 0:
    #             column_wise_mean_normalGroup = np.mean(normalGroup,axis=0)
    #             group_combineAll.append(tuple(list(column_wise_mean_normalGroup)+ ['normal']))
    #     startIndex = i

    ### group by fixed range of z score
    group_count = 80

    zScoreIntervals = []
    startZscore = -2
    endZscore = 2
    step = (endZscore-startZscore)/group_count
    cur_zScore = startZscore + step
    while cur_zScore <= endZscore:
        zScoreIntervals.append(cur_zScore)
        cur_zScore += step
    
    if zScoreIntervals[-1] < endZscore:
        zScoreIntervals.append(endZscore)

    startIndex = 0
    for zintv in zScoreIntervals:
        cur_group = []
        for i in range(startIndex, len(combineAll)):
            cur_tup = combineAll[i]
            cur_zScore = cur_tup[sortingColumn]
            if cur_zScore <= zintv:
                if targetType == 'all':
                    cur_group.append(cur_tup)
                else:
                    if cur_tup[-1] in disagreeAnswersIdList:
                        cur_group.append(cur_tup)
            else:
                startIndex = i
                break
        if len(cur_group)>0: 
            underGroup = [t for t in cur_group if t[-1] in underEstimatedAnswersIdList]
            overGroup = [t for t in cur_group if t[-1] in overEstimatedAnswersIdList]
            normalGroup = [t for t in cur_group if t[-1] not in underEstimatedAnswersIdList and t[-1] not in overEstimatedAnswersIdList]
            if len(underGroup) > 0:
                column_wise_mean_underGroup = np.mean(underGroup,axis=0)
                group_combineAll.append(tuple(list(column_wise_mean_underGroup) + ['under']))
            if len(overGroup) > 0:
                column_wise_mean_overGroup = np.mean(overGroup,axis=0)
                group_combineAll.append(tuple(list(column_wise_mean_overGroup)+ ['over']))
            if len(normalGroup) > 0:
                column_wise_mean_normalGroup = np.mean(normalGroup,axis=0)
                group_combineAll.append(tuple(list(column_wise_mean_normalGroup)+ ['normal']))

    # ### group by interval
    # group_count = 200
    # # group_count = len(combineAll) # equivalent to not grouping

    # percentages = [x / group_count for x in range(1, group_count+1)]
    # zScoreIntervals = [scipy.stats.norm.ppf(p) for p in percentages]

    # startIndex = 0
    # for zintv in zScoreIntervals:
    #     cur_group = []
    #     for i in range(startIndex, len(combineAll)):
    #         cur_tup = combineAll[i]
    #         cur_zScore = cur_tup[sortingColumn]
    #         if cur_zScore <= zintv:
    #             if cur_tup[-1] in disagreeAnswersIdList:
    #                 cur_group.append(cur_tup)
    #         else:
    #             startIndex = i
    #             break
    #     if len(cur_group)>0: 
    #         underGroup = [t for t in cur_group if t[-1] in underEstimatedAnswersIdList]
    #         overGroup = [t for t in cur_group if t[-1] in overEstimatedAnswersIdList]
    #         normalGroup = [t for t in cur_group if t[-1] not in underEstimatedAnswersIdList and t[-1] not in overEstimatedAnswersIdList]
    #         if len(underGroup) > 0:
    #             column_wise_mean_underGroup = np.mean(underGroup,axis=0)
    #             group_combineAll.append(tuple(list(column_wise_mean_underGroup) + ['under']))
    #         if len(overGroup) > 0:
    #             column_wise_mean_overGroup = np.mean(overGroup,axis=0)
    #             group_combineAll.append(tuple(list(column_wise_mean_overGroup)+ ['over']))
    #         if len(normalGroup) > 0:
    #             column_wise_mean_normalGroup = np.mean(normalGroup,axis=0)
    #             group_combineAll.append(tuple(list(column_wise_mean_normalGroup)+ ['normal']))
    
    # print(f"group z score sample count {len(group_combineAll)} sorted by column {sortingColumn}")
    return group_combineAll

def residulToDiagonalLine(y_pred, y_true):
    squredResiduals = []
    for i in range(len(y_true)):
        squredResiduals.append((y_pred[i]-y_true[i])**2)
    r2 = r2_score(y_true, y_pred)
    pearson = stats.pearsonr(y_true, y_pred)
    pearson_stat = pearson[0]
    pearson_p = pearson[1]
    return sum(squredResiduals), sum(squredResiduals)/len(y_true), squredResiduals, r2, pearson_stat, pearson_p

#######################################################################################################
# plot with all data point
def myPlot(commName, combineAll_forSentiment,combineAll_forHelpful, plotFileName, reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP, underEstimatedAnswersIdList_forSentiment, overEstimatedAnswersIdList_forSentiment, underEstimatedAnswersIdList_forHelpful, overEstimatedAnswersIdList_forHelpful, disagreeAnswersIdList):
    newModelName = 'CVA'
    """
    ### x axis as prediction and y axis as ground truth

    yAxisName_1 = 'sentiment rankZscore'
    yAxisName_2 = 'helpfulness rankZscore'

    plt.cla()
    fig, axs = plt.subplots(2, 3, figsize=(7.3, 4))
    fig.tight_layout(pad=1.5)

    # first row for sentiment as ground truth
    group_combineAll_basedOnVoteDiff = myGrouping(combineAll_forSentiment, sortingColumn=1)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiff]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnVoteDiff]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnVoteDiff]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[0, 0].scatter(group_voteDiff_rankZscores, group_priorQ_rankZscores,s = sizes)

    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    all_voteDiff_rankZscores = [t[1] for t in combineAll_forSentiment]
    
    axs[0, 0].set_xlabel('vote diff rankZscore', fontsize = 8)
    axs[0, 0].set_ylabel(yAxisName_1, fontsize = 8)
    axs[0, 0].set_xlim(-2,2)
    axs[0, 0].set_ylim(-2,2)

    # OLS fit
    # z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_voteDiff_forSentiment, norm_sumSquredResiduls_voteDiff_forSentiment, squredResiduals_voteDiff_forSentiment, r2_voteDiff_forSentiment, pearson_stat_voteDiff_forSentiment, pearson_p_voteDiff_forSentiment = residulToDiagonalLine(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    # axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1, 
    #               label=f'slope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,0].legend(loc="best", fontsize = 6)
    axs[0,0].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[0,0].text(2, -1.5, f'residual={round(sumSquredResiduls_voteDiff_forSentiment,1)}', fontsize=8, horizontalalignment='right')
    # axs[0,0].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_voteDiff_forSentiment,3)}', fontsize=8, horizontalalignment='right')


    group_combineAll_basedOnCVP = myGrouping(combineAll_forSentiment, sortingColumn=2)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVP]
    group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnCVP]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnCVP]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[0, 1].scatter(group_CVPsklearnQ_rankZscores, group_priorQ_rankZscores,s = sizes)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll_forSentiment]
    
    axs[0, 1].set_xlabel('CVP quality rankZscore', fontsize = 8)
    axs[0, 1].set_ylabel(yAxisName_1, fontsize = 8)
    axs[0, 1].set_xlim(-2,2)
    axs[0, 1].set_ylim(-2,2)

    # OLS fit
    # z1,p1 = curveFitOLS(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_CVP_forSentiment, norm_sumSquredResiduls_CVP_forSentiment, squredResiduals_CVP_forSentiment, r2_CVP_forSentiment, pearson_stat_CVP_forSentiment, pearson_p_CVP_forSentiment = residulToDiagonalLine(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    # axs[0,1].plot(all_CVPsklearnQ_rankZscores,p1(all_CVPsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,1].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[0,1].legend(loc="best", fontsize = 6)
    axs[0,1].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[0,1].text(2, -1.5, f'residual={round(sumSquredResiduls_CVP_forSentiment,1)}', fontsize=8, horizontalalignment='right')
    # axs[0,1].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_CVP_forSentiment,3)}', fontsize=8, horizontalalignment='right')

    group_combineAll_basedOnNewModel = myGrouping(combineAll_forSentiment, sortingColumn=3)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnNewModel]
    group_newModelsklearnQ_rankZscores = [t[3] for t in group_combineAll_basedOnNewModel]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnNewModel]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[0, 2].scatter(group_newModelsklearnQ_rankZscores, group_priorQ_rankZscores,s = sizes)
    
    # ### using all raw points to fit
    all_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    all_newModelsklearnQ_rankZscores = [t[3] for t in combineAll_forSentiment]
    
    axs[0, 2].set_xlabel(f'{newModelName} quality rankZscore', fontsize = 8)
    axs[0, 2].set_ylabel(yAxisName_1, fontsize = 8)
    axs[0, 2].set_xlim(-2,2)
    axs[0, 2].set_ylim(-2,2)

    # OLS fit
    # z3,p3 = curveFitOLS(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_newModel_forSentiment, norm_sumSquredResiduls_newModel_forSentiment, squredResiduals_newModel_forSentiment, r2_newModel_forSentiment, pearson_stat_newModel_forSentiment, pearson_p_newModel_forSentiment = residulToDiagonalLine(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    # axs[1,0].plot(all_newModelsklearnQ_rankZscores,p3(all_newModelsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[1,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[1,0].legend(loc="best", fontsize = 6)
    axs[0,2].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[0,2].text(2, -1.5, f'residual={round(sumSquredResiduls_newModel_forSentiment,1)}', fontsize=8, horizontalalignment='right')
    # axs[0,2].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_newModel_forSentiment,3)}', fontsize=8, horizontalalignment='right')

    
    # second row for helpful as ground truth
    group_combineAll_basedOnVoteDiff = myGrouping(combineAll_forHelpful, sortingColumn=1)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiff]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnVoteDiff]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnVoteDiff]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[1, 0].scatter(group_voteDiff_rankZscores, group_priorQ_rankZscores,s = sizes)

    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    all_voteDiff_rankZscores = [t[1] for t in combineAll_forHelpful]
    
    axs[1, 0].set_xlabel('vote diff rankZscore', fontsize = 8)
    axs[1, 0].set_ylabel(yAxisName_2, fontsize = 8)
    axs[1, 0].set_xlim(-2,2)
    axs[1, 0].set_ylim(-2,2)

    # OLS fit
    # z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_voteDiff_forHelpful, norm_sumSquredResiduls_voteDiff_forHelpful, squredResiduals_voteDiff_forHelpful, r2_voteDiff_forHelpful, pearson_stat_voteDiff_forHelpful, pearson_p_voteDiff_forHelpful = residulToDiagonalLine(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    # axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1, 
    #               label=f'slope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,0].legend(loc="best", fontsize = 6)
    axs[1,0].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[1,0].text(2, -1.5, f'residual={round(sumSquredResiduls_voteDiff_forHelpful,1)}', fontsize=8, horizontalalignment='right')
    # axs[1,0].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_voteDiff_forHelpful,3)}', fontsize=8, horizontalalignment='right')


    group_combineAll_basedOnCVP = myGrouping(combineAll_forHelpful, sortingColumn=2)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVP]
    group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnCVP]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnCVP]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[1, 1].scatter(group_CVPsklearnQ_rankZscores, group_priorQ_rankZscores,s = sizes)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll_forHelpful]
    
    axs[1, 1].set_xlabel('CVP quality rankZscore', fontsize = 8)
    axs[1, 1].set_ylabel(yAxisName_2, fontsize = 8)
    axs[1, 1].set_xlim(-2,2)
    axs[1, 1].set_ylim(-2,2)

    # OLS fit
    # z1,p1 = curveFitOLS(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_CVP_forHelpful, norm_sumSquredResiduls_CVP_forHelpful, squredResiduals_CVP_forHelpful, r2_CVP_forHelpful, pearson_stat_CVP_forHelpful, pearson_p_CVP_forHelpful = residulToDiagonalLine(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    # axs[0,1].plot(all_CVPsklearnQ_rankZscores,p1(all_CVPsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,1].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[0,1].legend(loc="best", fontsize = 6)
    axs[1,1].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[1,1].text(2, -1.5, f'residual={round(sumSquredResiduls_CVP_forHelpful,1)}', fontsize=8, horizontalalignment='right')
    # axs[1,1].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_CVP_forHelpful,3)}', fontsize=8, horizontalalignment='right')

    group_combineAll_basedOnNewModel = myGrouping(combineAll_forHelpful, sortingColumn=3)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnNewModel]
    group_newModelsklearnQ_rankZscores = [t[3] for t in group_combineAll_basedOnNewModel]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnNewModel]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[1, 2].scatter(group_newModelsklearnQ_rankZscores, group_priorQ_rankZscores,s = sizes)
    
    # ### using all raw points to fit
    all_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    all_newModelsklearnQ_rankZscores = [t[3] for t in combineAll_forHelpful]
    
    axs[1, 2].set_xlabel(f'{newModelName} quality rankZscore', fontsize = 8)
    axs[1, 2].set_ylabel(yAxisName_2, fontsize = 8)
    axs[1, 2].set_xlim(-2,2)
    axs[1, 2].set_ylim(-2,2)

    # OLS fit
    # z3,p3 = curveFitOLS(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_newModel_forHelpful, norm_sumSquredResiduls_newModel_forHelpful, squredResiduals_newModel_forHelpful, r2_newModel_forHelpful, pearson_stat_newModel_forHelpful, pearson_p_newModel_forHelpful = residulToDiagonalLine(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    # axs[1,0].plot(all_newModelsklearnQ_rankZscores,p3(all_newModelsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[1,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[1,0].legend(loc="best", fontsize = 6)
    axs[1,2].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[1,2].text(2, -1.5, f'residual={round(sumSquredResiduls_newModel_forHelpful,1)}', fontsize=8, horizontalalignment='right')
    # axs[1,2].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_newModel_forHelpful,3)}', fontsize=8, horizontalalignment='right')


    # fig.suptitle(f"{commName.replace('.stackexchange','')}", fontsize = 8)

    savePlot(fig, plotFileName)

    # compare answer-level residuals
    betterThanVoteDiff_forSentiment = 0
    betterThanCVP_forSentiment = 0
    betterThanBoth_forSentiment = 0
    for i, sr_newModel in enumerate(squredResiduals_newModel_forSentiment):
        sr_voteDiff = squredResiduals_voteDiff_forSentiment[i]
        sr_CVP = squredResiduals_CVP_forSentiment[i]
        if sr_newModel <= sr_voteDiff:
            betterThanVoteDiff_forSentiment += 1
        if sr_newModel <= sr_CVP:
            betterThanCVP_forSentiment += 1
        if sr_newModel <= sr_voteDiff and sr_newModel <= sr_CVP:
            betterThanBoth_forSentiment += 1
    
    betterThanVoteDiff_forHelpful = 0
    betterThanCVP_forHelpful = 0
    betterThanBoth_forHelpful = 0
    for i, sr_newModel in enumerate(squredResiduals_newModel_forHelpful):
        sr_voteDiff = squredResiduals_voteDiff_forHelpful[i]
        sr_CVP = squredResiduals_CVP_forHelpful[i]
        if sr_newModel <= sr_voteDiff:
            betterThanVoteDiff_forHelpful += 1
        if sr_newModel <= sr_CVP:
            betterThanCVP_forHelpful += 1
        if sr_newModel <= sr_voteDiff and sr_newModel <= sr_CVP:
            betterThanBoth_forHelpful += 1

    # T-test B > A ttest_ind(B, A)
    ttest_voteDiff_forSentiment_residaul = stats.ttest_ind(squredResiduals_voteDiff_forSentiment, squredResiduals_newModel_forSentiment).statistic
    p_ofTtest_voteDiff_forSentiment_residaul = stats.ttest_ind(squredResiduals_voteDiff_forSentiment, squredResiduals_newModel_forSentiment).pvalue /2 # one-tailed
    ttest_CVP_forSentiment_residaul = stats.ttest_ind(squredResiduals_CVP_forSentiment, squredResiduals_newModel_forSentiment).statistic
    p_ofTtest_CVP_forSentiment_residaul = stats.ttest_ind(squredResiduals_CVP_forSentiment, squredResiduals_newModel_forSentiment).pvalue /2 # one-tailed

    ttest_voteDiff_forHelpful_residaul = stats.ttest_ind(squredResiduals_voteDiff_forHelpful, squredResiduals_newModel_forHelpful).statistic
    p_ofTtest_voteDiff_forHelpful_residaul = stats.ttest_ind(squredResiduals_voteDiff_forHelpful, squredResiduals_newModel_forHelpful).pvalue /2 # one-tailed
    ttest_CVP_forHelpful_residaul = stats.ttest_ind(squredResiduals_CVP_forHelpful, squredResiduals_newModel_forHelpful).statistic
    p_ofTtest_CVP_forHelpful_residaul = stats.ttest_ind(squredResiduals_CVP_forHelpful, squredResiduals_newModel_forHelpful).pvalue /2 # one-tailed

    tup = (sumSquredResiduls_voteDiff_forSentiment, sumSquredResiduls_CVP_forSentiment, sumSquredResiduls_newModel_forSentiment, 
           sumSquredResiduls_voteDiff_forHelpful, sumSquredResiduls_CVP_forHelpful, sumSquredResiduls_newModel_forHelpful, 
           betterThanVoteDiff_forSentiment, betterThanCVP_forSentiment, betterThanBoth_forSentiment, 
           betterThanVoteDiff_forHelpful, betterThanCVP_forHelpful, betterThanBoth_forHelpful, 
           r2_voteDiff_forSentiment, r2_CVP_forSentiment, r2_newModel_forSentiment, 
           r2_voteDiff_forHelpful, r2_CVP_forHelpful, r2_newModel_forHelpful, 
           pearson_stat_voteDiff_forSentiment, pearson_p_voteDiff_forSentiment, pearson_stat_CVP_forSentiment, pearson_p_CVP_forSentiment, pearson_stat_newModel_forSentiment, pearson_p_newModel_forSentiment, 
           pearson_stat_voteDiff_forHelpful, pearson_p_voteDiff_forHelpful, pearson_stat_CVP_forHelpful, pearson_p_CVP_forHelpful, pearson_stat_newModel_forHelpful, pearson_p_newModel_forHelpful,
           ttest_voteDiff_forSentiment_residaul, p_ofTtest_voteDiff_forSentiment_residaul, ttest_CVP_forSentiment_residaul, p_ofTtest_CVP_forSentiment_residaul,
           ttest_voteDiff_forHelpful_residaul, p_ofTtest_voteDiff_forHelpful_residaul, ttest_CVP_forHelpful_residaul, p_ofTtest_CVP_forHelpful_residaul)
    """

    ### y axis as prediction and x axis as ground truth

    xAxisName_1 = 'sentiment rankZscore'
    xAxisName_2 = 'helpfulness rankZscore'

    plt.cla()
    fig, axs = plt.subplots(2, 3, figsize=(7.3, 4))
    fig.tight_layout(pad=1.5)

    # first row for sentiment as ground truth
    group_combineAll_basedOnVoteDiff = myGrouping(combineAll_forSentiment, sortingColumn=0)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiff]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnVoteDiff]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnVoteDiff]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[0, 0].scatter(group_priorQ_rankZscores, group_voteDiff_rankZscores,s = sizes)

    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    all_voteDiff_rankZscores = [t[1] for t in combineAll_forSentiment]
    
    axs[0, 0].set_ylabel('vote diff rankZscore', fontsize = 8)
    axs[0, 0].set_xlabel(xAxisName_1, fontsize = 8)
    axs[0, 0].set_xlim(-2,2)
    axs[0, 0].set_ylim(-2,2)

    # OLS fit
    # z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_voteDiff_forSentiment, norm_sumSquredResiduls_voteDiff_forSentiment, squredResiduals_voteDiff_forSentiment, r2_voteDiff_forSentiment, pearson_stat_voteDiff_forSentiment, pearson_p_voteDiff_forSentiment = residulToDiagonalLine(all_priorQ_rankZscores, all_voteDiff_rankZscores)
    # axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1, 
    #               label=f'slope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,0].legend(loc="best", fontsize = 6)
    axs[0,0].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[0,0].text(2, -1.5, f'residual={round(sumSquredResiduls_voteDiff_forSentiment,1)}', fontsize=8, horizontalalignment='right')
    # axs[0,0].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_voteDiff_forSentiment,3)}', fontsize=8, horizontalalignment='right')


    group_combineAll_basedOnCVP = myGrouping(combineAll_forSentiment, sortingColumn=0)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVP]
    group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnCVP]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnCVP]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[0, 1].scatter(group_priorQ_rankZscores,group_CVPsklearnQ_rankZscores, s = sizes)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll_forSentiment]
    
    axs[0, 1].set_ylabel('CVP quality rankZscore', fontsize = 8)
    axs[0, 1].set_xlabel(xAxisName_1, fontsize = 8)
    axs[0, 1].set_xlim(-2,2)
    axs[0, 1].set_ylim(-2,2)

    # OLS fit
    # z1,p1 = curveFitOLS(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_CVP_forSentiment, norm_sumSquredResiduls_CVP_forSentiment, squredResiduals_CVP_forSentiment, r2_CVP_forSentiment, pearson_stat_CVP_forSentiment, pearson_p_CVP_forSentiment = residulToDiagonalLine(all_priorQ_rankZscores, all_CVPsklearnQ_rankZscores)
    # axs[0,1].plot(all_CVPsklearnQ_rankZscores,p1(all_CVPsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,1].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[0,1].legend(loc="best", fontsize = 6)
    axs[0,1].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[0,1].text(2, -1.5, f'residual={round(sumSquredResiduls_CVP_forSentiment,1)}', fontsize=8, horizontalalignment='right')
    # axs[0,1].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_CVP_forSentiment,3)}', fontsize=8, horizontalalignment='right')

    group_combineAll_basedOnNewModel = myGrouping(combineAll_forSentiment, sortingColumn=0)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnNewModel]
    group_newModelsklearnQ_rankZscores = [t[3] for t in group_combineAll_basedOnNewModel]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnNewModel]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[0, 2].scatter(group_priorQ_rankZscores,group_newModelsklearnQ_rankZscores, s = sizes)
    
    # ### using all raw points to fit
    all_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    all_newModelsklearnQ_rankZscores = [t[3] for t in combineAll_forSentiment]
    
    axs[0, 2].set_ylabel(f'{newModelName} quality rankZscore', fontsize = 8)
    axs[0, 2].set_xlabel(xAxisName_1, fontsize = 8)
    axs[0, 2].set_xlim(-2,2)
    axs[0, 2].set_ylim(-2,2)

    # OLS fit
    # z3,p3 = curveFitOLS(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_newModel_forSentiment, norm_sumSquredResiduls_newModel_forSentiment, squredResiduals_newModel_forSentiment, r2_newModel_forSentiment, pearson_stat_newModel_forSentiment, pearson_p_newModel_forSentiment = residulToDiagonalLine(all_priorQ_rankZscores, all_newModelsklearnQ_rankZscores)
    # axs[1,0].plot(all_newModelsklearnQ_rankZscores,p3(all_newModelsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[1,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[1,0].legend(loc="best", fontsize = 6)
    axs[0,2].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[0,2].text(2, -1.5, f'residual={round(sumSquredResiduls_newModel_forSentiment,1)}', fontsize=8, horizontalalignment='right')
    # axs[0,2].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_newModel_forSentiment,3)}', fontsize=8, horizontalalignment='right')

    
    # second row for helpful as ground truth
    group_combineAll_basedOnVoteDiff = myGrouping(combineAll_forHelpful, sortingColumn=0)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiff]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnVoteDiff]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnVoteDiff]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[1, 0].scatter(group_priorQ_rankZscores,group_voteDiff_rankZscores, s = sizes)

    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    all_voteDiff_rankZscores = [t[1] for t in combineAll_forHelpful]
    
    axs[1, 0].set_ylabel('vote diff rankZscore', fontsize = 8)
    axs[1, 0].set_xlabel(xAxisName_2, fontsize = 8)
    axs[1, 0].set_xlim(-2,2)
    axs[1, 0].set_ylim(-2,2)

    # OLS fit
    # z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_voteDiff_forHelpful, norm_sumSquredResiduls_voteDiff_forHelpful, squredResiduals_voteDiff_forHelpful, r2_voteDiff_forHelpful, pearson_stat_voteDiff_forHelpful, pearson_p_voteDiff_forHelpful = residulToDiagonalLine(all_priorQ_rankZscores, all_voteDiff_rankZscores)
    # axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1, 
    #               label=f'slope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,0].legend(loc="best", fontsize = 6)
    axs[1,0].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[1,0].text(2, -1.5, f'residual={round(sumSquredResiduls_voteDiff_forHelpful,1)}', fontsize=8, horizontalalignment='right')
    # axs[1,0].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_voteDiff_forHelpful,3)}', fontsize=8, horizontalalignment='right')


    group_combineAll_basedOnCVP = myGrouping(combineAll_forHelpful, sortingColumn=0)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVP]
    group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnCVP]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnCVP]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[1, 1].scatter(group_priorQ_rankZscores,group_CVPsklearnQ_rankZscores, s = sizes)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll_forHelpful]
    
    axs[1, 1].set_ylabel('CVP quality rankZscore', fontsize = 8)
    axs[1, 1].set_xlabel(xAxisName_2, fontsize = 8)
    axs[1, 1].set_xlim(-2,2)
    axs[1, 1].set_ylim(-2,2)

    # OLS fit
    # z1,p1 = curveFitOLS(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_CVP_forHelpful, norm_sumSquredResiduls_CVP_forHelpful, squredResiduals_CVP_forHelpful, r2_CVP_forHelpful, pearson_stat_CVP_forHelpful, pearson_p_CVP_forHelpful = residulToDiagonalLine(all_priorQ_rankZscores, all_CVPsklearnQ_rankZscores)
    # axs[0,1].plot(all_CVPsklearnQ_rankZscores,p1(all_CVPsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,1].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[0,1].legend(loc="best", fontsize = 6)
    axs[1,1].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[1,1].text(2, -1.5, f'residual={round(sumSquredResiduls_CVP_forHelpful,1)}', fontsize=8, horizontalalignment='right')
    # axs[1,1].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_CVP_forHelpful,3)}', fontsize=8, horizontalalignment='right')

    group_combineAll_basedOnNewModel = myGrouping(combineAll_forHelpful, sortingColumn=0)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnNewModel]
    group_newModelsklearnQ_rankZscores = [t[3] for t in group_combineAll_basedOnNewModel]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnNewModel]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[1, 2].scatter(group_priorQ_rankZscores,group_newModelsklearnQ_rankZscores, s = sizes)
    
    # ### using all raw points to fit
    all_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    all_newModelsklearnQ_rankZscores = [t[3] for t in combineAll_forHelpful]
    
    axs[1, 2].set_ylabel(f'{newModelName} quality rankZscore', fontsize = 8)
    axs[1, 2].set_xlabel(xAxisName_2, fontsize = 8)
    axs[1, 2].set_xlim(-2,2)
    axs[1, 2].set_ylim(-2,2)

    # OLS fit
    # z3,p3 = curveFitOLS(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    sumSquredResiduls_newModel_forHelpful, norm_sumSquredResiduls_newModel_forHelpful, squredResiduals_newModel_forHelpful, r2_newModel_forHelpful, pearson_stat_newModel_forHelpful, pearson_p_newModel_forHelpful = residulToDiagonalLine(all_priorQ_rankZscores, all_newModelsklearnQ_rankZscores)
    # axs[1,0].plot(all_newModelsklearnQ_rankZscores,p3(all_newModelsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[1,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[1,0].legend(loc="best", fontsize = 6)
    axs[1,2].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[1,2].text(2, -1.5, f'residual={round(sumSquredResiduls_newModel_forHelpful,1)}', fontsize=8, horizontalalignment='right')
    # axs[1,2].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_newModel_forHelpful,3)}', fontsize=8, horizontalalignment='right')


    # fig.suptitle(f"{commName.replace('.stackexchange','')}", fontsize = 8)

    savePlot(fig, plotFileName)

    # compare answer-level residuals
    betterThanVoteDiff_forSentiment = 0
    betterThanCVP_forSentiment = 0
    betterThanBoth_forSentiment = 0
    for i, sr_newModel in enumerate(squredResiduals_newModel_forSentiment):
        sr_voteDiff = squredResiduals_voteDiff_forSentiment[i]
        sr_CVP = squredResiduals_CVP_forSentiment[i]
        if sr_newModel <= sr_voteDiff:
            betterThanVoteDiff_forSentiment += 1
        if sr_newModel <= sr_CVP:
            betterThanCVP_forSentiment += 1
        if sr_newModel <= sr_voteDiff and sr_newModel <= sr_CVP:
            betterThanBoth_forSentiment += 1
    
    betterThanVoteDiff_forHelpful = 0
    betterThanCVP_forHelpful = 0
    betterThanBoth_forHelpful = 0
    for i, sr_newModel in enumerate(squredResiduals_newModel_forHelpful):
        sr_voteDiff = squredResiduals_voteDiff_forHelpful[i]
        sr_CVP = squredResiduals_CVP_forHelpful[i]
        if sr_newModel <= sr_voteDiff:
            betterThanVoteDiff_forHelpful += 1
        if sr_newModel <= sr_CVP:
            betterThanCVP_forHelpful += 1
        if sr_newModel <= sr_voteDiff and sr_newModel <= sr_CVP:
            betterThanBoth_forHelpful += 1

    # T-test B > A ttest_ind(B, A)
    ttest_voteDiff_forSentiment_residaul = stats.ttest_ind(squredResiduals_voteDiff_forSentiment, squredResiduals_newModel_forSentiment).statistic
    p_ofTtest_voteDiff_forSentiment_residaul = stats.ttest_ind(squredResiduals_voteDiff_forSentiment, squredResiduals_newModel_forSentiment).pvalue /2 # one-tailed
    ttest_CVP_forSentiment_residaul = stats.ttest_ind(squredResiduals_CVP_forSentiment, squredResiduals_newModel_forSentiment).statistic
    p_ofTtest_CVP_forSentiment_residaul = stats.ttest_ind(squredResiduals_CVP_forSentiment, squredResiduals_newModel_forSentiment).pvalue /2 # one-tailed

    ttest_voteDiff_forHelpful_residaul = stats.ttest_ind(squredResiduals_voteDiff_forHelpful, squredResiduals_newModel_forHelpful).statistic
    p_ofTtest_voteDiff_forHelpful_residaul = stats.ttest_ind(squredResiduals_voteDiff_forHelpful, squredResiduals_newModel_forHelpful).pvalue /2 # one-tailed
    ttest_CVP_forHelpful_residaul = stats.ttest_ind(squredResiduals_CVP_forHelpful, squredResiduals_newModel_forHelpful).statistic
    p_ofTtest_CVP_forHelpful_residaul = stats.ttest_ind(squredResiduals_CVP_forHelpful, squredResiduals_newModel_forHelpful).pvalue /2 # one-tailed

    tup = (sumSquredResiduls_voteDiff_forSentiment, sumSquredResiduls_CVP_forSentiment, sumSquredResiduls_newModel_forSentiment, 
           sumSquredResiduls_voteDiff_forHelpful, sumSquredResiduls_CVP_forHelpful, sumSquredResiduls_newModel_forHelpful, 
           betterThanVoteDiff_forSentiment, betterThanCVP_forSentiment, betterThanBoth_forSentiment, 
           betterThanVoteDiff_forHelpful, betterThanCVP_forHelpful, betterThanBoth_forHelpful, 
           r2_voteDiff_forSentiment, r2_CVP_forSentiment, r2_newModel_forSentiment, 
           r2_voteDiff_forHelpful, r2_CVP_forHelpful, r2_newModel_forHelpful, 
           pearson_stat_voteDiff_forSentiment, pearson_p_voteDiff_forSentiment, pearson_stat_CVP_forSentiment, pearson_p_CVP_forSentiment, pearson_stat_newModel_forSentiment, pearson_p_newModel_forSentiment, 
           pearson_stat_voteDiff_forHelpful, pearson_p_voteDiff_forHelpful, pearson_stat_CVP_forHelpful, pearson_p_CVP_forHelpful, pearson_stat_newModel_forHelpful, pearson_p_newModel_forHelpful,
           ttest_voteDiff_forSentiment_residaul, p_ofTtest_voteDiff_forSentiment_residaul, ttest_CVP_forSentiment_residaul, p_ofTtest_CVP_forSentiment_residaul,
           ttest_voteDiff_forHelpful_residaul, p_ofTtest_voteDiff_forHelpful_residaul, ttest_CVP_forHelpful_residaul, p_ofTtest_CVP_forHelpful_residaul)
    
    return tup


def myPlot_new(commName, combineAll_forSentiment,combineAll_forHelpful, plotFileName, reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP, underEstimatedAnswersIdList_forSentiment, overEstimatedAnswersIdList_forSentiment, underEstimatedAnswersIdList_forHelpful, overEstimatedAnswersIdList_forHelpful, disagreeAnswersIdList):
    newModelName = 'CVA'
    # newModelName = 'proposed model'

    yAxisName_1 = 'sentiment rankZscore'
    yAxisName_2 = 'helpfulness rankZscore'
    # targetType = 'disagree'
    targetType = 'all'
    

    plt.cla()
    fig, axs = plt.subplots(2, 3, figsize=(7.3, 4))
    fig.tight_layout(pad=1.5)

    # first row for sentiment as ground truth
    if targetType == 'under':
        targetAnswerIdList = underEstimatedAnswersIdList_forSentiment
    elif targetType == 'over':
        targetAnswerIdList = overEstimatedAnswersIdList_forSentiment
    elif targetType == 'normal':
        targetAnswerIdList = [t[-1] for t in combineAll_forSentiment if t[-1] not in underEstimatedAnswersIdList_forSentiment and t[-1] not in overEstimatedAnswersIdList_forSentiment]
    elif targetType == 'disagree':
        targetAnswerIdList = disagreeAnswersIdList
    else: # all
        targetAnswerIdList = [t[-1] for t in combineAll_forSentiment]

    group_combineAll_basedOnVoteDiff = myGrouping_new(combineAll_forSentiment, 1, underEstimatedAnswersIdList_forSentiment, overEstimatedAnswersIdList_forSentiment, disagreeAnswersIdList, targetType)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiff]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnVoteDiff]
    colors = [] 
    for t in group_combineAll_basedOnVoteDiff:
        if t[-1] == 'under':
            colors.append('blue')
        elif t[-1] == 'over':
            colors.append('red')
        else:
            colors.append('green')

    # # when not grouping
    # group_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    # group_voteDiff_rankZscores = [t[1] for t in combineAll_forSentiment]
    # colors = [] 
    # for t in combineAll_forSentiment:
    #     if t[-1] in underEstimatedAnswersIdList_forSentiment:
    #         colors.append('blue')
    #     elif t[-1] in overEstimatedAnswersIdList_forSentiment:
    #         colors.append('red')
    #     else:
    #         colors.append('black')
            
    if targetType in ['disagree','all']:
        axs[0, 0].scatter(group_voteDiff_rankZscores, group_priorQ_rankZscores,s = 2, c=colors)
    else:
        group_voteDiff_rankZscores_targetType = [t[1] for t in group_combineAll_basedOnVoteDiff if t[-1] == targetType]
        group_priorQ_rankZscores_targetType = [t[0] for t in group_combineAll_basedOnVoteDiff if t[-1] == targetType]
        colors_targetType = [colors[i] for i,t in enumerate(group_combineAll_basedOnVoteDiff) if t[-1] == targetType]
        axs[0, 0].scatter(group_voteDiff_rankZscores_targetType, group_priorQ_rankZscores_targetType,s = 2, c=colors_targetType)

    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    all_voteDiff_rankZscores = [t[1] for t in combineAll_forSentiment]
    
    axs[0, 0].set_xlabel('vote diff rankZscore', fontsize = 8)
    axs[0, 0].set_ylabel(yAxisName_1, fontsize = 8)
    axs[0, 0].set_xlim(-2,2)
    axs[0, 0].set_ylim(-2,2)

    # OLS fit
    # z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    # sumSquredResiduls_voteDiff_forSentiment, norm_sumSquredResiduls_voteDiff_forSentiment, squredResiduals_voteDiff_forSentiment, r2_voteDiff_forSentiment, pearson_stat_voteDiff_forSentiment, pearson_p_voteDiff_forSentiment = residulToDiagonalLine(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    all_voteDiff_rankZscores_targetType = [t[1] for t in combineAll_forSentiment if t[-1] in targetAnswerIdList]
    all_priorQ_rankZscores_targetType = [t[0] for t in combineAll_forSentiment if t[-1] in targetAnswerIdList]
    sumSquredResiduls_voteDiff_forSentiment, norm_sumSquredResiduls_voteDiff_forSentiment, squredResiduals_voteDiff_forSentiment, r2_voteDiff_forSentiment, pearson_stat_voteDiff_forSentiment, pearson_p_voteDiff_forSentiment = residulToDiagonalLine(all_voteDiff_rankZscores_targetType, all_priorQ_rankZscores_targetType)
    # axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1, 
    #               label=f'slope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,0].legend(loc="best", fontsize = 6)
    axs[0,0].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[0,0].text(2, -1.5, f'residual={round(sumSquredResiduls_voteDiff_forSentiment,1)}', fontsize=8, horizontalalignment='right')
    # axs[0,0].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_voteDiff_forSentiment,3)}', fontsize=8, horizontalalignment='right')


    group_combineAll_basedOnCVP = myGrouping_new(combineAll_forSentiment, 2, underEstimatedAnswersIdList_forSentiment, overEstimatedAnswersIdList_forSentiment, disagreeAnswersIdList, targetType)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVP]
    group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnCVP]
    colors = [] 
    for t in group_combineAll_basedOnCVP:
        if t[-1] == 'under':
            colors.append('blue')
        elif t[-1] == 'over':
            colors.append('red')
        else:
            colors.append('green')
    # # when not grouping
    # group_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    # group_CVPsklearnQ_rankZscores = [t[2] for t in combineAll_forSentiment]
            
    if targetType in ['disagree','all']:
        axs[0, 1].scatter(group_CVPsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2, c=colors)
    else:
        group_CVPsklearnQ_rankZscores_targetType = [t[2] for t in group_combineAll_basedOnCVP if t[-1] == targetType]
        group_priorQ_rankZscores_targetType = [t[0] for t in group_combineAll_basedOnCVP if t[-1] == targetType]
        colors_targetType = [colors[i] for i,t in enumerate(group_combineAll_basedOnCVP) if t[-1] == targetType]
        axs[0, 1].scatter(group_CVPsklearnQ_rankZscores_targetType, group_priorQ_rankZscores_targetType,s = 2, c=colors_targetType)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll_forSentiment]
    
    axs[0, 1].set_xlabel('CVP quality rankZscore', fontsize = 8)
    axs[0, 1].set_ylabel(yAxisName_1, fontsize = 8)
    axs[0, 1].set_xlim(-2,2)
    axs[0, 1].set_ylim(-2,2)

    # OLS fit
    # z1,p1 = curveFitOLS(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    # sumSquredResiduls_CVP_forSentiment, norm_sumSquredResiduls_CVP_forSentiment, squredResiduals_CVP_forSentiment, r2_CVP_forSentiment, pearson_stat_CVP_forSentiment, pearson_p_CVP_forSentiment = residulToDiagonalLine(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    all_CVPsklearnQ_rankZscores_targetType = [t[2] for t in combineAll_forSentiment if t[-1] in targetAnswerIdList]
    all_priorQ_rankZscores_targetType = [t[0] for t in combineAll_forSentiment if t[-1] in targetAnswerIdList]
    sumSquredResiduls_CVP_forSentiment, norm_sumSquredResiduls_CVP_forSentiment, squredResiduals_CVP_forSentiment, r2_CVP_forSentiment, pearson_stat_CVP_forSentiment, pearson_p_CVP_forSentiment = residulToDiagonalLine(all_CVPsklearnQ_rankZscores_targetType, all_priorQ_rankZscores_targetType)
    # axs[0,1].plot(all_CVPsklearnQ_rankZscores,p1(all_CVPsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,1].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[0,1].legend(loc="best", fontsize = 6)
    axs[0,1].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[0,1].text(2, -1.5, f'residual={round(sumSquredResiduls_CVP_forSentiment,1)}', fontsize=8, horizontalalignment='right')
    # axs[0,1].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_CVP_forSentiment,3)}', fontsize=8, horizontalalignment='right')

    group_combineAll_basedOnNewModel = myGrouping_new(combineAll_forSentiment, 3, underEstimatedAnswersIdList_forSentiment, overEstimatedAnswersIdList_forSentiment, disagreeAnswersIdList, targetType)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnNewModel]
    group_newModelsklearnQ_rankZscores = [t[3] for t in group_combineAll_basedOnNewModel]
    colors = [] 
    for t in group_combineAll_basedOnNewModel:
        if t[-1] == 'under':
            colors.append('blue')
        elif t[-1] == 'over':
            colors.append('red')
        else:
            colors.append('green')
    
    # # when not grouping
    # group_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    # group_newModelsklearnQ_rankZscores = [t[3] for t in combineAll_forSentiment]
    
    if targetType in ['disagree','all']:
        axs[0, 2].scatter(group_newModelsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2, c=colors)
    else:
        group_newModelsklearnQ_rankZscores_targetType = [t[3] for t in group_combineAll_basedOnNewModel if t[-1] == targetType]
        group_priorQ_rankZscores_targetType = [t[0] for t in group_combineAll_basedOnNewModel if t[-1] == targetType]
        colors_targetType = [colors[i] for i,t in enumerate(group_combineAll_basedOnNewModel) if t[-1] == targetType]
        axs[0, 2].scatter(group_newModelsklearnQ_rankZscores_targetType, group_priorQ_rankZscores_targetType,s = 2, c=colors_targetType)
    # ### using all raw points to fit
    all_priorQ_rankZscores = [t[0] for t in combineAll_forSentiment]
    all_newModelsklearnQ_rankZscores = [t[3] for t in combineAll_forSentiment]
    
    axs[0, 2].set_xlabel(f'{newModelName} quality rankZscore', fontsize = 8)
    axs[0, 2].set_ylabel(yAxisName_1, fontsize = 8)
    axs[0, 2].set_xlim(-2,2)
    axs[0, 2].set_ylim(-2,2)

    # OLS fit
    # z3,p3 = curveFitOLS(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    # sumSquredResiduls_newModel_forSentiment, norm_sumSquredResiduls_newModel_forSentiment, squredResiduals_newModel_forSentiment, r2_newModel_forSentiment, pearson_stat_newModel_forSentiment, pearson_p_newModel_forSentiment = residulToDiagonalLine(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    all_newModelsklearnQ_rankZscores_targetType = [t[3] for t in combineAll_forSentiment if t[-1] in targetAnswerIdList]
    all_priorQ_rankZscores_targetType = [t[0] for t in combineAll_forSentiment if t[-1] in targetAnswerIdList]
    sumSquredResiduls_newModel_forSentiment, norm_sumSquredResiduls_newModel_forSentiment, squredResiduals_newModel_forSentiment, r2_newModel_forSentiment, pearson_stat_newModel_forSentiment, pearson_p_newModel_forSentiment = residulToDiagonalLine(all_newModelsklearnQ_rankZscores_targetType, all_priorQ_rankZscores_targetType)
    # axs[1,0].plot(all_newModelsklearnQ_rankZscores,p3(all_newModelsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[1,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[1,0].legend(loc="best", fontsize = 6)
    axs[0,2].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[0,2].text(2, -1.5, f'residual={round(sumSquredResiduls_newModel_forSentiment,1)}', fontsize=8, horizontalalignment='right')
    # axs[0,2].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_newModel_forSentiment,3)}', fontsize=8, horizontalalignment='right')

    
    # second row for helpful as ground truth
    if targetType == 'under':
        targetAnswerIdList = underEstimatedAnswersIdList_forHelpful
    elif targetType == 'over':
        targetAnswerIdList = overEstimatedAnswersIdList_forHelpful
    elif targetType == 'normal':
        targetAnswerIdList = [t[-1] for t in combineAll_forHelpful if t[-1] not in underEstimatedAnswersIdList_forHelpful and t[-1] not in overEstimatedAnswersIdList_forHelpful]
    elif targetType == 'disagree':
        targetAnswerIdList = disagreeAnswersIdList
    else:
        targetAnswerIdList = [t[-1] for t in combineAll_forHelpful]


    group_combineAll_basedOnVoteDiff = myGrouping_new(combineAll_forHelpful, 1, underEstimatedAnswersIdList_forHelpful, overEstimatedAnswersIdList_forHelpful, disagreeAnswersIdList, targetType)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiff]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnVoteDiff]
    colors = [] 
    for t in group_combineAll_basedOnVoteDiff:
        if t[-1] == 'under':
            colors.append('blue')
        elif t[-1] == 'over':
            colors.append('red')
        else:
            colors.append('green')
    # # when not grouping
    # group_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    # group_voteDiff_rankZscores = [t[1] for t in combineAll_forHelpful]
    
    if targetType in ['disagree','all']:
        axs[1, 0].scatter(group_voteDiff_rankZscores, group_priorQ_rankZscores,s = 2, c=colors)
    else:
        group_voteDiff_rankZscores_targetType = [t[1] for t in group_combineAll_basedOnVoteDiff if t[-1] == targetType]
        group_priorQ_rankZscores_targetType = [t[0] for t in group_combineAll_basedOnVoteDiff if t[-1] == targetType]
        colors_targetType = [colors[i] for i,t in enumerate(group_combineAll_basedOnVoteDiff) if t[-1] == targetType]
        axs[1, 0].scatter(group_voteDiff_rankZscores_targetType, group_priorQ_rankZscores_targetType,s = 2, c=colors_targetType)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    all_voteDiff_rankZscores = [t[1] for t in combineAll_forHelpful]
    
    axs[1, 0].set_xlabel('vote diff rankZscore', fontsize = 8)
    axs[1, 0].set_ylabel(yAxisName_2, fontsize = 8)
    axs[1, 0].set_xlim(-2,2)
    axs[1, 0].set_ylim(-2,2)

    # OLS fit
    # z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    # sumSquredResiduls_voteDiff_forHelpful, norm_sumSquredResiduls_voteDiff_forHelpful, squredResiduals_voteDiff_forHelpful, r2_voteDiff_forHelpful, pearson_stat_voteDiff_forHelpful, pearson_p_voteDiff_forHelpful = residulToDiagonalLine(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    all_voteDiff_rankZscores_targetType = [t[1] for t in combineAll_forHelpful if t[-1] in targetAnswerIdList]
    all_priorQ_rankZscores_targetType = [t[0] for t in combineAll_forHelpful if t[-1] in targetAnswerIdList]
    sumSquredResiduls_voteDiff_forHelpful, norm_sumSquredResiduls_voteDiff_forHelpful, squredResiduals_voteDiff_forHelpful, r2_voteDiff_forHelpful, pearson_stat_voteDiff_forHelpful, pearson_p_voteDiff_forHelpful = residulToDiagonalLine(all_voteDiff_rankZscores_targetType, all_priorQ_rankZscores_targetType)
    # axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1, 
    #               label=f'slope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,0].legend(loc="best", fontsize = 6)
    axs[1,0].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[1,0].text(2, -1.5, f'residual={round(sumSquredResiduls_voteDiff_forHelpful,1)}', fontsize=8, horizontalalignment='right')
    # axs[1,0].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_voteDiff_forHelpful,3)}', fontsize=8, horizontalalignment='right')


    group_combineAll_basedOnCVP = myGrouping_new(combineAll_forHelpful, 2, underEstimatedAnswersIdList_forHelpful, overEstimatedAnswersIdList_forHelpful, disagreeAnswersIdList, targetType)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVP]
    group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnCVP]
    colors = [] 
    for t in group_combineAll_basedOnCVP:
        if t[-1] == 'under':
            colors.append('blue')
        elif t[-1] == 'over':
            colors.append('red')
        else:
            colors.append('green')
    # # when not grouping
    # group_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    # group_CVPsklearnQ_rankZscores = [t[2] for t in combineAll_forHelpful]
    
    if targetType in ['disagree','all']:
        axs[1, 1].scatter(group_CVPsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2, c=colors)
    else:
        group_CVPsklearnQ_rankZscores_targetType = [t[2] for t in group_combineAll_basedOnCVP if t[-1] == targetType]
        group_priorQ_rankZscores_targetType = [t[0] for t in group_combineAll_basedOnCVP if t[-1] == targetType]
        colors_targetType = [colors[i] for i,t in enumerate(group_combineAll_basedOnCVP) if t[-1] == targetType]
        axs[1, 1].scatter(group_CVPsklearnQ_rankZscores_targetType, group_priorQ_rankZscores_targetType,s = 2, c=colors_targetType)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll_forHelpful]
    
    axs[1, 1].set_xlabel('CVP quality rankZscore', fontsize = 8)
    axs[1, 1].set_ylabel(yAxisName_2, fontsize = 8)
    axs[1, 1].set_xlim(-2,2)
    axs[1, 1].set_ylim(-2,2)

    # OLS fit
    # z1,p1 = curveFitOLS(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    # sumSquredResiduls_CVP_forHelpful, norm_sumSquredResiduls_CVP_forHelpful, squredResiduals_CVP_forHelpful, r2_CVP_forHelpful, pearson_stat_CVP_forHelpful, pearson_p_CVP_forHelpful = residulToDiagonalLine(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    all_CVPsklearnQ_rankZscores_targetType = [t[2] for t in combineAll_forHelpful if t[-1] in targetAnswerIdList]
    all_priorQ_rankZscores_targetType = [t[0] for t in combineAll_forHelpful if t[-1] in targetAnswerIdList]
    sumSquredResiduls_CVP_forHelpful, norm_sumSquredResiduls_CVP_forHelpful, squredResiduals_CVP_forHelpful, r2_CVP_forHelpful, pearson_stat_CVP_forHelpful, pearson_p_CVP_forHelpful = residulToDiagonalLine(all_CVPsklearnQ_rankZscores_targetType, all_priorQ_rankZscores_targetType)
    # axs[0,1].plot(all_CVPsklearnQ_rankZscores,p1(all_CVPsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[0,1].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[0,1].legend(loc="best", fontsize = 6)
    axs[1,1].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[1,1].text(2, -1.5, f'residual={round(sumSquredResiduls_CVP_forHelpful,1)}', fontsize=8, horizontalalignment='right')
    # axs[1,1].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_CVP_forHelpful,3)}', fontsize=8, horizontalalignment='right')

    group_combineAll_basedOnNewModel = myGrouping_new(combineAll_forHelpful, 3, underEstimatedAnswersIdList_forHelpful, overEstimatedAnswersIdList_forHelpful, disagreeAnswersIdList, targetType)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnNewModel]
    group_newModelsklearnQ_rankZscores = [t[3] for t in group_combineAll_basedOnNewModel]
    colors = [] 
    for t in group_combineAll_basedOnNewModel:
        if t[-1] == 'under':
            colors.append('blue')
        elif t[-1] == 'over':
            colors.append('red')
        else:
            colors.append('green')
    # # when not grouping
    # group_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    # group_newModelsklearnQ_rankZscores = [t[3] for t in combineAll_forHelpful]
    
    if targetType in ['disagree','all']:
        axs[1, 2].scatter(group_newModelsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2, c=colors)
    else:
        group_newModelsklearnQ_rankZscores_targetType = [t[3] for t in group_combineAll_basedOnNewModel if t[-1] == targetType]
        group_priorQ_rankZscores_targetType = [t[0] for t in group_combineAll_basedOnNewModel if t[-1] == targetType]
        colors_targetType = [colors[i] for i,t in enumerate(group_combineAll_basedOnNewModel) if t[-1] == targetType]
        axs[1, 2].scatter(group_newModelsklearnQ_rankZscores_targetType, group_priorQ_rankZscores_targetType,s = 2, c=colors_targetType)
    # ### using all raw points to fit
    all_priorQ_rankZscores = [t[0] for t in combineAll_forHelpful]
    all_newModelsklearnQ_rankZscores = [t[3] for t in combineAll_forHelpful]
    
    axs[1, 2].set_xlabel(f'{newModelName} quality rankZscore', fontsize = 8)
    axs[1, 2].set_ylabel(yAxisName_2, fontsize = 8)
    axs[1, 2].set_xlim(-2,2)
    axs[1, 2].set_ylim(-2,2)

    # OLS fit
    # z3,p3 = curveFitOLS(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    # sumSquredResiduls_newModel_forHelpful, norm_sumSquredResiduls_newModel_forHelpful, squredResiduals_newModel_forHelpful, r2_newModel_forHelpful, pearson_stat_newModel_forHelpful, pearson_p_newModel_forHelpful = residulToDiagonalLine(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    all_newModelsklearnQ_rankZscores_targetType = [t[3] for t in combineAll_forHelpful if t[-1] in targetAnswerIdList]
    all_priorQ_rankZscores_targetType = [t[0] for t in combineAll_forHelpful if t[-1] in targetAnswerIdList]
    sumSquredResiduls_newModel_forHelpful, norm_sumSquredResiduls_newModel_forHelpful, squredResiduals_newModel_forHelpful, r2_newModel_forHelpful, pearson_stat_newModel_forHelpful, pearson_p_newModel_forHelpful = residulToDiagonalLine(all_newModelsklearnQ_rankZscores_targetType, all_priorQ_rankZscores_targetType)
    # axs[1,0].plot(all_newModelsklearnQ_rankZscores,p3(all_newModelsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[1,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    # axs[1,0].legend(loc="best", fontsize = 6)
    axs[1,2].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[1,2].text(2, -1.5, f'residual={round(sumSquredResiduls_newModel_forHelpful,1)}', fontsize=8, horizontalalignment='right')
    # axs[1,2].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_newModel_forHelpful,3)}', fontsize=8, horizontalalignment='right')


    # fig.suptitle(f"{commName.replace('.stackexchange','')}", fontsize = 8)
    plotFileName = plotFileName.strip('.pdf')+'_'+targetType+'_fixedZscoreRange.pdf'

    savePlot(fig, plotFileName)

    return 
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
        bestRegAlpha_CVP = sorted(CVP_testAccuracyAndRegList, key=lambda x:x[0], reverse=True)[0][1]
        bestRegAlpha_newModel = sorted(newModel_testAccuracyAndRegList, key=lambda x:x[0], reverse=True)[0][1]
        bestRegAlpha_newModelInteraction = sorted(newModelInteraction_testAccuracyAndRegList, key=lambda x:x[0], reverse=True)[0][1]
    
    sampleCount = return_trainSuccess_dict_newModel[list(return_trainSuccess_dict_newModelInteraction.keys())[0]]['dataShape'][0]

    commName2bestRegAlphas[commName] = [bestRegAlpha_newModelInteraction, bestRegAlpha_newModel, bestRegAlpha_CVP, sampleCount]
    return


#####################################################################################################################
def myFun(commName, commDir, rootDir, roundIndex, variation, reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP, sampled_comms, sampleCount):
   
    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())
    print(f"processing {commName}")

    # load intermediate_data filesdd
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

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
        topAnswersIdList = tup[15]
        underEstimatedAnswersIdList_forSentiment = tup[16]
        overEstimatedAnswersIdList_forSentiment = tup[17]
        underEstimatedAnswersIdList_forHelpful = tup[18]
        overEstimatedAnswersIdList_forHelpful = tup[19]
        disagreeAnswersIdList = tup[20]
        aid2quality_CVP = tup[21]
        aid2quality_newModel = tup[22]
        aid2quality_newModelInteration = tup[23]
        print( f"loaded intermediate outputs for {commName}.")
    
    combineAll_forSentiment = []

    for aid in aid2sentiment_rankZscore.keys():
        try:
            tup = (aid2sentiment_rankZscore[aid],aid2VoteDiff_rankZscore[aid],aid2CVP_sklearn_q_rankZscore[aid],aid2newModel_sklearn_q_rankZscore[aid],aid2newModelInteraction_sklearn_q_rankZscore[aid], aid)
            combineAll_forSentiment.append(tup)
        except Exception as e:
            print(e)

    combineAll_forHelpful = []

    for aid in aid2helpfulScore_rankZscore.keys():
        try:
            tup = (aid2helpfulScore_rankZscore[aid],aid2VoteDiff_rankZscore[aid],aid2CVP_sklearn_q_rankZscore[aid],aid2newModel_sklearn_q_rankZscore[aid],aid2newModelInteraction_sklearn_q_rankZscore[aid], aid)
            combineAll_forHelpful.append(tup)
            
        except Exception as e:
            print(e)

    ### for using sentiment scores as ground truth
    # plotFileName = f"temperalOrderTraining19_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).pdf" # group by interval , 1layer grouping on x_axis
    plotFileName = f"temperalOrderTraining19_groupByFixedGroupSize_flipXY.pdf" # group by fixed range of zscore , 1layer grouping on x_axis
    tup = myPlot(commName, combineAll_forSentiment,combineAll_forHelpful, plotFileName, reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP,
                 underEstimatedAnswersIdList_forSentiment, overEstimatedAnswersIdList_forSentiment, 
                 underEstimatedAnswersIdList_forHelpful, overEstimatedAnswersIdList_forHelpful, disagreeAnswersIdList)
    
    # plotFileName = f"temperalOrderTraining19_colors.pdf" # group by interval , 1layer grouping on x_axis
    # myPlot_new(commName, combineAll_forSentiment,combineAll_forHelpful, plotFileName, reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP,
    #              underEstimatedAnswersIdList_forSentiment, overEstimatedAnswersIdList_forSentiment, 
    #              underEstimatedAnswersIdList_forHelpful, overEstimatedAnswersIdList_forHelpful, disagreeAnswersIdList)
    """
    sumSquredResiduls_voteDiff_forSentiment, sumSquredResiduls_CVP_forSentiment, sumSquredResiduls_newModel_forSentiment, sumSquredResiduls_voteDiff_forHelpful, sumSquredResiduls_CVP_forHelpful, sumSquredResiduls_newModel_forHelpful, betterThanVoteDiff_forSentiment, betterThanCVP_forSentiment, betterThanBoth_forSentiment, betterThanVoteDiff_forHelpful, betterThanCVP_forHelpful, betterThanBoth_forHelpful, r2_voteDiff_forSentiment, r2_CVP_forSentiment, r2_newModel_forSentiment, r2_voteDiff_forHelpful, r2_CVP_forHelpful, r2_newModel_forHelpful, pearson_stat_voteDiff_forSentiment, pearson_p_voteDiff_forSentiment, pearson_stat_CVP_forSentiment, pearson_p_CVP_forSentiment, pearson_stat_newModel_forSentiment, pearson_p_newModel_forSentiment, pearson_stat_voteDiff_forHelpful, pearson_p_voteDiff_forHelpful, pearson_stat_CVP_forHelpful, pearson_p_CVP_forHelpful, pearson_stat_newModel_forHelpful, pearson_p_newModel_forHelpful, ttest_voteDiff_forSentiment_residaul, p_ofTtest_voteDiff_forSentiment_residaul, ttest_CVP_forSentiment_residaul, p_ofTtest_CVP_forSentiment_residaul,ttest_voteDiff_forHelpful_residaul, p_ofTtest_voteDiff_forHelpful_residaul, ttest_CVP_forHelpful_residaul, p_ofTtest_CVP_forHelpful_residaul = tup

    # # compare prior q and learned q ranks (question-level aggregate)
    # convert involved answer2parentQ into qid2aidList
    qid2aidList = defaultdict()
    for (qid,aid) in total_answersWithVotes_ids:
        if aid not in aid2sentiment_rank.keys():
            continue
        if qid in qid2aidList.keys():
            qid2aidList[qid].append(aid)
        else:
            qid2aidList[qid] = [aid]
    # compute question-level kendalltau distance of ranks
    # kendalltauList_voteDiff_forSentiment = []
    # kendalltauList_CVP_forSentiment = []
    # kendalltauList_newModel_forSentiment = []
    # kendalltauList_voteDiff_forHelpful = []
    # kendalltauList_CVP_forHelpful = []
    # kendalltauList_newModel_forHelpful = []

    # pList_voteDiff_forSentiment = []
    # pList_CVP_forSentiment = []
    # pList_newModel_forSentiment = []
    # pList_voteDiff_forHelpful = []
    # pList_CVP_forHelpful = []
    # pList_newModel_forHelpful = []
            
    sentimentRanks_asWhole = []
    helpfulRanks_asWhole = []
    voteDiffRanks_asWhole = []
    CVPqualityRanks_asWhole = []
    newModelqualityRanks_asWhole = []

    kendalltauList_voteDiff_forSentiment = []
    kendalltauList_CVP_forSentiment = []
    kendalltauList_newModel_forSentiment = []

    kendalltauList_voteDiff_forHelpful = []
    kendalltauList_CVP_forHelpful = []
    kendalltauList_newModel_forHelpful = []

    rank_base = 0
    for qid, aidList in qid2aidList.items():
        if len(aidList) <=3:
            continue

        # for kT as whole
        cur_ranks = [ aid2sentiment_rank[aid]+rank_base for aid in aidList]
        sentimentRanks_asWhole.extend( cur_ranks )

        cur_ranks = [ aid2helpfulScore_rank[aid]+rank_base for aid in aidList]
        helpfulRanks_asWhole.extend( cur_ranks )

        cur_ranks = [aid2VoteDiff_rank[aid]+rank_base for aid in aidList]
        voteDiffRanks_asWhole.extend(cur_ranks  )

        cur_ranks = [aid2CVP_sklearn_q_rank[aid]+rank_base for aid in aidList]
        CVPqualityRanks_asWhole.extend(cur_ranks) 

        cur_ranks = [aid2newModel_sklearn_q_rank[aid]+rank_base for aid in aidList]
        newModelqualityRanks_asWhole.extend( cur_ranks )

        rank_base += len(aidList)

        # for KT per question
        cur_sentimetRanks = [ aid2sentiment_rank[aid] for aid in aidList]
        cur_helpfulRanks = [ aid2helpfulScore_rank[aid] for aid in aidList]
        cur_voteDiffRanks = [aid2VoteDiff_rank[aid] for aid in aidList]
        cur_CVPRanks = [aid2CVP_sklearn_q_rank[aid] for aid in aidList]
        cur_newModelRanks = [aid2newModel_sklearn_q_rank[aid] for aid in aidList]

        kendalltau_voteDiff_forSentiment = stats.kendalltau(cur_sentimetRanks, cur_voteDiffRanks).statistic
        # p_voteDiff_forSentment = stats.kendalltau(cur_sentimetRanks, cur_voteDiffRanks).pvalue
        kendaltau_CVP_forSentiment = stats.kendalltau(cur_sentimetRanks, cur_CVPRanks).statistic
        # p_CVP_forSentiment = stats.kendalltau(cur_sentimetRanks, cur_CVPRanks).pvalue
        kendalltau_newModel_forSentiment = stats.kendalltau(cur_sentimetRanks, cur_newModelRanks).statistic
        # p_newModel_forSentiment = stats.kendalltau(cur_sentimetRanks, cur_newModelRanks).pvalue

        kendalltau_voteDiff_forHelpful = stats.kendalltau(cur_helpfulRanks, cur_voteDiffRanks).statistic
        # p_voteDiff_forHelpful = stats.kendalltau(cur_helpfulRanks, cur_voteDiffRanks).pvalue
        kendaltau_CVP_forHelpful = stats.kendalltau(cur_helpfulRanks, cur_CVPRanks).statistic
        # p_CVP_forHelpful = stats.kendalltau(cur_helpfulRanks, cur_CVPRanks).pvalue
        kendalltau_newModel_forHelpful = stats.kendalltau(cur_helpfulRanks, cur_newModelRanks).statistic
        # p_newModel_forHelpful = stats.kendalltau(cur_helpfulRanks, cur_newModelRanks).pvalue

        kendalltauList_voteDiff_forSentiment.append(kendalltau_voteDiff_forSentiment)
        kendalltauList_CVP_forSentiment.append(kendaltau_CVP_forSentiment)
        kendalltauList_newModel_forSentiment.append(kendalltau_newModel_forSentiment)

        kendalltauList_voteDiff_forHelpful.append(kendalltau_voteDiff_forHelpful)
        kendalltauList_CVP_forHelpful.append(kendaltau_CVP_forHelpful)
        kendalltauList_newModel_forHelpful.append(kendalltau_newModel_forHelpful)


    # for KT as whole
    kendalltau_voteDiff_forSentiment_asWhole = stats.kendalltau(sentimentRanks_asWhole, voteDiffRanks_asWhole).statistic
    kendalltau_CVP_forSentiment_asWhole = stats.kendalltau(sentimentRanks_asWhole, CVPqualityRanks_asWhole).statistic
    kendalltau_newModel_forSentiment_asWhole = stats.kendalltau(sentimentRanks_asWhole, newModelqualityRanks_asWhole).statistic

    p_voteDiff_forSentiment_asWhole = stats.kendalltau(sentimentRanks_asWhole, voteDiffRanks_asWhole).pvalue
    p_CVP_forSentiment_asWhole = stats.kendalltau(sentimentRanks_asWhole, CVPqualityRanks_asWhole).pvalue
    p_newModel_forSentiment_asWhole = stats.kendalltau(sentimentRanks_asWhole, newModelqualityRanks_asWhole).pvalue

    kendalltau_voteDiff_forHelpful_asWhole = stats.kendalltau(helpfulRanks_asWhole, voteDiffRanks_asWhole).statistic
    kendalltau_CVP_forHelpful_asWhole = stats.kendalltau(helpfulRanks_asWhole, CVPqualityRanks_asWhole).statistic
    kendalltau_newModel_forHelpful_asWhole = stats.kendalltau(helpfulRanks_asWhole, newModelqualityRanks_asWhole).statistic

    p_voteDiff_forHelpful_asWhole = stats.kendalltau(helpfulRanks_asWhole, voteDiffRanks_asWhole).pvalue
    p_CVP_forHelpful_asWhole = stats.kendalltau(helpfulRanks_asWhole, CVPqualityRanks_asWhole).pvalue
    p_newModel_forHelpful_asWhole = stats.kendalltau(helpfulRanks_asWhole, newModelqualityRanks_asWhole).pvalue

    # for KT per question
    kendalltau_voteDiff_forSentiment_perQuestion = mean(kendalltauList_voteDiff_forSentiment)
    kendalltau_CVP_forSentiment_perQuestion = mean(kendalltauList_CVP_forSentiment)
    kendalltau_newModel_forSentiment_perQuestion = mean(kendalltauList_newModel_forSentiment)

    ttest_voteDiff_forSentiment_perQuestion = scipy.stats.ttest_ind(kendalltauList_newModel_forSentiment, kendalltauList_voteDiff_forSentiment).statistic
    p_ofTtest_voteDiff_forSentiment_perQuestion = scipy.stats.ttest_ind(kendalltauList_newModel_forSentiment, kendalltauList_voteDiff_forSentiment).pvalue /2 # one-tailed

    ttest_CVP_forSentiment_perQuestion = scipy.stats.ttest_ind(kendalltauList_newModel_forSentiment, kendalltauList_CVP_forSentiment).statistic
    p_ofTtest_CVP_forSentiment_perQuestion = scipy.stats.ttest_ind(kendalltauList_newModel_forSentiment, kendalltauList_CVP_forSentiment).pvalue /2 # one-tailed

    kendalltau_voteDiff_forHelpful_perQuestion = mean(kendalltauList_voteDiff_forHelpful)
    kendalltau_CVP_forHelpful_perQuestion = mean(kendalltauList_CVP_forHelpful)
    kendalltau_newModel_forHelpful_perQuestion = mean(kendalltauList_newModel_forHelpful)

    ttest_voteDiff_forHelpful_perQuestion = scipy.stats.ttest_ind(kendalltauList_newModel_forHelpful, kendalltauList_voteDiff_forHelpful).statistic
    p_ofTtest_voteDiff_forHelpful_perQuestion = scipy.stats.ttest_ind(kendalltauList_newModel_forHelpful, kendalltauList_voteDiff_forHelpful).pvalue /2 # one-tailed

    ttest_CVP_forHelpful_perQuestion = scipy.stats.ttest_ind(kendalltauList_newModel_forHelpful, kendalltauList_CVP_forHelpful).statistic
    p_ofTtest_CVP_forHelpful_perQuestion = scipy.stats.ttest_ind(kendalltauList_newModel_forHelpful, kendalltauList_CVP_forHelpful).pvalue /2 # one-tailed



    # save  into csv
    with open(rootDir+f'/allComm_temperalOrderTraining19_residuals.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( [commName, sampleCount, len(aid2sentiment_rank),len(topAnswersIdList), len(disagreeAnswersIdList), len(overEstimatedAnswersIdList_forHelpful), len(underEstimatedAnswersIdList_forHelpful),
                          sumSquredResiduls_voteDiff_forSentiment, sumSquredResiduls_CVP_forSentiment, sumSquredResiduls_newModel_forSentiment, 
                          ttest_voteDiff_forSentiment_residaul, p_ofTtest_voteDiff_forSentiment_residaul, ttest_CVP_forSentiment_residaul, p_ofTtest_CVP_forSentiment_residaul,
                          sumSquredResiduls_voteDiff_forHelpful, sumSquredResiduls_CVP_forHelpful, sumSquredResiduls_newModel_forHelpful,
                          ttest_voteDiff_forHelpful_residaul, p_ofTtest_voteDiff_forHelpful_residaul, ttest_CVP_forHelpful_residaul, p_ofTtest_CVP_forHelpful_residaul,
                          betterThanVoteDiff_forSentiment, betterThanCVP_forSentiment, betterThanBoth_forSentiment,
                          betterThanVoteDiff_forHelpful, betterThanCVP_forHelpful, betterThanBoth_forHelpful,
                        #   r2_voteDiff_forSentiment, r2_CVP_forSentiment, r2_newModel_forSentiment, 
                        #   r2_voteDiff_forHelpful, r2_CVP_forHelpful, r2_newModel_forHelpful,
                        #   pearson_stat_voteDiff_forSentiment, pearson_p_voteDiff_forSentiment, pearson_stat_CVP_forSentiment, pearson_p_CVP_forSentiment, pearson_stat_newModel_forSentiment, pearson_p_newModel_forSentiment, 
                        #   pearson_stat_voteDiff_forHelpful, pearson_p_voteDiff_forHelpful, pearson_stat_CVP_forHelpful, pearson_p_CVP_forHelpful, pearson_stat_newModel_forHelpful, pearson_p_newModel_forHelpful
                        kendalltau_voteDiff_forSentiment_asWhole, p_voteDiff_forSentiment_asWhole,kendalltau_CVP_forSentiment_asWhole,p_CVP_forSentiment_asWhole, kendalltau_newModel_forSentiment_asWhole,p_newModel_forSentiment_asWhole,
                        kendalltau_voteDiff_forHelpful_asWhole, p_voteDiff_forHelpful_asWhole,kendalltau_CVP_forHelpful_asWhole,p_CVP_forHelpful_asWhole, kendalltau_newModel_forHelpful_asWhole,p_newModel_forHelpful_asWhole,
                        kendalltau_voteDiff_forSentiment_perQuestion,kendalltau_CVP_forSentiment_perQuestion, kendalltau_newModel_forSentiment_perQuestion, ttest_voteDiff_forSentiment_perQuestion, p_ofTtest_voteDiff_forSentiment_perQuestion, ttest_CVP_forSentiment_perQuestion, p_ofTtest_CVP_forSentiment_perQuestion,
                        kendalltau_voteDiff_forHelpful_perQuestion,kendalltau_CVP_forHelpful_perQuestion, kendalltau_newModel_forHelpful_perQuestion, ttest_voteDiff_forHelpful_perQuestion, p_ofTtest_voteDiff_forHelpful_perQuestion, ttest_CVP_forHelpful_perQuestion, p_ofTtest_CVP_forHelpful_perQuestion])
    """
def main():

    t0=time.time()
    rootDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    # # # save  into csv
    # with open('allComm_temperalOrderTraining19_residuals.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow( ["commName", "sample Count", "answer count", "top answer count", "disagree answer count", "over estimate count", "under estimate count",
    #                       "sumSquredResiduls_voteDiff_forSentiment", "sumSquredResiduls_CVP_forSentiment", "sumSquredResiduls_newModel_forSentiment", 
    #                       "ttest_voteDiff_forSentiment_residaul", "p_ofTtest_voteDiff_forSentiment_residaul", "ttest_CVP_forSentiment_residaul", "p_ofTtest_CVP_forSentiment_residaul",
    #                       "sumSquredResiduls_voteDiff_forHelpful", "sumSquredResiduls_CVP_forHelpful", "sumSquredResiduls_newModel_forHelpful",
    #                       "ttest_voteDiff_forHelpful_residaul", "p_ofTtest_voteDiff_forHelpful_residaul", "ttest_CVP_forHelpful_residaul", "p_ofTtest_CVP_forHelpful_residaul",
    #                       "betterThanVoteDiff_forSentiment count", "betterThanCVP_forSentiment count", "betterThanBoth_forSentiment count",
    #                       "betterThanVoteDiff_forHelpful count", "betterThanCVP_forHelpful count", "betterThanBoth_forHelpful count",
    #                     #   "r2_voteDiff_forSentiment", "r2_CVP_forSentiment", "r2_newModel_forSentiment", 
    #                     #   "r2_voteDiff_forHelpful", "r2_CVP_forHelpful", "r2_newModel_forHelpful",
    #                     #   "pearson_stat_voteDiff_forSentiment", "pearson_p_voteDiff_forSentiment", "pearson_stat_CVP_forSentiment", "pearson_p_CVP_forSentiment", "pearson_stat_newModel_forSentiment", "pearson_p_newModel_forSentiment",
    #                     #   "pearson_stat_voteDiff_forHelpful", "pearson_p_voteDiff_forHelpful", "pearson_stat_CVP_forHelpful", "pearson_p_CVP_forHelpful", "pearson_stat_newModel_forHelpful", "pearson_p_newModel_forHelpful"
    #                       "kendalltau_voteDiff_forSentiment_asWhole", "p_voteDiff_forSentiment_asWhole","kendalltau_CVP_forSentiment_asWhole","p_CVP_forSentiment_asWhole", "kendalltau_newModel_forSentiment_asWhole","p_newModel_forSentiment_asWhole",
    #                         "kendalltau_voteDiff_forHelpful_asWhole", "p_voteDiff_forHelpful_asWhole","kendalltau_CVP_forHelpful_asWhole","p_CVP_forHelpful_asWhole", "kendalltau_newModel_forHelpful_asWhole","p_newModel_forHelpful_asWhole",
    #                         "kendalltau_voteDiff_forSentiment_perQuestion","kendalltau_CVP_forSentiment_perQuestion", "kendalltau_newModel_forSentiment_perQuestion", "ttest_voteDiff_forSentiment_perQuestion", "p_ofTtest_voteDiff_forSentiment_perQuestion", "ttest_CVP_forSentiment_perQuestion", "p_ofTtest_CVP_forSentiment_perQuestion",
    #                         "kendalltau_voteDiff_forHelpful_perQuestion","kendalltau_CVP_forHelpful_perQuestion", "kendalltau_newModel_forHelpful_perQuestion", "ttest_voteDiff_forHelpful_perQuestion", "p_ofTtest_voteDiff_forHelpful_perQuestion", "ttest_CVP_forHelpful_perQuestion", "p_ofTtest_CVP_forHelpful_perQuestion"])
                        

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
    
    
    # load commName2selected_reg_strengthList (extracted from temperalOrderTraining15_verifyingQualities_GPT.py)
    with open(f'allComm_bestRegAlphas_fixedTau.dict', 'rb') as inputFile:
        commName2selected_reg_strengthList = pickle.load( inputFile)

    selected_comms =['cstheory.stackexchange','codegolf.meta.stackexchange','stackoverflow','politics.stackexchange',
                     'math.meta.stackexchange','mathoverflow.net','askubuntu','philosophy.stackexchange']
    
    
    # prepare args
    argsList = []
    for commName, tup in commName2selected_reg_strengthList.items():
        if commName not in selected_comms:
            continue
        reg_alpha_newModelInteraction, reg_alpha_newModel, reg_alpha_CVP, sampleCount = tup
        for comm in commDir_sizes_sortedlist:
            if comm[0] == commName:
                commDir = comm[1]
                break
        argsList.append((commName, commDir, rootDir, roundIndex, variation, reg_alpha_newModelInteraction, reg_alpha_newModel, reg_alpha_CVP, sampled_comms, sampleCount))

    
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
        if len(processes)==20:
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
    
def temp():  # for descriptive model and CVA comparision on top 120 communities 
    des_trendiness_ranks = [35,	104,	9,	6,	2,	66,	72,	37,	68,	84,	78,	77,	82,	71,	52,	18,	120,	86,	107,	57,	40,	50,	97,	111,	32,	29,	67,	36,	58,	106,	38,	51,	61,	102,	16,	69,	75,	26,	20,	110,	93,	76,	49,	43,	60,	83,	59,	23,	91,	41,	79,	90,	11,	89,	31,	116,	100,	92,	115,	65,	108,	112,	95,	13,	1,	101,	27,	33,	62,	81,	48,	42,	39,	113,	24,	105,	74,	47,	114,	87,	64,	88,	10,	53,	44,	103,	98,	80,	28,	25,	117,	15,	8,	12,	54,	21,	94,	85,	70,	73,	46,	4,	55,	3,	34,	119,	63,	118,	7,	109,	22,	19,	17,	30,	5,	14,	96,	45,	99,	56]
    cva_trendiness_ranks = [68,	113,	34,	48,	23,	13,	14,	76,	25,	55,	28,	87,	86,	18,	73,	96,	115,	65,	109,	24,	67,	54,	36,	92,	7,	75,	10,	8,	56,	118,	41,	12,	60,	117,	38,	62,	83,	32,	61,	120,	52,	89,	49,	3,	97,	59,	70,	6,	85,	37,	20,	82,	42,	47,	81,	95,	80,	84,	91,	29,	100,	101,	99,	102,	104,	110,	16,	79,	11,	31,	17,	57,	21,	116,	71,	106,	50,	64,	103,	69,	5,	22,	30,	40,	66,	111,	108,	74,	53,	39,	114,	35,	4,	58,	94,	46,	105,	19,	33,	78,	43,	27,	90,	2,	72,	93,	77,	119,	45,	112,	15,	1,	63,	51,	26,	44,	107,	88,	98,	9]

    des_conformity_ranks = [31,	112,	69,	21,	6,	14,	45,	91,	33,	50,	94,	84,	48,	90,	101,	1,	107,	9,	86,	42,	81,	68,	56,	5,	37,	32,	20,	63,	25,	113,	74,	39,	73,	102,	51,	70,	89,	62,	66,	120,	24,	79,	36,	61,	115,	100,	88,	64,	80,	75,	87,	77,	41,	85,	23,	109,	47,	3,	11,	52,	106,	105,	93,	83,	104,	82,	30,	59,	35,	46,	58,	96,	34,	98,	71,	117,	16,	95,	97,	53,	65,	55,	67,	13,	57,	108,	118,	92,	44,	29,	116,	28,	18,	54,	99,	27,	119,	76,	26,	78,	8,	10,	15,	22,	2,	43,	19,	110,	4,	111,	17,	38,	72,	60,	49,	40,	114,	12,	103,	7]
    cva_conformity_ranks = [72,	112,	52,	41,	8,	13,	56,	80,	7,	64,	97,	67,	75,	83,	98,	12,	106,	25,	95,	58,	46,	45,	31,	20,	3,	36,	30,	23,	51,	119,	76,	5,	87,	111,	6,	55,	85,	49,	74,	120,	17,	89,	54,	32,	118,	99,	86,	61,	93,	62,	66,	78,	38,	81,	60,	108,	70,	2,	43,	48,	105,	102,	101,	92,	107,	104,	39,	69,	59,	50,	63,	94,	47,	88,	77,	116,	21,	96,	91,	44,	57,	16,	40,	11,	82,	113,	117,	84,	34,	65,	114,	29,	14,	37,	90,	71,	110,	42,	28,	79,	9,	15,	27,	24,	4,	22,	53,	109,	26,	100,	18,	1,	73,	19,	33,	68,	115,	35,	103,	10]
    
    kendaltau_trendiness = stats.kendalltau(des_trendiness_ranks, cva_trendiness_ranks)
    kendaltau_conformity = stats.kendalltau(des_conformity_ranks, cva_conformity_ranks)

    print(kendaltau_trendiness)
    print(kendaltau_conformity)

if __name__ == "__main__":
  
    # # calling main function
    main()
    # temp()
