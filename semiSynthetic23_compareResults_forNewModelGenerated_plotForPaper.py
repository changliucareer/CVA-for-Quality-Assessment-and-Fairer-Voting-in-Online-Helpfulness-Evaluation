import os
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
from itertools import groupby
import re
import psutil
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import sys
from statistics import mean
from scipy.stats import norm, bernoulli, pearsonr
import numpy as np
import scipy
import seaborn as sns
from scipy.optimize import fsolve
from scipy.special import psi # digamma
import matplotlib.pyplot as plt
from scipy.special import kl_div
import random
from scipy import stats 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
from matplotlib import cm
from sklearn.metrics import r2_score

def residulToDiagonalLine(y_pred, y_true):
    squredResiduals = []
    for i in range(len(y_true)):
        squredResiduals.append((y_pred[i]-y_true[i])**2)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)
    pearson_stat = pearson[0]
    pearson_p = pearson[1]
    return sum(squredResiduals), sum(squredResiduals)/len(y_true), squredResiduals, r2, pearson_stat, pearson_p


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

    group_combineAll = []

    # qid2combingAll = defaultdict()
    # for t in combineAll:
    #     qid = t[-1]
    #     if qid in qid2combingAll.keys():
    #         qid2combingAll[qid].append(t)
    #     else:
    #         qid2combingAll[qid]=[t]

    # ## get answerCount2qids 
    # answerCount2qids = defaultdict()
    # for qid, cur_combineAll in qid2combingAll.items():
    #     answerCount= len(cur_combineAll)
    #     if answerCount in answerCount2qids.keys():
    #         answerCount2qids[answerCount].append(qid)
    #     else:
    #         answerCount2qids[answerCount]= [qid]

    group_count = 150

    # ### group by fixed number of answers
    # group_size = int(len(combineAll)/group_count) # the number of answers in each group
    
    # startIndex = 0
    # for i in range(startIndex, len(combineAll), group_size):
    #     cur_group = combineAll[i:group_size+i]
    #     if len(cur_group)>0: 
    #         # groupCount = len(cur_group)
    #         column_wise_mean = np.mean(cur_group,axis=0)
    #         group_combineAll.append(tuple(list(column_wise_mean) + [group_size])) # force as the same group size
    #     startIndex = i

    # ### group by fixed range of z score
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

    ### group by interval
    percentages = [x / group_count for x in range(1, group_count+1)]
    zScoreIntervals = [scipy.stats.norm.ppf(p) for p in percentages]

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
            # # group cur_group by qid
            # qid2cur_group = defaultdict()
            # for t in cur_group:
            #     qid = t[-1]
            #     if qid in qid2cur_group.keys():
            #         qid2cur_group[qid].append(t)
            #     else:
            #         qid2cur_group[qid]=[t]
            # # further grouping the cur_group by answerCount
            # for answerCount, qids in answerCount2qids.items():
            #     cur_group_curAnswerCount = []
            #     for qid, cur_question_group in qid2cur_group.items():
            #         if qid in qids:
            #             cur_group_curAnswerCount.extend(cur_question_group)

            #     if len(cur_group_curAnswerCount)>0:
            #         column_wise_mean = np.mean(cur_group_curAnswerCount,axis=0)
            #         group_combineAll.append(tuple(column_wise_mean))

            # 1layer grouping based on x-axis
            groupCount = len(cur_group)
            column_wise_mean = np.mean(cur_group,axis=0)
            group_combineAll.append(tuple(list(column_wise_mean) + [groupCount]))
    
    print(f"group z score sample count {len(group_combineAll)} sorted by column {sortingColumn}")
    return group_combineAll

def myGridGrouping(combineAll, sortingColumn, xColumn):
    combineAll.sort(key=lambda t:t[sortingColumn]) # sorted by zScore
    interval_count = 100

    # percentages = [x / interval_count for x in range(1, interval_count+1)]
    # zScoreIntervals = [scipy.stats.norm.ppf(p) for p in percentages]
    zScoreIntervals = list(np.arange(-1,1.1,0.1))
    zScoreIntervals += [np.inf]

    group_x = []
    group_y = []

    pointsCount = 0

    previous_zintv_y = zScoreIntervals[0]-1
    previous_zintv_x = zScoreIntervals[0]-1

    for zintv_y in zScoreIntervals:
        for zintv_x in zScoreIntervals:
            # clear cur group
            cur_group_y = []
            cur_group_x = []
            
            for cur_tup in combineAll:
                cur_zScore_y = cur_tup[sortingColumn]
                cur_zScore_x = cur_tup[xColumn]
                if (previous_zintv_y < cur_zScore_y) and (cur_zScore_y <= zintv_y):
                    if (previous_zintv_x < cur_zScore_x) and (cur_zScore_x <= zintv_x):
                        cur_group_y.append(cur_zScore_y)
                        cur_group_x.append(cur_zScore_x)
                   
            if len(cur_group_y)>0: 
                group_y.append( np.mean(cur_group_y) )
                group_x.append( np.mean(cur_group_x) )
                pointsCount += len(cur_group_x)
            
            previous_zintv_x =zintv_x # next interval x
        
        previous_zintv_y =zintv_y # next interval y
        previous_zintv_x = zScoreIntervals[0]-1 # restart interval x

    assert pointsCount==len(combineAll)

    print(f"group z score sample count {len(group_x)} sorted by column {sortingColumn}")
    return group_y, group_x

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]
##########################################################################
# plot with all data point
def myPlot(commName, combineAll, plotFileName):
    # newModelName = 'proposed model'
    newModelName = 'CVA'
    overOrUnderEstimatedFlag = ('overEstimated' in plotFileName) or ('underEstimated' in plotFileName)

    plt.cla()
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 2.1))
    fig.tight_layout(pad=1.5)


    ### x axis as prediction and y axis as ground truth
    group_combineAll_basedOnVoteDiff = myGrouping(combineAll, sortingColumn=1)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiff]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnVoteDiff]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnVoteDiff]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[0].scatter(group_voteDiff_rankZscores, group_priorQ_rankZscores,s = sizes)

    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_voteDiff_rankZscores = [t[1] for t in combineAll]

    axs[0].set_xlabel('vote diff rankZscore', fontsize = 8)
    axs[0].set_ylabel('true quality rankZscore', fontsize = 8)
    axs[0].set_xlim(-2,2)
    axs[0].set_ylim(-2,2)

    # OLS fit
    sumSquredResiduls_voteDiff, norm_sumSquredResiduls_voteDiff, squredResiduals_voteDiff, r2_voteDiff, pearson_stat_voteDiff, pearson_p_voteDiff = residulToDiagonalLine(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    axs[0].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[0].text(2, -1.5, f'residual={round(sumSquredResiduls_voteDiff,1)}', fontsize=8, horizontalalignment='right')
    # axs[0].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_voteDiff,3)}', fontsize=8, horizontalalignment='right')

    # group_combineAll_basedOnPriorQAndCVP = myGrouping(combineAll, sortingColumn1=0,sortingColumn2=2)
    # group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnPriorQAndCVP]
    # group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnPriorQAndCVP]

    group_combineAll_basedOnCVP = myGrouping(combineAll, sortingColumn=2)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVP]
    group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnCVP]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnCVP]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[1].scatter(group_CVPsklearnQ_rankZscores, group_priorQ_rankZscores,s = sizes)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll]

    
    axs[1].set_xlabel('CVP quality rankZscore', fontsize = 8)
    axs[1].set_ylabel('true quality rankZscore', fontsize = 8)
    axs[1].set_xlim(-2,2)
    axs[1].set_ylim(-2,2)

    # OLS fit
    sumSquredResiduls_CVP, norm_sumSquredResiduls_CVP, squredResiduals_CVP, r2_CVP, pearson_stat_CVP, pearson_p_CVP = residulToDiagonalLine(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
    axs[1].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[1].text(2, -1.5, f'residual={round(sumSquredResiduls_CVP,1)}', fontsize=8, horizontalalignment='right')
    # axs[1].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_CVP,3)}', fontsize=8, horizontalalignment='right')


    group_combineAll_basedOnNewModel = myGrouping(combineAll, sortingColumn=5)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnNewModel]
    group_newModelsklearnQ_rankZscores = [t[5] for t in group_combineAll_basedOnNewModel]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnNewModel]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[2].scatter(group_newModelsklearnQ_rankZscores, group_priorQ_rankZscores,s = sizes)
    
    # ### using all raw points to fit
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_newModelsklearnQ_rankZscores = [t[5] for t in combineAll]
    
    axs[2].set_xlabel(f'{newModelName} rankZscore', fontsize = 8)
    axs[2].set_ylabel('true quality rankZscore', fontsize = 8)
    axs[2].set_xlim(-2,2)
    axs[2].set_ylim(-2,2)

    # OLS fit
    sumSquredResiduls_newModel, norm_sumSquredResiduls_newModel, squredResiduals_newModel, r2_newModel, pearson_stat_newModel, pearson_p_newModel = residulToDiagonalLine(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
    axs[2].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[2].text(2, -1.5, f'residual={round(sumSquredResiduls_newModel,1)}', fontsize=8, horizontalalignment='right')
    # axs[2].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_newModel,3)}', fontsize=8, horizontalalignment='right')


    # compare answer-level residuals
    betterThanVoteDiff = 0
    betterThanCVP = 0
    betterThanBoth = 0
    for i, sr_newModel in enumerate(squredResiduals_newModel):
        sr_voteDiff = squredResiduals_voteDiff[i]
        sr_CVP = squredResiduals_CVP[i]
        if sr_newModel <= sr_voteDiff:
            betterThanVoteDiff += 1
        if sr_newModel <= sr_CVP:
            betterThanCVP += 1
        if sr_newModel <= sr_voteDiff and sr_newModel <= sr_CVP:
            betterThanBoth += 1


    # group_combineAll_basedOnPriorQAndNewModelInteraction = myGrouping(combineAll, sortingColumn1=0,sortingColumn2=7)
    # group_priorQ_rankZscores = [t[0] for t in  group_combineAll_basedOnPriorQAndNewModelInteraction]
    # group_newModelInteractionsklearnQ_rankZscores = [t[7] for t in  group_combineAll_basedOnPriorQAndNewModelInteraction]
    # group_combineAll_basedOnNewModelInteraction = myGrouping(combineAll, sortingColumn=7)
    # group_priorQ_rankZscores = [t[0] for t in  group_combineAll_basedOnNewModelInteraction]
    # group_newModelInteractionsklearnQ_rankZscores = [t[7] for t in  group_combineAll_basedOnNewModelInteraction]
    # axs[1, 1].scatter(group_newModelInteractionsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2)
    
    # # ### using all raw points to fit
    # all_priorQ_rankZscores = [t[0] for t in combineAll]
    # all_newModelInteractionsklearnQ_rankZscores = [t[7] for t in combineAll]
    # # # axs[1, 1].scatter(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2)

    # # ### using grouped points to fit
    # # all_priorQ_rankZscores = group_priorQ_rankZscores
    # # all_newModelInteractionsklearnQ_rankZscores = group_newModelInteractionsklearnQ_rankZscores
    
    # axs[1, 1].set_xlabel('newModelInteraction sklearn q rankZscore', fontsize = 8)
    # axs[1, 1].set_ylabel('prior q rankZscore', fontsize = 8)
    # axs[1, 1].set_xlim(-2,2)
    # axs[1, 1].set_ylim(-2,2)

    # # OLS fit
    # z4,p4 = curveFitOLS(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores)
    # sumSquredResiduls = residulToDiagonalLine(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores)
    # axs[1,1].plot(all_newModelInteractionsklearnQ_rankZscores,p4(all_newModelInteractionsklearnQ_rankZscores),"r-", linewidth=1, 
    #               label=f'slope={round(z4[0][0],4)}\nresidual={round(z3[1][0],4)}\nresidualToDiagonal={round(sumSquredResiduls,4)}')
    # axs[1,1].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"g-", linewidth=1)
    

    # # # robust regression
    # # Huber_robust_param_newModelInteraction,pred_y,Huber_robust_resid_newModelInteraction = robust_HuberT(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores)
    # # axs[1,1].plot(all_newModelInteractionsklearnQ_rankZscores,pred_y,"g-", label=f'robust_Huber slope={round(Huber_robust_param_newModelInteraction,4)}\nresidual={round(Huber_robust_resid_newModelInteraction,4)}')
    # # TheilSen_robust_param_newModelInteraction,pred_y,TheilSen_robust_resid_newModelInteraction = robust_TheilSen(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores)
    # # axs[1,1].plot(all_newModelInteractionsklearnQ_rankZscores,pred_y,"y-", label=f'robust_TheilSen slope={round(TheilSen_robust_param_newModelInteraction,4)}\nresidual={round(TheilSen_robust_resid_newModelInteraction,4)}')
    # # RANSAC_robust_param_newModelInteraction,pred_y,RANSAC_robust_resid_newModelInteraction, inlier_count = robust_RANSAC(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores)
    # # inlier_ratio = inlier_count/len(all_newModelInteractionsklearnQ_rankZscores)
    # # axs[1,1].plot(all_newModelInteractionsklearnQ_rankZscores,pred_y,"m-", label=f'robust_RANSAC slope={round(RANSAC_robust_param_newModelInteraction,4)}\nresidual={round(RANSAC_robust_resid_newModelInteraction,4)}\ninlierRatio={round(inlier_ratio,4)}')
    
    # axs[1,1].legend(loc="best", fontsize = 6)


    # fig.suptitle(f"{commName.replace('.stackexchange','')}\n({len(combineAll)})")
    """
    ### y axis as prediction and x axis as ground truth
    group_combineAll_basedOnVoteDiff = myGrouping(combineAll, sortingColumn=0)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiff]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnVoteDiff]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnVoteDiff]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[0].scatter(group_priorQ_rankZscores,group_voteDiff_rankZscores, s = sizes)

    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_voteDiff_rankZscores = [t[1] for t in combineAll]

    axs[0].set_ylabel('vote diff rankZscore', fontsize = 8)
    axs[0].set_xlabel('true quality rankZscore', fontsize = 8)
    axs[0].set_xlim(-2,2)
    axs[0].set_ylim(-2,2)

    # OLS fit
    sumSquredResiduls_voteDiff, norm_sumSquredResiduls_voteDiff, squredResiduals_voteDiff, r2_voteDiff, pearson_stat_voteDiff, pearson_p_voteDiff = residulToDiagonalLine(all_priorQ_rankZscores, all_voteDiff_rankZscores)
    axs[0].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[0].text(2, -1.5, f'residual={round(sumSquredResiduls_voteDiff,1)}', fontsize=8, horizontalalignment='right')
    # axs[0].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_voteDiff,3)}', fontsize=8, horizontalalignment='right')

    # group_combineAll_basedOnPriorQAndCVP = myGrouping(combineAll, sortingColumn1=0,sortingColumn2=2)
    # group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnPriorQAndCVP]
    # group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnPriorQAndCVP]

    group_combineAll_basedOnCVP = myGrouping(combineAll, sortingColumn=0)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVP]
    group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnCVP]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnCVP]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[1].scatter(group_priorQ_rankZscores,group_CVPsklearnQ_rankZscores, s = sizes)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll]

    
    axs[1].set_ylabel('CVP quality rankZscore', fontsize = 8)
    axs[1].set_xlabel('true quality rankZscore', fontsize = 8)
    axs[1].set_xlim(-2,2)
    axs[1].set_ylim(-2,2)

    # OLS fit
    sumSquredResiduls_CVP, norm_sumSquredResiduls_CVP, squredResiduals_CVP, r2_CVP, pearson_stat_CVP, pearson_p_CVP = residulToDiagonalLine(all_priorQ_rankZscores, all_CVPsklearnQ_rankZscores)
    axs[1].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[1].text(2, -1.5, f'residual={round(sumSquredResiduls_CVP,1)}', fontsize=8, horizontalalignment='right')
    # axs[1].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_CVP,3)}', fontsize=8, horizontalalignment='right')


    group_combineAll_basedOnNewModel = myGrouping(combineAll, sortingColumn=0)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnNewModel]
    group_newModelsklearnQ_rankZscores = [t[5] for t in group_combineAll_basedOnNewModel]
    gropu_sizes = [t[-1] for t in group_combineAll_basedOnNewModel]
    min_size = min(gropu_sizes)
    max_size = max(gropu_sizes)
    if max_size == min_size:
        sizes = [1 for s in gropu_sizes]
    else:
        sizes = [1 + 10*(s-min_size)/(max_size-min_size) for s in gropu_sizes]
    axs[2].scatter(group_priorQ_rankZscores,group_newModelsklearnQ_rankZscores, s = sizes)
    
    # ### using all raw points to fit
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_newModelsklearnQ_rankZscores = [t[5] for t in combineAll]
    
    axs[2].set_ylabel(f'{newModelName} rankZscore', fontsize = 8)
    axs[2].set_xlabel('true quality rankZscore', fontsize = 8)
    axs[2].set_xlim(-2,2)
    axs[2].set_ylim(-2,2)

    # OLS fit
    sumSquredResiduls_newModel, norm_sumSquredResiduls_newModel, squredResiduals_newModel, r2_newModel, pearson_stat_newModel, pearson_p_newModel = residulToDiagonalLine(all_priorQ_rankZscores, all_newModelsklearnQ_rankZscores)
    axs[2].plot([-2, 2], [-2, 2], ls="--", c='black', linewidth=0.5)
    axs[2].text(2, -1.5, f'residual={round(sumSquredResiduls_newModel,1)}', fontsize=8, horizontalalignment='right')
    # axs[2].text(2, -1.9, f'normalized residual={round(norm_sumSquredResiduls_newModel,3)}', fontsize=8, horizontalalignment='right')


    # compare answer-level residuals
    betterThanVoteDiff = 0
    betterThanCVP = 0
    betterThanBoth = 0
    for i, sr_newModel in enumerate(squredResiduals_newModel):
        sr_voteDiff = squredResiduals_voteDiff[i]
        sr_CVP = squredResiduals_CVP[i]
        if sr_newModel <= sr_voteDiff:
            betterThanVoteDiff += 1
        if sr_newModel <= sr_CVP:
            betterThanCVP += 1
        if sr_newModel <= sr_voteDiff and sr_newModel <= sr_CVP:
            betterThanBoth += 1
    """

    savePlot(fig, plotFileName)

    return len(all_priorQ_rankZscores), sumSquredResiduls_voteDiff, sumSquredResiduls_CVP, sumSquredResiduls_newModel, betterThanVoteDiff, betterThanCVP, betterThanBoth, r2_voteDiff, r2_CVP, r2_newModel, pearson_stat_voteDiff, pearson_p_voteDiff, pearson_stat_CVP, pearson_p_CVP, pearson_stat_newModel, pearson_p_newModel

##########################################################################
# plot fitting lines per question
def questionLevelPlot(commName, combineAll, compareType):
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

    # plt.cla()
    # fig, axs = plt.subplots(2, 2)
    # fig.tight_layout(pad=3.0)
    # axs[0, 0].set_xlabel(f'vote diff {compareType}', fontsize = 8)
    # axs[0, 0].set_ylabel(f'prior q {compareType}', fontsize = 8)
    # axs[0, 1].set_xlabel(f'CVP sklearn q {compareType}', fontsize = 8)
    # axs[0, 1].set_ylabel(f'prior q {compareType}', fontsize = 8)
    # axs[1, 0].set_xlabel(f'new model sklearn q {compareType}', fontsize = 8)
    # axs[1, 0].set_ylabel(f'prior q {compareType}', fontsize = 8)
    # axs[1, 1].set_xlabel(f'newModelInteraction sklearn q {compareType}', fontsize = 8)
    # axs[1, 1].set_ylabel(f'prior q {compareType}', fontsize = 8)

    answerCountThreshold = 3  # when answer count <=2, no residual
    selectedQids = []
    for qid, combineAll in qid2combingAll.items():
        if len(combineAll) >= answerCountThreshold:
            selectedQids.append(qid)

    print(f"selected {len(selectedQids)} questions for {commName}")


    voteDiffSlopes = []
    CVPSlopes = []
    newModelSlopes = []
    newModelInteractionSlopes = []
    voteDiffResiduals = []
    CVPResiduals = []
    newModelResiduals = []
    newModelInteractionResiduals = []

    # maxPlotTimes = 2
    # plotQids = random.sample(selectedQids, maxPlotTimes)

    n_colors = len(selectedQids)
    colours = cm.rainbow(np.linspace(0, 1, n_colors))
    myColorDict = dict([(qid,colours[i]) for i, qid in enumerate(selectedQids)])

    for qid in selectedQids:
        combineAll = qid2combingAll[qid]
        if len(combineAll) >= answerCountThreshold:
            
            cur_color = myColorDict[qid]
   
            ### using all raw points
            all_priorQ_rankZscores = [t[tupIndex_prior] for t in combineAll]
            all_voteDiff_rankZscores = [t[tupIndex_voteDiff] for t in combineAll]
            # axs[0, 0].scatter(all_voteDiff_rankZscores, all_priorQ_rankZscores,s = 2, color=cur_color)

            # compute Pearson's Product Moment Correlation Coefficient 
            pearsonCor_voteDiff = stats.pearsonr(all_priorQ_rankZscores, all_voteDiff_rankZscores,alternative='two-sided')[0]
            # compute Spearman's Rank Correlation Coefficient
            spearmanCor_voteDiff = stats.spearmanr(all_priorQ_rankZscores, all_voteDiff_rankZscores,alternative='two-sided')[0]

            # OLS fit
            z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
            voteDiffSlopes.append(z0[0][0])
            voteDiffResiduals.append(z0[1][0])

            # axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"-", linewidth=0.5, color=cur_color, label=f'question:{qid}\nslope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}\npearson Correlation: {round(pearsonCor_voteDiff,4)}')
            # axs[0,0].legend(loc="best", fontsize = 4)
            

            ### using all raw points
            all_priorQ_rankZscores = [t[tupIndex_prior] for t in combineAll]
            all_CVPsklearnQ_rankZscores = [t[tupIndex_CVP] for t in combineAll]
            # axs[0, 1].scatter(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2, color=cur_color)

            # compute Pearson's Product Moment Correlation Coefficient 
            pearsonCor_CVP = stats.pearsonr(all_priorQ_rankZscores, all_CVPsklearnQ_rankZscores,alternative='two-sided')[0]
            # compute Spearman's Rank Correlation Coefficient
            spearmanCor_CVP = stats.spearmanr(all_priorQ_rankZscores, all_CVPsklearnQ_rankZscores,alternative='two-sided')[0]

            # OLS fit
            z1,p1 = curveFitOLS(all_CVPsklearnQ_rankZscores, all_priorQ_rankZscores)
            CVPSlopes.append(z1[0][0])
            CVPResiduals.append(z1[1][0])
            # axs[0,1].plot(all_CVPsklearnQ_rankZscores,p1(all_CVPsklearnQ_rankZscores),"-", linewidth=0.5, color=cur_color, label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}\npearson Correlation: {round(pearsonCor_CVP,4)}')
            # axs[0,1].legend(loc="best", fontsize = 4)

            
            # ### using all raw points to fit
            all_priorQ_rankZscores = [t[tupIndex_prior] for t in combineAll]
            all_newModelsklearnQ_rankZscores = [t[tupIndex_newModel] for t in combineAll]
            # axs[1, 0].scatter(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2, color=cur_color)

            # compute Pearson's Product Moment Correlation Coefficient 
            pearsonCor_newModel = stats.pearsonr(all_priorQ_rankZscores, all_newModelsklearnQ_rankZscores,alternative='two-sided')[0]
            # compute Spearman's Rank Correlation Coefficient
            spearmanCor_newModel = stats.spearmanr(all_priorQ_rankZscores, all_newModelsklearnQ_rankZscores,alternative='two-sided')[0]

            # OLS fit
            z3,p3 = curveFitOLS(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores)
            newModelSlopes.append(z3[0][0])
            newModelResiduals.append(z3[1][0])
            # axs[1,0].plot(all_newModelsklearnQ_rankZscores,p3(all_newModelsklearnQ_rankZscores),"-", linewidth=0.5, color=cur_color, label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}\npearson Correlation: {round(pearsonCor_newModel,4)}')
            # axs[1,0].legend(loc="best", fontsize = 4)

            # ### using all raw points to fit
            all_priorQ_rankZscores = [t[tupIndex_prior] for t in combineAll]
            all_newModelInteractionsklearnQ_rankZscores = [t[tupIndex_newModelInteraction] for t in combineAll]
            # axs[1, 1].scatter(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2, color=cur_color)

            # compute Pearson's Product Moment Correlation Coefficient 
            pearsonCor_newModelInteraction = stats.pearsonr(all_priorQ_rankZscores, all_newModelInteractionsklearnQ_rankZscores,alternative='two-sided')[0]
            # compute Spearman's Rank Correlation Coefficient
            spearmanCor_newModelInteraction = stats.spearmanr(all_priorQ_rankZscores, all_newModelInteractionsklearnQ_rankZscores,alternative='two-sided')[0]

            # OLS fit
            z4,p4 = curveFitOLS(all_newModelInteractionsklearnQ_rankZscores, all_priorQ_rankZscores)
            newModelInteractionSlopes.append(z4[0][0])
            newModelInteractionResiduals.append(z4[1][0])
            # axs[1,1].plot(all_newModelInteractionsklearnQ_rankZscores,p4(all_newModelInteractionsklearnQ_rankZscores),"-", linewidth=0.5, color=cur_color, label=f'slope={round(z4[0][0],4)}\nresidual={round(z4[1][0],4)}\npearson Correlation: {round(pearsonCor_newModelInteraction,4)}')
            # axs[1,1].legend(loc="best", fontsize = 4)


    if len(voteDiffSlopes) >0:
        voteDiffAvgSlope = mean(voteDiffSlopes)
        CVPAvgSlope = mean(CVPSlopes)
        newModelAvgSlope = mean(newModelSlopes)
        newModelInteractionAvgSlope = mean(newModelInteractionSlopes)
        voteDiffAvgResidual = mean(voteDiffResiduals)
        CVPAvgResidual = mean(CVPResiduals)
        newModelAvgResidual= mean(newModelResiduals)
        newModelInteractionAvgResidual = mean(newModelInteractionResiduals)

    else:
        voteDiffAvgSlope = None
        CVPAvgSlope = None
        newModelAvgSlope = None
        newModelInteractionAvgSlope = None
        voteDiffAvgResidual = None
        CVPAvgResidual = None
        newModelAvgResidual = None
        newModelInteractionAvgResidual = None

    percentageOfPositiveSlope_voteDiff = sum([1 for s in voteDiffSlopes if s>0])/len(voteDiffSlopes)
    percentageOfPositiveSlope_CVP = sum([1 for s in CVPSlopes if s>0])/len(CVPSlopes)
    percentageOfPositiveSlope_newModel = sum([1 for s in newModelSlopes if s>0])/len(newModelSlopes)
    percentageOfPositiveSlope_newModelInteraction = sum([1 for s in newModelInteractionSlopes if s>0])/len(newModelInteractionSlopes)

    # fig.suptitle(f"{commName.replace('.stackexchange','')} (question with {answerCountThreshold} answers)")
    # savePlot(fig, plotFileName)
    return voteDiffAvgSlope,CVPAvgSlope,newModelAvgSlope,newModelInteractionAvgSlope,voteDiffAvgResidual,CVPAvgResidual,newModelAvgResidual,newModelInteractionAvgResidual, percentageOfPositiveSlope_voteDiff, percentageOfPositiveSlope_CVP, percentageOfPositiveSlope_newModel, percentageOfPositiveSlope_newModelInteraction
    
##########################################################################

def myFun(commName, commDir, rootDir, roundIndex, variation, reg_alpha1, reg_alpha2, try_reg_strengthList, sampled_comms):

    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "semiSynthetic19_compareResults_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # prior coefs
    if variation == "_fixedTau":
        if roundIndex in [1,2,3,4]:
            if commName == 'mathoverflow.net' and reg_alpha1==600: # special case
                with open(intermediate_directory+'/'+f"temperalOrderTraining12_newModel_fixedTau_regAlpha({reg_alpha1})_return.dict", 'rb') as inputFile:
                    return_trainSuccess_dict_newModel = pickle.load( inputFile)
            else:
                try: # for sampled comms
                    with open(intermediate_directory+'/'+f"temperalOrderTraining12_newModel_fixedTau_regAlpha({reg_alpha1})_forSampledQuestion_return.dict", 'rb') as inputFile:
                        return_trainSuccess_dict_newModel = pickle.load( inputFile)
                except:
                    with open(intermediate_directory+'/'+f"temperalOrderTraining12_newModel_fixedTau_regAlpha({reg_alpha1})_return.dict", 'rb') as inputFile:
                        return_trainSuccess_dict_newModel = pickle.load( inputFile)
           

    print(f"return train success dict newModel loaded. length {len(return_trainSuccess_dict_newModel)}")

    # learned coefs by CVP
    if roundIndex in [1,2,3,4]:
        with open(intermediate_directory+'/'+f'semiSynthetic9_CVP{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_training_return.dict', 'rb') as inputFile:
            semiSynthetic_return_trainSuccess_dict_CVP = pickle.load( inputFile)
    print(f"semiSynthetic_return train success dict CVP loaded. ")

    # learned coefs by new model
    if roundIndex in [1,2,3,4]:
        with open(intermediate_directory+'/'+f"semiSynthetic10_newModelTraining{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_return.dict", 'rb') as inputFile:
            semiSynthetic_return_trainSuccess_dict_newModel = pickle.load( inputFile)
    print(f"semiSynthetic_return train success dict new Model loaded. ")

    # learned coefs by new model with interaction
    if roundIndex in [1,2,3,4]:
        with open(intermediate_directory+'/'+f"semiSynthetic21_newModelInteractionTraining{variation}_newModelGenerated_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_return.dict", 'rb') as inputFile:
            semiSynthetic_return_trainSuccess_dict_newModelInteraction = pickle.load( inputFile)
    print(f"semiSynthetic_return train success dict new Model with interaction loaded. ")

    # get prior coefs
    try:
        if len(return_trainSuccess_dict_newModel)==1:
            simplifiedCommName = list(return_trainSuccess_dict_newModel.keys())[0]
        prior_coefs = return_trainSuccess_dict_newModel[simplifiedCommName]['coefs_sklearn']
        prior_lamb = prior_coefs[0] # for one side training
        prior_beta = prior_coefs[1] # for one side training
        prior_nus = return_trainSuccess_dict_newModel[simplifiedCommName]['nus_sklearn']
        prior_tau = 1
    except:
        print(f"No new model prior training results for {commName}")
        return

    # get learned coefs by training CVP with SKLEARN
    try:
        semiSynthetic_coefs_CVP_sklearn = semiSynthetic_return_trainSuccess_dict_CVP[commName]['coefs_sklearn']
        semiSynthetic_lamb_CVP_sklearn = semiSynthetic_coefs_CVP_sklearn[0] # for one side training
        semiSynthetic_beta_CVP_sklearn = None 
        semiSynthetic_qs_CVP_sklearn = semiSynthetic_return_trainSuccess_dict_CVP[commName]['qs_sklearn']
        semiSynthetic_nus_CVP_sklearn = semiSynthetic_return_trainSuccess_dict_CVP[commName]['nus_sklearn']
        semiSynthetic_CVscoresList_CVP = semiSynthetic_return_trainSuccess_dict_CVP[commName]['CV_scores'] # cross validation scores, a list of tup (KL, accuarcy), list length = length of try_reg_strengthList
        semiSynthetic_CVscores_CVP = semiSynthetic_CVscoresList_CVP[try_reg_strengthList.index(reg_alpha2)]
    except:
        print(f"No semiSynthetic CVP voting stage training with SKLEARN results for {commName}")
        return
    
    # get learned coefs by training CVP with torchLBFGS
    try:
        semiSynthetic_coefs_CVP_torchLBFGS = semiSynthetic_return_trainSuccess_dict_CVP[commName]['coefs_lbfgs']
        if semiSynthetic_coefs_CVP_torchLBFGS == None:
            semiSynthetic_lamb_CVP_torchLBFGS = None
        else:
            semiSynthetic_lamb_CVP_torchLBFGS = semiSynthetic_coefs_CVP_torchLBFGS[0] # for one side training
        semiSynthetic_beta_CVP_torchLBFGS = None
        semiSynthetic_qs_CVP_torchLBFGS = semiSynthetic_return_trainSuccess_dict_CVP[commName]['qs_lbfgs']
        semiSynthetic_nus_CVP_torchLBFGS = semiSynthetic_return_trainSuccess_dict_CVP[commName]['nus_lbfgs']
    except:
        print(f"No semiSynthetic CVP voting stage training with torch SGD results for {commName}")
        semiSynthetic_lamb_CVP_torchLBFGS = None
        semiSynthetic_beta_CVP_torchLBFGS = None
        semiSynthetic_qs_CVP_torchLBFGS = None
        semiSynthetic_nus_CVP_torchLBFGS = None

    # get learned coefs by training newModel with SKLEARN
    try:
        semiSynthetic_coefs_newModel_sklearn = semiSynthetic_return_trainSuccess_dict_newModel[commName]['coefs_sklearn']
        semiSynthetic_lamb_newModel_sklearn = semiSynthetic_coefs_newModel_sklearn[0] # for one side training
        semiSynthetic_beta_newModel_sklearn = semiSynthetic_coefs_newModel_sklearn[1] # for one side training
        semiSynthetic_qs_newModel_sklearn = semiSynthetic_return_trainSuccess_dict_newModel[commName]['qs_sklearn']
        semiSynthetic_nus_newModel_sklearn = semiSynthetic_return_trainSuccess_dict_newModel[commName]['nus_sklearn']
        semiSynthetic_tau_newModel_sklearn = 1
        semiSynthetic_CVscoresList_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['CV_scores'] # cross validation scores, a list of tup (KL, accuarcy), list length = length of try_reg_strengthList
        semiSynthetic_CVscores_newModel = semiSynthetic_CVscoresList_newModel[try_reg_strengthList.index(reg_alpha2)]
    except:
        print(f"No semiSynthetic new model training results for {commName}")
        return
    
    # get learned coefs by training newModel with torchLBFGS
    try:
        semiSynthetic_coefs_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['coefs_lbfgs']
        if semiSynthetic_coefs_newModel == None:
            semiSynthetic_lamb_newModel = None
            semiSynthetic_beta_newModel = None
        else:
            semiSynthetic_lamb_newModel = semiSynthetic_coefs_newModel[0] # for one side training
            semiSynthetic_beta_newModel = semiSynthetic_coefs_newModel[1] # for one side training
        semiSynthetic_qs_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['qs_lbfgs']
        semiSynthetic_nus_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['nus_lbfgs']
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
        semiSynthetic_CVscoresList_newModelInteraction = semiSynthetic_return_trainSuccess_dict_newModelInteraction[commName]['CV_scores'] # cross validation scores, a list of tup (KL, accuarcy), list length = length of try_reg_strengthList
        semiSynthetic_CVscores_newModelInteraction = semiSynthetic_CVscoresList_newModelInteraction[try_reg_strengthList.index(reg_alpha2)]
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
    
    
    # get learned tau by CVP
    if roundIndex in [20]:
        result_directory = os.path.join(commDir, r'result_folder')
        with open(result_directory+'/'+ f'semiSynthetic_CVP{variation}_newModelGenerated_round{roundIndex}_regAlpha({reg_alpha1})_selectionPhaseTrainingResults.dict', 'rb')  as inputFile:
            CVP_selectionPhaseResults= pickle.load( inputFile)
            learned_tau, tau_record, ll_record, convergeFlag, convergeIter = CVP_selectionPhaseResults
        
        semiSynthetic_tau_CVP = learned_tau 
    else:
        semiSynthetic_tau_CVP = 1


    # get prior qs
    if roundIndex in [1,2,3,4]:
        with open(intermediate_directory+'/'+f'simulated_data_byNewModel{variation}_round{roundIndex}_regAlpha({reg_alpha1}).dict', 'rb') as inputFile:
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
    if roundIndex in [1,2, 3,4]:
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
    
    
    if roundIndex in [1,2, 3,4]: 
        # compare ranking order based on different qs and voteDiff

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

        topAnswersIdList = []
        topThreshold = 5

        underEstimatedAnswersIdList = []
        overEstimatedAnswersIdList = []
        changedThreshold = 3

       
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
                if i < topThreshold: # filter out the top answers
                    topAnswersIdList.append(aid)
            
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

                # get over estimated answers and under estimated answers
                change = involved_aid2rankBasedOnPriorQ[aid] - involved_aid2rankBasedOnVoteDiff[aid] # pos: over estimated, neg: under estimated
                if change >= changedThreshold:
                    overEstimatedAnswersIdList.append(aid)
                elif change <= -changedThreshold:
                    underEstimatedAnswersIdList.append(aid)

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
            
            # get ranks based on newModel_q lbfgs
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
        with open(intermediate_directory+f"/semiSynthetic19_newModelGenerated{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2})_outputs.dict", 'wb') as outputFile:
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
        combineAll_topAnswers = []
        combineAll_overEstimatedAnswers = []
        combineAll_underEstimatedAnswers = []

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
            if aid in topAnswersIdList:
                combineAll_topAnswers.append(tup)
            if aid in overEstimatedAnswersIdList:
                combineAll_overEstimatedAnswers.append(tup)
            if aid in underEstimatedAnswersIdList:
                combineAll_underEstimatedAnswers.append(tup)
        
        # # compare prior q and learned q ranks (question-level aggregate)
        # convert involved answer2parentQ into qid2aidList
        qid2aidList = defaultdict()
        for aid, qid in answer2parentQ.items():
            if qid in qid2aidList.keys():
                qid2aidList[qid].append(aid)
            else:
                qid2aidList[qid] = [aid]
        # compute question-level kendalltau distance of ranks
        kendalltauList_voteDiff = []
        kendalltauList_CVP = []
        kendalltauList_newModel = []
        kendalltauList_newModelInteraction = []
        for qid, aidList in qid2aidList.items():
            if len(aidList) <=1:
                continue
            priorQualityRanks = [aid2Prior_q_rank[aid] for aid in aidList]
            voteDiffRanks = [aid2VoteDiff_rank[aid] for aid in aidList]
            CVPqualityRanks = [aid2CVP_sklearn_q_rank[aid] for aid in aidList]
            newModelqualityRanks = [aid2newModel_sklearn_q_rank[aid] for aid in aidList]
            newModelInteractionqualityRanks = [aid2newModelInteraction_sklearn_q_rank[aid] for aid in aidList]
            
            kendalltauList_voteDiff.append(stats.kendalltau(priorQualityRanks, voteDiffRanks).statistic)
            kendalltauList_CVP.append(stats.kendalltau(priorQualityRanks, CVPqualityRanks).statistic)
            kendalltauList_newModel.append(stats.kendalltau(priorQualityRanks, newModelqualityRanks).statistic)
            kendalltauList_newModelInteraction.append(stats.kendalltau(priorQualityRanks, newModelInteractionqualityRanks).statistic)

        kendalltau_voteDiff = mean(kendalltauList_voteDiff)
        kendalltau_CVP = mean(kendalltauList_CVP)
        kendalltau_newModel = mean(kendalltauList_newModel)
        kendalltau_newModelInteraction = mean(kendalltauList_newModelInteraction)

        # group
        # group points by intervals based on voteDiff zscores
        # group_combineAll_basedOnPriorQ = myGrouping(combineAll, sortingColumn=0)
        # group_combineAll_basedOnPriorQ = myEvenGrouping(combineAll, sortingColumn=0)
        # group_combineAll_basedOnVoteDiff = myGrouping(combineAll, sortingColumn=1)
        # group_combineAll_basedOnCVPsklearnQ = myGrouping(combineAll, sortingColumn=2)
        # group_combineAll_basedOnCVPtorchQ = myGrouping(combineAll, sortingColumn=3)
        # group_combineAll_basedOnNewModelQ = myGrouping(combineAll, sortingColumn=4)

        # plotFileName = f"semiSynthetic23{variation}_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2}).pdf" # group by interval , 1layer grouping on x_axis
        # plotFileName = f"semiSynthetic6{variation}_round{roundIndex}_groupByFixedGroupSize_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2}).png" # group by fixed group size , 1layer grouping on x_axis
        # z0,z1,z3,z4 = myPlot(commName, group_combineAll_basedOnPriorQ, combineAll, plotFileName)
        plotFileName = f"semiSynthetic23_groupByInterval.pdf"
        tup = myPlot(commName, combineAll, plotFileName)
        # # z0,z1,z3,z4, Huber_robust_param_voteDiff,Huber_robust_param_CVP,Huber_robust_param_newModel,Huber_robust_param_newModelInteraction,TheilSen_robust_param_voteDiff, TheilSen_robust_param_CVP, TheilSen_robust_param_newModel, TheilSen_robust_param_newModelInteraction,RANSAC_robust_param_voteDiff, RANSAC_robust_param_CVP, RANSAC_robust_param_newModel, RANSAC_robust_param_newModelInteraction,Huber_robust_resid_voteDiff,Huber_robust_resid_CVP,Huber_robust_resid_newModel,Huber_robust_resid_newModelInteraction,TheilSen_robust_resid_voteDiff, TheilSen_robust_resid_CVP, TheilSen_robust_resid_newModel, TheilSen_robust_resid_newModelInteraction,RANSAC_robust_resid_voteDiff, RANSAC_robust_resid_CVP, RANSAC_robust_resid_newModel, RANSAC_robust_resid_newModelInteraction = tup
        answerCount, sumSquredResiduls_voteDiff, sumSquredResiduls_CVP, sumSquredResiduls_newModel, betterThanVoteDiff, betterThanCVP, betterThanBoth, r2_voteDiff, r2_CVP, r2_newModel, pearson_stat_voteDiff, pearson_p_voteDiff, pearson_stat_CVP, pearson_p_CVP, pearson_stat_newModel, pearson_p_newModel = tup


        # with open(rootDir+'/'+f'allComm_semiSynthetic_newModel{variation}_round{roundIndex}_resultComparison_forPaper.csv', 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     writer.writerow( 
        #                     [commName, reg_alpha1,reg_alpha2, answerCount, sumSquredResiduls_voteDiff, sumSquredResiduls_CVP, sumSquredResiduls_newModel, betterThanVoteDiff, betterThanCVP, betterThanBoth, r2_voteDiff, r2_CVP, r2_newModel, pearson_stat_voteDiff, pearson_p_voteDiff, pearson_stat_CVP, pearson_p_CVP, pearson_stat_newModel, pearson_p_newModel])
        

        # # plot for top answers
        # myPlot(commName, combineAll_topAnswers, f"semiSynthetic6{variation}_round{roundIndex}_topAnswers_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2}).png")
        # # plot for changedMost answers
        # myPlot(commName, combineAll_overEstimatedAnswers, f"semiSynthetic6{variation}_round{roundIndex}_overEstimatedAnswers_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2}).png")
        # myPlot(commName, combineAll_underEstimatedAnswers, f"semiSynthetic6{variation}_round{roundIndex}_underEstimatedAnswers_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2}).png")

        # # question-level plot
        # # questionLevelPlotFileNameForRankZscoreComparison = f"semiSynthetic6{variation}_questionLevelRankZscoreComparison_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2}).png"
        # avgSlopesForRankZscoreComparison = questionLevelPlot(commName, combineAll,   'rankZscore')
        # # questionLevelPlotFileNameForRankComparison = f"semiSynthetic6{variation}_questionLevelRankComparison_round{roundIndex}_PriorRegAlpha({reg_alpha1})_regAlpha({reg_alpha2}).png"
        # avgSlopesForRankComparison = questionLevelPlot(commName, combineAll,   'rank')

        #####################################################################################################################

    
    # # report in csv
    # if roundIndex in [1,2,3,4]:
    #     with open(rootDir+'/'+f'allComm_semiSynthetic_newModel{variation}_round{roundIndex}_resultComparison.csv', 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #         writer.writerow( [commName,generated_event_count, reg_alpha1,reg_alpha2,
    #                           len(answer2parentQ), len(overEstimatedAnswersIdList),len(underEstimatedAnswersIdList),len(topAnswersIdList),
    #                         prior_lamb, semiSynthetic_lamb_CVP_sklearn, semiSynthetic_lamb_CVP_torchLBFGS, semiSynthetic_lamb_newModel_sklearn, semiSynthetic_lamb_newModel, semiSynthetic_lamb_newModelInteraction_sklearn, semiSynthetic_lamb_newModelInteraction,
    #                         prior_beta, semiSynthetic_beta_CVP_sklearn, semiSynthetic_beta_CVP_torchLBFGS, semiSynthetic_beta_newModel_sklearn,semiSynthetic_beta_newModel, semiSynthetic_beta_newModelInteraction_sklearn,semiSynthetic_beta_newModelInteraction,
    #                         mean(aid2prior_q.values()), mean(aid2CVP_sklearn_q.values()), mean(aid2CVP_torchLBFGS_q.values()),mean(aid2newModel_sklearn_q.values()), mean(aid2newModel_q.values()),mean(aid2newModelInteraction_sklearn_q.values()), mean(aid2newModelInteraction_q.values()), 
    #                         mean(qid2prior_nu.values()), mean(qid2CVP_sklearn_nu.values()), mean(qid2CVP_torchLBFGS_nu.values()),mean(qid2newModel_sklearn_nu.values()),mean(qid2newModel_nu.values()),mean(qid2newModelInteraction_sklearn_nu.values()),mean(qid2newModelInteraction_nu.values()), 
    #                         round(z0[0][0],4),round(z1[0][0],4),round(z3[0][0],4),round(z4[0][0],4),
    #                         round(z0[1][0],4),round(z1[1][0],4),round(z3[1][0],4),round(z4[1][0],4),
    #                         # round(KL_voteDiffToPriorQ_rankZscore,4),round(KL_CVPlearnedQToPriorQ_rankZscore,4), round(KL_newModellearnedQToPriorQ_rankZscore,4), round(KL_newModelInteractionlearnedQToPriorQ_rankZscore,4),
    #                         round(kendalltau_voteDiff,4),round(kendalltau_CVP,4), round(kendalltau_newModel,4), round(kendalltau_newModelInteraction,4),
    #                         avgSlopesForRankZscoreComparison[0],avgSlopesForRankZscoreComparison[1],avgSlopesForRankZscoreComparison[2],avgSlopesForRankZscoreComparison[3],
    #                         avgSlopesForRankZscoreComparison[4],avgSlopesForRankZscoreComparison[5],avgSlopesForRankZscoreComparison[6],avgSlopesForRankZscoreComparison[7],
    #                         avgSlopesForRankZscoreComparison[8],avgSlopesForRankZscoreComparison[9],avgSlopesForRankZscoreComparison[10],avgSlopesForRankZscoreComparison[11],
    #                         avgSlopesForRankComparison[0],avgSlopesForRankComparison[1],avgSlopesForRankComparison[2],avgSlopesForRankComparison[3],
    #                         avgSlopesForRankComparison[4],avgSlopesForRankComparison[5],avgSlopesForRankComparison[6],avgSlopesForRankComparison[7],
    #                         avgSlopesForRankComparison[8],avgSlopesForRankComparison[9],avgSlopesForRankComparison[10],avgSlopesForRankComparison[11],
    #                         # Huber_robust_param_voteDiff,Huber_robust_param_CVP,Huber_robust_param_newModel,Huber_robust_param_newModelInteraction,
    #                         # TheilSen_robust_param_voteDiff, TheilSen_robust_param_CVP, TheilSen_robust_param_newModel, TheilSen_robust_param_newModelInteraction,
    #                         # RANSAC_robust_param_voteDiff, RANSAC_robust_param_CVP, RANSAC_robust_param_newModel, RANSAC_robust_param_newModelInteraction,
    #                         # Huber_robust_resid_voteDiff,Huber_robust_resid_CVP,Huber_robust_resid_newModel,Huber_robust_resid_newModelInteraction,
    #                         # TheilSen_robust_resid_voteDiff, TheilSen_robust_resid_CVP, TheilSen_robust_resid_newModel, TheilSen_robust_resid_newModelInteraction,
    #                         # RANSAC_robust_resid_voteDiff,RANSAC_robust_resid_CVP, RANSAC_robust_resid_newModel, RANSAC_robust_resid_newModelInteraction,
    #                         semiSynthetic_CVscores_CVP[0], semiSynthetic_CVscores_newModel[0], semiSynthetic_CVscores_newModelInteraction[0],
    #                         semiSynthetic_CVscores_CVP[1], semiSynthetic_CVscores_newModel[1], semiSynthetic_CVscores_newModelInteraction[1]])

    


def main():
    rootDir = os.getcwd()
    t0=time.time()
    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # roundIndex = 1 ## multiple question multiple answer, original total event count, fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    roundIndex = 2 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # roundIndex = 3 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and DOUBLED beta (for different regularization strength) selected_reg_strengthList of each comm
    # roundIndex = 4 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and TRIPLED beta (for different regularization strength) selected_reg_strengthList of each comm


    commName2selected_reg_strengthList = {
                                          'stackoverflow':[1000],
                                          'politics.stackexchange':[200, 300, 400, 500, 600, 700, 800,900,1000],
                                          'cstheory.stackexchange':[700, 800, 900, 1000],
                                        #   'lifehacks.stackexchange':[300, 400, 500, 600, 700, 800, 900],
                                        #   'meta.askubuntu':[20, 30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                        #   'unix.meta.stackexchange':[30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                        #   '3dprinting.stackexchange':[20, 30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700],
                                        #   'latin.stackexchange':[40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                        'math.meta.stackexchange':[30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                          'mathoverflow.net':[500,600],
                                        #   'mathematica.stackexchange':[80,90,100],
                                          'askubuntu':[300,400,500,600],
                                          'philosophy.stackexchange':[60, 70,80,90,100, 200, 300, 400, 500, 600],
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
    
    # for roundIndex 1
    selected_reg_strengthPairs = []
    for commName, selected_reg_strengthList in commName2selected_reg_strengthList.items():
        for reg_1 in selected_reg_strengthList:
            for reg_2 in try_reg_strengthList:
                selected_reg_strengthPairs.append((commName, reg_1, reg_2)) 

    # for paper
    selected_reg_strengthPairs = [('stackoverflow',1000,0.1),
                                 ('politics.stackexchange',500,0.2),
                                 ('cstheory.stackexchange',1000,0.3),
                                 ('math.meta.stackexchange',500,0.3),
                                 ('mathoverflow.net',500,600),
                                 ('askubuntu',600,600),
                                 ('philosophy.stackexchange',400,0.2),
                                 ('codegolf.meta.stackexchange',800,0.1)]


    # # save csv for NEW MODEL generating semi-synthetic dataset statistics
    # with open(rootDir+'/'+f'allComm_semiSynthetic_newModel{variation}_round{roundIndex}_resultComparison_forPaper.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow( 
    #                     ["commName", "reg_alpha1","reg_alpha2", "answer count","sumSquaredResidual_voteDiff","sumSquaredResidual_CVP","sumSquaredResidual_newModel", "better than voteDiff count","better than CVP count","better than both count","r2_voteDiff", "r2_CVP", "r2_newModel", "pearson_stat_voteDiff", "pearson_p_voteDiff", "pearson_stat_CVP", "pearson_p_CVP", "pearson_stat_newModel", "pearson_p_newModel"])
                         
    """
    # save csv for NEW MODEL generating semi-synthetic dataset statistics
    with open(rootDir+'/'+f'allComm_semiSynthetic_newModel{variation}_round{roundIndex}_resultComparison.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( 
                        ["commName","total generated event Count", "reg_alpha1",'reg_alpha2',
                              "total Answer Count", "over estimated answer count", "under esitmated answer count", "top ranked answer count",
                            # "prior_tau","learned_tau_CVP_selectionPhase","learned_tau_newModel_torchLBFGS",
                            "prior_lamb", "learned_lamb_CVP_sklearn", "learned_lamb_CVP_torchLBFGS","learned_lamb_newModel_sklearn","learned_lamb_newModel_torchLBFGS","learned_lamb_newModelInteraction_sklearn","learned_lamb_newModelInteraction_torchLBFGS",
                            "prior_beta", "learned_beta_CVP_sklearn", "learned_beta_CVP_torchLBFGS","learned_beta_newModel_sklearn","learned_beta_newModel_torchLBFGS","learned_beta_newModelInteraction_sklearn","learned_beta_newModelInteraction_torchLBFGS",
                            "prior_q_mean", "learned_q_CVP_sklearn_mean", "learned_q_CVP_torchLBFGS_mean","learned_q_newModel_sklearn_mean","learned_q_newModel_torchLBFGS_mean","learned_q_newModelInteraction_sklearn_mean","learned_q_newModelInteraction_torchLBFGS_mean",
                            "prior_nu_mean", "learned_nu_CVP_sklearn_mean", "learned_nu_CVP_torchLBFGS_mean","learned_nu_newModel_sklearn_mean","learned_nu_newModel_torchLBFGS_mean","learned_nu_newModelInteraction_sklearn_mean","learned_nu_newModelInteraction_torchLBFGS_mean", 
                            "slope_voteDiff", "slope_CVPsklearnQuality","slope_newModelsklearn_Quality","slope_newModelInteractionsklearn_Quality",
                            "residual_voteDiff", "residual_CVPsklearnQuality","residual_newModelsklearn_Quality","residual_newModelInteractionsklearn_Quality",
                            # "KL_voteDiffRankZScoreToPrior","KL_CVPqualityRankZScoreToPrior","KL_newModelqualityRankZScoreToPrior","KL_newModelInteractionqualityRankZScoreToPrior",
                            "avgKT_voteDiffRankToPrior","avgKT_CVPqualityRankToPrior","avgKT_newModelqualityRankToPrior","avgKT_newModelInteractionqualityRankToPrior",
                            "voteDiff_zscore_AvgSlope","CVP_zscore_AvgSlope","newModel_zscore_AvgSlope","newModelInteraction_zscore_AvgSlope",
                            "voteDiff_zscore_AvgResidual","CVP_zscore_AvgResidual","newModel_zscore_AvgResidual","newModelInteraction_zscore_AvgResidual",
                             "voteDiff_zscore_PositiveSlopePercentage","CVP_zscore_PositiveSlopePercentage","newModel_zscore_PositiveSlopePercentage","newModelInteraction_zscore_PositiveSlopePercentage",
                             "voteDiff_rank_AvgSlope","CVP_rank_AvgSlope","newModel_rank_AvgSlope","newModelInteraction_rank_AvgSlope",
                             "voteDiff_rank_AvgResidual","CVP_rank_AvgResidual","newModel_rank_AvgResidual","newModelInteraction_rank_AvgResidual",
                             "voteDiff_rank_PositiveSlopePercentage","CVP_rank_PositiveSlopePercentage","newModel_rank_PositiveSlopePercentage","newModelInteraction_rank_PositiveSlopePercentage",
                            #  "HuberT_slope_voteDiff", "HuberT_slope_CVPsklearnQuality","HuberT_slope_newModelsklearn_Quality","HuberT_slope_newModelInteractionsklearn_Quality",
                            #  "TheilSen_slope_voteDiff", "TheilSen_slope_CVPsklearnQuality","TheilSen_slope_newModelsklearn_Quality","TheilSen_slope_newModelInteractionsklearn_Quality",
                            #  "RANSAC_slope_voteDiff", "RANSAC_slope_CVPsklearnQuality","RANSAC_slope_newModelsklearn_Quality","RANSAC_slope_newModelInteractionsklearn_Quality",
                            #  "HuberT_residual_voteDiff", "HuberT_residual_CVPsklearnQuality","HuberT_residual_newModelsklearn_Quality","HuberT_residual_newModelInteractionsklearn_Quality",
                            #  "TheilSen_residual_voteDiff", "TheilSen_residual_CVPsklearnQuality","TheilSen_residual_newModelsklearn_Quality","TheilSen_residual_newModelInteractionsklearn_Quality",
                            #  "RANSAC_residual_voteDiff", "RANSAC_residual_CVPsklearnQuality","RANSAC_residual_newModelsklearn_Quality","RANSAC_residual_newModelInteractionsklearn_Quality",
                             "crossValidation_KL_CVP","crossValidation_KL_newModel","crossValidation_KL_newModelInteraction",
                             "crossValidation_accuracy_CVP","crossValidation_accuracy_newModel","crossValidation_accuracy_newModelInteraction"])
    """
    
    # prepare args
    argsList = []
    for tup in selected_reg_strengthPairs:
        commName, reg_alpha1, reg_alpha2 = tup
        for comm in commDir_sizes_sortedlist:
            if comm[0] == commName:
                commDir = comm[1]
                break
        argsList.append((commName, commDir, rootDir, roundIndex, variation, reg_alpha1, reg_alpha2, try_reg_strengthList, sampled_comms))


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
    

    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('semiSynthetic19_compareResults Done completely.    Elapsed: {:}.\n'.format(elapsed))
    
      
if __name__ == "__main__":
  
    # calling main function
    main()


##########################################################################