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


def myFun(commName, commDir, rootDir, roundIndex, variation, reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP):
   
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
                with open(intermediate_directory+'/'+f"temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha_NewModelInteraction})_SGD_return.dict", 'rb') as inputFile:
                    return_trainSuccess_dict_newModelInteraction = pickle.load( inputFile)
            else:
                return_trainSuccess_dict_newModelInteraction = None
            if reg_alpha_NewModel != None:
                with open(intermediate_directory+'/'+f"temperalOrderTraining12_newModel_fixedTau_regAlpha({reg_alpha_NewModel})_SGD_return.dict", 'rb') as inputFile:
                    return_trainSuccess_dict_newModel = pickle.load( inputFile)
            else:
                return_trainSuccess_dict_newModel = None
            if reg_alpha_CVP != None:
                with open(intermediate_directory+'/'+f"temperalOrderTraining11_CVP_regAlpha({reg_alpha_CVP})_SGD_return.dict", 'rb') as inputFile:
                    return_trainSuccess_dict_CVP = pickle.load( inputFile)
            else:
                return_trainSuccess_dict_CVP = None
    
    ############################################################################################
        
    print(f"loading total_answersWithVotes_indice... for {commName}")
    if commName != 'stackoverflow':
        try:
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
        with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
            Questions = pickle.load( inputFile)
    else: # using subcomm to represent stackoverflow
        subComm_QuestionsWithEventList_directory = subCommDir+'/'+f'QuestionsWithEventList_tag_reactjs.dict'
        with open(subComm_QuestionsWithEventList_directory, 'rb') as inputFile:
            Questions = pickle.load( inputFile)

    # load userIds for comment filtering
    try:
        with open(intermediate_directory+'/'+'postId2UserId.dict', 'rb') as inputFile:
            postId2UserId = pickle.load( inputFile)
    except:
        print(f"no postId2UserId for {commName}")
        return
    try:
        with open(intermediate_directory+'/'+'postId2commentIdAndCreationTimeAndUserId_tillCurChunk.dict', 'rb') as inputFile:
            post2commentIdAndCreationTimeAndUserId = pickle.load( inputFile)
    except:
        print(f"no post2commentIdAndCreationTimeAndUserId for {commName}")
        return
    ############################################################################################
    
    # get max answer count of one question
    maxAnswerCount = 0
    for qid, content in Questions.items():
        answerCount = len(content['filtered_answerList'])
        if answerCount > maxAnswerCount:
            maxAnswerCount = answerCount
    
    if len(return_trainSuccess_dict_newModel)==1:
        simplifiedCommName = list(return_trainSuccess_dict_newModel.keys())[0]

    # get learned qs (without bias)
    try:
        learned_qs_interaction = return_trainSuccess_dict_newModelInteraction[simplifiedCommName]['qs']  # after retrain with full data
    except:
        learned_qs_interaction = return_trainSuccess_dict_newModelInteraction[commName]['qs']  # after retrain with full data
        
    try:
        learned_qs = return_trainSuccess_dict_newModel[simplifiedCommName]['qs']  # after retrain with full data
    except:
        learned_qs = return_trainSuccess_dict_newModel[commName]['qs']  # after retrain with full data

    try:
        learned_qs_CVP = return_trainSuccess_dict_CVP[simplifiedCommName]['qs']  # after retrain with full data
    except:
        learned_qs_CVP = return_trainSuccess_dict_CVP[commName]['qs']  # after retrain with full data
    
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

    if commName == 'stackoverflow':
        with open(commDir+ f'/promptJson_folder/prompt_template_{promptType}_reactjs.json') as json_file:
            prompts_Dict = json.load(json_file)
        with open(commDir+ f'/GPTresponse_folder/qid2resDict_prompt{promptType}_{my_model}_reactjs.json') as json_file:
            qid2resDict = json.load(json_file)
    else:
        with open(commDir+ f'/promptJson_folder/prompt_template_{promptType}.json') as json_file:
            prompts_Dict = json.load(json_file)

        with open(commDir+ f'/GPTresponse_folder/qid2resDict_prompt{promptType}_{my_model}.json') as json_file:
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
    aid2VoteDiff_rankZscore = defaultdict()
    aid2CVP_q_rankZscore  = defaultdict()
    aid2newModel_q_rankZscore  = defaultdict()
    aid2newModelInteraction_q_rankZscore  = defaultdict()


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
        involved_aid2ranksBasedOnVoteDiff = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2voteDiff.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) : # sort by the vote score and then by the answer index
            aid = kv[0]
            involved_aid2ranksBasedOnVoteDiff[aid] = i+1 # rank
        
        zScores = get_zScore(involved_aid2ranksBasedOnVoteDiff.values())
        for i,aid in enumerate(involved_aid2ranksBasedOnVoteDiff.keys()):
            aid2VoteDiff_rankZscore[aid] = zScores[i]
        
        # get ranks based on sentiment GPT
        involved_aid2sentimentScore = {aid : aid2sentimentScore[aid] for aid in involved_aid2ranksBasedOnVoteDiff.keys()}
        involved_aid2ranksBasedOnSentimentScore = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2sentimentScore.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
            aid = kv[0]
            involved_aid2ranksBasedOnSentimentScore[aid] = i+1 # rank
        
        zScores = get_zScore(involved_aid2ranksBasedOnSentimentScore.values())
        for i,aid in enumerate(involved_aid2ranksBasedOnSentimentScore.keys()):
            aid2sentiment_rankZscore[aid] = zScores[i]
        
        # get ranks based on CVP_q
        involved_aid2CVP_q = {aid : aid2quality_CVP[aid] for aid in involved_aid2ranksBasedOnVoteDiff.keys()}
        involved_aid2ranksBasedOnCVPQ = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2CVP_q.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
            aid = kv[0]
            involved_aid2ranksBasedOnCVPQ[aid] = i+1 # rank
        
        zScores = get_zScore(involved_aid2ranksBasedOnCVPQ.values())
        for i,aid in enumerate(involved_aid2ranksBasedOnCVPQ.keys()):
            aid2CVP_q_rankZscore[aid] = zScores[i]

        
        # get ranks based on newModel_q_sgd
        involved_aid2newModel_q = {aid : aid2quality_newModel[aid] for aid in involved_aid2ranksBasedOnVoteDiff.keys()}
        involved_aid2ranksBasedOnNewModelQ = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2newModel_q.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
            aid = kv[0]
            involved_aid2ranksBasedOnNewModelQ[aid] = i+1 # rank
        
        zScores = get_zScore(involved_aid2ranksBasedOnNewModelQ.values())
        for i,aid in enumerate(involved_aid2ranksBasedOnNewModelQ.keys()):
            aid2newModel_q_rankZscore[aid] = zScores[i]
        

        # get ranks based on newModelInteraction_q
        involved_aid2newModelInteraction_q = {aid : aid2quality_newModelInteration[aid] for aid in involved_aid2ranksBasedOnVoteDiff.keys()}
        involved_aid2ranksBasedOnNewModelInteractionQ = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2newModelInteraction_q.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)) :
            aid = kv[0]
            involved_aid2ranksBasedOnNewModelInteractionQ[aid] = i+1 # rank
        
        zScores = get_zScore(involved_aid2ranksBasedOnNewModelInteractionQ.values())
        for i,aid in enumerate(involved_aid2ranksBasedOnNewModelInteractionQ.keys()):
            aid2newModelInteraction_q_rankZscore[aid] = zScores[i]

    Questions.clear()

    combineAll = []
    for aid in aid2sentiment_rankZscore.keys():
        try:
            tup = (aid2sentiment_rankZscore[aid],aid2VoteDiff_rankZscore[aid],aid2CVP_q_rankZscore[aid], aid2newModel_q_rankZscore[aid],aid2newModelInteraction_q_rankZscore[aid])
            combineAll.append(tup)
        except Exception as e:
            print(e)


    # plot with all data point
    plt.cla()
    fig, axs = plt.subplots(2,2)
    fig.tight_layout(pad=3.0)


    group_combineAll_basedOnVoteDiff = myGrouping(combineAll, sortingColumn=1)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiff ]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnVoteDiff ]
    axs[0,0].scatter(group_voteDiff_rankZscores, group_priorQ_rankZscores,s = 2)

    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_voteDiff_rankZscores = [t[1] for t in combineAll]
    # axs[0, 0].scatter(all_voteDiff_rankZscores, all_priorQ_rankZscores,s = 2)
    
    axs[0,0].set_xlabel('vote diff rankZscore', fontsize = 8)
    axs[0,0].set_ylabel('sentiment rankZscore', fontsize = 8)
    axs[0,0].set_xlim(-2,2)
    axs[0,0].set_ylim(-2,2)

    # OLS fit
    z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"r-", linewidth=1, label=f'slope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}')
    axs[0,0].legend(loc="best", fontsize = 6)

    group_combineAll_basedOnCVP = myGrouping(combineAll, sortingColumn=2)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVP]
    group_CVPQ_rankZscores = [t[2] for t in group_combineAll_basedOnCVP]
    axs[0, 1].scatter(group_CVPQ_rankZscores, group_priorQ_rankZscores,s = 2)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_CVPQ_rankZscores = [t[2] for t in combineAll]
    # axs[0, 1].scatter(all_CVPQ_rankZscores, all_priorQ_rankZscores,s = 2)
    
    axs[0, 1].set_xlabel('CVP sgd q rankZscore', fontsize = 8)
    axs[0, 1].set_ylabel('sentiment rankZscore', fontsize = 8)
    axs[0, 1].set_xlim(-2,2)
    axs[0, 1].set_ylim(-2,2)

    # OLS fit
    z1,p1 = curveFitOLS(all_CVPQ_rankZscores, all_priorQ_rankZscores)
    axs[0,1].plot(all_CVPQ_rankZscores,p1(all_CVPQ_rankZscores),"r-", linewidth=1, label=f'slope={round(z1[0][0],4)}\nresidual={round(z1[1][0],4)}')
    axs[0,1].legend(loc="best", fontsize = 6)

    group_combineAll_basedOnNewModel = myGrouping(combineAll, sortingColumn=2)
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnNewModel]
    group_newModelQ_rankZscores = [t[3] for t in group_combineAll_basedOnNewModel]
    axs[1,0].scatter(group_newModelQ_rankZscores, group_priorQ_rankZscores,s = 2)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_newModelQ_rankZscores = [t[3] for t in combineAll]
    # axs[1, 0].scatter(all_newModelsklearnQ_rankZscores, all_priorQ_rankZscores,s = 2)
    
    axs[1,0].set_xlabel('new model sgd q rankZscore', fontsize = 8)
    axs[1,0].set_ylabel('sentiment rankZscore', fontsize = 8)
    axs[1,0].set_xlim(-2,2)
    axs[1,0].set_ylim(-2,2)

    # OLS fit
    z3,p3 = curveFitOLS(all_newModelQ_rankZscores, all_priorQ_rankZscores)
    axs[1,0].plot(all_newModelQ_rankZscores,p3(all_newModelQ_rankZscores),"r-", linewidth=1, label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}')
    axs[1,0].legend(loc="best", fontsize = 6)
    
    group_combineAll_basedOnNewModelInteraction = myGrouping(combineAll, sortingColumn=4)
    group_priorQ_rankZscores = [t[0] for t in  group_combineAll_basedOnNewModelInteraction]
    group_newModelInteractionQ_rankZscores = [t[4] for t in  group_combineAll_basedOnNewModelInteraction]
    axs[1, 1].scatter(group_newModelInteractionQ_rankZscores, group_priorQ_rankZscores,s = 2)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_newModelInteractionQ_rankZscores = [t[4] for t in combineAll]
    # axs[1, 1].scatter(all_newModelInteractionQ_rankZscores, all_priorQ_rankZscores,s = 2)
    
    axs[1, 1].set_xlabel('newModelInteraction sgd q rankZscore', fontsize = 8)
    axs[1, 1].set_ylabel('sentiment rankZscore', fontsize = 8)

    # OLS fit
    z4,p4 = curveFitOLS(all_newModelInteractionQ_rankZscores, all_priorQ_rankZscores)
    axs[1,1].plot(all_newModelInteractionQ_rankZscores,p4(all_newModelInteractionQ_rankZscores),"r-", linewidth=1, label=f'slope={round(z4[0][0],4)}\nresidual={round(z3[1][0],4)}')
    axs[1,1].legend(loc="best", fontsize = 6)
    axs[1, 1].set_xlim(-2,2)
    axs[1, 1].set_ylim(-2,2)

    fig.suptitle(f"{commName.replace('.stackexchange','')}\nnewModelInteractionReg({reg_alpha_NewModelInteraction}) newModelReg({reg_alpha_NewModel}) CVPReg({reg_alpha_CVP})", fontsize = 6)
    
    # # savePlot(fig, f"temperalOrderTraining15_(ignoreByGPT)_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).png")
    # # savePlot(fig, f"temperalOrderTraining15_(ignoreStartedWithAt)_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).png")
    # savePlot(fig, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).png")
    savePlot(fig, f"temperalOrderTraining15_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})_SGD.png")
    # print(f"saved plot for {commName}, round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP})")
    
    
   
def main():

    t0=time.time()
    rootDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

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
                                        # # 'math.meta.stackexchange':(None,0.1,None),
                                        # 'politics.stackexchange':(None,0.1,None),
                                        # 'stackoverflow':(None,0.1,None),
                                            'mathoverflow.net':(400,700,800),
                                          }
    variation = '_fixedTau'
    
    
    # prepare args
    argsList = []
    for commName, tup in commName2selected_reg_strengthList.items():
        reg_alpha_newModelInteraction, reg_alpha_newModel, reg_alpha_CVP = tup
        for comm in commDir_sizes_sortedlist:
            if comm[0] == commName:
                commDir = comm[1]
                break
        argsList.append((commName, commDir, rootDir, roundIndex, variation, reg_alpha_newModelInteraction, reg_alpha_newModel, reg_alpha_CVP))


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

    # Report progress.
    elapsed = format_time(time.time() - t0)
    print('verify qualities  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
