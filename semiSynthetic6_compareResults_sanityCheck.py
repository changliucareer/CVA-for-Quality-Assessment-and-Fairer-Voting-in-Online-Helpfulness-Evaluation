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
from scipy.optimize import fsolve
from scipy.special import psi # digamma
import matplotlib.pyplot as plt
import scipy

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


##########################################################################

def myFun(commName, commDir, rootDir):

    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "semiSynthetic6_compareResults_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # prior coefs
    # load CVP one-side temperal training result
    with open(intermediate_directory+'/'+'temperalOrderTraining8_CVP_return.dict', 'rb') as inputFile:
        return_trainSuccess_dict_CVP = pickle.load( inputFile)
    print(f"return train success dict CVP loaded. length {len(return_trainSuccess_dict_CVP)}")

    # learned coefs by CVP
    with open(intermediate_directory+'/'+'semiSynthetic9_sanityCheckRound_CVP_votingStage_training_return.dict', 'rb') as inputFile:
        semiSynthetic_return_trainSuccess_dict_CVP = pickle.load( inputFile)
    print(f"semiSynthetic_return train success dict CVP loaded. ")

    # learned coefs by new model
    with open(intermediate_directory+f"/semiSynthetic10_sanityCheckRound_newModelTraining_posTau_return.dict", 'rb') as inputFile:
        semiSynthetic_return_trainSuccess_dict_newModel = pickle.load( inputFile)
    print(f"semiSynthetic_return train success dict new Model loaded. ")

    # get prior coefs
    try:
        prior_coefs = return_trainSuccess_dict_CVP[commName]['coefs_sklearn']
        prior_lamb = prior_coefs[0] # for one side training
        prior_nus = return_trainSuccess_dict_CVP[commName]['nus_sklearn']
    except:
        print(f"No CVP voting stage training results for {commName}")
        return

    # get learned coefs by training CVP with SKLEARN
    try:
        semiSynthetic_coefs_CVP_sklearn = semiSynthetic_return_trainSuccess_dict_CVP[commName]['coefs_sklearn']
        semiSynthetic_lamb_CVP_sklearn = semiSynthetic_coefs_CVP_sklearn[0] # for one side training
        semiSynthetic_nus_CVP_sklearn = semiSynthetic_return_trainSuccess_dict_CVP[commName]['nus_sklearn']
        semiSynthetic_qs_CVP_sklearn = semiSynthetic_return_trainSuccess_dict_CVP[commName]['qs_sklearn']
    except:
        print(f"No semiSynthetic CVP voting stage training with SKLEARN results for {commName}")
        return
    
    # get learned coefs by training CVP with torchSGD
    try:
        semiSynthetic_coefs_CVP_torchSGD = semiSynthetic_return_trainSuccess_dict_CVP[commName]['coefs']
        semiSynthetic_lamb_CVP_torchSGD = semiSynthetic_coefs_CVP_torchSGD[0] # for one side training
        semiSynthetic_nus_CVP_torchSGD = semiSynthetic_return_trainSuccess_dict_CVP[commName]['nus']
        semiSynthetic_qs_CVP_torchSGD = semiSynthetic_return_trainSuccess_dict_CVP[commName]['qs']
    except:
        print(f"No semiSynthetic CVP voting stage training with torch SGD results for {commName}")
        return

    # get learned coefs by training newModel with torchSGD
    try:
        semiSynthetic_coefs_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['coefs']
        semiSynthetic_lamb_newModel = semiSynthetic_coefs_newModel[0] # for one side training
        semiSynthetic_nus_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['nus']
        semiSynthetic_qs_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['qs']
        semiSynthetic_tau_newModel = semiSynthetic_return_trainSuccess_dict_newModel[commName]['tau']
    except:
        print(f"No semiSynthetic new model training results for {commName}")
        return
    
    # get prior tau
    try:
        result_directory = os.path.join(commDir, r'result_folder')
        with open(result_directory+'/'+ 'CVP1_selectionPhaseTrainingResults.dict', 'rb')  as inputFile:
            CVP_selectionPhaseResults= pickle.load( inputFile)
            learned_tau, tau_record, ll_record, convergeFlag, convergeIter = CVP_selectionPhaseResults

        prior_tau = learned_tau
    except:
        print(f"{commName} hasn't finished the CVP selectionPhase training.")
        return
    
    # get learned tau by CVP
    try:
        result_directory = os.path.join(commDir, r'result_folder')
        with open(result_directory+'/'+ 'semiSynthetic_CVP1_selectionPhaseTrainingResults.dict', 'rb')  as inputFile:
            CVP_selectionPhaseResults= pickle.load( inputFile)
            learned_tau, tau_record, ll_record, convergeFlag, convergeIter = CVP_selectionPhaseResults

        semiSynthetic_tau_CVP = learned_tau
    except:
        print(f"{commName} hasn't finished the CVP selectionPhase training.")
        return


    # get prior qs
    with open(intermediate_directory+'/'+'simulated_data_byCVP_sanityCheckRound.dict', 'rb') as inputFile:
        loadedFile = pickle.load( inputFile)
    simulatedQuestions = loadedFile[0]
    generated_neg_vote_count = loadedFile[4]
    generated_pos_vote_count = loadedFile[5]
    generated_vote_count = generated_neg_vote_count  +  generated_pos_vote_count
    generated_answer_count = loadedFile[6]
    generated_event_count = generated_vote_count + generated_answer_count
    
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
    with open(intermediate_directory+'/'+f'semiSynthetic_sanityCheck_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
        total_answersWithVotes_indice = pickle.load( inputFile)
    
    total_answersWithVotes_ids = []
    for i,(qid, ai) in enumerate(total_answersWithVotes_indice):
        answerList = simulatedQuestions[qid]['answerList']
        aid = answerList[ai][0]
        total_answersWithVotes_ids.append((qid,aid))
    
    aid2CVP_sklearn_q = defaultdict()
    aid2CVP_torchSGD_q = defaultdict()
    aid2newModel_q = defaultdict()
    for i, tup in enumerate(total_answersWithVotes_ids):
        qid, aid = tup
        q_CVP_sklearn = semiSynthetic_qs_CVP_sklearn[i]
        q_CVP_torchSGD = semiSynthetic_qs_CVP_torchSGD[i]
        q_newModel = semiSynthetic_qs_newModel[i]
        aid2CVP_sklearn_q[aid] = q_CVP_sklearn
        aid2CVP_torchSGD_q[aid] = q_CVP_torchSGD
        aid2newModel_q[aid] = q_newModel
    
    """
    prior_qs = []
    semiSynthetic_qs_CVP_sklearn = []
    semiSynthetic_qs_CVP_torchSGD = []
    semiSynthetic_qs_newModel = []
    for aid in aid2CVP_sklearn_q.keys():
        prior_qs.append(aid2prior_q[aid])
        semiSynthetic_qs_CVP_sklearn.append(aid2CVP_sklearn_q[aid])
        semiSynthetic_qs_CVP_torchSGD.append(aid2CVP_torchSGD_q[aid])
        semiSynthetic_qs_newModel.append(aid2newModel_q[aid])


    # compute Pearson's Product Moment Correlation Coefficient 
    correlation_CVP_sklearn_to_prior = pearsonr(prior_qs, semiSynthetic_qs_CVP_sklearn,alternative='two-sided').statistic
    correlation_CVP_sklearn_to_prior_pvalue = pearsonr(prior_qs, semiSynthetic_qs_CVP_sklearn,alternative='two-sided').pvalue
    correlation_CVP_torchSGD_to_prior = pearsonr(prior_qs, semiSynthetic_qs_CVP_torchSGD,alternative='two-sided').statistic
    correlation_CVP_torchSGD_to_prior_pvalue = pearsonr(prior_qs, semiSynthetic_qs_CVP_torchSGD,alternative='two-sided').pvalue
    correlation_newModel_to_prior = pearsonr(prior_qs, semiSynthetic_qs_newModel,alternative='two-sided').statistic
    correlation_newModel_to_prior_pvalue = pearsonr(prior_qs, semiSynthetic_qs_newModel,alternative='two-sided').pvalue
    """
    
    
    # compare ranking order based on different qs and voteDiff
        
    # load voteDiff before the last vote of each answer for torchSGD training
    with open(intermediate_directory+f"/semiSynthetic8_sanityCheckRound_votingStage_trainingTestingSplitting_outputs.dict", 'rb') as inputFile:
        _, qid2aid2voteDiffBeforeLastTestingVote = pickle.load( inputFile)

    aid2Prior_q_rankZscore = defaultdict()
    aid2VoteDiff_rankZscore = defaultdict()
    aid2CVP_sklearn_q_rankZscore  = defaultdict()
    aid2VoteDiffBeforeLastVote_rankZscore = defaultdict()
    aid2CVP_torchSGD_q_rankZscore  = defaultdict()
    aid2newModel_q_rankZscore  = defaultdict()

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
        involved_aid2ranksBasedOnVoteDiff = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2voteDiff.items(), key=lambda kv: kv[1],reverse=True)) :
            aid = kv[0]
            involved_aid2ranksBasedOnVoteDiff[aid] = i+1 # rank
        
        zScores = get_zScore(involved_aid2ranksBasedOnVoteDiff.values())
        for i,aid in enumerate(involved_aid2ranksBasedOnVoteDiff.keys()):
            aid2VoteDiff_rankZscore[aid] = zScores[i]
        
        # get voteDiff before the last vote of each answer
        aid2voteDiffBeforeLastTestingVote = qid2aid2voteDiffBeforeLastTestingVote[qid]
        involved_aid2voteDiffBeforeLastTestingVote = {aid : aid2voteDiffBeforeLastTestingVote[aid] for aid in involved_answerList}

        # get ranks based on voteDiffBeforeLastVote
        involved_aid2ranksBasedOnVoteDiffBeforeLastVote = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2voteDiffBeforeLastTestingVote.items(), key=lambda kv: kv[1],reverse=True)) :
            aid = kv[0]
            involved_aid2ranksBasedOnVoteDiffBeforeLastVote[aid] = i+1 # rank
        
        zScores = get_zScore(involved_aid2ranksBasedOnVoteDiffBeforeLastVote.values())
        for i,aid in enumerate(involved_aid2ranksBasedOnVoteDiffBeforeLastVote.keys()):
            aid2VoteDiffBeforeLastVote_rankZscore[aid] = zScores[i]

        # get ranks based on prior_q
        involved_aid2prior_q = {aid : aid2prior_q[aid] for aid in involved_answerList}
        involved_aid2ranksBasedOnPriorQ = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2prior_q.items(), key=lambda kv: kv[1],reverse=True)) :
            aid = kv[0]
            involved_aid2ranksBasedOnPriorQ[aid] = i+1 # rank
        
        zScores = get_zScore(involved_aid2ranksBasedOnPriorQ.values())
        for i,aid in enumerate(involved_aid2ranksBasedOnPriorQ.keys()):
            aid2Prior_q_rankZscore[aid] = zScores[i]
        
        # get ranks based on CVP_sklearn_q
        involved_aid2CVP_sklearn_q = {aid : aid2CVP_sklearn_q[aid] for aid in involved_answerList}
        involved_aid2ranksBasedOnCVPsklearnQ = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2CVP_sklearn_q.items(), key=lambda kv: kv[1],reverse=True)) :
            aid = kv[0]
            involved_aid2ranksBasedOnCVPsklearnQ[aid] = i+1 # rank
        
        zScores = get_zScore(involved_aid2ranksBasedOnCVPsklearnQ.values())
        for i,aid in enumerate(involved_aid2ranksBasedOnCVPsklearnQ.keys()):
            aid2CVP_sklearn_q_rankZscore[aid] = zScores[i]

        # get ranks based on CVP_torchSGD_q
        involved_aid2CVP_torchSGD_q = {aid : aid2CVP_torchSGD_q[aid] for aid in involved_answerList}
        involved_aid2ranksBasedOnCVPtorchQ = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2CVP_torchSGD_q.items(), key=lambda kv: kv[1],reverse=True)) :
            aid = kv[0]
            involved_aid2ranksBasedOnCVPtorchQ[aid] = i+1 # rank
        
        zScores = get_zScore(involved_aid2ranksBasedOnCVPtorchQ.values())
        for i,aid in enumerate(involved_aid2ranksBasedOnCVPtorchQ.keys()):
            aid2CVP_torchSGD_q_rankZscore[aid] = zScores[i]
        
        # get ranks based on newModel_q
        involved_aid2newModel_q = {aid : aid2newModel_q[aid] for aid in involved_answerList}
        involved_aid2ranksBasedOnNewModelQ = defaultdict()
        for i, kv in enumerate(sorted(involved_aid2newModel_q.items(), key=lambda kv: kv[1],reverse=True)) :
            aid = kv[0]
            involved_aid2ranksBasedOnNewModelQ[aid] = i+1 # rank
        
        zScores = get_zScore(involved_aid2ranksBasedOnNewModelQ.values())
        for i,aid in enumerate(involved_aid2ranksBasedOnNewModelQ.keys()):
            aid2newModel_q_rankZscore[aid] = zScores[i]
    
    # save results
    with open(intermediate_directory+f"/semiSynthetic6_sanityCheckRound_outputs.dict", 'wb') as outputFile:
        pickle.dump((aid2Prior_q_rankZscore,aid2VoteDiff_rankZscore,aid2CVP_sklearn_q_rankZscore,aid2VoteDiffBeforeLastVote_rankZscore,aid2CVP_torchSGD_q_rankZscore,aid2newModel_q_rankZscore), outputFile)
        print( f"saved aid2zScore_outputs of {commName}")
    

    # prepare for plot
    # prior_q_rankZscores = []
    # voteDiff_rankZscores = []
    # CVP_sklearn_q_rankZscores  = []
    # voteDiffBeforeLastVote_rankZscores = []
    # CVP_torchSGD_q_rankZscores  = []
    # newModel_q_rankZscores  = []
    combineAll = [] # a list of tuple corresponding to each answer (prior_q_rankZscore, voteDff_rankZscore, CVP_sklearn_q_rankZscore, voteDiffBeforeLastVote_rankZscore, CVP_torchSGD_q_rankZscore, newModel_q_rankZscore)

    for aid in aid2Prior_q_rankZscore.keys():
        tup = (aid2Prior_q_rankZscore[aid],aid2VoteDiff_rankZscore[aid],aid2CVP_sklearn_q_rankZscore[aid],aid2VoteDiffBeforeLastVote_rankZscore[aid],aid2CVP_torchSGD_q_rankZscore[aid],aid2newModel_q_rankZscore[aid])
        combineAll.append(tup)

    # group
    # group points by intervals based on voteDiff zscores
    group_combineAll_basedOnPriorQ = myGrouping(combineAll, sortingColumn=0)
    group_combineAll_basedOnVoteDiff = myGrouping(combineAll, sortingColumn=1)
    group_combineAll_basedOnCVPsklearnQ = myGrouping(combineAll, sortingColumn=2)
    group_combineAll_basedOnVoteDiffBeforeLastVote = myGrouping(combineAll, sortingColumn=3)
    group_combineAll_basedOnCVPtorchQ = myGrouping(combineAll, sortingColumn=4)
    group_combineAll_basedOnNewModelQ = myGrouping(combineAll, sortingColumn=5)

    
    # plot
    plt.cla()
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=3.0)
    
    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnVoteDiff]
    group_voteDiff_rankZscores = [t[1] for t in group_combineAll_basedOnVoteDiff]
    axs[0, 0].scatter(group_voteDiff_rankZscores, group_priorQ_rankZscores,s = 2)
    axs[0, 0].set_xlabel('vote diff rankZscore', fontsize = 8)
    axs[0, 0].set_ylabel('prior q rankZscore', fontsize = 8)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_voteDiff_rankZscores = [t[1] for t in combineAll]
    # OLS fit
    z0,p0 = curveFitOLS(all_voteDiff_rankZscores, all_priorQ_rankZscores)
    axs[0,0].plot(all_voteDiff_rankZscores,p0(all_voteDiff_rankZscores),"r-", linewidth=1, label=f'slope={round(z0[0][0],4)}\nresidual={round(z0[1][0],4)}')
    axs[0,0].legend(loc="best", fontsize = 6)

    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVPsklearnQ]
    group_CVPsklearnQ_rankZscores = [t[2] for t in group_combineAll_basedOnCVPsklearnQ]
    axs[0, 1].scatter(group_CVPsklearnQ_rankZscores, group_priorQ_rankZscores,s = 2)
    axs[0, 1].set_xlabel('CVP sklearn q rankZscore', fontsize = 8)
    axs[0, 1].set_ylabel('prior q rankZscore', fontsize = 8)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_CVPsklearnQ_rankZscores = [t[2] for t in combineAll]
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

    group_priorQ_rankZscores = [t[0] for t in group_combineAll_basedOnCVPtorchQ]
    group_CVPtorchQ_rankZscores = [t[4] for t in group_combineAll_basedOnCVPtorchQ]
    axs[1, 0].scatter(group_CVPtorchQ_rankZscores, group_priorQ_rankZscores,s = 2)
    axs[1, 0].set_xlabel('CVP torchSGD q rankZscore', fontsize = 8)
    axs[1, 0].set_ylabel('prior q rankZscore', fontsize = 8)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_CVPtorchQ_rankZscores = [t[4] for t in combineAll]
    # OLS fit
    z3,p3 = curveFitOLS(all_CVPtorchQ_rankZscores, all_priorQ_rankZscores)
    axs[1,0].plot(all_CVPtorchQ_rankZscores,p3(all_CVPtorchQ_rankZscores),"r-", linewidth=1, label=f'slope={round(z3[0][0],4)}\nresidual={round(z3[1][0],4)}')
    axs[1,0].legend(loc="best", fontsize = 6)

    group_priorQ_rankZscores = [t[0] for t in  group_combineAll_basedOnNewModelQ]
    group_newModelQ_rankZscores = [t[5] for t in  group_combineAll_basedOnNewModelQ]
    axs[1, 1].scatter(group_newModelQ_rankZscores, group_priorQ_rankZscores,s = 2)
    axs[1, 1].set_xlabel('newModel q rankZscore', fontsize = 8)
    axs[1, 1].set_ylabel('prior q rankZscore', fontsize = 8)
    ### using all raw points
    all_priorQ_rankZscores = [t[0] for t in combineAll]
    all_newModelQ_rankZscores = [t[5] for t in combineAll]
    # OLS fit
    z4,p4 = curveFitOLS(all_newModelQ_rankZscores, all_priorQ_rankZscores)
    axs[1,1].plot(all_newModelQ_rankZscores,p4(all_newModelQ_rankZscores),"r-", linewidth=1, label=f'slope={round(z4[0][0],4)}\nresidual={round(z3[1][0],4)}')
    axs[1,1].legend(loc="best", fontsize = 6)

    fig.suptitle(f"{commName}")
    savePlot(fig, "semiSynthetic6_sanityCheckRound.png")


    # save csv statistics
    with open(rootDir+'/'+'allComm_semiSynthetic_sanityCheckRound_resultComparison.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( [commName,generated_event_count, 
                          prior_lamb, semiSynthetic_lamb_CVP_sklearn, semiSynthetic_lamb_CVP_torchSGD, semiSynthetic_lamb_newModel,
                          prior_tau,semiSynthetic_tau_CVP,semiSynthetic_tau_newModel,
                          round(z0[0][0],4),round(z1[0][0],4),round(z3[0][0],4),round(z4[0][0],4)])
                        #   f"{correlation_CVP_sklearn_to_prior} ({correlation_CVP_sklearn_to_prior_pvalue})", 
                        #   f"{correlation_CVP_torchSGD_to_prior} ({correlation_CVP_torchSGD_to_prior_pvalue})", 
                        #   f"{correlation_newModel_to_prior} ({correlation_newModel_to_prior_pvalue})"])
    

    


def main():
    rootDir = os.getcwd()
    t0=time.time()
    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    
    # save csv for CVP generating semi-synthetic dataset statistics
    with open(rootDir+'/'+'allComm_semiSynthetic_sanityCheckRound_resultComparison.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","total generated event Count", 
                          "prior_lamb", "semiSynthetic_lamb_CVP_sklearn", "semiSynthetic_lamb_CVP_torchSGD", "semiSynthetic_lamb_newModel",
                          "prior_tau","semiSynthetic_tau_CVP","semiSynthetic_tau_newModel",
                          "slope_voteDiff", "slope_CVPsklearnQuality",
                          "slope_CVPtorchSGD_Quality", "slope_newModel_Quality"])
    
    """
    # test on comm "coffee.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1], rootDir)
    # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1], rootDir)
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1], rootDir)

    
    """

    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]
 
        try:
            p = mp.Process(target=myFun, args=(commName,commDir, rootDir))
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
    print('semiSynthetic6_compareResults Done completely.    Elapsed: {:}.\n'.format(elapsed))
    
      
if __name__ == "__main__":
  
    # calling main function
    main()


##########################################################################