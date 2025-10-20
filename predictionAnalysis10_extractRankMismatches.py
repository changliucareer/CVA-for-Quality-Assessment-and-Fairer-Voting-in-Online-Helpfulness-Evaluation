import os
import sys
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################
from toolFunctions import format_time, writeIntoLog, savePlot
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
from statistics import mean, pstdev
from sklearn.model_selection import train_test_split
import sklearn
from CustomizedNN import LRNN_1layer, LRNN_1layer_withoutRankTerm
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.preprocessing import normalize
import random

from collections import Counter

#################################################################
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

def resultFormat_CVP(weights, bias, oneside, ori_questionCount):
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

def plotAnswerCountAtEachRankDiff (percentage2rankDiff2aids, commName, commDir):

    fig, ax = plt.subplots(4, figsize=(10,20))
    fig.suptitle(commName)
    

    for i, tup in enumerate(percentage2rankDiff2aids.items()):
        percentage, rankDiff2aids = tup
        #sort by rank Diff
        rankDiff2aids = dict(sorted(rankDiff2aids.items()))
        x = [rankDiff for rankDiff, aids in rankDiff2aids.items() if rankDiff !=0]
        y = [len(aids) for rankDiff, aids in rankDiff2aids.items() if rankDiff !=0]

        totalAnswerCount =  sum([len(aids) for rankDiff, aids in rankDiff2aids.items()])

        ax[i].bar(x, y)
        ax[i].set_ylabel('answer count')
        ax[i].set_xlabel('rank difference')

        for index, count in enumerate(y):
            ax[i].text(x[index]-0.1,count, f" {count}\n({int(count*100/totalAnswerCount)}%)", color = 'black',fontsize=10)

        ax[i].set_title(f'after train {percentage}%')
    
    savePlot(fig, 'answerCountAtEachRankDiff.pdf')

def my_analyze(percentage2aid2differentRanks, percentage2rankDiff2aids, commName):

    rankDiffStartPoint = 2

    percentage2statistics = defaultdict()
    for percentage, rankDiff2aids in percentage2rankDiff2aids.items():
        aid2differentRanks = percentage2aid2differentRanks[percentage]
        overEstimated = []
        underEstimated = []
        for rankDiff, aids in rankDiff2aids.items():
            if rankDiff > rankDiffStartPoint:
                overEstimated.extend(aids)
            elif rankDiff < - rankDiffStartPoint:
                underEstimated.extend(aids)

        newModelDemoted = []
        CVPDemoted = []

        newModelPromoted = []
        CVPPromoted = []

        for aid in overEstimated:
            d = aid2differentRanks[aid]
            vdRank = d['vdRank']
            sentimentRank = d['sentimentRank']
            newModelQualityRank = d['newModelQualityRank']
            CVPQualityRank = d['CVPQualityRank']

            if newModelQualityRank > vdRank:
                newModelDemoted.append(aid)

            if CVPQualityRank > vdRank:
                CVPDemoted.append(aid)

        for aid in underEstimated:
            d = aid2differentRanks[aid]
            vdRank = d['vdRank']
            sentimentRank = d['sentimentRank']
            newModelQualityRank = d['newModelQualityRank']
            CVPQualityRank = d['CVPQualityRank']

            if newModelQualityRank < vdRank:
                newModelPromoted.append(aid)

            if CVPQualityRank < vdRank:
                CVPPromoted.append(aid)

        percentage2statistics[percentage] = {'overEstimatedCount':len(overEstimated), 
                                             'underEstimatedCount':len(underEstimated),
                                             'newModel_DemotedRate': len(newModelDemoted) / len(overEstimated),
                                             'CVP_DemotedRate':len(CVPDemoted)/len(overEstimated),
                                             'newModel_PromotedRate':len(newModelPromoted)/len(underEstimated),
                                             'CVP_PromotedRate':len(CVPPromoted)/len(underEstimated)}
         
    return percentage2statistics

def create_qid2aidAndScoresList(answerWithVotes_ids_and_scores, percentage, newModel, CVPmodel, ori_questionCount):
    if percentage != 100:
        # output learned parameters
        parm = defaultdict()
        for name, param in CVPmodel.named_parameters():
            parm[name]=param.cpu().detach().numpy() 
        bias = None
        weights = parm['linear.weight'][0]
        _, _, _, qs_CVP = resultFormat_CVP(weights, bias, True, ori_questionCount)

        parm = defaultdict()
        for name, param in newModel.named_parameters():
            parm[name]=param.cpu().detach().numpy() 
        bias = None
        tau = np.exp(parm['tau'].item())
        weights = parm['linear.weight'][0]
        _, _, _, qs = resultFormat(weights, bias, tau, True, True, ori_questionCount)

    qid2aidAndScoresList = defaultdict()
    for i, tup in enumerate(answerWithVotes_ids_and_scores):
        qid = tup[0][0]
        aid = tup[0][1]
        learned_q = tup[1]
        voteDiff = tup[2]
        useful_sentimentScores = tup[4]
        if isinstance(useful_sentimentScores,list):
            avgSentiment = mean(useful_sentimentScores)
        else: # when there's no sentiment score, skip this answer
            continue

        learned_q_CVP = tup[7]

        if percentage != 100:
            learned_q = qs[i]
            learned_q_CVP = qs_CVP[i]
    
        if qid not in qid2aidAndScoresList.keys():
            qid2aidAndScoresList[qid] = [(aid, voteDiff, learned_q, avgSentiment, learned_q_CVP)]
        else:
            qid2aidAndScoresList[qid].append((aid, voteDiff, learned_q, avgSentiment, learned_q_CVP))

    return qid2aidAndScoresList

def myAction (qid, aidAndScoresList, percentage, commName, commDir):
    print(f"processing {commName} percent {percentage} questions {qid}...")
    # generate aid2differentRanks
    aid2differentRanks = defaultdict()

    # aidAndScoresList is a tuple list. tuple as (aid, voteDiff, learned_q, avgSentiment, learned_q_CVP)
    sortedByVoteDiff = copy.deepcopy(aidAndScoresList)
    sortedByVoteDiff.sort(key=lambda t:t[1], reverse=True)

    for i, tup in enumerate(sortedByVoteDiff):
        aid = tup[0]
        aid2differentRanks[aid]={'vdRank':(i+1), 'sentimentRank':None, 'newModelQualityRank': None, 'CVPQualityRank':None}

    sortedByNewModelQuality = copy.deepcopy(aidAndScoresList)
    sortedByNewModelQuality.sort(key=lambda t:t[2], reverse=True)

    for i, tup in enumerate(sortedByNewModelQuality):
        aid = tup[0]
        aid2differentRanks[aid]['newModelQualityRank'] = i+1

    sortedBySentiment = copy.deepcopy(aidAndScoresList)
    sortedBySentiment.sort(key=lambda t:t[3], reverse=True)

    for i, tup in enumerate(sortedBySentiment):
        aid = tup[0]
        aid2differentRanks[aid]['sentimentRank'] = i+1

    sortedByCVPQuality = copy.deepcopy(aidAndScoresList)
    sortedByCVPQuality.sort(key=lambda t:t[4], reverse=True)

    for i, tup in enumerate(sortedByCVPQuality):
        aid = tup[0]
        aid2differentRanks[aid]['CVPQualityRank'] = i+1


    # a list to store the vote count of each answer
    rankDiff2aids = defaultdict()

    for aid, r in aid2differentRanks.items():
        rankDiff = r['sentimentRank'] -  r['vdRank'] # when >0, over estimated, when <0 under estimated
        if rankDiff not in rankDiff2aids.keys():
            rankDiff2aids[rankDiff] = [aid]
        else:
            rankDiff2aids[rankDiff].append(aid)

    return aid2differentRanks, rankDiff2aids
         

def myFun(commIndex, commName, commDir, commSize, root_dir):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())


    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    print(f"loading data... for {commName}")
    """
    # get original_n_feature
    with open(intermediate_directory+'/'+'whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'rb') as inputFile:
        try:
            questions_Data = pickle.load( inputFile)
        except Exception as e:
            print(f"for {commName} error when load the Questions data: {e}")
            return
    
    for qid, tup in questions_Data.items():
        X = tup[0].todense()
        y = tup[1]
        if len(y) != 0:
            original_n_feature = X.shape[1] # the column number of X is the original number of features
            break
    questions_Data.clear()

    # get n_features
    n_feature = original_n_feature - 2
    n_feature_CVP = original_n_feature - 3

    # to get the origianl question count when generating nus, load eventlist file
    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        ori_Questions = pickle.load( inputFile)
        ori_questionCount = len(ori_Questions)

    # load sentiment data
    with open(intermediate_directory+'/'+'verifyQualities_newModelAndCVP_QualitywithFullData.dict', 'rb') as inputFile:
        try:
            answerWithVotes_ids_and_scores = pickle.load( inputFile) # a tuple list, (ids, learned_q, voteDiff, voteCount, useful_sentimentScores,helpful_sentimentScores,correct_sentimentScores, learned_q_CVP), ids is qid and aid tuple
        except Exception as e:
            print(f"for {commName} error when load the file: {e}")
            return
        
    # load models
    percentage2newModel = defaultdict() 
    percentage2CVPmodel = defaultdict()
    model_folder = os.path.join(commDir, r'trained_model_folder')

    for percentage in [25,50,75]:
        newModelDir = model_folder + f'/predictionAnalysis2_negLogLikelyhoods_newModel_posTau_trainWith{percentage}_model.sav'
        CVPmodelDir = model_folder + f'/predictionAnalysis3_negLogLikelyhoods_CVP_trainWith{percentage}_model.sav'

        initial_tau =1
        tauColumnIndex =1
        positiveTau= True
        newModel = LRNN_1layer(n_feature,initial_tau,tauColumnIndex, positiveTau)
        newModel.load_state_dict(torch.load(newModelDir, map_location='cpu'))
        percentage2newModel[percentage] = copy.deepcopy(newModel)
        
        CVPmodel = LRNN_1layer_withoutRankTerm(n_feature_CVP)
        CVPmodel.load_state_dict(torch.load(CVPmodelDir, map_location='cpu'))
        percentage2CVPmodel[percentage] = copy.deepcopy(CVPmodel)
    
    
    
    # get misMatched statistics
    percentage2aid2differentRanks = defaultdict()
    percentage2rankDiff2aids = defaultdict()

    for percentage in [25,50,75,100]:
        if percentage == 100:
            newModel = None
            CVPmodel = None
        else:
            newModel = percentage2newModel[percentage]
            CVPmodel = percentage2CVPmodel[percentage]
        # create qid2aidAndScoresList
        qid2aidAndScoresList = create_qid2aidAndScoresList(answerWithVotes_ids_and_scores, percentage, newModel, CVPmodel, ori_questionCount)
        
        # prepare args
        args =[]
        for qid, d in qid2aidAndScoresList.items():
            args.append((qid, d, percentage, commName, commDir))

        # process Questions chunk by chunk
        n_proc = mp.cpu_count() # left 2 cores to do others
        with mp.Pool(processes=n_proc) as pool:
            # issue tasks to the process pool and wait for tasks to complete
            all_outputs = pool.starmap(myAction, args , chunksize=10)

        # combine all outputs
        aid2differentRanks_total = None
        rankDiff2aids_total = None

        for output in all_outputs:
            aid2differentRanks, rankDiff2aids = output
            if rankDiff2aids_total == None:
                aid2differentRanks_total = aid2differentRanks
                rankDiff2aids_total = rankDiff2aids
            else:
                aid2differentRanks_total.update(aid2differentRanks)
                for rankDiff, aids in rankDiff2aids.items():
                    if rankDiff not in rankDiff2aids_total.keys():
                        rankDiff2aids_total[rankDiff] = aids
                    else:
                        rankDiff2aids_total[rankDiff].extend(aids)


        percentage2aid2differentRanks[percentage] = aid2differentRanks_total
        percentage2rankDiff2aids[percentage] = rankDiff2aids_total

    # saved the results
    with open(intermediate_directory+f"/percentage2aid2differentRanks.dict", 'wb') as outputFile:
        pickle.dump(percentage2aid2differentRanks, outputFile)

    with open(intermediate_directory+f"/percentage2rankDiff2aids.dict", 'wb') as outputFile:
        pickle.dump(percentage2rankDiff2aids, outputFile)
    """
    # load intermediate results
    with open(intermediate_directory+f"/percentage2aid2differentRanks.dict", 'rb') as inputFile:
        percentage2aid2differentRanks = pickle.load( inputFile)
    
    with open(intermediate_directory+f"/percentage2rankDiff2aids.dict", 'rb') as inputFile:
        percentage2rankDiff2aids = pickle.load( inputFile)
    
    # plotAnswerCountAtEachRankDiff (percentage2rankDiff2aids, commName, commDir)
        
    percentage2statistics = my_analyze(percentage2aid2differentRanks, percentage2rankDiff2aids, commName)

    csvfile = open(root_dir+'/predictionAnalysis10_statistics_rankDiffGreaterThan2.csv', 'a', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([commName, commSize, 
                     percentage2statistics[25]['overEstimatedCount'], percentage2statistics[25]['newModel_DemotedRate'],percentage2statistics[25]['CVP_DemotedRate'],
                     percentage2statistics[25]['underEstimatedCount'], percentage2statistics[25]['newModel_PromotedRate'],percentage2statistics[25]['CVP_PromotedRate'],
                     percentage2statistics[50]['overEstimatedCount'], percentage2statistics[50]['newModel_DemotedRate'],percentage2statistics[50]['CVP_DemotedRate'],
                     percentage2statistics[50]['underEstimatedCount'], percentage2statistics[50]['newModel_PromotedRate'],percentage2statistics[50]['CVP_PromotedRate'],
                     percentage2statistics[75]['overEstimatedCount'], percentage2statistics[75]['newModel_DemotedRate'],percentage2statistics[75]['CVP_DemotedRate'],
                     percentage2statistics[75]['underEstimatedCount'], percentage2statistics[75]['newModel_PromotedRate'],percentage2statistics[75]['CVP_PromotedRate'],
                     percentage2statistics[100]['overEstimatedCount'], percentage2statistics[100]['newModel_DemotedRate'],percentage2statistics[100]['CVP_DemotedRate'],
                     percentage2statistics[100]['underEstimatedCount'], percentage2statistics[100]['newModel_PromotedRate'],percentage2statistics[100]['CVP_PromotedRate']])
    csvfile.close()


def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")


    csvfile = open('predictionAnalysis10_statistics_rankDiffGreaterThan2.csv', 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow( ["commName","commSize", 
                      "25% overEstimated Count", "25% newModel DemotedRate", "25% CVP DemotedRate",
                      "25% underEstimated Count","25% newModel PromotedRate", "25% CVP PromotedRate",
                      "50% overEstimated Count", "50% newModel DemotedRate", "50% CVP DemotedRate",
                      "50% underEstimated Count","50% newModel PromotedRate", "50% CVP PromotedRate",
                      "75% overEstimated Count", "75% newModel DemotedRate", "75% CVP DemotedRate",
                      "75% underEstimated Count","75% newModel PromotedRate", "75% CVP PromotedRate",
                      "100% overEstimated Count", "100% newModel DemotedRate", "100% CVP DemotedRate",
                      "100% underEstimated Count","100% newModel PromotedRate", "100% CVP PromotedRate"])
    csvfile.close()
    
    
    """
    try:
        # test on comm "coffee.stackexchange" to debug
        myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1], commDir_sizes_sortedlist[166][2], root_dir)
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
    
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]
        commSize = tup[2]

        if commName in splitted_comms: # skip splitted big communities
            print(f"{commName} was splitted, skip.")
            continue

        try:
            p = mp.Process(target=myFun, args=(commIndex, commName,commDir, commSize, root_dir))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()
            return

        processes.append(p)
        if len(processes)==24:
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
    print('prediction analysis 10 extract rank mismatches  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
