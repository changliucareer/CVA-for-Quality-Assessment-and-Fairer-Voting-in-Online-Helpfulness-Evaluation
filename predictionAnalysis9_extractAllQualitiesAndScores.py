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
# import statsmodels.api as sm
# from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
from scipy import stats
import json


def myFun(commName, commDir, commSize, return_trainSuccess_item, return_trainSuccess_item_CVP):
   
    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())
    print(f"processing {commName}")

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')

    # check whether already done this step, skip
    resultFiles = ['verifyQualities_newModelAndCVP_QualitywithFullData.dict']
    resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
    # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    if os.path.exists(resultFiles[0]):
        print(f"{commName} has already done this step.")
        return
    
    print(f"loading total_answersWithVotes_indice... for {commName}")
    if commName != 'stackoverflow':
        try:
            with open(intermediate_directory+'/'+'whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
                total_answersWithVotes_indice = pickle.load( inputFile)
        except Exception as e:
            print(f"for {commName} error when load the total_answersWithVotes_indice: {e}")
            return
    else: # stackoverflow using 1% sampled data
        with open(intermediate_directory+'/'+'sampled1percent_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
            total_answersWithVotes_indice = pickle.load( inputFile)
    ############################################################################################
    

    # with open(final_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
    #     Questions = pickle.load( inputFile)
    # # get max answer count of one question
    # maxAnswerCount = 0
    # for qid, content in Questions.items():
    #     answerCount = len(content['filtered_answerList'])
    #     if answerCount > maxAnswerCount:
    #         maxAnswerCount = answerCount

    # get learned qs (without bias)
    if commName != 'stackoverflow': 
        # learned_qs = return_trainSuccess_item['qs']  
        learned_qs = return_trainSuccess_item['qs_withFullData']  # after retrain with full data

        # learned_qs_CVP = return_trainSuccess_item_CVP['qs'] 
        learned_qs_CVP = return_trainSuccess_item_CVP['qs_withFullData']  # after retrain with full data
    
    else: # for stackoverflow use old version learned qs
        learned_qs = return_trainSuccess_item['qs'] 
        learned_qs_CVP = return_trainSuccess_item_CVP['qs'] 


    assert len(learned_qs)==len(total_answersWithVotes_indice)

    # extract answer ids 
    if commName != 'stackoverflow': 
        with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
            Questions = pickle.load( inputFile)
    
    else: # for stackoverflow, scan part files
        splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
        split_sampled_QuestionsWithEventList_files_directory = os.path.join(splitFolder_directory, r'Sampled1percent_QuestionsPartsWithEventList')
        partFiles = [ f.path for f in os.scandir(split_sampled_QuestionsWithEventList_files_directory) if f.path.endswith('.dict') ]
        # combine all parts to one dict
        Questions = defaultdict()
        for i, partDir in enumerate(partFiles):
            # get question count of each part
            with open(partDir, 'rb') as inputFile:
                Questions_part = pickle.load( inputFile)
                if len(Questions_part) >0:
                    Questions.update(Questions_part)
                Questions_part.clear()
    
    total_answersWithVotes_ids = []
    for i,(qid, ai) in enumerate(total_answersWithVotes_indice):
        filtered_answerList = Questions[qid]['filtered_answerList']
        aid = filtered_answerList[ai]
        total_answersWithVotes_ids.append((qid,aid))


    # get corresponding sentiment and corresponding vote differences
    print(f"loading devertaV3 sentiment scores ... for {commName}")

    if commName != 'stackoverflow': 
        try:
            with open(intermediate_directory+'/'+'debertaV3_large_SentimentScores_of_posts_replaceTags.dict', 'rb') as inputFile:
                post2sentiment = pickle.load( inputFile)
        except:
                print(f"c{commName} hasn't debertaV3 large sentiment.")
                return
    else: # for stackoverflow, using base deberta sentiment and sampled questions
        with open(intermediate_directory+'/'+'debertaV3_base_SentimentScores_of_sampled1percent_posts_replaceTags_tillChunk39.dict', 'rb') as inputFile:
            post2sentiment = pickle.load( inputFile)
        with open(intermediate_directory+'/'+'debertaV3_base_SentimentScores_of_sampled1percent_posts_replaceTags_fromChunk40_new.dict', 'rb') as inputFile:
            post2sentiment_secondPart = pickle.load( inputFile)
        for k,v in post2sentiment_secondPart.items():
            if k in post2sentiment.keys():
                for s in v['sentimentScores']:
                    if s not in post2sentiment[k]['sentimentScores']:
                        post2sentiment[k]['sentimentScores'] = post2sentiment[k]['sentimentScores'] + [s]
            else:
                post2sentiment[k] = v
        post2sentiment_secondPart.clear()

    # convert post2sentiment keys to int
    if isinstance(list(post2sentiment.keys())[0], str):
        print(f"converting post2sentiment keys to int...")
        new_post2sentiment = defaultdict()
        for k,v in post2sentiment.items():
            new_post2sentiment[int(k)]=v
        post2sentiment = copy.deepcopy(new_post2sentiment)
        new_post2sentiment.clear()

    print(f"extracting aspact2sentimentScores...")
    useful_sentimentList = []
    helpful_sentimentList = []
    correct_sentimentList = []
    for qid, aid in total_answersWithVotes_ids:
        aspact2sentimentScores = {'useful':[], 'helpful':[],'correct':[]}
        if aid in post2sentiment.keys(): # current answer has comment sentiment
            # sentimentScores = post2sentiment[str(aid)]['sentimentScores']
            sentimentScores = post2sentiment[aid]['sentimentScores']
            for sentDict in sentimentScores:
                for aspact, value in sentDict.items():
                    label = value[0]['label']
                    score = value[0]['score']
                    if label == 'Negative':
                        aspact2sentimentScores[aspact].append(-1*score)
                    elif label == 'Neutral':
                        aspact2sentimentScores[aspact].append(0)
                    else: # positive
                        aspact2sentimentScores[aspact].append(1*score)
            useful_sentimentList.append(aspact2sentimentScores['useful'])
            helpful_sentimentList.append(aspact2sentimentScores['helpful'])
            correct_sentimentList.append(aspact2sentimentScores['correct'])
        else:
            useful_sentimentList.append(None)
            helpful_sentimentList.append(None)
            correct_sentimentList.append(None)
            

    # filter None
    print(f"filtering None scores...")
    answerWithVotes_ids_and_scores = []
    filtered_answerWithVotes_ids_and_scores = []
    for i, sentimentScores in enumerate(useful_sentimentList):
        learned_q = learned_qs[i]
        learned_q_CVP = learned_qs_CVP[i]
        ids = total_answersWithVotes_ids[i] # qid and aid tuple
        ai = total_answersWithVotes_indice[i][1] # answer index in filtered answer list
        # extract the vote difference
        for qid, content in Questions.items():
            if qid == ids[0]:
                voteMatrix = content['vote_matrix'].todense()
                voteDiff = np.sum(voteMatrix[ai,:])
                voteCount = np.count_nonzero(voteMatrix[ai,:])
                break
        if sentimentScores==None:
            answerWithVotes_ids_and_scores.append((ids, learned_q, voteDiff, voteCount, sentimentScores,helpful_sentimentList[i],correct_sentimentList[i], learned_q_CVP))
        else:
            answerWithVotes_ids_and_scores.append((ids, learned_q, voteDiff, voteCount, sentimentScores,helpful_sentimentList[i],correct_sentimentList[i], learned_q_CVP))
            filtered_answerWithVotes_ids_and_scores.append((ids, learned_q, voteDiff, voteCount, sentimentScores,helpful_sentimentList[i],correct_sentimentList[i], learned_q_CVP))
    
    if len(filtered_answerWithVotes_ids_and_scores)==0:
        print(f"no answer has sentiment score in comm {commName}.")
        return
    
    # save total_answersWithVotes_ids, sentimentList and voteDiffList
    with open(intermediate_directory+'/'+'verifyQualities_newModelAndCVP_QualitywithFullData.dict', 'wb') as outputFile:
        pickle.dump(answerWithVotes_ids_and_scores, outputFile)
        print(f"saved answerWithVotes_ids_and_scores.")
    
   
def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    return_slopeAndResidual = manager.dict() # to save the used train mode (wholebatch or minibatch) of each community

    """
    # old training results (don't constrain tau and with bias, and rank bug)
    with open('training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_return_trainSuccess.dict', 'rb') as inputFile:
        return_trainSuccess_dict = pickle.load( inputFile)
    print(f"return train success dict loaded. length {len(return_trainSuccess_dict)}")

    with open('training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_return_trainSuccess_round2.dict', 'rb') as inputFile:
        return_trainSuccess_round2_dict = pickle.load( inputFile)
    print("return train success round 2 dict loaded.")

    for qid, content in return_trainSuccess_round2_dict.items():
        if qid not in return_trainSuccess_dict.keys():
            return_trainSuccess_dict[qid] = content

    with open('training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_return_trainSuccess_forSplittedData.dict', 'rb') as inputFile:
        return_trainSuccess_forSplittedData_dict = pickle.load( inputFile)
    print("return train success forSplittedData dict loaded.")

    for qid, content in return_trainSuccess_forSplittedData_dict.items():
        if qid not in return_trainSuccess_dict.keys():
            return_trainSuccess_dict[qid] = content

    with open('training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_return_trainSuccess_forSplittedData_codegolf.dict', 'rb') as inputFile:
        return_trainSuccess_codegolf_dict = pickle.load( inputFile)
    print("return train success codegolf dict loaded.")

    for qid, content in return_trainSuccess_codegolf_dict.items():
        if qid not in return_trainSuccess_dict.keys():
            return_trainSuccess_dict[qid] = content
    """
    """
    # updated training results (constrain positive tau and without bias)
    with open('training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_return_trainSuccess_first300Comm_posTau.dict', 'rb') as inputFile:
        return_trainSuccess_dict = pickle.load( inputFile)
    print(f"return train success dict loaded. length {len(return_trainSuccess_dict)}")

    with open('training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_return_trainSuccess_remain52Comm_posTau.dict', 'rb') as inputFile:
        return_trainSuccess_round2_dict = pickle.load( inputFile)
    print("return train success round 2 dict loaded.")

    for commName, content in return_trainSuccess_round2_dict.items():
        if commName not in return_trainSuccess_dict.keys():
            return_trainSuccess_dict[commName] = content

    with open('training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_return_trainSuccess_forSplittedData_posTau.dict', 'rb') as inputFile:
        return_trainSuccess_forSplittedData_dict = pickle.load( inputFile)
    print("return train success forSplittedData dict loaded.")

    for commName, content in return_trainSuccess_forSplittedData_dict.items():
        if commName not in return_trainSuccess_dict.keys():
            return_trainSuccess_dict[commName] = content
    
    with open('training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_return_trainSuccess_forSplittedData_posTau(codegolf).dict', 'rb') as inputFile:
        return_trainSuccess_codegolf_dict = pickle.load( inputFile)
    print("return train success codegolf dict loaded.")

    for commName, content in return_trainSuccess_codegolf_dict.items():
        if commName not in return_trainSuccess_dict.keys():
            return_trainSuccess_dict[commName] = content
    
    print(f"updated return train success dict loaded. length {len(return_trainSuccess_dict)}")

    with open('SOF_7_training_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_for1PercentSampledQuestions_forSplittedData_posTau.dict', 'rb') as inputFile:
        return_trainSuccess_1percentSOF_dict = pickle.load( inputFile)
    print("return train success 1percentSOF dict loaded.")

    for commName, content in return_trainSuccess_1percentSOF_dict.items():
        if commName not in return_trainSuccess_dict.keys():
            return_trainSuccess_dict[commName] = content
    
    print(f"updated return train success dict loaded. length {len(return_trainSuccess_dict)}")
    """

    # temperal order trained
    with open('temperalOrderTraining_newModel_return.dict', 'rb') as inputFile:
        return_trainSuccess_dict = pickle.load( inputFile)
    print(f"return train success dict loaded. length {len(return_trainSuccess_dict)}")

    with open('temperalOrderTraining_CVP_return.dict', 'rb') as inputFile:
        return_trainSuccess_dict_CVP = pickle.load( inputFile)
    print(f"return train success dict CVP loaded. length {len(return_trainSuccess_dict_CVP)}")

    # add 1% sampled SOF results from old learning
    with open('SOF_7_training_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_for1PercentSampledQuestions_forSplittedData_posTau.dict', 'rb') as inputFile:
        return_trainSuccess_1percentSOF_dict = pickle.load( inputFile)
    print("return train success 1percentSOF dict loaded.")

    for commName, content in return_trainSuccess_1percentSOF_dict.items():
        if commName not in return_trainSuccess_dict.keys():
            return_trainSuccess_dict[commName] = content

    with open('SOF_7_1_training_CVP_removeFirstRealVote_oneside_temperalTesting_for1PercentSampledQuestions_forSplittedData.dict', 'rb') as inputFile:
        return_trainSuccess_1percentSOF_dict_CVP = pickle.load( inputFile)
    print("return train success 1percentSOF CVP dict loaded.")

    for commName, content in return_trainSuccess_1percentSOF_dict_CVP.items():
        if commName not in return_trainSuccess_dict_CVP.keys():
            return_trainSuccess_dict_CVP[commName] = content

    ## Load top120CommNames
    with open('top120CommNames.dict', 'rb') as inputFile:
        top120CommNames = pickle.load( inputFile)
        print(top120CommNames)
    print("top120CommNames loaded.")
    
    
    # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1],return_trainSuccess_dict[commDir_sizes_sortedlist[166][0]])
    # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1],return_trainSuccess_dict[commDir_sizes_sortedlist[301][0]])
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1],return_trainSuccess_dict[commDir_sizes_sortedlist[305][0]])
    # test on comm "english.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[349][0], commDir_sizes_sortedlist[349][1],return_trainSuccess_dict[commDir_sizes_sortedlist[349][0]])
    # test on comm "math.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[358][0], commDir_sizes_sortedlist[358][1],return_trainSuccess_dict[commDir_sizes_sortedlist[358][0]])
    # test on comm "codegolf.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[333][0], commDir_sizes_sortedlist[333][1],return_trainSuccess_dict[commDir_sizes_sortedlist[333][0]])
    # test on comm "stackoverflow" to debug
    # myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1],return_trainSuccess_dict[commDir_sizes_sortedlist[359][0]])
    
    # test on comm "meta.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[336][0], commDir_sizes_sortedlist[336][1],return_trainSuccess_dict[commDir_sizes_sortedlist[336][0]])
    # test on comm "workplace.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[323][0], commDir_sizes_sortedlist[323][1],return_trainSuccess_dict[commDir_sizes_sortedlist[323][0]])
    
    
    # splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist[359:]):
        commName = tup[0]
        commDir = tup[1]
        commSize = tup[2]

        # if commName not in splitted_comms:
        #     print(f"{commName} is not splitted, skip")
        #     continue
        
        # if commName not in top120CommNames:
        #     print(f"{commName} not in top 120, skip")
        #     continue

        if commName in return_trainSuccess_dict.keys():
            return_trainSuccess_item = return_trainSuccess_dict[commName]
        else:
            print(f"{commName} not in return_trainSuccess_dict, skip")
            continue
        
        if commName in return_trainSuccess_dict_CVP.keys():
            return_trainSuccess_item_CVP = return_trainSuccess_dict_CVP[commName]
        else:
            print(f"{commName} not in return_trainSuccess_dict_CVP, skip")
            continue

        try:
            p = mp.Process(target=myFun, args=(commName,commDir,commSize, return_trainSuccess_item, return_trainSuccess_item_CVP))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()

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
    print('predictionAnalysis 9 Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
