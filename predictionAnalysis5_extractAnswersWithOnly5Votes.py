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
from CustomizedNN import LRNN_1layer, LRNN_1layer_bias, LRNN_1layer_bias_specify,LRNN_1layer_bias_withoutRankTerm
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.preprocessing import normalize
import random

from collections import Counter

#################################################################
def mySorting (percentage, commName, commDir, percentData, predictionAnalysis_hardSamples_folder):
    # sort by sorting base
    percentData.sort(key=lambda t:t[0])

    # extract only data sample
    X = []
    y = []
    for tup in percentData:
        cur_x = tup[1][0].todense()
        cur_y = tup[1][1]
        max_votesCountOfComm = tup[1][2]
        if len(X)==0:
            X = cur_x
        else:
            X = np.vstack((X,cur_x))
        y.append(cur_y)
    
    new_X = lil_matrix( X, dtype=np.float16 )

    # save new version of part data
    with open(predictionAnalysis_hardSamples_folder+f"/precent{percentage}HardSamplesPureData.dict", 'wb') as outputFile:
        pickle.dump((new_X, y, max_votesCountOfComm), outputFile)
        print( f"saved conveted hard samples data of percent{percentage} {commName}")

def myAction (qid, question_data, question_content, commName, commDir):
    eventList = question_content['eventList']
    lts = question_content['local_timesteps']
    filtered_answer_list = question_content['filtered_answerList']
    voteMatrix = question_content['vote_matrix'].todense()
    question_data_X = question_data[0].todense()
    question_data_y = question_data[1]
    max_votesCountOfComm = question_data[2]


    # a list to store the vote count of each answer
    answersWithOnly5Votes = []
    aiListWihtOnly5votes = []
    for ai in range(len(filtered_answer_list)):
        voteCount = np.count_nonzero(voteMatrix[ai,:])
        if voteCount == 5:
            aid = filtered_answer_list[ai]
            answersWithOnly5Votes.append((qid,aid,ai))
            aiListWihtOnly5votes.append(ai)

    # align data according to answer
    ai2data = defaultdict() # store a dict, key as answer index of filtered answers, data as a tuple list of (sorting base, training data sample)
    dataSampleIndex = 0

    firstVoteRemovedFlagForEachAnswer =  []  # the first vote of each answer was removed, must skip
    for i,e in enumerate(eventList):
        t = lts[i]
        if commName == 'stackoverflow':
            cur_year = t[2].year
            t = t[1] # which is vid, can be used to sort
        eventType = e['et']
        if eventType != 'v': # skip all event that is not a vote
            continue
        ai = e['ai']
        if ai in firstVoteRemovedFlagForEachAnswer: # current vote's target answer already has the first vote removed
            if ai in aiListWihtOnly5votes:
                cur_x = lil_matrix(question_data_X[dataSampleIndex], dtype=np.float16)
                if ai in ai2data.keys():
                    ai2data[ai] = ai2data[ai] + [(t,(cur_x,question_data_y[dataSampleIndex], max_votesCountOfComm))]
                else:
                    ai2data[ai] = [(t,(cur_x,question_data_y[dataSampleIndex], max_votesCountOfComm))]
            dataSampleIndex += 1
        else: # current vote is its target answer's first vote, don't use as sample
            firstVoteRemovedFlagForEachAnswer.append(ai)
    
    assert dataSampleIndex == len(question_data_y)

    # divide data samples by quantiles
    percent25HardSamples = []
    percent50HardSamples = []
    percent75HardSamples = []
    percent100HardSamples = []

    skipVoteCount = 0 # the count of votes that for answers less than 5 votes
    for ai, tupList in ai2data.items():
        voteCount = len(tupList)
        if voteCount != 4: # Exception, only extracted answers with 5 votes before removing the first vote
            print(f"Exception!")
        else:
            voteCountForEachPart = int(voteCount/4)
            residualVoteCount = voteCount - voteCountForEachPart*4
            percent25HardSamples.extend(tupList[:(voteCountForEachPart+residualVoteCount)])
            percent50HardSamples.extend(tupList[(voteCountForEachPart+residualVoteCount):(voteCountForEachPart*2+residualVoteCount)])
            percent75HardSamples.extend(tupList[(voteCountForEachPart*2+residualVoteCount):(voteCountForEachPart*3+residualVoteCount)])
            percent100HardSamples.extend(tupList[(voteCountForEachPart*3+residualVoteCount):])
    
    return qid, answersWithOnly5Votes, percent25HardSamples, percent50HardSamples, percent75HardSamples, percent100HardSamples


def myFun(commIndex, commName, commDir):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "predictionAnalysis5_extractAnswersWithOnly5Votes_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # check whether already done this step, skip
    # result_directory = os.path.join(commDir, r'result_folder')
    # resultFiles = ['training1_allTimeSteps_removeFirstRealVote_oneside_temperalTesting_results.txt']
    # resultFiles = [result_directory+'/'+f for f in resultFiles]
    # # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    # if os.path.exists(resultFiles[0]):
    #     print(f"{commName} has already done this step.")
    #     return

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    print(f"loading data... for {commName}")
    with open(intermediate_directory+'/'+'whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'rb') as inputFile:
        try:
            questions_Data = pickle.load( inputFile)
        except Exception as e:
            print(f"for {commName} error when load the Questions data: {e}")
            return
    questionCount = len(questions_Data) # this not equal to the origianl question count when generating nus
    # to get the origianl question count when generating nus, load eventlist file
    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        ori_Questions = pickle.load( inputFile)
    ori_questionCount = len(ori_Questions)

    print(f"{commName} has {questionCount} questions in training data, but has {ori_questionCount} questions corresponding to nus.")

    # mkdir to keep predictionAnalysis data
    predictionAnalysis_hardSamples_folder = os.path.join(intermediate_directory, r'predictionAnalysis_hardSamples_folder')
    if not os.path.exists(predictionAnalysis_hardSamples_folder): # don't have dataset splitted intermediate data folder, skip
        print(f"{commName} has no  predictionAnalysis_hardSamples_folder, create one.")
        os.makedirs(predictionAnalysis_hardSamples_folder)

    # prepare args
    args =[]
    for qid, d in questions_Data.items():
        if qid not in ori_Questions.keys():
            print(f"Exception: no question {qid} in QuestionsWithEventList!")
            continue
        args.append((qid, d, ori_Questions[qid], commName, commDir))

    # process Questions chunk by chunk
    n_proc = mp.cpu_count() # left 2 cores to do others
    with mp.Pool(processes=n_proc) as pool:
        # issue tasks to the process pool and wait for tasks to complete
        all_outputs = pool.starmap(myAction, args , chunksize=10)
    
    questions_Data.clear()
    ori_Questions.clear()

    # combine all outputs
    percent25HardSamples_total = []
    percent50HardSamples_total = []
    percent75HardSamples_total = []
    percent100HardSamples_total = []
    answersWithOnly5Votes_total = []

    for tup in all_outputs:
        qid, answersWithOnly5Votes, percent25HardSamples, percent50HardSamples, percent75HardSamples, percent100HardSamples = tup
        answersWithOnly5Votes_total.extend(answersWithOnly5Votes)
        percent25HardSamples_total.extend(percent25HardSamples)
        percent50HardSamples_total.extend(percent50HardSamples)
        percent75HardSamples_total.extend(percent75HardSamples)
        percent100HardSamples_total.extend(percent100HardSamples)

    all_outputs.clear()

    # sort and save the data
    mySorting(25,commName, commDir, percent25HardSamples_total, predictionAnalysis_hardSamples_folder)
    mySorting(50,commName, commDir, percent50HardSamples_total, predictionAnalysis_hardSamples_folder)
    mySorting(75,commName, commDir, percent75HardSamples_total, predictionAnalysis_hardSamples_folder)
    mySorting(100,commName, commDir, percent100HardSamples_total, predictionAnalysis_hardSamples_folder)
    
    # saved the extracted answers info 
    with open(predictionAnalysis_hardSamples_folder+f"/answersWithOnly5Votes.dict", 'wb') as outputFile:
        pickle.dump(answersWithOnly5Votes_total, outputFile)
        log_text = f"saved answersWithOnly5Votes_total with total {len(answersWithOnly5Votes_total)} answers\n"
    

    writeIntoLog(log_text, commDir, logFileName)
    print(log_text + f"for {commName}")
    

def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    
    try:
        # test on comm "coffee.stackexchange" to debug
        myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
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

        if commName in splitted_comms: # skip splitted big communities
            print(f"{commName} was splitted.")
            continue

        try:
            p = mp.Process(target=myFun, args=(commIndex, commName,commDir))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()
            return

        processes.append(p)
        if len(processes)==4:
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
    """

    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('prediction analysis 5 extract answers with only 5 votes before removing the first vote  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
