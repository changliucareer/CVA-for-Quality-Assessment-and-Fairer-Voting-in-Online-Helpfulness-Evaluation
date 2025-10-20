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
def plotHisto ( voteCountOfEachAnswerBeforeRemoveTheFirstVote, commName, commDir):
    meanVoteCount = mean(voteCountOfEachAnswerBeforeRemoveTheFirstVote)
    std = pstdev(voteCountOfEachAnswerBeforeRemoveTheFirstVote) 
    plt.clf()
    fig, axs = plt.subplots(2, 1)
    fig.tight_layout(pad=3.0)
    counted = Counter(voteCountOfEachAnswerBeforeRemoveTheFirstVote)
    counted = dict(sorted(counted.items()))
    x = list(counted.keys())
    y = list(counted.values())
    rects = axs[0].bar(x, y)

    for i,yy in enumerate(y):
        if yy == max(y):
            axs[0].text(x[i],yy+1, str(yy)+f"(vc={x[i]})", color = 'black',fontsize=8, horizontalalignment='left')
        elif x[i]== round(meanVoteCount):
            axs[0].text(x[i],yy+1, str(yy)+f"(vc={x[i]})", color = 'black',fontsize=8, horizontalalignment='left')
        
    axs[0].grid(axis='y', alpha=0.75)
    axs[0].set_xlabel('vote count of an answer')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Histogram')
    axs[0].text(max(x)/2, max(y)/2, f"mean:{round(meanVoteCount,4)}, std:{round(std,4)}")

    yy = [ round(i/ len(voteCountOfEachAnswerBeforeRemoveTheFirstVote),2) for i in y]
    rects_ = axs[1].bar(x, yy)
    for i,yyy in enumerate(yy):
        if yyy == max(yy):
            axs[1].text(x[i],yyy+0.01, yyy, color = 'black',fontsize=8, horizontalalignment='center')
        elif x[i]== round(meanVoteCount):
            axs[1].text(x[i],yyy+0.01, yyy, color = 'black',fontsize=8, horizontalalignment='center')
    axs[1].grid(axis='y', alpha=0.75)
    axs[1].set_xlabel('vote count of an answer')
    axs[1].set_ylabel('Proportion')
    # axs[1].set_title('Histogram')
    savePlot(plt, "voteCountOfEachAnswerBeforeRemoveTheFirstVote.png")


def myAction (qid, question_data, question_content, commName, commDir):
    eventList = question_content['eventList']
    lts = question_content['local_timesteps']
    filtered_answer_list = question_content['filtered_answerList']
    voteMatrix = question_content['vote_matrix'].todense()
    question_data_X = question_data[0].todense()
    question_data_y = question_data[1]
    max_votesCountOfComm = question_data[2]


    # a list to store the vote count of each answer
    voteCountOfEachAnswerBeforeRemoveTheFirstVote = []
    for ai in range(len(filtered_answer_list)):
        voteCount = np.count_nonzero(voteMatrix[ai,:])
        voteCountOfEachAnswerBeforeRemoveTheFirstVote.append(voteCount)

    # extract universal time stamp index or vid as sorting base
    sortingBaseList = [] # should correspond to each data sample

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
            sortingBaseList.append(t)
            cur_x = lil_matrix(question_data_X[dataSampleIndex], dtype=np.float16)
            if ai in ai2data.keys():
                ai2data[ai] = ai2data[ai] + [(t,(cur_x,question_data_y[dataSampleIndex], max_votesCountOfComm))]
            else:
                ai2data[ai] = [(t,(cur_x,question_data_y[dataSampleIndex], max_votesCountOfComm))]
            dataSampleIndex += 1
        else: # current vote is its target answer's first vote, don't use as sample
            firstVoteRemovedFlagForEachAnswer.append(ai)
    
    assert len(sortingBaseList) == len(question_data_y)

    # divide data samples by quantiles
    percent25Data = []
    percent50Data = []
    percent75Data = []
    percent100Data = []

    aid2dataOfAnswersWithLessThan5Votes = defaultdict() 

    skipVoteCount = 0 # the count of votes that for answers less than 5 votes
    for ai, tupList in ai2data.items():
        voteCount = len(tupList)
        assert voteCount == voteCountOfEachAnswerBeforeRemoveTheFirstVote[ai] - 1
        if voteCount < 4: # unable to divide evenly into 4 parts after removing the first vote
            aid = filtered_answer_list[ai]
            skipVoteCount += voteCount
            if aid not in aid2dataOfAnswersWithLessThan5Votes.keys():
                aid2dataOfAnswersWithLessThan5Votes[aid] = tupList
            else:
                print(f"Exception!")
        else:
            voteCountForEachPart = int(voteCount/4)
            residualVoteCount = voteCount - voteCountForEachPart*4
            percent25Data.extend(tupList[:(voteCountForEachPart+residualVoteCount)])
            percent50Data.extend(tupList[(voteCountForEachPart+residualVoteCount):(voteCountForEachPart*2+residualVoteCount)])
            percent75Data.extend(tupList[(voteCountForEachPart*2+residualVoteCount):(voteCountForEachPart*3+residualVoteCount)])
            percent100Data.extend(tupList[(voteCountForEachPart*3+residualVoteCount):])

    assert len(percent25Data)+ len(percent50Data) + len(percent75Data) + len(percent100Data) + skipVoteCount == len(sortingBaseList)
    
    return qid, voteCountOfEachAnswerBeforeRemoveTheFirstVote, sortingBaseList, percent25Data, percent50Data, percent75Data, percent100Data, aid2dataOfAnswersWithLessThan5Votes


def myFun(commIndex, commName, commDir):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "predictionAnalysis0_dataSortingAndDividing_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # check whether already done this step, skip
    predictionAnalysis_data_folder = os.path.join(intermediate_directory, r'predictionAnalysis_data_folder')
    result_directory = predictionAnalysis_data_folder
    resultFiles = ['percent25Data.dict','percent50Data.dict','percent75Data.dict','percent100Data.dict']
    resultFiles = [result_directory+'/'+f for f in resultFiles]
    # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    if all([os.path.exists(rf) for rf in resultFiles]):
        print(f"{commName} has already done this step.")
        return

    print(f"loading data... for {commName}")
    try:
        with open(intermediate_directory+'/'+'whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'rb') as inputFile:
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
    predictionAnalysis_data_folder = os.path.join(intermediate_directory, r'predictionAnalysis_data_folder')
    if not os.path.exists(predictionAnalysis_data_folder): # don't have dataset splitted intermediate data folder, skip
        print(f"{commName} has no  predictionAnalysis_data_folder, create one.")
        os.makedirs(predictionAnalysis_data_folder)

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
    voteCountOfEachAnswerBeforeRemoveTheFirstVote_total = []
    qid2sortingBaseList = defaultdict()
    percent25Data_total = []
    percent50Data_total = []
    percent75Data_total = []
    percent100Data_total = []
    aid2dataOfAnswersWithLessThan5Votes_total = defaultdict()

    for tup in all_outputs:
        qid, voteCountOfEachAnswerBeforeRemoveTheFirstVote, sortingBaseList, percent25Data, percent50Data, percent75Data, percent100Data, aid2dataOfAnswersWithLessThan5Votes = tup
        voteCountOfEachAnswerBeforeRemoveTheFirstVote_total.extend(voteCountOfEachAnswerBeforeRemoveTheFirstVote)
        qid2sortingBaseList[qid] = sortingBaseList
        percent25Data_total.extend(percent25Data)
        percent50Data_total.extend(percent50Data)
        percent75Data_total.extend(percent75Data)
        percent100Data_total.extend(percent100Data)

        aid2dataOfAnswersWithLessThan5Votes_total.update(aid2dataOfAnswersWithLessThan5Votes)
        

    all_outputs.clear()

    # save the data parts
    with open(predictionAnalysis_data_folder+f"/percent25Data.dict", 'wb') as outputFile:
        pickle.dump(percent25Data_total, outputFile)
        log_text = f"saved percent25Data_total with total {len(percent25Data_total)} samples\n"
    
    with open(predictionAnalysis_data_folder+f"/percent50Data.dict", 'wb') as outputFile:
        pickle.dump(percent50Data_total, outputFile)
        log_text += f"saved percent50Data_total with total {len(percent50Data_total)} samples\n"
    
    with open(predictionAnalysis_data_folder+f"/percent75Data.dict", 'wb') as outputFile:
        pickle.dump(percent75Data_total, outputFile)
        log_text += f"saved percent75Data_total with total {len(percent75Data_total)} samples\n"
    
    with open(predictionAnalysis_data_folder+f"/percent100Data.dict", 'wb') as outputFile:
        pickle.dump(percent100Data_total, outputFile)
        log_text += f"saved percent100Data_total with total {len(percent100Data_total)} samples\n"
    
    with open(predictionAnalysis_data_folder+f"/dataOfAnswersWithLessThan5Votes.dict", 'wb') as outputFile:
        pickle.dump(aid2dataOfAnswersWithLessThan5Votes_total, outputFile)
        log_text += f"saved aid2dataOfAnswersWithLessThan5Votes_total with total {sum([len(tupList) for tupList in aid2dataOfAnswersWithLessThan5Votes_total.values()])} samples\n"

    log_text += f"The count of answer less than 5 votes:{len(aid2dataOfAnswersWithLessThan5Votes_total)}.\n"

    writeIntoLog(log_text, commDir, logFileName)
    print(log_text + f"for {commName}")

    # save the other results
    with open(intermediate_directory+f"/predictionAnalysis_dataSortingAndDividingOutputs.dict", 'wb') as outputFile:
        pickle.dump((voteCountOfEachAnswerBeforeRemoveTheFirstVote_total, qid2sortingBaseList), outputFile)
        print( f"saved other outputs of {commName}")
    
    """
    # load
    with open(intermediate_directory+f"/predictionAnalysis_dataSortingAndDividingOutputs.dict", 'rb') as inputFile:
        tup = pickle.load( inputFile)
        voteCountOfEachAnswerBeforeRemoveTheFirstVote_total, qid2sortingBaseList = tup
    """
    # plot histo of vote count of each answer
    plotHisto (voteCountOfEachAnswerBeforeRemoveTheFirstVote_total, commName, commDir)
    

def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    """
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
    for commIndex, tup in enumerate(commDir_sizes_sortedlist[253:]):
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
    print('prediction analysis 0 data dividing  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
