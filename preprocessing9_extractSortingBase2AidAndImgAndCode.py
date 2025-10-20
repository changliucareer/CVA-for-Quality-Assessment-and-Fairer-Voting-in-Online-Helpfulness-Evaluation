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

def myAction (qid, question_content, commName, commDir):
    eventList = question_content['eventList']
    lts = question_content['local_timesteps']
    filtered_answer_list = question_content['filtered_answerList']
    voteMatrix = question_content['vote_matrix'].todense()

    # extract universal time stamp index or vid as sorting base
    sortingBase2Aid = defaultdict() # should correspond to each data sample

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
        aid = filtered_answer_list[ai]
        if ai in firstVoteRemovedFlagForEachAnswer: # current vote's target answer already has the first vote removed
            sortingBase2Aid[t] = aid
        else: # current vote is its target answer's first vote, don't use as sample
            firstVoteRemovedFlagForEachAnswer.append(ai)

    return qid, sortingBase2Aid


def myFun(commIndex, commName, commDir):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "preprocessing9_extractSortingBase2AidAndImgAndCode_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # # check whether already done this step, skip
    # predictionAnalysis_data_folder = os.path.join(intermediate_directory, r'predictionAnalysis_data_folder')
    # result_directory = predictionAnalysis_data_folder
    # resultFiles = ['percent25Data.dict','percent50Data.dict','percent75Data.dict','percent100Data.dict']
    # resultFiles = [result_directory+'/'+f for f in resultFiles]
    # # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    # if all([os.path.exists(rf) for rf in resultFiles]):
    #     print(f"{commName} has already done this step.")
    #     return

    print(f"loading data... for {commName}")
    try:
        with open(intermediate_directory+'/'+'whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'rb') as inputFile:
            questions_Data = pickle.load( inputFile)
    except Exception as e:
        print(f"for {commName} error when load the Questions data: {e}")
        return
    
    QidsWithData = list(questions_Data.keys())
    questions_Data.clear()


    # load sorting base list (a list of tuple (qid,i,sortingBase)) (already sorted by sortingBase)
    try:
        with open(intermediate_directory+f"/sorted_qidAndSampleIndexAndSortingBase.dict", 'rb') as inputFile:
            sorted_qidAndSampleIndex = pickle.load( inputFile)
    except:
        print(f"{commName} hasn't done temperalOrderSorting yet, skip")
        return
    dataSampleCount = len(sorted_qidAndSampleIndex) 

    # load aid2img and aid2code
    with open(intermediate_directory+f"/answerId2phId_img.dict", 'rb') as inputFile:
        answerId2phId_img = pickle.load( inputFile)
    
    with open(intermediate_directory+f"/answerId2phId_code.dict", 'rb') as inputFile:
        answerId2phId_code = pickle.load( inputFile)
    
    # to get the origianl question count when generating nus, load eventlist file
    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        ori_Questions = pickle.load( inputFile)
    ori_questionCount = len(ori_Questions)

    # prepare args
    args =[]
    for qid, d in ori_Questions.items():
        if qid in QidsWithData:
            args.append((qid, d, commName, commDir))

    # process Questions chunk by chunk
    n_proc = mp.cpu_count() # left 2 cores to do others
    with mp.Pool(processes=n_proc) as pool:
        # issue tasks to the process pool and wait for tasks to complete
        all_outputs = pool.starmap(myAction, args , chunksize=10)
    
    ori_Questions.clear()

    # combine all outputs
    sortingBase2Aid_total = defaultdict()

    for tup in all_outputs:
        qid, sortingBase2Aid = tup
        sortingBase2Aid_total.update(sortingBase2Aid)

    all_outputs.clear()

    assert(len(sortingBase2Aid_total) == dataSampleCount)

    # save the data parts
    with open(intermediate_directory+f"/sortingBase2Aid.dict", 'wb') as outputFile:
        pickle.dump(sortingBase2Aid_total, outputFile)
        log_text = f"saved sortingBase2Aid_total with total {len(sortingBase2Aid_total)} samples. "

    # create img and code columns of data
    imgAndCode = []
    imgCount = 0
    codeCount = 0
    for sortingBase, aid in sortingBase2Aid_total.items():
        img = 0
        code = 0
        if aid in answerId2phId_img.keys():
            img = 1
            imgCount += 1
        if aid in answerId2phId_code.keys():
            code = 1
            codeCount += 1
        imgAndCode.append([img, code])
    
    imgAndCode_data = lil_matrix( imgAndCode, dtype=np.float16 )

    # save imgAndCode data 
    with open(intermediate_directory+f"/imgAndCode_datasets_sorted_removeFirstRealVote.dict", 'wb') as outputFile:
        pickle.dump(imgAndCode_data, outputFile)
        log_text += f"saved imgAndCode_datasets_sorted_removeFirstRealVote.dict, imgCount:{imgCount}, codeCount:{codeCount} for {commName}"

        writeIntoLog(log_text, commDir, logFileName)
        print(log_text)



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
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        if commName in splitted_comms: # skip splitted big communities
            print(f"{commName} was splitted, skip")
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
    print('preprocessing 9   Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
