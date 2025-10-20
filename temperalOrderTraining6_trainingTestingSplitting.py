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
    
    # extract universal time stamp index or vid as sorting base
    ai2sortingBaseList = defaultdict()

    # for counting vote diff before the last vote of each answer as the testing sample
    ai2voteList = defaultdict()

    firstVoteRemovedFlagForEachAnswer =  []  # the first vote of each answer was removed, must skip
    for i,e in enumerate(eventList):
        t = lts[i]
        if commName == 'stackoverflow':
            cur_year = t[2].year
            t = t[1] # which is vid, can be used to sort
        eventType = e['et']
        if eventType != 'v': # skip all event that is not a vote
            continue
        
        vote = e['v']
        if vote == 1:
            cur_vote = 1
        else: # vote == 0 is negative vote
            cur_vote = -1

        ai = e['ai']
        if ai in firstVoteRemovedFlagForEachAnswer: # current vote's target answer already has the first vote removed
            
            if ai in ai2sortingBaseList.keys():
                ai2sortingBaseList[ai] = ai2sortingBaseList[ai] + [t]
            else:
                ai2sortingBaseList[ai] = [t]

        else: # current vote is its target answer's first vote, don't use as sample
            firstVoteRemovedFlagForEachAnswer.append(ai)

        if ai in ai2voteList.keys():
            ai2voteList[ai].append(cur_vote)
        else:
            ai2voteList[ai] = [cur_vote]

        if i%1000==0:
            print(f"processed the {i}th event of question {qid} of {commName}")

    sortingBaseListInTesting = [] # the sortingBase of samples that in aids for testing
    aid2voteDiffBeforeLastTestingVote = defaultdict() # only contains the answers that have at least 2 votes (means at least 1 sample in the trainning data)
    for ai, sortingBaseList in ai2sortingBaseList.items():
        if len(sortingBaseList)<2: # don't use as testing
            aid2voteDiffBeforeLastTestingVote[filtered_answer_list[ai]] = sum(ai2voteList[ai])
            continue
        # only use the last vote of each answers as testing samples in whole dataset
        sortingBaseListInTesting.append(sortingBaseList[-1])
        aid2voteDiffBeforeLastTestingVote[filtered_answer_list[ai]] = sum(ai2voteList[ai][:-1])

    
    return qid, sortingBaseListInTesting, aid2voteDiffBeforeLastTestingVote


def myFun(commIndex, commName, commDir, splitted_comms):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "temperalOrderTraining6_trainingTestingSplittingAndExtractingVoteDiffBeforeTesting_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    if commName == 'stackoverflow': # using subcomm to represent stackoverflow
        subComms_data_folder = os.path.join(commDir, f'subCommunities_folder')
        ## Load all sub community direcotries 
        with open(subComms_data_folder+'/'+f'subCommName2subCommDir.dict', 'rb') as inputFile:
            subCommName2commDir = pickle.load( inputFile)
        subCommDir = subCommName2commDir['reactjs']
        subComm_intermediate_directory = os.path.join(subCommDir, r'intermediate_data_folder')

    # check whether already done this step, skip
    result_directory = intermediate_directory
    resultFiles = ['temperalOrderTraining6_outputs.dict']
    resultFiles = [result_directory+'/'+f for f in resultFiles]
    # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    if os.path.exists(resultFiles[0]):
        print(f"{commName} has already done this step.")
        return

    print(f"loading data... for {commName}")
    try:
        # get full data file list
        if commName != 'stackoverflow':
            if commName in splitted_comms: # for splitted comms
                splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
                split_datasets_intermediatefiles_directory = os.path.join(splitFolder_directory, r'Datasets_forEachQuestion_parts_folder')
                partFiles = [ f.path for f in os.scandir(split_datasets_intermediatefiles_directory) if f.path.split('/')[-1].startswith('whole_datasets_forEachQuestion_removeFirstRealVote')]
                partFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
                partsCount = len(partFiles)
                qidsInWholeDatasets = []
                for i, partDir in enumerate(partFiles):
                    part = i+1
                    print(f"scanning part {part}/{partsCount} of {commName}...")
                    with open(partDir, 'rb') as inputFile:
                        questions_Data_part = pickle.load( inputFile)
                        qidsInWholeDatasets.extend(list(questions_Data_part.keys()))
                        questions_Data_part.clear()
            else: # for not splitted comms
                with open(intermediate_directory+'/'+'whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'rb') as inputFile:
                    questions_Data = pickle.load( inputFile)
                    qidsInWholeDatasets = list(questions_Data.keys())
                    questions_Data.clear()
        else:
            # # in case using subcomm to represent stackoverflow
            # with open(subComm_intermediate_directory+'/'+'whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'rb') as inputFile:
            #     questions_Data = pickle.load( inputFile)
            #     qidsInWholeDatasets = list(questions_Data.keys())
            #     questions_Data.clear()

            # in case using the whole SOF
            splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
            split_datasets_intermediatefiles_directory = os.path.join(splitFolder_directory, r'Datasets_forEachQuestion_parts_folder')
            partFiles = [ f.path for f in os.scandir(split_datasets_intermediatefiles_directory) if f.path.split('/')[-1].startswith('whole_datasets_forEachQuestion_removeFirstRealVote')]
            partFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
            partsCount = len(partFiles)
            qidsInWholeDatasets = []
            for i, partDir in enumerate(partFiles):
                part = i+1
                print(f"scanning part {part}/{partsCount} of {commName}...")
                with open(partDir, 'rb') as inputFile:
                    questions_Data_part = pickle.load( inputFile)
                    qidsInWholeDatasets.extend(list(questions_Data_part.keys()))
                    questions_Data_part.clear()

    except Exception as e:
        print(f"for {commName} error when load the Questions data: {e}")
        return
    

    questionCount = len(qidsInWholeDatasets) # this not equal to the origianl question count when generating nus
    
    # to get the origianl question count when generating nus, load eventlist file
    if commName != 'stackoverflow':
        with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
            ori_Questions = pickle.load( inputFile)
            ori_questionCount = len(ori_Questions)
    else: 
        # # in case  using subcomm to represent stackoverflow
        # subComm_QuestionsWithEventList_directory = subCommDir+'/'+f'QuestionsWithEventList_tag_reactjs.dict'
        # with open(subComm_QuestionsWithEventList_directory, 'rb') as inputFile:
        #     ori_Questions = pickle.load( inputFile)
        #     ori_questionCount = len(ori_Questions)
        # in case using the whole SOF
        ori_questionCount = 410516

    print(f"{commName} has {questionCount} questions in training data, but has {ori_questionCount} questions corresponding to nus.")


    if commName != 'stackoverflow':
        # prepare args
        args =[]
        for qid, d in ori_Questions.items():
            if qid not in qidsInWholeDatasets:
                print(f"skip question {qid} which is not in whole dataset for {commName}!")
                continue
            args.append((qid, d, commName, commDir))

        # process Questions chunk by chunk
        n_proc = mp.cpu_count() # left 2 cores to do others
        with mp.Pool(processes=n_proc) as pool:
            # issue tasks to the process pool and wait for tasks to complete
            all_outputs = pool.starmap(myAction, args , chunksize=10)
        
        ori_Questions.clear()
    
    else: # for stackoverflow whole
        all_outputs = []
        splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
        split_QuestionsWithEventList_files_directory = os.path.join(splitFolder_directory, r'QuestionsPartsWithEventList')
        if not os.path.exists(split_QuestionsWithEventList_files_directory): # didn't find the parts files
            print("Exception: no split_QuestionsWithEventList_files_directory!")
        partFiles = [ f.path for f in os.scandir(split_QuestionsWithEventList_files_directory) if f.path.endswith('.dict') ]
        # sort csvFiles paths based on part number
        partFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
        partsCount = len(partFiles)
        assert partsCount == int(partFiles[-1].strip(".dict").split("_")[-1]) # last part file's part number should equal to the parts count                          
        print(f"there are {partsCount} splitted event list files in {commName}")
        for i, subDir in enumerate(partFiles):
            part = i+1
            partDir = subDir
            # get question count of each part
            with open(partDir, 'rb') as inputFile:
                Questions_part = pickle.load( inputFile)
                # prepare args for cur part
                args =[]
                for qid, d in Questions_part.items():
                    if qid not in qidsInWholeDatasets:
                        print(f"skip question {qid} which is not in whole dataset for {commName}!")
                        continue
                    args.append((qid, d, commName, commDir))

                # process Questions chunk by chunk
                n_proc = mp.cpu_count() # left 2 cores to do others
                with mp.Pool(processes=n_proc) as pool:
                    # issue tasks to the process pool and wait for tasks to complete
                    part_outputs = pool.starmap(myAction, args , chunksize=10)
                    all_outputs.extend(part_outputs)
                Questions_part.clear()


    # combine all_outputs
    print(f"combining all_outputs for {commName}...")
    qid2TestingSortingBaseList = defaultdict()
    qid2aid2voteDiffBeforeLastTestingVote = defaultdict()
    for tup in all_outputs:
        qid, sortingBaseListInTesting, aid2voteDiffBeforeLastTestingVote = tup
        qid2TestingSortingBaseList[qid] = sortingBaseListInTesting
        qid2aid2voteDiffBeforeLastTestingVote[qid] = aid2voteDiffBeforeLastTestingVote

    # save the testing aids and sortingBase List
    with open(intermediate_directory+f"/temperalOrderTraining6_outputs.dict", 'wb') as outputFile:
        pickle.dump((qid2TestingSortingBaseList, qid2aid2voteDiffBeforeLastTestingVote), outputFile)
        print( f"saved qid2TestingSortingBaseList of {commName}")
    


def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    """
    try:
        # test on comm "coffee.stackexchange" to debug
        # myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1],splitted_comms )
        # test on comm "datascience.stackexchange" to debug
        # myFun(301,commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1],splitted_comms)
        # test on comm "webapps.stackexchange" to debug
        # myFun(305,commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1],splitted_comms)
        # test on comm "travel.stackexchange" to debug
        # myFun(319,commDir_sizes_sortedlist[319][0], commDir_sizes_sortedlist[319][1],splitted_comms)
        # test on comm "stackoverflow" to debug
        # myFun(359,commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1],splitted_comms)
        # test on comm "math" to debug
        # myFun(358,commDir_sizes_sortedlist[358][0], commDir_sizes_sortedlist[358][1], splitted_comms)
        # test on comm "stackoverflow" to debug
        # myFun(359,commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1], splitted_comms)
    except Exception as e:
        print(e)
        sys.exit()
    """
    
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        try:
            p = mp.Process(target=myFun, args=(commIndex, commName,commDir, splitted_comms))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()
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
    print('temperalOrderTraing6 training/testing dividing  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
