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
from scipy.sparse import csr_matrix, lil_matrix
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
from collections import Counter

#################################################################

def myFun(commIndex, commName, commDir):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "temperalOrderTraning0_sortingData_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # check whether already done this step, skip
    result_directory = intermediate_directory
    resultFiles = ['whole_datasets_sorted_removeFirstRealVote.dict']
    resultFiles = [result_directory+'/'+f for f in resultFiles]
    if all([os.path.exists(rf) for rf in resultFiles]):
        print(f"{commName} has already done this step.")
        return

    print(f"loading data... for {commName}")
    with open(intermediate_directory+'/'+'whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'rb') as inputFile:
        questions_Data = pickle.load( inputFile)
    
    # load sorting base list
    with open(intermediate_directory+f"/predictionAnalysis_dataSortingAndDividingOutputs.dict", 'rb') as inputFile:
        tup = pickle.load( inputFile)
        _, qid2sortingBaseList = tup
    
    # sort the sample index according to sorting base
    sorted_qidAndSampleIndex = []
    for qid, sortingBaseList in qid2sortingBaseList.items():
        if qid not in questions_Data.keys():
            print("Exception!")
        
        assert questions_Data[qid][0].shape[0] == len(sortingBaseList) # X sample count should equals sorting base list length

        for i, sortingBase in enumerate(sortingBaseList):
            sorted_qidAndSampleIndex.append((qid,i,sortingBase))
    
    sorted_qidAndSampleIndex.sort(key=lambda t:t[2])

    # save the sorted_qidAndSampleIndex
    with open(intermediate_directory+f"/sorted_qidAndSampleIndexAndSortingBase.dict", 'wb') as outputFile:
        pickle.dump(sorted_qidAndSampleIndex, outputFile)

    print(f"sorted_qidAndSampleIndex is ready for {commName}, start to rearrange training data...")
    totalSampleCount = len(sorted_qidAndSampleIndex)

    # rearrange training data according to sorted_qidAndSampleIndex
    new_X = []
    new_y = []
    max_votesCountOfComm = None
    sampleIndex = 0

    for tup in sorted_qidAndSampleIndex:
        qid, i, _ = tup
        X,y,max_votesCountOfComm = questions_Data[qid]
        new_y.append(y[i])
        X = X.todense()
        cur_x = X[i]
        if len(new_X)==0:
            new_X = cur_x
        else:
            new_X = np.vstack((new_X,cur_x))
        sampleIndex +=1
        
        if sampleIndex%100 == 0: # report progress every 100 samples
            print(f"added {sampleIndex}th/{totalSampleCount} sample to {commName}")

    questions_Data.clear()

    new_X = lil_matrix( new_X, dtype=np.float16 )

    # save whole sorted data
    with open(intermediate_directory+f"/whole_datasets_sorted_removeFirstRealVote.dict", 'wb') as outputFile:
        pickle.dump((new_X, new_y, max_votesCountOfComm), outputFile)
        print( f"saved whole_datasets_sorted_removeFirstRealVote.dict (X shape: {new_X.shape}, y length: {len(new_y)}) for {commName}")

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
    print('temperal order 0 sorting data Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
