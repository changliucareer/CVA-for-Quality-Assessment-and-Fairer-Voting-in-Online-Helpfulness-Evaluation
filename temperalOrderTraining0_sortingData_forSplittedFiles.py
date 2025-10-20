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

    logFileName = "temperalOrderTraning0_sortingData_forSplitted_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    splitted_intermediate_data_folder = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    if not os.path.exists(splitted_intermediate_data_folder): # don't have splitted intermediate data folder, skip
        print(f"{commName} has no splitted intermediate data folder.")
        return
    

    split_datasets_intermediatefiles_directory = os.path.join(splitted_intermediate_data_folder, r'Datasets_forEachQuestion_parts_folder')
    if not os.path.exists(split_datasets_intermediatefiles_directory): # don't have dataset splitted intermediate data folder, skip
        print(f"{commName} has no splitted datasets folder.")
        return


    partFiles = [ f.path for f in os.scandir(split_datasets_intermediatefiles_directory) if f.path.split('/')[-1].startswith('whole_datasets_forEachQuestion_removeFirstRealVote')]
    # sort csvFiles paths based on part number
    partFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
    partsCount = len(partFiles)
    print(f"found {partsCount} parts of intermediate data for {commName}")
    
    # load sorting base list
    with open(intermediate_directory+f"/predictionAnalysis_dataSortingAndDividingOutputs.dict", 'rb') as inputFile:
        tup = pickle.load( inputFile)
        _, qid2sortingBaseList = tup

    # make a folder to store temperal ordered data files
    final_directory = os.path.join(splitted_intermediate_data_folder, r'Datasets_sorted_batches_folder')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # check whether already done this step, skip
    resultFileDir = os.path.join(commDir, r'log_folder')+'/'+ logFileName
    # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    if os.path.exists(resultFileDir):
        f = open(resultFileDir, "r")
        lines = f.readlines()
        for line in lines:
            if "all batches complete." in line:
                print(f"{commName} has already done this step.")
                return
        f.close()
    

    # get qid2partFileDir
    qid2partFileDir = defaultdict()
    for i, partDir in enumerate(partFiles):
        part = i+1
        print(f"scanning part {part} of {commName}...")
        with open(partDir, 'rb') as inputFile:
            questions_Data_part = pickle.load( inputFile)
            for qid in questions_Data_part.keys():
                qid2partFileDir[qid] = partDir
        

    # sort the sample index according to sorting base
    sorted_qidAndSampleIndex = []
    for qid, sortingBaseList in qid2sortingBaseList.items():
        if qid not in qid2partFileDir.keys():
            print("Exception!")

        for i, sortingBase in enumerate(sortingBaseList):
            sorted_qidAndSampleIndex.append((qid, qid2partFileDir[qid], i,sortingBase))
    
    sorted_qidAndSampleIndex.sort(key=lambda t:t[3])
    # save the sorted_qidAndSampleIndex
    with open(intermediate_directory+f"/sorted_qidAndSampleIndexAndSortingBase.dict", 'wb') as outputFile:
        pickle.dump(sorted_qidAndSampleIndex, outputFile)

    print(f"sorted_qidAndSampleIndex is ready for {commName}, start to rearrange training data...")
    totalSampleCount = len(sorted_qidAndSampleIndex)

    # rearrange training data according to sorted_qidAndSampleIndex
    # batchSize = 200 # for debug with coffee
    batchSize = 100000
    new_X = []
    new_y = []
    max_votesCountOfComm = None
    batch = 1
    previousPartDir = None
    sampleIndex = 0
    for tup in sorted_qidAndSampleIndex:
        qid, partDir, i, _ = tup
        if partDir != previousPartDir:
            with open(partDir, 'rb') as inputFile:
                questions_Data_part = pickle.load( inputFile)
                previousPartDir = partDir
        X,y,max_votesCountOfComm = questions_Data_part[qid]
        new_y.append(y[i])
        X = X.todense()
        cur_x = X[i]
        if len(new_X)==0:
            new_X = cur_x
        else:
            new_X = np.vstack((new_X,cur_x))
        sampleIndex +=1
        
        if sampleIndex%100 == 0: # report progress every 100 samples
            print(f"added {sampleIndex+1}th/{totalSampleCount} sample to {commName} batch {batch}")
        
        if new_X.shape[0] == batchSize: # current batch is full, save as a file
            new_X = lil_matrix( new_X, dtype=np.float16 )
            # save whole sorted data
            with open(final_directory+f"/whole_datasets_sorted_removeFirstRealVote_batch_{batch}.dict", 'wb') as outputFile:
                pickle.dump((new_X, new_y, max_votesCountOfComm), outputFile)
                log_text = f"saved whole_datasets_sorted_removeFirstRealVote_batch_{batch}.dict (X shape: {new_X.shape}, y length: {len(new_y)})\n"
                writeIntoLog(log_text, commDir, logFileName)
                print(f"for {commName}, "+log_text)
            
            #clear new_X
            new_X = []
            new_y = []
            max_votesCountOfComm = None
            # update batch
            batch += 1
    
    # save the last batch file
    if len(y)>0:
        new_X = lil_matrix( new_X, dtype=np.float16 )
        # save whole sorted data
        with open(final_directory+f"/whole_datasets_sorted_removeFirstRealVote_batch_{batch}.dict", 'wb') as outputFile:
            pickle.dump((new_X, new_y, max_votesCountOfComm), outputFile)
            log_text = f"saved whole_datasets_sorted_removeFirstRealVote_batch_{batch}.dict (X shape: {new_X.shape}, y length: {len(new_y)})\n"
            writeIntoLog(log_text, commDir, logFileName)
            print(f"for {commName}, "+log_text)

    writeIntoLog(f"\nall batches complete.\n", commDir, logFileName)
    questions_Data_part.clear()

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
        # myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
        # test on comm "datascience.stackexchange" to debug
        # myFun(301,commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
        # test on comm "webapps.stackexchange" to debug
        # myFun(305,commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
        # test on comm "travel.stackexchange" to debug
        # myFun(319,commDir_sizes_sortedlist[319][0], commDir_sizes_sortedlist[319][1])
        # test on comm "stackoverflow" (5% samples) to debug
        myFun(359,commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1])
    except Exception as e:
        print(e)
        sys.exit()
    """
     # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    selected_comms = ['meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange']
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        if commName not in splitted_comms: 
            print(f"{commName} was not splitted, skip.")
            continue

        if commName not in selected_comms: 
            print(f"{commName} was not seleceted, skip.")
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
        if len(processes)==8:
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
    print('temperal order trainning data sorting for splitted files Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
