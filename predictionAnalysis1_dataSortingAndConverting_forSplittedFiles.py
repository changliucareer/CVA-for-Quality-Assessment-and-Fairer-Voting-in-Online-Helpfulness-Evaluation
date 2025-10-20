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
import tensorflow as tf
from collections import Counter

#################################################################
def my_iterable(batchFileDir):
    with open(batchFileDir, 'rb') as inputFile:
        percentData= pickle.load( inputFile)
        for tup in percentData:
            yield tup

def myAction (percentage, batchFileDirList, commName, commDir, predictionAnalysis_data_folder):

    # sorting within each batch file
    batchFileSizes = []
    for i, subDir in enumerate(batchFileDirList):
        batch = i+1
        print(f"sorting within batch {batch} of percent {percentage} data of {commName}...")
        batchFileDir = subDir
        with open(batchFileDir, 'rb') as inputFile:
            percentData= pickle.load( inputFile)
            batchFileSizes.append(len(percentData))
            print(f"raw data of percent{percentage} batch {batch} of {commName} loaded.")
        
        # sort by sorting base
        percentData.sort(key=lambda t:t[0])

        # store the sorted batch file
        with open(batchFileDir, 'wb') as outputFile:
            pickle.dump(percentData, outputFile)
            print( f"saved sorted data of percent{percentage} batch {batch} of {commName}.")
    
    assert len(batchFileSizes) == len(batchFileDirList)


    if len(batchFileSizes) > 1: # more than one batch, sort cross batch files using bucket sorting
        print(f"{len(batchFileSizes)} batches, sorting cross batch files for {commName} percent {percentage}...")
        # sorting cross batch files
        batch_iterables = [my_iterable(batchFileDir) for batchFileDir in batchFileDirList] # generators of each batch file

        cur_sorted_Batch = []
        cur_batch_size = batchFileSizes[0] # initialized as the first original batch size
        cur_batch = 1
        buckets = [next(iter) for iter in batch_iterables] # initialize with the first tup of each batch file

        while(True):
            # get the min of current bucket
            copy_buckets = [(i,b) for i,b in enumerate(buckets)]
            bucketIndex, cur_min_tup = sorted(copy_buckets, key=lambda t : float('inf') if t[1]==None else t[1][0])[0]
            cur_sorted_Batch.append(cur_min_tup)

            if len(cur_sorted_Batch) == cur_batch_size: # current batch full
                print(f"start to convert data for {commName} percent {percentage} batch {cur_batch}...")
                # extract only data sample
                X = []
                y = []
                for tup in cur_sorted_Batch:
                    cur_x = tup[1][0].todense()
                    cur_y = tup[1][1]
                    max_votesCountOfComm = tup[1][2]
                    if len(X)==0:
                        X = cur_x
                    else:
                        X = np.vstack((X,cur_x))
                    y.append(cur_y)
                
                new_X = lil_matrix( X, dtype=np.float16 )

                # save new version of part data for current batch
                with open(predictionAnalysis_data_folder+f"/precent{percentage}PureData_batch_{cur_batch}.dict", 'wb') as outputFile:
                    pickle.dump((new_X, y, max_votesCountOfComm), outputFile)
                    print( f"saved conveted data of percent{percentage} batch {cur_batch} {commName}")
                
                if cur_batch < len(batchFileSizes):
                    cur_batch_size = batchFileSizes[cur_batch] # update cur_batch size
                    cur_batch += 1 # update cur_batch number
                    cur_sorted_Batch = []
            
            # refill the min bucket 
            try:
                buckets[bucketIndex] = next(batch_iterables[bucketIndex])
            except: # no more for this batch file
                buckets[bucketIndex] = None
                # check whether all the buckets are empty
                if sum([0 if b==None else 1 for b in buckets]) == 1: # only remain the last batch, add this sample and stop loop
                    remainBucketIndex = [i for i,b in enumerate(buckets) if b!=None][0]
                    cur_sorted_Batch.append(buckets[remainBucketIndex])
                    break
        
        # add remaining data samples in the remaining batch file
        while (True):
            try:
                tup = next(batch_iterables[remainBucketIndex])
                cur_sorted_Batch.append(tup)
            except:
                break


        # process the last batch
        if len(cur_sorted_Batch) >0:
            print(f"start to convert data for {commName} percent {percentage} batch {cur_batch}...")
            # extract only data sample
            X = []
            y = []
            for tup in cur_sorted_Batch:
                cur_x = tup[1][0].todense()
                cur_y = tup[1][1]
                max_votesCountOfComm = tup[1][2]
                if len(X)==0:
                    X = cur_x
                else:
                    X = np.vstack((X,cur_x))
                y.append(cur_y)
            
            new_X = lil_matrix( X, dtype=np.float16 )

            # save new version of part data for current batch
            with open(predictionAnalysis_data_folder+f"/precent{percentage}PureData_batch_{cur_batch}.dict", 'wb') as outputFile:
                pickle.dump((new_X, y, max_votesCountOfComm), outputFile)
                print( f"saved conveted data of percent{percentage} batch {cur_batch} {commName}")

    else: # only one batch, directly load and sort it
        cur_batch = 1
        with open(batchFileDir, 'rb') as inputFile:
            percentData= pickle.load( inputFile)
        
        print(f"only 1 batch for {commName} percent {percentage}, start to convert the data...")
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

        # save new version of part data for current batch
        with open(predictionAnalysis_data_folder+f"/precent{percentage}PureData_batch_{cur_batch}.dict", 'wb') as outputFile:
            pickle.dump((new_X, y, max_votesCountOfComm), outputFile)
            print( f"saved conveted data of percent{percentage} batch {cur_batch} {commName}")


def myFun(commIndex, commName, commDir):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "predictionAnalysis1_dataSortingAndConverting_forSplittedFiles_Log.txt"

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

    # mkdir to keep predictionAnalysis data
    predictionAnalysis_data_folder = os.path.join(intermediate_directory, r'predictionAnalysis_data_folder')
    if not os.path.exists(predictionAnalysis_data_folder): # don't have dataset splitted intermediate data folder, skip
        print(f"{commName} has no  predictionAnalysis_data_folder.")
    

    # prepare args
    args =[]
    for percentage in [25,50,75,100]:
        batchFileDirList = [ f.path for f in os.scandir(predictionAnalysis_data_folder) if f.path.split('/')[-1].startswith(f"percent{percentage}Data_batch_")]
        batchFileDirList.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
        args.append((percentage, batchFileDirList, commName, commDir, predictionAnalysis_data_folder))

    # process Questions chunk by chunk
    n_proc = mp.cpu_count() # left 2 cores to do others
    with mp.Pool(processes=n_proc) as pool:
        # issue tasks to the process pool and wait for tasks to complete
        pool.starmap(myAction, args )

def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    
    try:
        # test on comm "coffee.stackexchange" to debug
        # myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
        # test on comm "datascience.stackexchange" to debug
        # myFun(301,commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
        # test on comm "webapps.stackexchange" to debug
        # myFun(305,commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
        # test on comm "math.stackexchange" to debug
        myFun(358,commDir_sizes_sortedlist[358][0], commDir_sizes_sortedlist[358][1])
    except Exception as e:
        print(e)
        sys.exit()
    """
     # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist[:359]):
        commName = tup[0]
        commDir = tup[1]

        if commName not in splitted_comms: # skip splitted big communities
            print(f"{commName} was not splitted, skip.")
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
    print('prediction analysis 1 data sorting and converting  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
