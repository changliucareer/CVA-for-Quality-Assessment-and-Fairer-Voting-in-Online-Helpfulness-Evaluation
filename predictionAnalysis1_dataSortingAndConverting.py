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

def myAction (percentage, partFileDir, commName, commDir, predictionAnalysis_data_folder):
    with open(partFileDir, 'rb') as inputFile:
        percentData= pickle.load( inputFile)
        print(f"raw data of percent{percentage} of {commName} loaded.")
    
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
    with open(predictionAnalysis_data_folder+f"/percent{percentage}PureData.dict", 'wb') as outputFile:
        pickle.dump((new_X, y, max_votesCountOfComm), outputFile)
        print( f"saved conveted data of percent{percentage} {commName}")


def myFun(commIndex, commName, commDir):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "predictionAnalysis1_dataSortingAndConverting_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # mkdir to keep predictionAnalysis data
    predictionAnalysis_data_folder = os.path.join(intermediate_directory, r'predictionAnalysis_data_folder')
    if not os.path.exists(predictionAnalysis_data_folder): # don't have dataset splitted intermediate data folder, skip
        print(f"{commName} has no  predictionAnalysis_data_folder.")

    # check whether already done this step, skip
    resultFiles = [f"percent{percentage}PureData.dict" for percentage in [25,50,75,100]]
    resultFiles = [predictionAnalysis_data_folder+'/'+f for f in resultFiles]
    # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    if all([os.path.exists(r) for r in resultFiles]):
        print(f"{commName} has already done this step.")
        return
    

    # prepare args
    args =[]
    for percentage in [25,50,75,100]:
        partFileDir = predictionAnalysis_data_folder+f"/percent{percentage}Data.dict"
        args.append((percentage, partFileDir, commName, commDir, predictionAnalysis_data_folder))

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
    print('prediction analysis 1 data sorting and converting  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
