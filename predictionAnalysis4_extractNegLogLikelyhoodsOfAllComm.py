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
import math
import scipy.stats

def myFun(commName, commDir, return_trainSuccess_dict):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    try:
        # get total voteCount
        with open(intermediate_directory+f"/predictionAnalysis_dataSortingAndDividingOutputs.dict", 'rb') as inputFile:
            tuple = pickle.load( inputFile)
            voteCountOfEachAnswerBeforeRemoveTheFirstVote_total, qid2sortingBaseList = tuple
            total_voteCount = sum(voteCountOfEachAnswerBeforeRemoveTheFirstVote_total)

        with open(intermediate_directory+f"/predictionAnalysis2_newModel_return.dict", 'rb') as inputFile:
            return_dict_newModel = pickle.load( inputFile)
        with open(intermediate_directory+f"/predictionAnalysis3_CVP_return.dict", 'rb') as inputFile:
            return_dict_CVP = pickle.load( inputFile)
        
        return_trainSuccess_dict[commName] = {'newModel':return_dict_newModel, 'CVP':return_dict_CVP, 'total_voteCount':total_voteCount}
        print(f"saved return_trainSuccess_dict for {commName}")
    except:
        print(f"no result dict file for {commName}")
   
def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    return_trainSuccess_dict = manager.dict() # to save the used train mode (wholebatch or minibatch) of each community


    # splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        if commName in splitted_comms:
            print(f"{commName} is splitted, skip")
            continue
        
        try:
            p = mp.Process(target=myFun, args=(commName,commDir,return_trainSuccess_dict))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()

        processes.append(p)
        if len(processes)==16:
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
    
    # save 
    return_trainSuccess_normalDict = defaultdict()
    for commName, d in return_trainSuccess_dict.items():
        return_trainSuccess_normalDict[commName] = {'newModel':d['newModel'], 'CVP':d['CVP'], 'total_voteCount':d['total_voteCount']}
    os.chdir(root_dir) # go back to root directory
    with open('predictionAnalysis.dict', 'wb') as outputFile:
        pickle.dump(return_trainSuccess_normalDict, outputFile)
        print(f"saved return_trainSuccess_normalDict")
        
    statisticsList = []
    for commName, content in return_trainSuccess_dict.items():
        total_voteCount = content['total_voteCount']
        return_dict_newModel = content['newModel']
        return_dict_CVP = content['CVP']
        statisticsList.append((commName, total_voteCount,
                               return_dict_newModel[(25,50)]['avg_negloglikelyhood'],return_dict_CVP[(25,50)]['avg_negloglikelyhood'],
                               return_dict_newModel[(50,75)]['avg_negloglikelyhood'],return_dict_CVP[(50,75)]['avg_negloglikelyhood'],
                               return_dict_newModel[(75,100)]['avg_negloglikelyhood'],return_dict_CVP[(75,100)]['avg_negloglikelyhood'],
                               return_dict_newModel[(25,50)]['test_accuracy'],return_dict_CVP[(25,50)]['test_accuracy'],
                               return_dict_newModel[(50,75)]['test_accuracy'],return_dict_CVP[(50,75)]['test_accuracy'],
                               return_dict_newModel[(75,100)]['test_accuracy'],return_dict_CVP[(75,100)]['test_accuracy']))


    import csv
    print(f"start to save the results as csv...")
    with open('predictionAnalysis.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","total_voteCount", 
                          "newModel_train25%/test50%_avgNLL","CVP_train25%/test50%_avgNLL",
                          "newModel_train50%/test75%_avgNLL","CVP_train50%/test75%_avgNLL",
                          "newModel_train75%/test100%_avgNLL","CVP_train75%/test100%_avgNLL",
                          "newModel_train25%/test50%_accuracy","CVP_train25%/test50%_accuracy",
                          "newModel_train50%/test75%_accuracy","CVP_train50%/test75%_accuracy",
                          "newModel_train75%/test100%_accuracy","CVP_train75%/test100%_accuracy",])

        for sl in statisticsList:
            writer.writerow(sl)
    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('predictionAnalysis4_extractNegLogLikelyhoodsOfAllComm Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
