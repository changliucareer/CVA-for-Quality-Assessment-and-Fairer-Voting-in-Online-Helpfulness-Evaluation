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

def myFun(commName, commDir, return_trainSuccess_dict, modelType):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    try:
        if modelType == 'newModel':
            with open(intermediate_directory+f"/temperalOrderTraining1_newModel_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_comm = pickle.load( inputFile)
        elif modelType == 'CVP': # CVP
            with open(intermediate_directory+f"/temperalOrderTraining3_CVP_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_comm = pickle.load( inputFile)
        elif modelType == 'newModel_interaction_withD': 
            with open(intermediate_directory+f"/temperalOrderTraining4_newModel_interactionWithD_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_comm = pickle.load( inputFile)
        elif modelType == 'newModel_interaction_withReciprocalD': 
            with open(intermediate_directory+f"/temperalOrderTraining4_newModel_interactionWithReciprocalD_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_comm = pickle.load( inputFile)
        elif modelType == 'newModel_bias_interaction_withD': 
            with open(intermediate_directory+f"/temperalOrderTraining4_newModel_bias_interactionWithD_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_comm = pickle.load( inputFile)
        elif modelType == 'newModel_bias_interaction_withReciprocalD': 
            with open(intermediate_directory+f"/temperalOrderTraining4_newModel_bias_interactionWithReciprocalD_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_comm = pickle.load( inputFile)
        elif modelType == 'newModel_interaction_withD_imgAndCode': 
            with open(intermediate_directory+f"/temperalOrderTraining5_newModel_interactionWithD_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_comm = pickle.load( inputFile)
        elif modelType == 'newModel_interaction_withReciprocalD_imgAndCode': 
            with open(intermediate_directory+f"/temperalOrderTraining5_newModel_interactionWithReciprocalD_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_comm = pickle.load( inputFile)
        elif modelType == 'newModel_bias_interaction_withD_imgAndCode': 
            with open(intermediate_directory+f"/temperalOrderTraining5_newModel_bias_interactionWithD_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_comm = pickle.load( inputFile)
        elif modelType == 'newModel_bias_interaction_withReciprocalD_imgAndCode': 
            with open(intermediate_directory+f"/temperalOrderTraining5_newModel_bias_interactionWithReciprocalD_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_comm = pickle.load( inputFile)
        
        return_trainSuccess_dict[commName] = return_trainSuccess_dict_comm[commName]
    except:
        print(f"no result dict file for {commName}")
   
def main():

    t0=time.time()
    root_dir = os.getcwd()

    # modelType = 'newModel'
    # modelType = 'CVP'
    # modelType = 'newModel_interaction_withD'
    # modelType = 'newModel_interaction_withReciprocalD'
    # modelType = 'newModel_bias_interaction_withD'
    # modelType = 'newModel_bias_interaction_withReciprocalD'
    modelType = 'newModel_interaction_withD_imgAndCode'
    modelType = 'newModel_interaction_withReciprocalD_imgAndCode'
    modelType = 'newModel_bias_interaction_withD_imgAndCode'
    modelType = 'newModel_bias_interaction_withReciprocalD_imgAndCode'

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

        # if commName in splitted_comms:
        #     print(f"{commName} is splitted, skip")
        #     continue
        
        try:
            p = mp.Process(target=myFun, args=(commName,commDir,return_trainSuccess_dict, modelType))
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
    
    # save 
    return_trainSuccess_normalDict = defaultdict()
    if modelType == 'CVP':
        for commName, d in return_trainSuccess_dict.items():
            return_trainSuccess_normalDict[commName] = {'dataShape':(d['dataShape'][0], d['dataShape'][1]), 'trainLosses':d['trainLosses'], 'testLosses':d['testLosses'], 'epoch':d['epoch'], 'lowest_test_loss':d['lowest_test_loss'], 'testAcc':d['testAcc'],
                                            'coefs':d['coefs'],'bias':d['bias'], 'nus':d['nus'], 'qs':d['qs'],
                                            'coefs_withFullData':d['coefs_withFullData'],'bias_withFullData':d['bias_withFullData'], 'nus_withFullData':d['nus_withFullData'], 'qs_withFullData':d['qs_withFullData'],
                                            'conformity':d['conformity']}
            
    else: # 'newModel'
        for commName, d in return_trainSuccess_dict.items():
            return_trainSuccess_normalDict[commName] = {'dataShape':(d['dataShape'][0], d['dataShape'][1]), 'trainLosses':d['trainLosses'], 'testLosses':d['testLosses'], 'epoch':d['epoch'], 'lowest_test_loss':d['lowest_test_loss'], 'testAcc':d['testAcc'],
                                            'coefs':d['coefs'],'bias':d['bias'],'tau':d['tau'], 'nus':d['nus'], 'qs':d['qs'],
                                            'coefs_withFullData':d['coefs_withFullData'],'bias_withFullData':d['bias_withFullData'],'tau_withFullData':d['tau_withFullData'], 'nus_withFullData':d['nus_withFullData'], 'qs_withFullData':d['qs_withFullData'],
                                            'conformity':d['conformity']}
            
    os.chdir(root_dir) # go back to root directory
    with open(f'temperalOrderTraining_{modelType}_return.dict', 'wb') as outputFile:
        pickle.dump(return_trainSuccess_normalDict, outputFile)
        print(f"saved return_trainSuccess_normalDict of {modelType}, with {len(return_trainSuccess_normalDict)} comms.")
    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('temperalOrderTraining2_extractResultsOfAllComms Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
