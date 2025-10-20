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

def myFun(commName, commDir, roundIndex, selected_reg_strengthList, variation, CVPgenerated):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "semiSynthetic7_sortingData_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    
    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    for reg_alpha in selected_reg_strengthList:

        # # check whether already done this step, skip
        # resultFiles = [f'semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_sorted_removeFirstRealVote.dict']
        # resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
        # if os.path.exists(resultFiles[0]):
        #     # target date
        #     target_date = datetime.datetime(2024, 12, 16)
        #     # file last modification time
        #     timestamp = os.path.getmtime(resultFiles[0])
        #     # convert timestamp into DateTime object
        #     datestamp = datetime.datetime.fromtimestamp(timestamp)
        #     print(f'{commName} Modified Date/Time:{datestamp}')
        #     if datestamp >= target_date:
        #         print(f"{commName} has already done this step.")
        #         return

        print(f"loading data... for {commName}")
        if CVPgenerated:
            with open(intermediate_directory+'/'+f'semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'rb') as inputFile:
                questions_Data = pickle.load( inputFile)
        
            # load sorting base list
            with open(intermediate_directory+f"/semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha})_qid2sortingBaseList_removeFirstRealVote.dict", 'rb') as inputFile:
                qid2sortingBaseList = pickle.load( inputFile)
        
        else: # new model generated
            with open(intermediate_directory+'/'+f'semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'rb') as inputFile:
                questions_Data = pickle.load( inputFile)
        
            # load sorting base list
            with open(intermediate_directory+f"/semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha})_qid2sortingBaseList_removeFirstRealVote.dict", 'rb') as inputFile:
                qid2sortingBaseList = pickle.load( inputFile)
        
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
        if CVPgenerated:
            with open(intermediate_directory+f"/semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha})_sorted_qidAndSampleIndexAndSortingBase.dict", 'wb') as outputFile:
                pickle.dump(sorted_qidAndSampleIndex, outputFile)
        else:
            with open(intermediate_directory+f"/semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha})_sorted_qidAndSampleIndexAndSortingBase.dict", 'wb') as outputFile:
                pickle.dump(sorted_qidAndSampleIndex, outputFile)

        print(f"sorted_qidAndSampleIndex of reg_alpha({reg_alpha}) is ready for {commName}, start to rearrange training data...")
        totalSampleCount = len(sorted_qidAndSampleIndex)

        # rearrange training data according to sorted_qidAndSampleIndex
        new_X = []
        new_y = []
        max_votesCountOfComm = None
        sampleIndex = 0

        for tup in sorted_qidAndSampleIndex:
            qid, i, _ = tup
            X,y,max_votesCountOfComm = questions_Data[qid]
            if y[i]== -1:
                new_y.append(0)
            elif y[i] == 1:
                new_y.append(1)
            else:
                print(f"EXCEPTION: raw y is not 1 nor 0!  for {commName} tup ({tup}) of sorted_qidAndSampleIndex")
                return
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
        if CVPgenerated:
            with open(intermediate_directory+f"/semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_sorted_removeFirstRealVote.dict", 'wb') as outputFile:
                pickle.dump((new_X, new_y, max_votesCountOfComm), outputFile)
        else: # new model generated
            with open(intermediate_directory+f"/semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_sorted_removeFirstRealVote.dict", 'wb') as outputFile:
                pickle.dump((new_X, new_y, max_votesCountOfComm), outputFile)
            print( f"saved semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_sorted_removeFirstRealVote.dict (X shape: {new_X.shape}, y length: {len(new_y)}) for {commName}")

def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

     ### using CVP to generate semisynthetic dataset
    # CVPgenerated = True
    # roundIndex = 1 # sampling round 
    # roundIndex = 2  # one question one answer
    # roundIndex = 3  # one question one answer, 100 times vote count per answer
    # roundIndex = 4  # one question one answer, 1000 times vote count per answer, q_std = 1
    # roundIndex = 5 # one question one answer, 1000 times vote count per answer, q_std = 1, fix lambda
    # roundIndex = 6 # one question one answer for toy example +++---, generate 1000 votes, fix lambda = -0.14189369976520538  prior_beta: -0.07276562601327896 prior_q_0: -0.14553125202655792 
    # roundIndex = 7 # one question one answer for toy example +++---, generate 1000 votes, fix lambda: -0.2684609591960907, beta: -0.058872684836387634, q_0: -0.11774536967277527
    # variation = '_fixedTau_noRL'
    # roundIndex = 8 # one question one answer, 1000 times vote count per answer, q_std = 1, fix lambda (for different regularization strength)
    # try_reg_strengthList = [0.005, 0.05, 0.5, 5, 50, 500, 5000]
    
    # roundIndex = 9 # one question one answer, 1000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    # roundIndex = 10 # one question one answer, 5000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    # roundIndex = 11 # one question one answer, 10000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    #roundIndex = 12 # one question one answer, 50000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    
    # roundIndex = 13 # one question one answer, 1000 votes, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    # roundIndex = 14 # one question one answer, 5000 votes, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    # roundIndex = 15 # one question one answer, 10000 votes, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    
    # roundIndex = 16 # one question multiple answer, 10000 events, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]

    # roundIndex = 17 # multiple question multiple answer, amplified 10 times of original total event count, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]

    # roundIndex = 18 # multiple question multiple answer, amplified 10 times of original total event count, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    # if roundIndex in [18]:
    #     variation = '_noRL'

    # selected_reg_strengthList = [300, 500, 700]

    # roundIndex = 19 ## multiple question multiple answer, original total event count, fix tau = 1, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    # roundIndex = 20 ## multiple question multiple answer, original total event count, learn tau, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    # if roundIndex in [19, 20]:
    #     variation = ''

    # selected_reg_strengthList = [500, 700]
    
    # roundIndex = 21 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # commName2selected_reg_strengthList = {'cstheory.stackexchange':[800, 900, 1000],
    #                                       'stackoverflow':[1000],
    #                                       'unix.meta.stackexchange':[60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
    #                                       'politics.stackexchange':[900,1000]}
    # commName2selected_reg_strengthList = {'3dprinting.stackexchange':[30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700],
    #                                       'latin.stackexchange':[50, 60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
    #                                       'meta.askubuntu':[70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
    #                                       'lifehacks.stackexchange':[400, 500, 600, 700, 800,900,1000]}
    # variation = ''

    ### using new Model to generate semi-synthetic dataset statistics
    CVPgenerated = False
    # roundIndex = 1 ## multiple question multiple answer, original total event count, fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # roundIndex = 2 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # roundIndex = 3 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and DOUBLED beta (for different regularization strength) selected_reg_strengthList of each comm
    roundIndex = 4 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and TRIPLED beta (for different regularization strength) selected_reg_strengthList of each comm


    commName2selected_reg_strengthList = {
                                        #   '3dprinting.stackexchange':[20, 30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700],
                                        #   'latin.stackexchange':[40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                        #   'meta.askubuntu':[20, 30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                        #   'lifehacks.stackexchange':[300, 400, 500, 600, 700, 800, 900],
                                        #   'cstheory.stackexchange':[700, 800, 900, 1000],
                                        #   'stackoverflow':[1000],
                                        #   'unix.meta.stackexchange':[30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                        #   'politics.stackexchange':[200, 300, 400, 500, 600, 700, 800,900,1000]
                                        # 'math.meta.stackexchange':[30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000]
                                          'codegolf.meta.stackexchange':[100, 200, 300, 400, 500, 600, 700, 800, 900,1000]
                                          }
    variation = '_fixedTau'

    """
    try:
        # test on comm "3dprinting.stackexchange" to debug
        myFun(commDir_sizes_sortedlist[227][0], commDir_sizes_sortedlist[227][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[227][0]], variation, CVPgenerated)
        # test on comm "latin.stackexchange" to debug
        # myFun(commDir_sizes_sortedlist[229][0], commDir_sizes_sortedlist[229][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[229][0]], variation, CVPgenerated)
        # test on comm "lifehacks.stackexchange" to debug
        # myFun(commDir_sizes_sortedlist[233][0], commDir_sizes_sortedlist[233][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[233][0]], variation, CVPgenerated)
        # test on comm "askubuntu.stackexchange" to debug
        # myFun(commDir_sizes_sortedlist[231][0], commDir_sizes_sortedlist[231][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[231][0]], variation, CVPgenerated)
        # test on comm "cstheory.stackexchange" to debug
        myFun(commDir_sizes_sortedlist[256][0], commDir_sizes_sortedlist[256][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[256][0]], variation, CVPgenerated)
        # test on comm "stackoverflow" to debug
        myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[359][0]], variation, CVPgenerated)
        # test on comm "unix.meta.stackexchange" to debug
        myFun(commDir_sizes_sortedlist[173][0], commDir_sizes_sortedlist[173][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[173][0]], variation, CVPgenerated)
        # # test on comm "politics.stackexchange" to debug
        myFun(commDir_sizes_sortedlist[283][0], commDir_sizes_sortedlist[283][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[283][0]], variation, CVPgenerated)
    
    except Exception as e:
        print(e)
        sys.exit()
    """
     # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    # selected_comms = ['3dprinting.stackexchange','latin.stackexchange','meta.askubuntu','lifehacks.stackexchange',
    #                   'cstheory.stackexchange','stackoverflow','unix.meta.stackexchange','politics.stackexchange']
    selected_comms = ['codegolf.meta.stackexchange']
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        if commName not in selected_comms: # skip 
            continue

        # if commName in splitted_comms: # skip splitted big communities
        #     print(f"{commName} was splitted.")
        #     continue

        selected_reg_strengthList = commName2selected_reg_strengthList[commName]

        try:
            p = mp.Process(target=myFun, args=(commName,commDir, roundIndex, selected_reg_strengthList, variation, CVPgenerated))
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
    print('semiSynthetic7_votingStage_trainingDataSorting Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
