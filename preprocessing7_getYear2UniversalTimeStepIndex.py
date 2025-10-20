import os
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################
from toolFunctions import format_time, writeIntoLog, insertEventInUniversalTimeStep
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
from itertools import groupby
import re
import psutil
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np


def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    with open(intermediate_directory+'/'+'universal_timesteps_afterCombineQAVH.dict', 'rb') as inputFile:
        universal_timesteps = pickle.load( inputFile)
    

    year = universal_timesteps[0][2].year

    year2UniversalTimeStepIndex = defaultdict()

    total_len_ut  = len(universal_timesteps)
    for i,ut in enumerate(universal_timesteps):
        print(f"processing {i+1}/{total_len_ut} time step of whole universal_timesteps of {commName}.")
        cur_year = ut[2].year
        if cur_year > year:
            # updated month
            year = cur_year
        
        year2UniversalTimeStepIndex[year]= i
    
    
    # save year2UniversalTimeStepIndex
    out_file_dir = intermediate_directory+'/'+'year2UniversalTimeStepIndex.dict'
    with open(out_file_dir, 'wb') as outputFile:
        pickle.dump(year2UniversalTimeStepIndex, outputFile)
        print(f"saved year 2 universal time step Index for {commName}.")


    
def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # # save the all comm sorted list into a csv file
    # import csv
    # with open('allComm_directories_sizes_sortedlist.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow( ["commName","totalPostCount"])
    #     for tup in commDir_sizes_sortedlist:
    #         writer.writerow([tup[0],tup[2]])

    # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1]) 
    # test on comm "stackoverflow" to debug
    # myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1]) 
    

    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]
        if commName == 'stackoverflow': # skip stackoverflow to run at the last
            stackoverflow_dir = commDir
            continue
        try:
            p = mp.Process(target=myFun, args=(commName,commDir))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
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


    # # run stackoverflow at the last separately
    print(f"start to process stackoverflow alone...")
    myFun('stackoverflow', stackoverflow_dir)

   
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('split universal time steps Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
