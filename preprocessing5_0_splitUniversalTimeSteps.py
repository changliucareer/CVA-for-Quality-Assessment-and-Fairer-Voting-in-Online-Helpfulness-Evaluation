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
    logFileName = 'preprocessing5_0_splitUniversalTimeSteps_Log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')

    with open(intermediate_directory+'/'+'universal_timesteps_afterCombineQAVH.dict', 'rb') as inputFile:
        universal_timesteps = pickle.load( inputFile)
    
    # creat a folder if not exists to store splitted raw data files
    splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    if not os.path.exists(splitFolder_directory):
        os.makedirs(splitFolder_directory)
    
    # under split_data_folder, create a folder to store specific splitted files of Votes.csv
    split_universalTS_files_directory = os.path.join(splitFolder_directory, r'UniversalTimeSteps_Yearly')
    if not os.path.exists(split_universalTS_files_directory):
        os.makedirs(split_universalTS_files_directory)

    year = universal_timesteps[0][2].year
    cur_uts = []
    total_len_ut = len(universal_timesteps)
    for i,ut in enumerate(universal_timesteps):
        print(f"processing {i+1}/{total_len_ut} time step of whole universal_timesteps.")
        cur_year = ut[2].year
        if cur_year == year:
            cur_uts.append(ut)
        elif cur_year < year:
            print(f"Exception: cur_year < year for {i+1}th time step")
        else: # cur_year > year
            # save previous year's universal time steps
            out_file_dir = split_universalTS_files_directory+'/'+'UniversalTimeSteps_year_'+ str(year) +'.dict'
            # save yearly universal time steps
            with open(out_file_dir, 'wb') as outputFile:
                pickle.dump(cur_uts, outputFile)
                logtext = f"saved year {year} universal time steps, length: {len(cur_uts)}"
                writeIntoLog(logtext, commDir, logFileName)
                print(logtext)
            # clear cur_uts
            cur_uts = []
            # start a new year
            year = cur_year
            cur_uts.append(ut)
    
    # save the last year of universal time steps'
    out_file_dir = split_universalTS_files_directory+'/'+'UniversalTimeSteps_year_'+ str(year) +'.dict'
    with open(out_file_dir, 'wb') as outputFile:
        pickle.dump(cur_uts, outputFile)
        logtext = f"saved year {year} universal time steps, length: {len(cur_uts)}"
        writeIntoLog(logtext, commDir, logFileName)
        print(logtext)


    
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
    myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1]) 
    

    # # run on all communities other than stackoverflow
    # finishedCount = 0
    # processes = []
    # for tup in commDir_sizes_sortedlist:
    #     commName = tup[0]
    #     commDir = tup[1]
    #     if commName == 'stackoverflow': # skip stackoverflow to run at the last
    #         stackoverflow_dir = commDir
    #         continue
    #     try:
    #         p = mp.Process(target=myFun, args=(commName,commDir))
    #         p.start()
    #     except Exception as e:
    #         print(e)
    #         pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
    #         print(f"current python3 processes count {pscount}.")
    #         return

    #     processes.append(p)
    #     if len(processes)==20:
    #         # make sure all p finish before main process finish
    #         for p in processes:
    #             p.join()
    #             finishedCount +=1
    #             print(f"finished {finishedCount} comm.")
    #         # clear processes
    #         processes = []
    
    # # join the last batch of processes
    # if len(processes)>0:
    #     # make sure all p finish before main process finish
    #     for p in processes:
    #         p.join()
    #         finishedCount +=1
    #         print(f"finished {finishedCount} comm.")


    # # run stackoverflow at the last separately
    # myFun('stackoverflow', stackoverflow_dir)

   
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('split universal time steps Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
