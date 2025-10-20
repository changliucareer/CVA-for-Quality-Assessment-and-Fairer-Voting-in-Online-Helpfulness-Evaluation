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
import copy
    

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'preprocessing5_1_2_getQid2UniversalTimeSteps_yearlyLength_Log.txt'
    logtext = ""

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # creat a folder if not exists to store splitted data files
    splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    if not os.path.exists(splitFolder_directory):
        print("Exception: no splitted_intermediate_data_folder")
        writeIntoLog("Exception: no splitted_intermediate_data_folder", commDir, logFileName)

    
    # under splitted_intermediate_data_folder, create a folder to store yearly saved qid2universal_timesteps
    split_qid2uts_files_directory = os.path.join(splitFolder_directory, r'qid2universalTimeSteps_Yearly')
    if not os.path.exists(split_qid2uts_files_directory):
        print("Exception: no split_qid2uts_files_directory")
    else: 
        accumulated_yearlyQid2uts_subfolders = [ f.path for f in os.scandir(split_qid2uts_files_directory) if f.path.split('/')[-1].startswith("qid2universalTimeSteps_tillYear_") ]
        yearlyQid2uts_subfolders = [ f.path for f in os.scandir(split_qid2uts_files_directory) if f.path.split('/')[-1].startswith("qid2universalTimeSteps_year_") ]

        # accumulated year's qid2universal_timestemps files
        if len(accumulated_yearlyQid2uts_subfolders)>0: # has some years done
            accumulated_yearlyQid2uts_subfolders.sort()
            print(f"accumulated_yearlyQid2uts length: {len(accumulated_yearlyQid2uts_subfolders)}")

        # single year's qid2universal_timestemps files
        if len(yearlyQid2uts_subfolders)>0: # has some years done
            yearlyQid2uts_subfolders.sort()
            print(f"yearlyQid2uts_subfolders length: {len(yearlyQid2uts_subfolders)}")

    # a dict to save qid to the length of universal time steps of each year since 2008
    # key is question id, value is a dictionary with year as key and length as value
    qid2utsYearlyLength = defaultdict()

    # exact uts length from accumulated files
    qid2preYearAccumulatedLength = defaultdict()

    for i,f in enumerate(accumulated_yearlyQid2uts_subfolders):
        year = int(f.split('/')[-1].split('_')[-1].split('.')[0])

        with open(f, 'rb') as inputFile:
            accumulated_filelTuple = pickle.load( inputFile)
            accumulated_qid2uts = accumulated_filelTuple[0]
            total_QuestionCount = len(accumulated_qid2uts)

        qIndex = 0
        for qid, cur_uts in accumulated_qid2uts.items():
            qIndex +=1
            print(f"processing question {qid} of year {year}. The {qIndex}th out of {total_QuestionCount}...")

            # check whether all eventtypes are 1,2,3
            eventTypesCheck = [0 if ts[0] in [1,2,3] else 1 for ts in cur_uts]
            assert sum(eventTypesCheck)==0

            total_len_ut = len(cur_uts)

            # for the first accumulated uts, scan through and extract length from 2008 to current year
            if i ==0:
                if qid not in qid2utsYearlyLength.keys():
                    # initialize a dict, each length as 0 from year 2008 to year 2022
                    initialYearlyLengthDict = defaultdict()
                    for y in range(2008,2023):
                        initialYearlyLengthDict[y]=0
                    qid2utsYearlyLength[qid] = initialYearlyLengthDict
                # update lengths for current qid
                qid2utsYearlyLength[qid][year] = total_len_ut
                
                qid2preYearAccumulatedLength[qid] = total_len_ut

            else: # for other accumulated uts, only need to get the length
                if qid not in qid2utsYearlyLength.keys():
                    # initialize a dict, each length as 0 from year 2008 to year 2022
                    initialYearlyLengthDict = defaultdict()
                    for y in range(2008,2023):
                        initialYearlyLengthDict[y]=0
                    qid2utsYearlyLength[qid] = initialYearlyLengthDict
                if qid not in qid2preYearAccumulatedLength.keys():
                    qid2preYearAccumulatedLength[qid] = 0
                qid2utsYearlyLength[qid][year] = total_len_ut - qid2preYearAccumulatedLength[qid]
                # update pre-year length
                qid2preYearAccumulatedLength[qid] = total_len_ut
        
        logtext += f"year {year} accumulated qid2uts processed.\n"
    
    # to save memory
    accumulated_qid2uts.clear()
    qid2preYearAccumulatedLength.clear()

    # exact uts length from single year files    
    for i,f in enumerate(yearlyQid2uts_subfolders):
        year = int(f.split('/')[-1].split('_')[-1].split('.')[0])

        with open(f, 'rb') as inputFile:
            filelTuple = pickle.load( inputFile)
            qid2uts = filelTuple[0]
            total_QuestionCount = len(qid2uts)

        qIndex = 0
        for qid, cur_uts in qid2uts.items():
            qIndex +=1
            print(f"processing question {qid} of year {year}. The {qIndex}th out of {total_QuestionCount}...")
            # check whether all eventtypes are 1,2,3
            eventTypesCheck = [0 if ts[0] in [1,2,3] else 1 for ts in cur_uts]
            assert sum(eventTypesCheck)==0
            
            if qid not in qid2utsYearlyLength.keys():
                # initialize a dict, each length as 0 from year 2008 to year 2022
                initialYearlyLengthDict = defaultdict()
                for y in range(2008,2023):
                    initialYearlyLengthDict[y]=0
                qid2utsYearlyLength[qid] = initialYearlyLengthDict

            qid2utsYearlyLength[qid][year] = len(cur_uts)

        logtext += f"year {year} single year qid2uts processed.\n"
        qid2uts.clear()
    
    writeIntoLog(logtext, commDir, logFileName)

    # save results every year
    saveFilename = 'qid2universalTimeSteps_yearlyLength.dict'
    with open(intermediate_directory+'/'+saveFilename, 'wb') as outputFile:
        pickle.dump(qid2utsYearlyLength, outputFile)
        print(f"saved qid2universalTimeSteps_yearlyLength")
    
    
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
    print('getQid2UniversalTimeSteps_yearlyLength Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
