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
    logFileName = 'preprocessing5_1_1_splitQuestionsWithAVH_Log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    
    print(f"loading Questions for {commName}")
    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistory.dict', 'rb') as inputFile:
        Questions = pickle.load( inputFile)

    # creat a folder if not exists to store splitted files
    splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    if not os.path.exists(splitFolder_directory):
        os.makedirs(splitFolder_directory)

    # under splitted_intermediate_data_folder, create a folder to store splitted Questions
    split_Questions_files_directory = os.path.join(splitFolder_directory, r'QuestionsWithAVH')
    if not os.path.exists(split_Questions_files_directory):
        print("no split_Questions_files_directory, create one")
        os.makedirs(split_Questions_files_directory)
    
    # convert dictionary to items list
    items = list(Questions.items())
    # clear Questions
    Questions.clear()

    partSize = 2000 # how many questions in one part
    print(f"set part size as {partSize} for {commName}")

    part2qids = defaultdict()

    part = 0
    logtext = f"for {commName} with total {len(items)} questions,\n"
    for partStartIndex in range(0, len(items), partSize):
        part += 1
        print(f"extract and saving part {part} for {commName}")
        curPart = dict(items[partStartIndex: (partStartIndex+partSize)])
        # save current part qids
        part2qids[part] = list(curPart.keys())
        # save current part
        out_file_dir = split_Questions_files_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistory_part_'+ str(part) +'.dict'
        with open(out_file_dir, 'wb') as outputFile:
            pickle.dump(curPart, outputFile)
            logtext += f"saved part {part} QuestionsWithAVH , length: {len(curPart)}\n"
            
    # save part2qids
    part2qids_dir = intermediate_directory+'/'+'partOfQuestionsWithAVH2qids.dict'
    with open(part2qids_dir, 'wb') as outputFile:
        pickle.dump(part2qids, outputFile)
        print( f"saved partOfQuestionsWithAVH2qids.dict with total {len(part2qids)} parts")

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
    print('splitQuestionsWithAVH Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
