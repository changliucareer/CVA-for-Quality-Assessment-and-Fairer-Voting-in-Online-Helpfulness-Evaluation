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
import psutil



def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    t1 = time.time()

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # creat a folder if not exists to store splitted raw data files
    splitFolder_directory = os.path.join(commDir, r'split_data_folder')
    if not os.path.exists(splitFolder_directory):
        os.makedirs(splitFolder_directory)
    
    # under split_data_folder, create a folder to store specific splitted files of Votes.csv
    split_votes_files_directory = os.path.join(splitFolder_directory, r'PostHistory.csv_files')
    if not os.path.exists(split_votes_files_directory):
        os.makedirs(split_votes_files_directory)

    chunk_size = 2600000  # total line count of post history is 152435942
    
    # load PostHistory.csv
    print(f"loading PostHistory.csv ... of {commName}")
    part = 1
    for df in pd.read_csv('PostHistory.csv', chunksize=chunk_size, engine='python',sep=','):
        out_file_dir = split_votes_files_directory+'/'+'PostHistroy_part_'+ str(part) +'.csv'
        df.to_csv(out_file_dir, index=False)
        print(f"part {part} saved.")
        part+=1


    elapsed = format_time(time.time() - t1)
    # Report progress.
    print(f"for {commName}, ")
    print('splitting PostHistory.csv Done.    Elapsed: {:}.\n'.format(elapsed))

def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print(f"all {len(commDir_sizes_sortedlist)} sorted CommDirList loaded. ")
    print(commDir_sizes_sortedlist[359:])
    time.sleep(5)
    
    # # test on comm "math.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[358][0], commDir_sizes_sortedlist[358][1])
    
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist[359:]:  # only need to split stackoverflow
        commName = tup[0]
        commDir = tup[1]
        try:
            p = mp.Process(target=myFun, args=(commName,commDir))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"Exception to start of new process: current python3 processes count {pscount}.")
            return

        processes.append(p)
        if len(processes)==60:
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
    print('splitting PostHistory.csv Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
