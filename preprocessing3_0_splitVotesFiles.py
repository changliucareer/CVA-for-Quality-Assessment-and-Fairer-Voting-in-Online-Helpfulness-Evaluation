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
    split_votes_files_directory = os.path.join(splitFolder_directory, r'Votes.csv_files')
    if not os.path.exists(split_votes_files_directory):
        os.makedirs(split_votes_files_directory)

    chunk_size = 100000

    # write a part
    def write_chunk(part, lines):
        print(f"writing part {part} of {commName}...")
        with open(split_votes_files_directory+'/'+'Votes_part_'+ str(part) +'.csv', 'w') as f_out:
            f_out.write(header)
            f_out.writelines(lines)
    
    # load Votes.csv
    print(f"loading Votes.csv ... of {commName}")
    with open(commDir+'/'+"Votes.csv", "r") as f:
        count = 0
        header = f.readline()
        lines = []
        for line in f:
            count += 1
            lines.append(line)
            if count % chunk_size == 0:
                write_chunk(count // chunk_size, lines)
                lines = []
        # write remainder
        if len(lines) > 0:
            write_chunk((count // chunk_size) + 1, lines)
            print(f"splitted {(count // chunk_size) + 1} parts in total.")
        else:
            print(f"splitted {count // chunk_size} parts in total.")


    elapsed = format_time(time.time() - t1)
    # Report progress.
    print(f"for {commName}, ")
    print('splitting Votes.csv Done.    Elapsed: {:}.\n'.format(elapsed))



def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # test on comm "math.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[358][0], commDir_sizes_sortedlist[358][1])
    # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # test on comm "datascience.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])


    """
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist[358:]:
        commName = tup[0]
        commDir = tup[1]
        try:
            p = mp.Process(target=myFun, args=(commName,commDir))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            return

        processes.append(p)
        if len(processes)==20:
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
    print('splitting Votes.csv Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
