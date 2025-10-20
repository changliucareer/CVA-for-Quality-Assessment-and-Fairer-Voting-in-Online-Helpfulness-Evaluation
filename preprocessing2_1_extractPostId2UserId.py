import os
#First print the current working directory
print("Current Working Directory", os.getcwd())
Original_Dir = os.getcwd()
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

def myAction (chunk,chunk_start_index,commName):
    print(f"{commName} current chunkn {chunk_start_index} running on {mp.current_process().name}")

    # initialize postId2UserId
    postId2UserId = defaultdict()

    for index,row in chunk.iterrows():

        postId = int(row['Id']) # current post Id
        userId = row['OwnerUserId']
        lastEditorUserId = row['LastEditorUserId']
        postId2UserId[postId] = {'userId':userId,'lastEditorUserId':lastEditorUserId}

    print(f"{commName} {mp.current_process().name} return")

    return postId2UserId
    

def myFun(commName, commDir):
    
    final_directory = os.path.join(commDir, r'intermediate_data_folder')
    # if already done for this comm, return
    # resultFile= final_directory+'/'+'QuestionsWithAnswers.dict'
    # if os.path.exists(resultFile):
    #     return

    print(f"comm {commName} running on {mp.current_process().name}")
    t1 = time.time()

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    #read data in chunks of 1 million rows at a time
    chunk_size = 1000
    chunksIter = pd.read_csv('Posts.csv',chunksize=chunk_size) # return type <class 'pandas.io.parsers.readers.TextFileReader'>

    done_looping = False
    chunk_batch = []
    # n_proc = mp.cpu_count() # n_proc as batch size
    n_proc = 1
    all_outputs = []
    chunk_index = 0
    while not done_looping:
        try:
            chunk = next(chunksIter) # chunk type is <class 'pandas.core.frame.DataFrame'>
        except StopIteration:
            done_looping = True
        else:
            # when the batch is full, do the action with multiprocessing pool
            if len(chunk_batch)==n_proc:
                args = zip(chunk_batch,[chunk_index*chunk_size]*len(chunk_batch),[commName]*len(chunk_batch))
                
                with mp.Pool(processes=n_proc) as pool:
                    # issue tasks to the process pool and wait for tasks to complete
                    #An iterator is returned with the result for each function call
                    results = pool.starmap(myAction, args, chunksize=100)
                    all_outputs.extend(results)
                    # process pool is closed automatically

                # increase the chunk_index
                chunk_index += 1
                # clear the chunk_batch
                chunk_batch = []
            chunk_batch.append(chunk)
    
    # process the last unfulled batch
    with mp.Pool(processes=n_proc) as pool:
        args = zip(chunk_batch,[chunk_index*chunk_size]*len(chunk_batch),[commName]*len(chunk_batch))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction, args)
        # process pool is closed automatically
        all_outputs.extend(results)

    # combine all_outputs
   
    all_postId2UserId = defaultdict()
    

    for output in all_outputs:
        postId2UserId = output
        all_postId2UserId.update(postId2UserId)
  
    # check whether have intermediate data folder, create one if not
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    # save to all_postId2UserId 
    with open(final_directory+'/'+'postId2UserId.dict', 'wb') as outputFile:
        pickle.dump(all_postId2UserId, outputFile)
    

    elapsed = format_time(time.time() - t1)
    # Report progress.
    print(f"for {commName}, ")
    print('processing Posts.csv and extract userId Done.    Elapsed: {:}.\n'.format(elapsed))


def main():

    t0=time.time()

   ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("all sorted CommDir loaded.")

    processes = []
    finishedCount = 0
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        p = mp.Process(target=myFun, args=(commName,commDir))
        p.start()
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
    print('processing Posts.csv and extract postId2UserId Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
