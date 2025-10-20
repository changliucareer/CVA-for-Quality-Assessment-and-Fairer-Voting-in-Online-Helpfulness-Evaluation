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
import json

def myAction (chunk,chunk_index,commName, logFileName):
    print(f"{commName} chunk {chunk_index} current chunk running on {mp.current_process().name}")

    # - VoteTypeId
    #     - ` 1`: AcceptedByOriginator
    #     - ` 2`: UpMod
    #     - ` 3`: DownMod
    #     - ` 4`: Offensive
    #     - ` 5`: Favorite - if VoteTypeId = 5 UserId will be populated
    #     - ` 6`: Close
    #     - ` 7`: Reopen
    #     - ` 8`: BountyStart
    #     - ` 9`: BountyClose
    #     - `10`: Deletion
    #     - `11`: Undeletion
    #     - `12`: Spam
    #     - `13`: InformModerator

    # spam posts
    postId2spamVotes = defaultdict()
    # offensive posts
    postId2offensiveVotes = defaultdict()

    for index,row in chunk.iterrows():
        
        # convert datetime info string into datetime object
        datetime_str = row['CreationDate'] 
        # date_time_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f") # because datetime is not serializable for json

        vid = int(row['Id'])
        pid = int(row['PostId'])

        voteTypeId = int(row['VoteTypeId'])

        if voteTypeId == 4: # offensive
            if pid in postId2offensiveVotes.keys(): 
                postId2offensiveVotes[pid].append((vid, datetime_str))
            else:
                postId2offensiveVotes[pid] = [(vid, datetime_str)]
            
        elif voteTypeId == 12: # spam
            if pid in postId2spamVotes.keys(): 
                postId2spamVotes[pid].append((vid, datetime_str))
            else:
                postId2spamVotes[pid] = [(vid, datetime_str)]

    print(f"{mp.current_process().name} return for {commName}")
    return postId2offensiveVotes, postId2spamVotes
    

def myFun(commName, commDir):

    logFileName = "preprocessing11_extractPostsThatvotedSpamOrOffensive_Log.txt"
    print(f"comm {commName} running on {mp.current_process().name}")
    t1 = time.time()

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    final_directory = os.path.join(commDir, r'intermediate_data_folder')

    # # if already done for this comm, return
    resultFiles = ['postId2offensiveVotes.json','postId2spamVotes.json']
    resultFiles = [final_directory+'/'+f for f in resultFiles]
    if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
        print(f"{commName} has already done this step.")
        return

    #read data in chunks of 1 million rows at a time
    chunk_size = 10000
    chunksIter = pd.read_csv('Votes.csv',chunksize=chunk_size) # return type <class 'pandas.io.parsers.readers.TextFileReader'>

    done_looping = False
    chunk_batch = []
    n_proc = mp.cpu_count()-2 # n_proc as batch size
    all_outputs = []
    chunk_index = 0
    while not done_looping:
        try:
            chunk = next(chunksIter) # chunk type is <class 'pandas.core.frame.DataFrame'>
        except StopIteration:
            done_looping = True
        else:
            # # # use shared variable to communicate among all comm's process
            # manager = mp.Manager()
            # Question2VoteListDict = manager.dict() # to save the question count and answer count of each community

            # when the batch is full, do the action with multiprocessing pool
            if len(chunk_batch)==n_proc:
                args = zip(chunk_batch,range(chunk_index, chunk_index+len(chunk_batch)),[commName]*len(chunk_batch),[logFileName]*len(chunk_batch))
                
                with mp.Pool(processes=n_proc) as pool:
                    # issue tasks to the process pool and wait for tasks to complete
                    #An iterator is returned with the result for each function call
                    results = pool.starmap(myAction, args, chunksize=100)
                    all_outputs.extend(results)
                    # process pool is closed automatically

                # increase the chunk_index
                chunk_index += len(chunk_batch)
                # clear the chunk_batch
                chunk_batch = []
            chunk_batch.append(chunk)
    
    # process the last batch
    with mp.Pool(processes=n_proc) as pool:
        args = zip(chunk_batch,range(chunk_index, chunk_index+len(chunk_batch)),[commName]*len(chunk_batch),[logFileName]*len(chunk_batch))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction, args, chunksize=100)
        # process pool is closed automatically
        all_outputs.extend(results)

    # combine all_outputs
    postId2offensiveVotes_total = defaultdict()
    postId2spamVotes_total = defaultdict()

    for tup in all_outputs:
        postId2offensiveVotes, postId2spamVotes = tup

        # combine offensive post
        for pid, offensiveVotes in postId2offensiveVotes.items():
            if pid in postId2offensiveVotes_total.keys():
                postId2offensiveVotes_total[pid].extend(offensiveVotes)
            else:
                postId2offensiveVotes_total[pid] = offensiveVotes
            # sort votes list by vid
            postId2offensiveVotes_total[pid].sort(key=lambda t:t[0])
        
        # combine spam post
        for pid, spamVotes in postId2spamVotes.items():
            if pid in postId2spamVotes_total.keys():
                postId2spamVotes_total[pid].extend(spamVotes)
            else:
                postId2spamVotes_total[pid] = spamVotes
            # sort votes list by vid
            postId2spamVotes_total[pid].sort(key=lambda t:t[0])

    # save 
    with open(final_directory+'/'+'postId2offensiveVotes.json', 'w') as outputFile:
        json.dump(postId2offensiveVotes_total, outputFile)
        print(f"saved postId2offensiveVotes_total for {commName}")
    
    with open(final_directory+'/'+'postId2spamVotes.json', 'w') as outputFile:
        json.dump(postId2spamVotes_total, outputFile)
        print(f"savedpostId2spamVotes_total for {commName}")

    logText = f"get {len(postId2offensiveVotes_total)} posts voted offensive, {len(postId2spamVotes_total)} posts voted spam.\n"
    writeIntoLog(logText, commDir, logFileName)

    elapsed = format_time(time.time() - t1)
    # Report progress.
    print(f"for {commName}, ")
    print('preprocessing11_extractPostsThatvotedSpamOrOffensive Done.    Elapsed: {:}.\n'.format(elapsed))
    print(logText)



def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    
    
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


    # run stackoverflow at the last separately
    myFun('stackoverflow', stackoverflow_dir)

    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('preprocessing11_extractPostsThatvotedSpamOrOffensive Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
