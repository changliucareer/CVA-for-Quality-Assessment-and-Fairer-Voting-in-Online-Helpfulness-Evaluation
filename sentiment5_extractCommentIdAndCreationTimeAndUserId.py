import os

from numpy import average
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
from collections import Counter
import sys
import re

import pandas as pd
from bs4 import BeautifulSoup

##########################################################################################
def replace_img_tags(data):
    p = re.compile(r'<img.*(/img)?>')
    return p.sub('image', data)

def replace_code_tags(data):
    p = re.compile(r'<code.*(/code)?>')
    return p.sub('code', data)

def replace_URL(data):
    p = re.compile(r'\S*https?:\S*')
    return p.sub('URL', data)

def cleanComment(comment_text):
    if ('<img' in comment_text) or ('<code' in comment_text) or ('http' in comment_text):
        # replace <img> and <code> segment
        cleaned_comment_text = replace_img_tags(comment_text)
        cleaned_comment_text = replace_code_tags(cleaned_comment_text)
        cleaned_comment_text = replace_URL(cleaned_comment_text)
        return BeautifulSoup(cleaned_comment_text, "lxml").text
    else: 
        return BeautifulSoup(comment_text, "lxml").text

def myAction (commName, commDir, chunkIndex, df, post2commentIdAndCreationTimeAndUserId):
    for line_count, row in df.iterrows():
        print(f"processing processing {commName} chunk {chunkIndex} line {line_count}...")
        # lines of data 
        comment_text = row['Text']

        cleaned_comment_text = cleanComment(comment_text)
        if len(cleaned_comment_text)<2: # skip the comment that is too short
            continue
        targetPost = row['PostId']
        commentId = row['Id']
        creationTime = row['CreationDate']
        date_time_obj = datetime.datetime.strptime(creationTime, "%Y-%m-%dT%H:%M:%S.%f")
        userId = row['UserId']

        if targetPost not in post2commentIdAndCreationTimeAndUserId.keys():
            post2commentIdAndCreationTimeAndUserId[targetPost]={'commentIds':[commentId], 'creationTimes':[date_time_obj], 'userIds':[userId]}
        else:
            post2commentIdAndCreationTimeAndUserId[targetPost]={'commentIds': post2commentIdAndCreationTimeAndUserId[targetPost]['commentIds']+[commentId],
                                                       'creationTimes': post2commentIdAndCreationTimeAndUserId[targetPost]['creationTimes']+[date_time_obj],
                                                       'userIds':post2commentIdAndCreationTimeAndUserId[targetPost]['userIds']+[userId]}

    # save results till this chunk
    # convert and save the last post2commentIdAndCreationTimeAndUserId dict
    post2commentIdAndCreationTimeAndUserId_normalDict = defaultdict()
    for pid, d in post2commentIdAndCreationTimeAndUserId.items():
        post2commentIdAndCreationTimeAndUserId_normalDict[pid] = {'commentIds':d['commentIds'], 'creationTimes':d['creationTimes'], 'userIds':d['userIds']}
    
    os.chdir(commDir) # go back to comm directory
    with open('intermediate_data_folder/postId2commentIdAndCreationTimeAndUserId_tillCurChunk.dict', 'wb') as outputFile:
        pickle.dump(post2commentIdAndCreationTimeAndUserId_normalDict, outputFile)
        print(f"saved commentIdAndCreationTimeAndUserId of posts for {commName} till chunk {chunkIndex}")
        writeIntoLog(f'saved post2commentIdAndCreationTimeAndUserId_normalDict after chunk {chunkIndex} done\n',commDir,'sentiment5_extractCommentIdAndCreationTime_Log.txt')


# Attributions of comment CSV
# 'Comments':['Id','PostId','Score','Text','CreationDate','UserId']
def myFun(commIndex,commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # check the line counts of sentiment_training_data.csv and 
    commentsFileLength = len(pd.read_csv('Comments.csv'))
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    # Read the CSV file into a pandas DataFrame object
    sentiment_training_data_length = len(pd.read_csv(intermediate_directory + '/sentiment_training_data.csv'))

    assert commentsFileLength == sentiment_training_data_length

    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    post2commentIdAndCreationTimeAndUserId = manager.dict() # to save the used train mode (wholebatch or minibatch) of each community

    chunk_size = 10000
    processes = []
    chunkIndex = 0
    finishedCount =0
    # load sentiment training data
    try:
        for df in pd.read_csv('Comments.csv', chunksize=chunk_size, engine='python',sep=','):
            
            try:
                p = mp.Process(target=myAction, args=(commName, commDir, chunkIndex, df, post2commentIdAndCreationTimeAndUserId))
                p.start()
                chunkIndex += 1
            except Exception as e:
                print(e)
                sys.exit()
                return

            processes.append(p)
            if len(processes)==4:
                # make sure all p finish before main process finish
                for p in processes:
                    p.join()
                    finishedCount +=1
                    print(f"finished {finishedCount} chunks for {commName}.")
                # clear processes
                processes = []
        
        # join the last batch of processes
        if len(processes)>0:
            # make sure all p finish before main process finish
            for p in processes:
                p.join()
                finishedCount +=1
                print(f"finished {finishedCount} chunks for {commName}.")


        # convert and save the last post2sentiment dict
        post2commentIdAndCreationTimeAndUserId_normalDict = defaultdict()
        for pid, d in post2commentIdAndCreationTimeAndUserId.items():
            post2commentIdAndCreationTimeAndUserId_normalDict[pid] = {'commentIds':d['commentIds'], 'creationTimes':d['creationTimes'], 'userIds':d['userIds']}
        
        os.chdir(commDir) # go back to comm directory
        with open('intermediate_data_folder/postId2commentIdAndCreationTimeAndUserId.dict', 'wb') as outputFile:
            pickle.dump(post2commentIdAndCreationTimeAndUserId_normalDict, outputFile)
            print(f"saved commentIdAndCreationTimeAndUserId of posts for {commName} till chunk {chunkIndex} as whole")
            writeIntoLog(f'saved post2commentIdAndCreationTimeAndUserId_normalDict after chunk {chunkIndex} as whole done\n',commDir,'sentiment5_extractCommentIdAndCreationTime_Log.txt')

    except Exception as e:
        logfile = 'sentiment5_extractCommentIdAndCreationTime_Log.txt'
        logtext = f"fail for {commName}\n"
        logtext += str(e)+'\n'
        writeIntoLog(logtext, commDir, logfile)


def main():
    t0 = time.time()
    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")
    
    # test on comm "coffee.stackexchange" to debug
    myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # test on comm "datascience.stackexchange" to debug
    # myFun(0,commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    # myFun(3,commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    # test on comm "math.stackexchange" to debug
    # myFun(358,commDir_sizes_sortedlist[358][0], commDir_sizes_sortedlist[358][1])
    # test on comm "stackoverflow" to debug
    # myFun(359,commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1])
    
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        try:
            p = mp.Process(target=myFun, args=(commIndex,commName,commDir))
            p.start()
        except Exception as e:
            print(e)
            return

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
    print('sentiment5 extract comment id and creation time Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
