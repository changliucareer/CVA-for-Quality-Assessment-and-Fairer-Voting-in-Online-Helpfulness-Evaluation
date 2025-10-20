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
import json
    

def myFun(commName, commDir):
    
    final_directory = os.path.join(commDir, r'intermediate_data_folder')

    print(f"comm {commName} running on {mp.current_process().name}")
    t1 = time.time()

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load universal_timesteps
    with open(final_directory+'/'+'universal_timesteps_afterCombineQA.dict', 'rb') as inputFile:
        universal_timesteps = pickle.load( inputFile)
    
    # keep an universal timestamps sequence for a community
    # a list of tuples (eventType, targetId, datetime_obj, (optional:postId))
    # Define eventType = 0 when create a new question, targetId as question Id
    #        eventType = 1 when create a new answer, targetId as answer Id
    #        eventType = 2 when edit a post, targetId as post history Id, has optional element answerId (we only consider answer post)
    #        eventType = 3 when vote, targetId as vote Id, has optional element target answerId 
    #        eventType = 4 when accept an answer, targetId as vote Id, has optional element the accepted answer Id
    #        eventType = 5 when close a post, targetId as post history Id, has optional element postId (could be an answer or a question)
    #        eventType = 6 when lock a post, targetId as post history Id, has optional element postId (could be an answer or a question)
    #        eventType = 7 when delete a post, targetId as post history Id, has optional element postId (could be an answer or a question)
    
    postId2CreationTime = defaultdict()
    for tup in universal_timesteps:
        eventType = tup[0]
        targetId = tup[1]
        datetime_obj = tup[2]
        if eventType == 0 or eventType == 1:
            postId2CreationTime[targetId] = datetime_obj
    
    

    # save updated postId2CreationTime
    with open(final_directory+'/'+'postId2CreationTime.json', "w") as outfile: 
        json.dump(postId2CreationTime, outfile, default=str)
        print(f"saved postId2CreationTime (length: {len(postId2CreationTime)}) for {commName}")


    elapsed = format_time(time.time() - t1)
    # Report progress.
    print(f"for {commName}, ")
    print('processing Posts.csv and combine Q and A Done.    Elapsed: {:}.\n'.format(elapsed))



def myAction(parentCommName, parentCommDir, subComms_data_folder, subCommName, subCommDir, subComm_QuestionsWithEventList_directory, postId2CreationTime):
    print(f"sub comm {subCommName} running on {mp.current_process().name}")
    
    # go to current comm data directory
    os.chdir(subCommDir)
    print(os.getcwd())

    # load intermediate_data files
    subComm_intermediate_directory = os.path.join(subCommDir, r'intermediate_data_folder')
    if not os.path.exists(subComm_intermediate_directory):
        print(f"no intermediate_folder for sub_comm {subCommName}, create one")
        os.makedirs(subComm_intermediate_directory)

    
    # load sub_questionId2answerIdList
    with open(subComm_intermediate_directory+'/'+'subComm_questionId2answerIdList.dict', 'rb') as inputFile:
        subComm_questionId2answerIdList = pickle.load( inputFile)

    # get all postIds
    postIds = []
    for questionId, answerIdList in subComm_questionId2answerIdList.items():
        postIds.append(questionId)
        postIds.extend(answerIdList)

    # postId2CreationTime for subComm
    postId2CreationTime_subComm = {postId:postId2CreationTime[str(postId)] for postId in postIds if str(postId) in postId2CreationTime}
    
    # save 
    with open(subComm_intermediate_directory+'/'+'postId2CreationTime_subComm.json', 'w') as outputFile:
        json.dump(postId2CreationTime_subComm, outputFile)
        print(f"saved postId2CreationTime_subComm for {subCommName}")
    
def myFun_SOF(commName, commDir):
    t0=time.time()
    final_directory = os.path.join(commDir, r'intermediate_data_folder')

    print(f"comm {commName} running on {mp.current_process().name}")
    t1 = time.time()

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load 
    print("loading postId2CreationTime...")
    with open(final_directory+'/'+'postId2CreationTime.json') as json_file:
        postId2CreationTime  = json.load(json_file)

    # process sub-comms
    # go to StackOverflow data directory
    parentCommName = commName
    parentCommDir = commDir

    # go to the target splitted files folder
    subComms_data_folder = os.path.join(parentCommDir, f'subCommunities_folder')
    if not os.path.exists( subComms_data_folder):
        print("Exception: no subComms_data_folder!")
    
    ## Load all sub community direcotries 
    with open(subComms_data_folder+'/'+f'subCommName2subCommDir.dict', 'rb') as inputFile:
        subCommName2commDir = pickle.load( inputFile)


    # prepare args 
    argsList = []
    for subCommName, subCommDir in subCommName2commDir.items():

        # subComm_QuestionsWithEventList
        subComm_QuestionsWithEventList_directory = subCommDir+'/'+f'QuestionsWithEventList_tag_{subCommName}.dict'
        # prepare args
        argsList.append((parentCommName, parentCommDir, subComms_data_folder, subCommName, subCommDir, subComm_QuestionsWithEventList_directory, postId2CreationTime))
    
    
    # run on all sub communities of stackoverflow
    finishedCount = 0
    processes = []
    for args in argsList:

        try:
            p = mp.Process(target=myAction, args=args)
            p.start()
        except Exception as e:
            print(e)
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

            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('extract moderating info for sub comms Done completely.    Elapsed: {:}.\n'.format(elapsed))

def main():

    t0=time.time()

    ## Load community direcotries .dict files
    with open('commDirectories.dict', 'rb') as inputFile:
        commDirDict = pickle.load( inputFile)
    print("CommDir loaded.")
    

    # processes = []
    # finishedCount = 0
    # for commName, commDir in commDirDict.items():
    #     p = mp.Process(target=myFun, args=(commName,commDir))
    #     p.start()
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

    # run stackoverflow at the last separately
    print("start to process stackoverflow...")
    myFun_SOF('stackoverflow', commDirDict['stackoverflow'])
   
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('processing postId2Creation Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
