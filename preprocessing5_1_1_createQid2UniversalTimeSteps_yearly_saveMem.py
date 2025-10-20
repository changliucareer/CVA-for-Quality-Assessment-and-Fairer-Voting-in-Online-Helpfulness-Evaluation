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
    logFileName = 'preprocessing5_1_createQid2UniversalTimeSteps_yearly_Log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # whether accumulated from previous-done qid2universalTimeSteps
    accumulatedFlag = False
    print(f"start to process {commName}: accumulatedFlag={accumulatedFlag}.")

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    
    with open(intermediate_directory+'/'+'answer2parentQLookup.dict', 'rb') as inputFile:
        answer2parentQ = pickle.load( inputFile)

    # creat a folder if not exists to store splitted data files
    splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    if not os.path.exists(splitFolder_directory):
        print("Exception: no splitted_intermediate_data_folder")
        writeIntoLog("Exception: no splitted_intermediate_data_folder", commDir, logFileName)
    
    # under splitted_intermediate_data_folder, create a folder to store specific splitted files of UniversalTimeSteps_Yearly
    split_universalTS_files_directory = os.path.join(splitFolder_directory, r'UniversalTimeSteps_Yearly')
    if not os.path.exists(split_universalTS_files_directory):
        print("Exception: no split_universalTS_files_directory")
        writeIntoLog("Exception: no split_universalTS_files_directory", commDir, logFileName)

    
    # load yearly universal time steps files
    yearlyUniversalTimeSteps_subfolders = [ f.path for f in os.scandir(split_universalTS_files_directory) if f.path.split('/')[-1].startswith("UniversalTimeSteps_year_") ]
    yearlyUniversalTimeSteps_subfolders.sort()


    ### convert universal_timesteps to dict and qid as key
    qid2universal_timesteps = defaultdict()

    # variables for statistics will be accumulated no matter whether qid2universal_timesteps is accumulated
    deletedQ = 0
    deletedA = []
    lockedQ = 0
    lockedA = []
    closedQ = []
    closedA = []
    acceptedA = 0

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

    # extract timesteps related to saved question
    answer_to_remove = [] # list of answers that are deleted or accepted
    answer_to_stop = []  # to remove the events related to locked or closed answers after their locking or closing
    question_to_remove = [] # list of questions that are deleted 
    question_to_stop = [] # to remove the events related to locked or closed questions after their locking or closing
    preDone_year = 0 # initialize the pre finished year as small as 0
    
    # check whether already done years of universal time steps processing
    
    # under splitted_intermediate_data_folder, create a folder to store yearly saved qid2universal_timesteps
    split_qid2uts_files_directory = os.path.join(splitFolder_directory, r'qid2universalTimeSteps_Yearly')
    if not os.path.exists(split_qid2uts_files_directory):
        os.makedirs(split_qid2uts_files_directory)
    else: 
        accumulated_yearlyQid2uts_subfolders = [ f.path for f in os.scandir(split_qid2uts_files_directory) if f.path.split('/')[-1].startswith("qid2universalTimeSteps_tillYear_") ]
        yearlyQid2uts_subfolders = [ f.path for f in os.scandir(split_qid2uts_files_directory) if f.path.split('/')[-1].startswith("qid2universalTimeSteps_year_") ]

        if accumulatedFlag: # accumulated from preDone year's qid2universal_timestemps
            if len(accumulated_yearlyQid2uts_subfolders)>0: # has some years done
                accumulated_yearlyQid2uts_subfolders.sort()
                preDone_year = int(accumulated_yearlyQid2uts_subfolders[-1].split('/')[-1].split('_')[-1].split('.')[0])               
                with open(accumulated_yearlyQid2uts_subfolders[-1], 'rb') as inputFile:
                    try:
                        qid2universal_timesteps, deletedQ,deletedA,lockedQ,lockedA,closedQ,closedA,acceptedA,answer_to_remove,answer_to_stop,question_to_remove,question_to_stop = pickle.load( inputFile)
                    except Exception as e:
                        print(f"fail to load pre done qid2universal_timesteps for {commName}: {e}")
                        return
            else: # has folder, but has no file 
                preDone_year = 0
        else: # only save as current year qid2universal_timestemps
            if len(yearlyQid2uts_subfolders)>0: # has some years done
                yearlyQid2uts_subfolders.sort()
                preDone_year = int(yearlyQid2uts_subfolders[-1].split('/')[-1].split('_')[-1].split('.')[0])
                # print(f"only save as current year {preDone_year+1} qid2universal_timestemps for {commName}")
                with open(yearlyQid2uts_subfolders[-1], 'rb') as inputFile:
                    try:
                        qid2universal_timesteps, deletedQ,deletedA,lockedQ,lockedA,closedQ,closedA,acceptedA,answer_to_remove,answer_to_stop,question_to_remove,question_to_stop = pickle.load( inputFile)
                        # clear qid2universal_timesteps to not accumulate
                        qid2universal_timesteps = defaultdict()
                    except Exception as e:
                        print(f"fail to load pre done qid2universal_timesteps for {commName}: {e}")
                        return
            else: # has no single year result yet, but may have tillYear result
                if len(accumulated_yearlyQid2uts_subfolders)>0: # has some accumulated years done
                    accumulated_yearlyQid2uts_subfolders.sort()
                    preDone_year = int(accumulated_yearlyQid2uts_subfolders[-1].split('/')[-1].split('_')[-1].split('.')[0])
                    # print(f"only save as current year {preDone_year+1} qid2universal_timestemps for {commName}")
                    with open(accumulated_yearlyQid2uts_subfolders[-1], 'rb') as inputFile:
                        try:
                            qid2universal_timesteps, deletedQ,deletedA,lockedQ,lockedA,closedQ,closedA,acceptedA,answer_to_remove,answer_to_stop,question_to_remove,question_to_stop = pickle.load( inputFile)
                            # clear qid2universal_timesteps to not accumulate
                            qid2universal_timesteps = defaultdict()
                        except Exception as e:
                            print(f"fail to load pre done qid2universal_timesteps for {commName}: {e}")
                            return
                else: # has no single year result nor tillYear result
                    preDone_year = 0
    print(f"preDone_year is {preDone_year} for {commName}")
    time.sleep(5)

    for i,f in enumerate(yearlyUniversalTimeSteps_subfolders):
        year = int(f.split('/')[-1].split('_')[-1].split('.')[0])
        if year <= preDone_year: # skip the finished year
            continue

        with open(f, 'rb') as inputFile:
            cur_uts = pickle.load( inputFile)

        total_len_ut = len(cur_uts)
        for utIndex, ts in enumerate(cur_uts):
            print(f"converting {utIndex+1}/{total_len_ut} time step of year {year} universal_timesteps.")
            eventType = ts[0]
            if eventType==0: # event of creation of a question, no need
                continue
            elif eventType==1: # event of creation of an answer of a question
                aid = ts[1]
                if aid not in answer2parentQ.keys(): # not in saved answers, skip
                    continue
                qid = answer2parentQ[aid]
                if qid in question_to_stop: # a question has been closed or locked, stop adding new related event
                    continue
                if qid not in qid2universal_timesteps.keys():
                    qid2universal_timesteps[qid]=[]
                qid2universal_timesteps[qid].append(ts)

            elif eventType==2: # event of edition of an answer of a question
                aid = ts[3]
                if aid not in answer2parentQ.keys(): # not in saved answers, skip
                    continue
                qid = answer2parentQ[aid]
                if aid in answer_to_stop: # an event related to a locked or closed answer
                    continue
                if qid in question_to_stop: # a question has been closed or locked, stop adding new related event
                    continue
                if qid not in qid2universal_timesteps.keys():
                    # answer edition before answer creation for answer {aid} in question {qid}, because not accumulate
                    qid2universal_timesteps[qid]=[]
        
                qid2universal_timesteps[qid].append(ts)
                
            elif eventType==3 : # event of voting to an answer of a question
                aid = ts[3] 
                if aid not in answer2parentQ.keys(): # not in saved answers, skip
                    continue
                qid = answer2parentQ[aid]
                if aid in answer_to_stop: # an event related to a locked or closed answer
                    continue
                if qid in question_to_stop: # a question has been closed or locked, stop adding new related event
                    continue
                if qid not in qid2universal_timesteps.keys():
                    # answer vote before answer creation for answer {aid} in question {qid}, because not accumulate
                    qid2universal_timesteps[qid]=[]

                qid2universal_timesteps[qid].append(ts)

            elif eventType==4: # event of accepting an answer of a question
                aid = ts[3]
                acceptedA +=1
                answer_to_remove.append(aid)

            elif eventType==5:    # event of close a post which could be a question or an answer
                pid = ts[3]
                if pid in answer2parentQ.keys(): # an answer is closed, stop adding all the furture events about this answer
                    closedA.append(pid)
                    answer_to_stop.append(pid)
                elif pid in answer2parentQ.values(): # an question is closed, stop adding all the furture events about this question
                    closedQ.append(pid)
                    question_to_stop.append(pid)
                    
            elif eventType==6:    # event of lock a post which could be a question or an answer
                pid = ts[3]
                if pid in answer2parentQ.keys(): # an answer is locked , stop adding all the furture events about this answer
                    lockedA.append(pid)
                    answer_to_stop.append(pid)
                elif pid in answer2parentQ.values(): # an question is locked, stop adding all the furture events about this answer
                    lockedQ +=1
                    question_to_stop.append(pid)
                    
            elif eventType==7:    # event of delete a post which could be a question or an answer
                pid = ts[3]
                if pid in answer2parentQ.keys(): # an answer is delete, remove it
                    deletedA.append(pid)
                    answer_to_remove.append(pid)
                elif pid in answer2parentQ.values(): # an question is delete
                    deletedQ +=1
                    question_to_remove.append(pid)

        # save results every year
        if accumulatedFlag:
            saveFilename = 'qid2universalTimeSteps_tillYear_'+str(year)+'.dict'
        else:
            saveFilename = 'qid2universalTimeSteps_year_'+str(year)+'.dict'
        with open(split_qid2uts_files_directory+'/'+saveFilename, 'wb') as outputFile:
            pickle.dump((qid2universal_timesteps, deletedQ,deletedA,lockedQ,lockedA,closedQ,closedA,acceptedA,answer_to_remove,answer_to_stop,question_to_remove,question_to_stop), outputFile)
            print(f"saved year {year} qid2universal_timesteps and other intermediate results. AccumulatedFlag: {accumulatedFlag}.")

        # clear universal_timesteps to save memory
        cur_uts.clear()

    if accumulatedFlag: # when accumulated, save a whole result
        # remove questions in question to remove
        print(f"total question count of qid2universal_timesteps before removing: {len(qid2universal_timesteps)}")
        for qid in question_to_remove:
            if qid in qid2universal_timesteps.keys():
                del qid2universal_timesteps[qid]
        print(f"total question count of qid2universal_timesteps after removing: {len(qid2universal_timesteps)}. AccumulatedFlag: {accumulatedFlag}.")

        # saved the whole qid2universal_timesteps and answer_to_remove into intermediate folder
        with open(intermediate_directory+'/'+'qid2universalTimeSteps_answerToRemove.dict', 'wb') as outputFile:
            pickle.dump((qid2universal_timesteps,answer_to_remove), outputFile)
    
    
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
    print('createQid2UniversalTimeSteps_yearly Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
