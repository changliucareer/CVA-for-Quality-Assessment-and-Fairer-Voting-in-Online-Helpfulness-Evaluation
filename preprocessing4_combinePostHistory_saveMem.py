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

def combineLengthList(ori_ll, chunk_ll):
    assert(len(ori_ll)==len(chunk_ll)) # should has the same number of answers
    for ansIndex, length_events in enumerate(chunk_ll):
        ori_ll[ansIndex].extend(length_events)
        # sort by the post history id of length updated
        ori_ll[ansIndex].sort(key= lambda x: x[0])
    return ori_ll


def myAction (chunk,chunk_start_index,commName,QuestionId2AnswerListDict,answer2parentQ):
    print(f"{commName} current chunk running on {mp.current_process().name}")

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

    # create universal timesteps of post history
    # it's a list of tuple, in format (eventType, posthistoryId, datetime, postId)
    universal_timesteps = []

    # keep a map of postId to rowIndex of PostHistory.csv
    phId2rowIndexOfPostHistoryCSV = defaultdict()

    # keep a light-weighted dict to map questionId to lengthList
    QuestionId2LengthListDict = defaultdict()

    # keep a map of answerId to posthistoryId that contains img 
    answerId2phId_img = defaultdict()
    # keep a map of answerId to posthistoryId that contains code
    answerId2phId_code = defaultdict()
    
    for index,row in chunk.iterrows():

        if row["PostHistoryTypeId"]==2: # Initial Body - The first raw body text a post is submitted with.
            phid = int(row["Id"]) # post history Id
            pid = int( row["PostId"] )
            phId2rowIndexOfPostHistoryCSV[phid] = chunk_start_index + index
            # convert datetime info string into datetime object
            datetime_str = row['CreationDate'] 
            date_time_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
            if pid in answer2parentQ.keys(): # this is an answer
                aid = pid
                qid = answer2parentQ[aid] # corresponding question's id
                text = row['Text']
                if not isinstance(text, str): # text is nan, not a string, skip
                    continue
                if qid in QuestionId2AnswerListDict.keys(): # corresponding question is in our Questions
                    answerList =  QuestionId2AnswerListDict[qid]
                    if qid not in QuestionId2LengthListDict.keys(): # current question has no lenghtList yet
                        lengthList = [ [] for _ in range(len(answerList)) ] # create an empty list to store lengths of each answer
                        QuestionId2LengthListDict[qid] = lengthList
                    # find the corresponding answer index
                    try:
                        ansIndex = answerList.index(aid)
                    except Exception as e:
                        print(e)
                    # update the corresponding answer's length event as a tuple (lengthOfText, posthistoryId)
                    QuestionId2LengthListDict[qid][ansIndex].append((phid,len(text)))

                    # check whether has image or code in the answer text
                    if '<img' in text:
                        answerId2phId_img[aid] = phid
                    if '<code' in text:
                        answerId2phId_code[aid] = phid
                    continue

        elif row["PostHistoryTypeId"]==5: # Edit Body - modified post body (raw markdown)
            phid = int(row["Id"]) # post history Id
            pid = int( row["PostId"] )
            phId2rowIndexOfPostHistoryCSV[phid] = chunk_start_index + index
            # convert datetime info string into datetime object
            datetime_str = row['CreationDate'] 
            date_time_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
            if pid in answer2parentQ.keys(): # this is an answer
                aid = pid
                qid = answer2parentQ[aid] # corresponding question's id
                text = row['Text']
                if not isinstance(text, str): # text is nan, not a string, skip
                    continue
                if qid in QuestionId2AnswerListDict.keys(): # corresponding question is in our Questions
                    answerList =  QuestionId2AnswerListDict[qid]
                    # in case the initial Body event is not in the currect chunk, still need to check whether has lengthList
                    if qid not in QuestionId2LengthListDict.keys(): # current question has not lenghtList yet
                        lengthList = [ [] for _ in range(len(answerList)) ] # create an empty list to store lengths of each answer
                        QuestionId2LengthListDict[qid] = lengthList
                    # find the corresponding answer index
                    ansIndex = answerList.index(aid)
                    # update the corresponding answer's length event as a tuple (lengthOfText, posthistoryId)
                    QuestionId2LengthListDict[qid][ansIndex].append((phid,len(text)))
                    universal_timesteps.append((2, phid, date_time_obj, aid))
                    # check whether has image or code in the answer text
                    if '<img' in text:
                        answerId2phId_img[aid] = phid
                    if '<code' in text:
                        answerId2phId_code[aid] = phid
                    continue 

        elif row["PostHistoryTypeId"]==8: # Rollback Body - reverted body (raw markdown) (treated the same as edit body)
            phid = int(row["Id"]) # post history Id
            pid = int( row["PostId"] )
            phId2rowIndexOfPostHistoryCSV[phid] = chunk_start_index + index
            # convert datetime info string into datetime object
            datetime_str = row['CreationDate'] 
            date_time_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
            if pid in answer2parentQ.keys(): # this is an answer
                aid = pid
                qid = answer2parentQ[aid] # corresponding question's id
                text = row['Text']
                if not isinstance(text, str): # text is nan, not a string, skip
                    continue
                if qid in QuestionId2AnswerListDict.keys(): # corresponding question is in our Questions
                    answerList =  QuestionId2AnswerListDict[qid]
                    # in case the initial Body event is not in the currect chunk, still need to check whether has lengthList
                    if qid not in QuestionId2LengthListDict.keys(): # current question has not lenghtList yet
                        lengthList = [ [] for _ in range(len(answerList)) ] # create an empty list to store lengths of each answer
                        QuestionId2LengthListDict[qid] = lengthList
                    # find the corresponding answer index
                    ansIndex = answerList.index(aid)
                    # update the corresponding answer's length event as a tuple (lengthOfText, posthistoryId)
                    QuestionId2LengthListDict[qid][ansIndex].append((phid,len(text)))
                    universal_timesteps.append((2, phid, date_time_obj, aid))
                    # check whether has image or code in the answer text
                    if '<img' in text:
                        answerId2phId_img[aid] = phid
                    if '<code' in text:
                        answerId2phId_code[aid] = phid
                    continue 

        elif row["PostHistoryTypeId"]==12: # Post Deleted - post voted to be removed
            phid = int(row["Id"]) # post history Id
            pid = int( row["PostId"] ) # could be a question or an answer
            phId2rowIndexOfPostHistoryCSV[phid] = chunk_start_index + index
            # convert datetime info string into datetime object
            datetime_str = row['CreationDate'] 
            date_time_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
            universal_timesteps.append((7, phid, date_time_obj, pid))
            continue


        elif row["PostHistoryTypeId"]==10: # Post Closed - A post was voted to be closed.
            phid = int(row["Id"]) # post history Id
            pid = int( row["PostId"] ) # could be a question or an answer
            phId2rowIndexOfPostHistoryCSV[phid] = chunk_start_index + index
            # convert datetime info string into datetime object
            datetime_str = row['CreationDate'] 
            date_time_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
            universal_timesteps.append((5, phid, date_time_obj, pid))
            continue
            
        elif row["PostHistoryTypeId"]==14: # Post Locked - A post was locked by a moderator.
            phid = int(row["Id"]) # post history Id
            pid = int( row["PostId"] ) # could be a question or an answer
            phId2rowIndexOfPostHistoryCSV[phid] = chunk_start_index + index
            # convert datetime info string into datetime object
            datetime_str = row['CreationDate'] 
            date_time_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
            universal_timesteps.append((6, phid, date_time_obj, pid))
            continue


    print(f"{mp.current_process().name} return")
    return QuestionId2LengthListDict, universal_timesteps, phId2rowIndexOfPostHistoryCSV, answerId2phId_img, answerId2phId_code
    

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'intermediate_data_folder')
    
    print("loading Questions...")
    with open(final_directory+'/'+'QuestionsWithAnswersWithVotes.dict', 'rb') as inputFile:
        Questions = pickle.load( inputFile)

    with open(final_directory+'/'+'answer2parentQLookup.dict', 'rb') as inputFile:
        answer2parentQ = pickle.load( inputFile)

    # extract a light-weighted Questions only with answerList
    print("extracting a light-weighted Questions only with answerList...")
    QuestionId2AnswerListDict = defaultdict()
    for qid, content in Questions.items():
        if 'answerList' in content.keys():
            QuestionId2AnswerListDict[qid]=content['answerList']

    print(f"extracted light-weighted Questions only with answerList count:{len(QuestionId2AnswerListDict)} from {len(Questions)} original Questions.")

    #read data in chunks of 10 thousand rows at a time
    chunk_size = 10000
    chunksIter = pd.read_csv('PostHistory.csv',chunksize=chunk_size) # return type <class 'pandas.io.parsers.readers.TextFileReader'>

    done_looping = False
    chunk_batch = []
    # n_proc = mp.cpu_count()-2 # n_proc as batch size
    n_proc = 5
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
                args = zip(chunk_batch,[chunk_index*chunk_size]*len(chunk_batch),[commName]*len(chunk_batch),[QuestionId2AnswerListDict]*len(chunk_batch),[answer2parentQ]*len(chunk_batch))
                
                with mp.Pool(processes=n_proc) as pool:
                    # issue tasks to the process pool and wait for tasks to complete
                    #An iterator is returned with the result for each function call
                    results = pool.starmap(myAction, args)
                    all_outputs.extend(results)
                    # process pool is closed automatically

                # increase the chunk_index
                chunk_index += 1
                # clear the chunk_batch
                chunk_batch = []
            chunk_batch.append(chunk)
    
    # process the last batch
    with mp.Pool(processes=n_proc) as pool:
        args = zip(chunk_batch,[chunk_index*chunk_size]*len(chunk_batch),[commName]*len(chunk_batch),[QuestionId2AnswerListDict]*len(chunk_batch),[answer2parentQ]*len(chunk_batch))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction, args)
        # process pool is closed automatically
        all_outputs.extend(results)

    # combine all_outputs

    # combined universal_timestemps
    universal_timesteps_ofPH =[]

    # keep a map of voteId to rowIndex of Posts.csv
    phId2rowIndexOfPostHistoryCSV= defaultdict()
    # keep a map of answerId to posthistoryId that contains img 
    answerId2phId_img = defaultdict()
    # keep a map of answerId to posthistoryId that contains code
    answerId2phId_code = defaultdict()

    print("start to combine all outputs...")
    for tup in all_outputs:
        QuestionId2LengthListDict = tup[0]
        ut = tup[1]
        id2index = tup[2]
        aid2phid_img = tup[3]
        aid2phid_code = tup[4]

        # add length List to Questions
        for qid, lengthList in QuestionId2LengthListDict.items():
            if 'lengthList' not in Questions[qid].keys(): # when there's no lengthList for this question, initialized it
                Questions[qid]['lengthList'] = lengthList
            else:
                Questions[qid]['lengthList'] = combineLengthList(Questions[qid]['lengthList'], lengthList)

        # combine universal_timestemps
        universal_timesteps_ofPH.extend(ut)

        # combine phId2rowIndexOfPostHistoryCSV
        for id, index in id2index.items():
            if id not in phId2rowIndexOfPostHistoryCSV.keys(): # only update when id is not in keys()
                phId2rowIndexOfPostHistoryCSV[id] = index
        
        # combine answerId2phId_img
        for id, index in aid2phid_img.items():
            if id not in answerId2phId_img.keys(): # only update when id is not in keys()
                answerId2phId_img[id] = index

        # combine answerId2phId_code
        for id, index in aid2phid_code.items():
            if id not in answerId2phId_code.keys(): # only update when id is not in keys()
                answerId2phId_code[id] = index

    all_outputs.clear()

    logfilename = 'combinePostHistory_Log.txt'
    logtext = f"Count of Questions: {len(Questions)}\n"

    # check whether have intermediate data folder, create one if not
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'intermediate_data_folder')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    # save all Questions
    print("savine Questions with post history...")
    with open(final_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistory.dict', 'wb') as outputFile:
        pickle.dump(Questions, outputFile)
    Questions.clear()

    # save phId2rowIndexOfPostHistoryCSV
    with open(final_directory+'/'+'phId2rowIndexOfPostHistoryCSV.dict', 'wb') as outputFile:
        pickle.dump(phId2rowIndexOfPostHistoryCSV, outputFile)

    # save answerId2phId_img
    with open(final_directory+'/'+'answerId2phId_img.dict', 'wb') as outputFile:
        pickle.dump(answerId2phId_img, outputFile)

    # save answerId2phId_code
    with open(final_directory+'/'+'answerId2phId_code.dict', 'wb') as outputFile:
        pickle.dump(answerId2phId_code, outputFile)
    
    print("loading universal time steps...")
    with open(final_directory+'/'+'universal_timesteps_afterCombineQAV.dict', 'rb') as inputFile:
        universal_timesteps_ofQAV = pickle.load( inputFile)

    # combine and sort universal_timesteps
    print("combining universal time steps...")
    universal_timesteps = universal_timesteps_ofQAV
    universal_timesteps.extend(universal_timesteps_ofPH)
    universal_timesteps.sort(key=lambda x: (x[2],x[1])) # sort all, first based on the datetime, then based on the id
    
    # save universal_timesteps
    print("saving universal time steps...")
    with open(final_directory+'/'+'universal_timesteps_afterCombineQAVH.dict', 'wb') as outputFile:
        pickle.dump(universal_timesteps, outputFile)

    logtext += f"Total event timesteps: {len(universal_timesteps)}\n"
    logtext += f"Count of posthistory: {len(universal_timesteps_ofPH)}\n"
    writeIntoLog(logtext, commDir, logfilename)


def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    # test on comm "sitecore.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[253][0], commDir_sizes_sortedlist[253][1])
    
    # run on all communities other than stackoverflow
    # finishedCount = 0
    # processes = []
    # for tup in commDir_sizes_sortedlist[300:340]:
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
    #     if len(processes)==10:
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
    print('processing PostHistory.csv and combine Q and A and V and Post History Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
