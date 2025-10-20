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
import json

def myAction (chunk,chunk_index,commName, answer2parentQ, saveChunkDir):
    print(f"{commName} current chunk {chunk_index} running on {mp.current_process().name}")
    logText = f"when process chunk {chunk_index},\n"
    
    # keep a dict of postHistoryId to text body, for current chunk
    postHistoryId2text_chunk = defaultdict()

    # keep 2 dictionaries of answerId/questionId to posthistoryId and other needed info
    aid2Moderates = defaultdict()
    qid2Moderates = defaultdict()
    # Dict aid2Moderates has answerId as key, (qid2Moderates has questionId as key)
    # and the value is a dict, keys are "initialUserId" and "moderateList",
    # and the moderateList is a list of tuple (phId, phType, userId, comment, closedReasonId, votedUserIds, migrationDetails)
    # (The first time of the list for each answer must be an "Initial Body" event, and the corresponding userId as the initial user)

    #  - PostHistoryTypeId
	# 		- 1: Initial Title - The first title a question is asked with.
	# 		- 2: Initial Body - The first raw body text a post is submitted with.
	# 		- 3: Initial Tags - The first tags a question is asked with.
	# 		- 4: Edit Title - A question's title has been changed.
	# 		- 5: Edit Body - A post's body has been changed, the raw text is stored here as markdown.
	# 		- 6: Edit Tags - A question's tags have been changed.
	# 		- 7: Rollback Title - A question's title has reverted to a previous version.
	# 		- 8: Rollback Body - A post's body has reverted to a previous version - the raw text is stored here.
	# 		- 9: Rollback Tags - A question's tags have reverted to a previous version.
	# 		- 10: Post Closed - A post was voted to be closed.
	# 		- 11: Post Reopened - A post was voted to be reopened.
	# 		- 12: Post Deleted - A post was voted to be removed.
	# 		- 13: Post Undeleted - A post was voted to be restored.
	# 		- 14: Post Locked - A post was locked by a moderator.
	# 		- 15: Post Unlocked - A post was unlocked by a moderator.
	# 		- 16: Community Owned - A post has become community owned.
	# 		- 17: Post Migrated - A post was migrated.
	# 		- 18: Question Merged - A question has had another, deleted question merged into itself.
	# 		- 19: Question Protected - A question was protected by a moderator
	# 		- 20: Question Unprotected - A question was unprotected by a moderator
	# 		- 21: Post Disassociated - An admin removes the OwnerUserId from a post.
	# 		- 22: Question Unmerged - A previously merged question has had its answers and votes restored.
    # - CloseReasonId
	# 		- 1: Exact Duplicate - This question covers exactly the same ground as earlier questions on this topic; its answers may be merged with another identical question.
	# 		- 2: off-topic
	# 		- 3: subjective
	# 		- 4: not a real question
	# 		- 7: too localized
    
    for index,row in chunk.iterrows():
        pid = int( row["PostId"] ) # post Id (could be question or answer id)
        selectedPostIds = set(answer2parentQ.keys()).union(set(answer2parentQ.values()))
        if pid not in selectedPostIds: # the post is not we selected, skip
            continue 

        phId = int(row["Id"]) # post history Id
        phType = int(row["PostHistoryTypeId"]) # post hitory type
        uid = row["UserId"] # user Id
        if uid != uid: # uid is nan, not a string
            uid = None
        text = row["Text"] # text body
        
        comment = row["Comment"]
        if not isinstance(comment, str): # comment is nan, not a string
            comment = None
        
        closedReasonId = row["CloseReasonId"]
        if not isinstance(closedReasonId, str): # closedReasonId is nan, not a string
            closedReasonId = None

        revisionGUID = row["RevisionGUID"]
        if not isinstance(revisionGUID, str): # revisionGUID is nan, not a string
            revisionGUID = None

        ### start to process different post history types event
        
        if phType == 1: # Initial Title - The first title a question is asked with.
            if pid not in answer2parentQ.values(): # not a question, exception
                logText += f"!!! not a question Exception: post {pid} is not a question for phType {phType}\n"
                continue
            
            qid = pid
            votedUserIds = None
            migrationDetails = None
            
            # update qid2Moderates
            if qid not in qid2Moderates.keys(): # a new question
                curDict = defaultdict()
                if uid != None:
                    curDict['initialUserId'] = uid
                curDict['moderateList'] = []
                curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                qid2Moderates[qid] = curDict
            else: # update an exsit question
                if 'initialUserId' in qid2Moderates[qid].keys():
                    if qid2Moderates[qid]['initialUserId'] != uid: # initial user id exception
                        logText += f"!!!initial User Exception: previous one is {qid2Moderates[qid]['initialUserId']}, current is {uid} for question {qid} phid {phId}\n"
                        continue
                    else:
                        qid2Moderates[qid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                else:
                    qid2Moderates[qid]['initialUserId'] = uid # update the initial user id
                    qid2Moderates[qid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            # update postHistoryId2text_chunk
            postHistoryId2text_chunk[phId] = text


        elif phType == 2: # Initial Body - The first raw body text a post is submitted with.
            votedUserIds = None
            migrationDetails = None

            if pid in answer2parentQ.keys(): # the post is an answer
                aid = pid
                # update aid2Moderates
                if aid not in aid2Moderates.keys(): # a new answer
                    curDict = defaultdict()
                    if uid != None:
                        curDict['initialUserId'] = uid
                    curDict['moderateList'] = []
                    curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    aid2Moderates[aid] = curDict
                else: # update an exsit answer
                    if 'initialUserId' in aid2Moderates[aid].keys():
                        if aid2Moderates[aid]['initialUserId'] != uid: # initial user id exception
                            logText += f"!!!initial User Exception: previous one is {aid2Moderates[aid]['initialUserId']}, current is {uid} for answer {aid} phid {phId}\n"
                            continue
                        else:
                            aid2Moderates[aid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    else:
                        aid2Moderates[aid]["initialUserId"] = uid # update the initial user id
                        aid2Moderates[aid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            else: # the post is a question
                qid = pid
                # update qid2Moderates
                if qid not in qid2Moderates.keys(): # a new question
                    curDict = defaultdict()
                    if uid != None:
                        curDict['initialUserId'] = uid
                    curDict['moderateList'] = []
                    curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    qid2Moderates[qid] = curDict
                else: # update an exsit question
                    if 'initialUserId' in qid2Moderates[qid].keys():
                        if qid2Moderates[qid]['initialUserId'] != uid: # initial user id exception
                            logText += f"!!!initial User Exception: previous one is {qid2Moderates[qid]['initialUserId']}, current is {uid} for question {qid} phid {phId}\n"
                            continue
                        else:
                            qid2Moderates[qid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    else:
                        qid2Moderates[qid]['initialUserId'] = uid # update the initial user id
                        qid2Moderates[qid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            # update postHistoryId2text_chunk
            postHistoryId2text_chunk[phId] = text
        

        elif phType in [3,4,6]: # 3: Initial Tags - The first tags a question is asked with. 4: Edit Title - A question's title has been changed. 6: Edit Tags - A question's tags have been changed.
            if pid not in answer2parentQ.values(): # not a question, exception
                logText += f"!!! not a question Exception: post {pid} is not a question for phType {phType}\n"
                continue

            qid = pid
            votedUserIds = None
            migrationDetails = None

            if qid not in qid2Moderates.keys(): # a new question for current chunk, but not a new question for whole comm
                curDict = defaultdict()
                curDict['moderateList'] = []
                curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                qid2Moderates[qid] = curDict
            else:
                # update an exsit question's qid2Moderates
                qid2Moderates[qid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            # update postHistoryId2text_chunk
            postHistoryId2text_chunk[phId] = text
        

        elif phType in [5, 7,8,9]: # Edit Body - A post's body has been changed, the raw text is stored here as markdown. or Rollback Tile, Body or Tags
            votedUserIds = None
            migrationDetails = None

            if pid in answer2parentQ.keys(): # the post is an answer
                aid = pid
                if aid not in aid2Moderates.keys(): # a new answer for current chunk, but not a new answer for whole comm
                    curDict = defaultdict()
                    curDict['moderateList'] = []
                    curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    aid2Moderates[aid] = curDict
                else:
                    # update an exsit answer's aid2Moderates
                    aid2Moderates[aid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            else: # the post is a question
                qid = pid
                if qid not in qid2Moderates.keys(): # a new question for current chunk, but not a new question for whole comm
                    curDict = defaultdict()
                    curDict['moderateList'] = []
                    curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    qid2Moderates[qid] = curDict
                else:
                    # update an exsit question's qid2Moderates
                    qid2Moderates[qid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            # update postHistoryId2text_chunk
            postHistoryId2text_chunk[phId] = text


        elif phType in [10,11,12,13,14,15]: # A post was voted to be closed, or reopened, or removed or restored , or locked or unlocked
            try:
                votedUserIds = [float(d['Id']) for d in json.loads(text)['Voters']]
            except:
                votedUserIds = None
                logText += f"when extract votedUserIds for phId {phId} phType={phType}, TypeError occurred. text is '{text}'\n"
            
            migrationDetails = None
            
            if pid in answer2parentQ.keys(): # the post is an answer
                aid = pid
                if aid not in aid2Moderates.keys(): # a new answer for current chunk, but not a new answer for whole comm
                    curDict = defaultdict()
                    curDict['moderateList'] = []
                    curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    aid2Moderates[aid] = curDict
                else:
                    # update an exsit answer's aid2Moderates
                    aid2Moderates[aid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            else: # the post is a question
                qid = pid
                if qid not in qid2Moderates.keys(): # a new question for current chunk, but not a new question for whole comm
                    curDict = defaultdict()
                    curDict['moderateList'] = []
                    curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    qid2Moderates[qid] = curDict
                else:
                    # update an exsit question's qid2Moderates
                    qid2Moderates[qid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                

        elif phType in [16,21]: # 16: Community Owned - A post has become community owned. 21: Post Disassociated - An admin removes the OwnerUserId from a post.
            votedUserIds = None
            migrationDetails = None

            if pid in answer2parentQ.keys(): # the post is an answer
                aid = pid
                if aid not in aid2Moderates.keys(): # a new answer for current chunk, but not a new answer for whole comm
                    curDict = defaultdict()
                    curDict['moderateList'] = []
                    curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    aid2Moderates[aid] = curDict
                else:
                    # update an exsit answer's aid2Moderates
                    aid2Moderates[aid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            else: # the post is a question
                qid = pid
                if qid not in qid2Moderates.keys(): # a new question for current chunk, but not a new question for whole comm
                    curDict = defaultdict()
                    curDict['moderateList'] = []
                    curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    qid2Moderates[qid] = curDict
                else:
                    # update an exsit question's qid2Moderates
                    qid2Moderates[qid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
            
            
        
        elif phType == 17: # Post Migrated - A post was migrated.
            votedUserIds = None
            migrationDetails = text
            
            if pid in answer2parentQ.keys(): # the post is an answer
                aid = pid
                if aid not in aid2Moderates.keys(): # a new answer for current chunk, but not a new answer for whole comm
                    curDict = defaultdict()
                    curDict['moderateList'] = []
                    curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    aid2Moderates[aid] = curDict
                else:
                    # update an exsit answer's aid2Moderates
                    aid2Moderates[aid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            else: # the post is a question
                qid = pid
                if qid not in qid2Moderates.keys(): # a new question for current chunk, but not a new question for whole comm
                    curDict = defaultdict()
                    curDict['moderateList'] = []
                    curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                    qid2Moderates[qid] = curDict
                else:
                    # update an exsit question's qid2Moderates
                    qid2Moderates[qid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
        
        elif phType in [18, 19, 20, 22]: # - 18: Question Merged - A question has had another, deleted question merged into itself.- 19: Question Protected - A question was protected by a moderator - 20: Question Unprotected - A question was unprotected by a moderator - 22: Question Unmerged - A previously merged question has had its answers and votes restored.
            if pid not in answer2parentQ.values(): # not a question, exception
                logText += f"!!! not a question Exception: post {pid} is not a question for phType {phType}\n"
                continue
            
            qid = pid
            votedUserIds = None
            migrationDetails = None
            
            # update qid2Moderates
            if qid not in qid2Moderates.keys(): # a new question for current chunk, but not a new question for whole comm
                curDict = defaultdict()
                curDict['moderateList'] = []
                curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                qid2Moderates[qid] = curDict
            else:
                # update an exsit question's qid2Moderates
                qid2Moderates[qid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
        
        else: # phType is not in 1 to 22 which are mentioned in Readme.txt
            logText += f"!!!phType Exception: phType {phType} is not mentioned in Readme.txt for phId {phId} \n"


    # save postHistoryId2text_chunk as json in subfolder
    with open(saveChunkDir + f'/postHistoryId2text_chunk_{chunk_index}.json', "w") as outfile: 
        json.dump( postHistoryId2text_chunk, outfile)        
        print(f"saved postHistoryId2text_chunk_{chunk_index} for {commName}")


    print(f"{mp.current_process().name} return")
    logText += f"finished chunck {chunk_index}\n"
    print(logText)
    return aid2Moderates, qid2Moderates, logText
    

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = "preprocessing10_extractModeratingInfo_Log.txt"
    logText = ""

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'intermediate_data_folder')

    # check whether already done this step, skip
    resultFiles = ['pid2Moderates.csv']
    resultFiles = [final_directory+'/'+f for f in resultFiles]
    # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    if os.path.exists(resultFiles[0]):
        print(f"{commName} has already done this step.")
        # print(f"fixing csv files for {commName}...")
        # with open(final_directory+'/'+'qid2Moderates.json', 'r') as inputFile:
        #     qid2Moderates_total = json.load( inputFile)
        # with open(final_directory+'/'+'aid2Moderates.json', 'r') as inputFile:
        #     aid2Moderates_total = json.load( inputFile)
        # # # save csv
        # with open(final_directory+'/'+'pid2Moderates.csv', 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',',
        #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     writer.writerow( ["postId","postType", "postHistoryType","postHistoryId","userId","comment","closedReasonId","votedUserIds", "migrationDetails", "revisionGUID"])

        #     for qid, mDict in qid2Moderates_total.items():
        #         tupList = mDict['moderateList']
        #         for tup in tupList:
        #             commentString = str(tup[3]).replace(',',' ')
        #             closedReasonIdString = str(tup[4]).replace(',',' ')
        #             votedUserIdsString = str(tup[5]).replace(',',' ')
        #             migrationDetailsString = str(tup[6]).replace(',',' ')
        #             writer.writerow((qid, 'Q',tup[1], tup[0], tup[2], commentString, closedReasonIdString, votedUserIdsString, migrationDetailsString, tup[7]))
            
        #     for aid, mDict in aid2Moderates_total.items():
        #         tupList = mDict['moderateList']
        #         for tup in tupList:
        #             commentString = str(tup[3]).replace(',',' ')
        #             closedReasonIdString = str(tup[4]).replace(',',' ')
        #             votedUserIdsString = str(tup[5]).replace(',',' ')
        #             migrationDetailsString = str(tup[6]).replace(',',' ')
        #             writer.writerow((aid, 'A',tup[1], tup[0], tup[2], commentString, closedReasonIdString, votedUserIdsString, migrationDetailsString, tup[7]))

        return

    # create a sub folder to store phId2text for each chunk files
    saveChunkDir = os.path.join(final_directory, r'phId2Text_chunks_folder')
    if not os.path.exists(saveChunkDir):
        print("no phId2Text_chunks_folder, create one")
        os.makedirs(saveChunkDir)

    
    print("loading ...")
    with open(final_directory+'/'+'answer2parentQLookup.dict', 'rb') as inputFile:
        answer2parentQ = pickle.load( inputFile)
    
    # extract questionId2answerIdList
    questionId2answerIdList = defaultdict()
    for aid, qid in answer2parentQ.items():
        if qid not in questionId2answerIdList.keys():
            questionId2answerIdList[qid] = [aid]
        else:
            questionId2answerIdList[qid].append(aid)
    # save
    with open(final_directory+'/'+'questionId2answerIdList.dict', 'wb') as outputFile:
        pickle.dump(questionId2answerIdList, outputFile)
        print(f"saved questionId2answerIdList for {commName}")
    questionId2answerIdList.clear() # to save memory

    #read data in chunks of 10 thousand rows at a time
    chunk_size = 10000
    chunksIter = pd.read_csv('PostHistory.csv',chunksize=chunk_size) # return type <class 'pandas.io.parsers.readers.TextFileReader'>

    done_looping = False
    chunk_batch = []
    n_proc = 1
    # n_proc = 5
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
                args = zip(chunk_batch,range(chunk_index, chunk_index+len(chunk_batch)),[commName]*len(chunk_batch),[answer2parentQ]*len(chunk_batch), [saveChunkDir]*len(chunk_batch))
                
                with mp.Pool(processes=n_proc) as pool:
                    # issue tasks to the process pool and wait for tasks to complete
                    #An iterator is returned with the result for each function call
                    results = pool.starmap(myAction, args)
                    all_outputs.extend(results)
                    # process pool is closed automatically

                # increase the chunk_index
                chunk_index += len(chunk_batch)
                # clear the chunk_batch
                chunk_batch = []
            
            chunk_batch.append(chunk)
    
    # process the last batch
    with mp.Pool(processes=n_proc) as pool:
        args = zip(chunk_batch,range(chunk_index, chunk_index+len(chunk_batch)),[commName]*len(chunk_batch),[answer2parentQ]*len(chunk_batch),[saveChunkDir]*len(chunk_batch))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction, args)
        # process pool is closed automatically
        all_outputs.extend(results)

    # combine all_outputs
    aid2Moderates_total = defaultdict()
    qid2Moderates_total = defaultdict()
   
    print("start to combine all outputs...")
    for tup in all_outputs:
        aid2Moderates, qid2Moderates, logText_chunk = tup
        logText += logText_chunk
        
        # combine 
        for aid,mDict in aid2Moderates.items():
            if aid not in aid2Moderates_total.keys(): 
                aid2Moderates_total[aid] = mDict
            else:
                if 'initialUserId' not in aid2Moderates_total[aid].keys():
                    if 'initialUserId' in mDict.keys():
                        aid2Moderates_total[aid]['initialUserId'] = mDict['initialUserId']
                
                aid2Moderates_total[aid]['moderateList'].extend(mDict['moderateList'])
                    

        for qid,mDict in qid2Moderates.items():
            if qid not in qid2Moderates_total.keys(): 
                qid2Moderates_total[qid] = mDict
            else:
                if 'initialUserId' not in qid2Moderates_total[qid].keys():
                    if 'initialUserId' in mDict.keys():
                        qid2Moderates_total[qid]['initialUserId'] = mDict['initialUserId']
                
                qid2Moderates_total[qid]['moderateList'].extend(mDict['moderateList'])

    all_outputs.clear()

    writeIntoLog(logText, commDir, logFileName)
    
    # sort moderate List
    for aid in aid2Moderates_total.keys():
        aid2Moderates_total[aid]['moderateList'].sort(key = lambda tup:tup[0])
    
    for qid in qid2Moderates_total.keys():
        qid2Moderates_total[qid]['moderateList'].sort(key = lambda tup:tup[0])
    
    # save 
    with open(final_directory+'/'+'aid2Moderates.json', 'w') as outputFile:
        json.dump(aid2Moderates_total, outputFile)
        print(f"saved aid2Moderates_total for {commName}")
    
    with open(final_directory+'/'+'qid2Moderates.json', 'w') as outputFile:
        json.dump(qid2Moderates_total, outputFile)
        print(f"saved qid2Moderates_total for {commName}")

    # # save csv
    with open(final_directory+'/'+'pid2Moderates.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["postId","postType", "postHistoryType","postHistoryId","userId","comment","closedReasonId","votedUserIds", "migrationDetails", "revisionGUID"])

        for qid, mDict in qid2Moderates_total.items():
            tupList = mDict['moderateList']
            for tup in tupList:
                commentString = str(tup[3]).replace(',',' ')
                closedReasonIdString = str(tup[4]).replace(',',' ')
                votedUserIdsString = str(tup[5]).replace(',',' ')
                migrationDetailsString = str(tup[6]).replace(',',' ')
                writer.writerow((qid, 'Q',tup[1], tup[0], tup[2], commentString, closedReasonIdString, votedUserIdsString, migrationDetailsString, tup[7]))
        
        for aid, mDict in aid2Moderates_total.items():
            tupList = mDict['moderateList']
            for tup in tupList:
                commentString = str(tup[3]).replace(',',' ')
                closedReasonIdString = str(tup[4]).replace(',',' ')
                votedUserIdsString = str(tup[5]).replace(',',' ')
                migrationDetailsString = str(tup[6]).replace(',',' ')
                writer.writerow((aid, 'A',tup[1], tup[0], tup[2], commentString, closedReasonIdString, votedUserIdsString, migrationDetailsString, tup[7]))


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
    # test on comm "sitecore.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[253][0], commDir_sizes_sortedlist[253][1])
    # test on comm "sound.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[272][0], commDir_sizes_sortedlist[272][1])
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist[359:]:
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

   
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('processing 10 extract Moderating Info Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
