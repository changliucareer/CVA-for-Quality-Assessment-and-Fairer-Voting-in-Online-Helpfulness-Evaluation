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


def myAction (chunk,chunk_index,commName, postIdsNotInModerates, saveChunkDir):
    print(f"{commName} current chunk running on {mp.current_process().name}")
    logText = f"when process chunk {chunk_index} for {commName},\n"
    
    # keep a dict of postHistoryId to text body, for current chunk
    postHistoryId2text_chunk = defaultdict()

    # keep a dictionary of postId to posthistoryId and other needed info
    postId2Moderates = defaultdict()
   
    # Dict postId2Moderates has postId as key, 
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

    postIdsNotInModerates_dict = dict(postIdsNotInModerates)
    
    for index,row in chunk.iterrows():
        pid = int( row["PostId"] ) # post Id (could be question or answer id)
        
        if pid not in postIdsNotInModerates_dict.keys(): # the post is not we selected, skip
            continue 

        type = postIdsNotInModerates_dict[pid]

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
            votedUserIds = None
            migrationDetails = None
            
            # update postId2Moderates
            if pid not in postId2Moderates.keys(): # a new question
                curDict = defaultdict()
                if uid != None:
                    curDict['initialUserId'] = uid
                curDict['moderateList'] = []
                curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                curDict['type'] = type
                postId2Moderates[pid] = curDict
            else: # update an exsit question
                if 'initialUserId' in postId2Moderates[pid].keys():
                    if postId2Moderates[pid]['initialUserId'] != uid: # initial user id exception
                        logText += f"!!!initial User Exception: previous one is {postId2Moderates[pid]['initialUserId']}, current is {uid} for question {qid} phid {phId}\n"
                        continue
                    else:
                        postId2Moderates[pid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                else:
                    postId2Moderates[pid]['initialUserId'] = uid # update the initial user id
                    postId2Moderates[pid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            # update postHistoryId2text_chunk
            postHistoryId2text_chunk[phId] = text


        elif phType == 2: # Initial Body - The first raw body text a post is submitted with.
            votedUserIds = None
            migrationDetails = None

            # update postId2Moderates
            if pid not in postId2Moderates.keys(): # a new post
                curDict = defaultdict()
                if uid != None:
                    curDict['initialUserId'] = uid
                curDict['moderateList'] = []
                curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                curDict['type'] = type
                postId2Moderates[pid] = curDict
            else: # update an exsit post
                if 'initialUserId' in postId2Moderates[aid].keys():
                    if postId2Moderates[pid]['initialUserId'] != uid: # initial user id exception
                        logText += f"!!!initial User Exception: previous one is {postId2Moderates[pid]['initialUserId']}, current is {uid} for answer {aid} phid {phId}\n"
                        continue
                    else:
                        postId2Moderates[pid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                else:
                    postId2Moderates[pid]["initialUserId"] = uid # update the initial user id
                    postId2Moderates[pid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            # update postHistoryId2text_chunk
            postHistoryId2text_chunk[phId] = text
        

        elif phType in [3,4,6]: # 3: Initial Tags - The first tags a question is asked with. 4: Edit Title - A question's title has been changed. 6: Edit Tags - A question's tags have been changed.
            votedUserIds = None
            migrationDetails = None

            if pid not in postId2Moderates.keys(): # a new question for current chunk, but not a new question for whole comm
                curDict = defaultdict()
                curDict['moderateList'] = []
                curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                curDict['type'] = type
                postId2Moderates[pid] = curDict
            else:
                # update an exsit question's qid2Moderates
                postId2Moderates[pid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            # update postHistoryId2text_chunk
            postHistoryId2text_chunk[phId] = text
        

        elif phType in [5, 7,8,9]: # Edit Body - A post's body has been changed, the raw text is stored here as markdown. or Rollback Tile, Body or Tags
            votedUserIds = None
            migrationDetails = None

            if pid not in postId2Moderates.keys(): # a new answer for current chunk, but not a new answer for whole comm
                curDict = defaultdict()
                curDict['moderateList'] = []
                curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                curDict['type'] = type
                postId2Moderates[pid] = curDict
            else:
                # update an exsit answer's aid2Moderates
                postId2Moderates[pid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

            # update postHistoryId2text_chunk
            postHistoryId2text_chunk[phId] = text


        elif phType in [10,11,12,13,14,15]: # A post was voted to be closed, or reopened, or removed or restored , or locked or unlocked
            try:
                votedUserIds = [float(d['Id']) for d in json.loads(text)['Voters']]
            except:
                votedUserIds = None
                logText += f"when extract votedUserIds for phId {phId} phType={phType}, TypeError occurred. text is '{text}'\n"
            
            migrationDetails = None
            
           
            if pid not in postId2Moderates.keys(): # a new answer for current chunk, but not a new answer for whole comm
                curDict = defaultdict()
                curDict['moderateList'] = []
                curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                curDict['type'] = type
                postId2Moderates[pid] = curDict
            else:
                # update an exsit answer's aid2Moderates
                postId2Moderates[pid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))


        elif phType in [16,21]: # 16: Community Owned - A post has become community owned. 21: Post Disassociated - An admin removes the OwnerUserId from a post.
            votedUserIds = None
            migrationDetails = None

            
            if pid not in postId2Moderates.keys(): # a new answer for current chunk, but not a new answer for whole comm
                curDict = defaultdict()
                curDict['moderateList'] = []
                curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                curDict['type'] = type
                postId2Moderates[pid] = curDict
            else:
                # update an exsit answer's aid2Moderates
                postId2Moderates[pid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
        
    
        elif phType == 17: # Post Migrated - A post was migrated.
            votedUserIds = None
            migrationDetails = text
            
            if pid not in postId2Moderates.keys(): # a new answer for current chunk, but not a new answer for whole comm
                curDict = defaultdict()
                curDict['moderateList'] = []
                curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                curDict['type'] = type
                postId2Moderates[pid] = curDict
            else:
                # update an exsit answer's aid2Moderates
                postId2Moderates[pid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))

        elif phType in [18, 19, 20, 22]: # - 18: Question Merged - A question has had another, deleted question merged into itself.- 19: Question Protected - A question was protected by a moderator - 20: Question Unprotected - A question was unprotected by a moderator - 22: Question Unmerged - A previously merged question has had its answers and votes restored.
            votedUserIds = None
            migrationDetails = None
            
            # update postId2Moderates
            if pid not in postId2Moderates.keys(): # a new question for current chunk, but not a new question for whole comm
                curDict = defaultdict()
                curDict['moderateList'] = []
                curDict["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
                curDict['type'] = type
                postId2Moderates[pid] = curDict
            else:
                # update an exsit question's postId2Moderates
                postId2Moderates[pid]["moderateList"].append((phId, phType, uid, comment, closedReasonId, votedUserIds, migrationDetails, revisionGUID))
        
        else: # phType is not in 1 to 22 which are mentioned in Readme.txt
            logText += f"!!!phType Exception: phType {phType} is not mentioned in Readme.txt for phId {phId} \n"


    # save postHistoryId2text_chunk as json in subfolder
    with open(saveChunkDir + f'/postHistoryId2text_chunk_{chunk_index}.json', "w") as outfile: 
        json.dump( postHistoryId2text_chunk, outfile)        
        print(f"saved postHistoryId2text_chunk_{chunk_index} for {commName}")


    print(f"{mp.current_process().name} return")
    logText += f"finished chunck {chunk_index}\n"
    print(logText)
    return postId2Moderates, logText
    

def myFun(commName, commDir, commSize, csvFileDir):

    logFileName = "preprocessing12_getHistoryOfPostsThatvotedSpamOrOffensive_Log.txt"
    print(f"comm {commName} running on {mp.current_process().name}")
    t1 = time.time()

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    final_directory = os.path.join(commDir, r'intermediate_data_folder')

    # # # if already done for this comm, return
    # resultFiles = ['postId2offensiveVotes.json','postId2spamVotes.json']
    # resultFiles = [final_directory+'/'+f for f in resultFiles]
    # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    #     print(f"{commName} has already done this step.")
    #     return

    # load
    with open(final_directory+'/'+'answer2parentQLookup.dict', 'rb') as inputFile:
        answer2parentQ = pickle.load( inputFile)
        allPostIds = set(answer2parentQ.keys()).union(set(answer2parentQ.values()))

    try:
        with open(final_directory+'/'+'postId2offensiveVotes.json') as json_file:
            postId2offensiveVotes = json.load(json_file)
        with open(final_directory+'/'+'postId2spamVotes.json') as json_file:
            postId2spamVotes = json.load(json_file)
    except:
        print(f"{commName} hasn't done preprocessing 11 yet.")
        return None
    
    try:
        with open(final_directory+'/'+'aid2Moderates.json') as json_file:
            aid2Moderates = json.load(json_file)
        with open(final_directory+'/'+'qid2Moderates.json') as json_file:
            qid2Moderates = json.load(json_file)
    except:
        print(f"{commName} hasn't done preprocessing 10 yet.")
        aid2Moderates = defaultdict()
        qid2Moderates = defaultdict()
    
    # first check whether in moderates data
    postIdsNotInModerates = []

    postId2Moderates_forOffensiveOrSpamPosts = defaultdict()

    # scan offensive posts
    offensivePostCountInPostHistory = 0
    offensivePostCountInPost = 0
    for pid in postId2offensiveVotes.keys():
        if pid in allPostIds:
            offensivePostCountInPost +=1

        mDict = defaultdict()
        if pid in aid2Moderates.keys():
            mDict = aid2Moderates[pid]
            offensivePostCountInPostHistory += 1
            mDict['type'] = 'offensive'
            postId2Moderates_forOffensiveOrSpamPosts[pid] = mDict
        elif pid in qid2Moderates.keys():
            mDict = qid2Moderates[pid]
            offensivePostCountInPostHistory += 1
            mDict['type'] = 'offensive'
            postId2Moderates_forOffensiveOrSpamPosts[pid] = mDict
        else:
            postIdsNotInModerates.append((pid, 'offensive'))
        
        

    # scan spam posts
    spamPostCountInPostHistory = 0
    spamPostCountInPost = 0
    for pid in postId2spamVotes.keys():
        if pid in allPostIds:
            spamPostCountInPost +=1

        mDict = defaultdict()
        if pid in aid2Moderates.keys():
            mDict = aid2Moderates[pid]
            spamPostCountInPostHistory +=1
            mDict['type'] = 'spam'
            postId2Moderates_forOffensiveOrSpamPosts[pid] = mDict

        elif pid in qid2Moderates.keys():
            mDict = qid2Moderates[pid]
            spamPostCountInPostHistory +=1
            mDict['type'] = 'spam'
            postId2Moderates_forOffensiveOrSpamPosts[pid] = mDict
        else:
            postIdsNotInModerates.append((pid, 'spam'))
        
    """
    # create a sub folder to store phId2text for each chunk files especially for offensive or spam posts
    saveChunkDir = os.path.join(final_directory, r'offensiveOrSpamPost_phId2Text_chunks_folder')
    if not os.path.exists(saveChunkDir):
        print("no offensiveOrSpamPost_phId2Text_chunks_folder, create one")
        os.makedirs(saveChunkDir)
    
    #read data in chunks of 10 thousand rows at a time
    chunk_size = 10000
    chunksIter = pd.read_csv('PostHistory.csv',chunksize=chunk_size) # return type <class 'pandas.io.parsers.readers.TextFileReader'>

    done_looping = False
    chunk_batch = []
    n_proc = mp.cpu_count()
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
                args = zip(chunk_batch,range(chunk_index, chunk_index+len(chunk_batch)),[commName]*len(chunk_batch),[postIdsNotInModerates]*len(chunk_batch), [saveChunkDir]*len(chunk_batch))
                
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
        args = zip(chunk_batch,range(chunk_index, chunk_index+len(chunk_batch)),[commName]*len(chunk_batch),[postIdsNotInModerates]*len(chunk_batch),[saveChunkDir]*len(chunk_batch))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction, args)
        # process pool is closed automatically
        all_outputs.extend(results)
    
    # combine all_output
    print("start to combine all outputs...")
    
    for tup in all_outputs:
        postId2Moderates, logText_chunk = tup
        logText += logText_chunk
        
        # combine 
        for pid,mDict in postId2Moderates.items():
            if pid not in postId2Moderates_forOffensiveOrSpamPosts.keys(): 
                postId2Moderates_forOffensiveOrSpamPosts[pid] = mDict
            else:
                if 'initialUserId' not in postId2Moderates_forOffensiveOrSpamPosts[pid].keys():
                    if 'initialUserId' in mDict.keys():
                        postId2Moderates_forOffensiveOrSpamPosts[pid]['initialUserId'] = mDict['initialUserId']
                
                postId2Moderates_forOffensiveOrSpamPosts[pid]['moderateList'].extend(mDict['moderateList'])

    all_outputs.clear()

    writeIntoLog(logText, commDir, logFileName)

    # save 
    with open(final_directory+'/'+'postId2Moderates_forOffensiveOrSpamPosts.json', 'w') as outputFile:
        json.dump(postId2Moderates_forOffensiveOrSpamPosts, outputFile)
        print(f"saved postId2Moderates_forOffensiveOrSpamPosts for {commName}")

    """

    if commName == 'stackoverflow':
        offensivePostCountInPostHistory = offensivePostCountInPost
        spamPostCountInPostHistory = offensivePostCountInPost

    logText = f"there are {len(postId2offensiveVotes)} posts having offensive votes, {offensivePostCountInPostHistory} are in post history;\n"
    logText += f"there are {len(postId2spamVotes)} posts having spam votes, {spamPostCountInPostHistory} are in post history.\n"
    writeIntoLog(logText, commDir, logFileName)

    # save in csv
    with open(csvFileDir, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( [commName,commSize,len(postId2offensiveVotes), offensivePostCountInPostHistory,len(postId2spamVotes), spamPostCountInPostHistory])


    elapsed = format_time(time.time() - t1)
    # Report progress.
    print(f"for {commName}, ")
    print('preprocessing12_getHistoryOfPostsThatvotedSpamOrOffensive Done.    Elapsed: {:}.\n'.format(elapsed))
    print(logText)




def main():
    rootDir = os.getcwd()
    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # # save csv
    csvFileDir = rootDir + '/allComm_OffensiveOrSpamPosts.csv'
    with open(csvFileDir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","commSize","offensive Post count", "offensive post in post history count","spam post count", "spam post in post history count"])


    # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1], commDir_sizes_sortedlist[166][2], csvFileDir)
    # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1], csvFileDir)
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1], csvFileDir)
    
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]
        commSize = tup[2]

        if commName == 'stackoverflow': # skip stackoverflow to run at the last
            stackoverflow_dir = commDir
            stackoverflow_size = commSize
            continue
        try:
            p = mp.Process(target=myFun, args=(commName,commDir, commSize, csvFileDir))
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
    myFun('stackoverflow', stackoverflow_dir, stackoverflow_size, csvFileDir)

    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('preprocessing12_getHistoryOfPostsThatvotedSpamOrOffensive Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
