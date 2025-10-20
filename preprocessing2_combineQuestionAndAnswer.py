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
    print(f"{commName} current chunk running on {mp.current_process().name}")

    # initialize Questions using Id in posts as the key with posttype=1
    Questions = defaultdict()

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
    universal_timesteps = []

    # keep a answerId to its parent questionId lookup dictionary key:aid, value:parent qid
    answer2parentQ = defaultdict()

    # keep a map of postId to rowIndex of Posts.csv
    postId2rowIndexOfPostsCSV = defaultdict()

    # keep a map of question Id to tags
    qid2tags = defaultdict()

    for index,row in chunk.iterrows():
        datetime_str = row['CreationDate'] 
        date_time_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")
        if row['PostTypeId']== 1 : # it's a question
            qid = int(row['Id']) # current question Id
            # Questions[qid]= {'answerList':[]} can't empty, there may be answers for this question before the question was created
            tagList = str(row['Tags']).replace('<', '').replace('>',' ').split()
            qid2tags[qid]= tagList
            universal_timesteps.append((0, qid ,date_time_obj))
            postId2rowIndexOfPostsCSV[qid] = chunk_start_index + index
        elif row['PostTypeId']== 2 : # it's an answer
            qid = int(row['ParentId'])
            aid = int(row['Id'])
            if qid not in Questions.keys():
                if qid in qid2tags.keys():
                    Questions[qid]= {'answerList':[aid], 'tagList':qid2tags[qid]}
                else:
                    Questions[qid]= {'answerList':[aid], 'tagList':[]}
            else:
                Questions[qid]['answerList'].append(aid)

            answer2parentQ[aid]=qid
            universal_timesteps.append((1, aid ,date_time_obj))
            postId2rowIndexOfPostsCSV[aid] = chunk_start_index + index
        else:# other types of post, ignore
            continue

    # fill out the empty tagList
    for qid, content in Questions.items():
        if len(content['tagList'])==0: # empty tagList
            if qid in qid2tags.keys():
                Questions[qid]['tagList'] = qid2tags[qid]

    print(f"{commName} {mp.current_process().name} return")

    return Questions, answer2parentQ, universal_timesteps, postId2rowIndexOfPostsCSV
    

def myFun(commName, commDir, return_size_dict):
    
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
    n_proc = mp.cpu_count()-2 # n_proc as batch size
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
    # combined all Questions
    all_Questions = defaultdict()
    # combined universal_timestemps
    universal_timesteps =[]
    # combined answer2parentQ lookup dict
    answer2parentQ = defaultdict()
    # keep a map of postId to rowIndex of Posts.csv
    postId2rowIndexOfPostsCSV = defaultdict()

    for tup in all_outputs:
        Questions = tup[0]
        a2q = tup[1]
        ut = tup[2]
        id2index = tup[3]

        # combine Questions
        for qid, value in Questions.items():
            if qid in all_Questions.keys():
                all_Questions[qid]['answerList'] = all_Questions[qid]['answerList'] + value['answerList']
                all_Questions[qid]['answerList'].sort()
                all_Questions[qid]['tagList'] = list( set(all_Questions[qid]['tagList'] + value['tagList']) )
            else:
                all_Questions[qid] = value
        
        # combine answer2parentQ
        for aid, pid in a2q.items():
            if aid not in answer2parentQ.keys(): # only update when aid is not in answer2parentQ
                answer2parentQ[aid] = pid

        # combine universal_timestemps
        universal_timesteps.extend(ut)

        # combine postId2rowIndexOfPostsCSV
        for id, index in id2index.items():
            if id not in postId2rowIndexOfPostsCSV.keys(): # only update when id is not in keys()
                postId2rowIndexOfPostsCSV[id] = index
    

    logfilename = 'combineQuestionAndAnwer_Log.txt'
    logtext = f"Total event timesteps: {len(universal_timesteps)}\n"
    logtext += f"Count of Questions: {len(all_Questions)}\n"
    logtext += f"Count of answer: {len(answer2parentQ)}\n"
    writeIntoLog(logtext, commDir, logfilename)
  
    # check whether have intermediate data folder, create one if not
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    # save to answer2parent Lookup 
    with open(final_directory+'/'+'answer2parentQLookup.dict', 'wb') as outputFile:
        pickle.dump(answer2parentQ, outputFile)
    
    # save all Questions
    with open(final_directory+'/'+'QuestionsWithAnswers.dict', 'wb') as outputFile:
        pickle.dump(all_Questions, outputFile)

    # save universal_timesteps
    with open(final_directory+'/'+'universal_timesteps_afterCombineQA.dict', 'wb') as outputFile:
        pickle.dump(universal_timesteps, outputFile)

    # save updatedQuestions
    with open(final_directory+'/'+'postId2rowIndexOfPostsCSV.dict', 'wb') as outputFile:
        pickle.dump(postId2rowIndexOfPostsCSV, outputFile)

    # save question and answer count into shared variable
    return_size_dict[commName] = {'questionCount': len(all_Questions), 'answerCount': len(answer2parentQ)}

    elapsed = format_time(time.time() - t1)
    # Report progress.
    print(f"for {commName}, ")
    print('processing Posts.csv and combine Q and A Done.    Elapsed: {:}.\n'.format(elapsed))


def main():

    t0=time.time()

    ## Load community direcotries .dict files
    with open('commDirectories.dict', 'rb') as inputFile:
        commDirDict = pickle.load( inputFile)
    print("CommDir loaded.")

    # for debug, only run on certain commu
    # commDirDict = {'stackoverflow':commDirDict['stackoverflow'], 'english.stackexchange':commDirDict['english.stackexchange']}
    # commDirDict = {'bicycles.stackexchange':commDirDict['bicycles.stackexchange'], 'english.stackexchange':commDirDict['english.stackexchange']}
    # commDirDict = {'webapps.stackexchange':commDirDict['webapps.stackexchange']}
    # commDirDict = {'datascience.stackexchange':commDirDict['datascience.stackexchange']}
    # commDirDict = {'coffee.stackexchange':commDirDict['coffee.stackexchange']}
    # commDirDict = {'ru.stackoverflow':commDirDict['ru.stackoverflow']}
    
    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    return_size_dict = manager.dict() # to save the question count and answer count of each community

    processes = []
    finishedCount = 0
    for commName, commDir in commDirDict.items():
        if commName == 'stackoverflow': # skip stackoverflow to run at the last
            continue
        p = mp.Process(target=myFun, args=(commName,commDir, return_size_dict))
        p.start()
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
    print("start to process stackoverflow...")
    myFun('stackoverflow', commDirDict['stackoverflow'], return_size_dict)

    # save return_size_dict
    # save the results to folder "SE_codes_2022"
    os.chdir(Original_Dir)
    with open('QandA_sizes_of_allComm.dict', 'wb') as outputFile:
        pickle.dump(return_size_dict, outputFile)

    # combine commDirDict and return_size_dict and sort by the size
    allComm_directories_sizes_sortedlist = []
    for commName, commDir in commDirDict.items():
        qCount = return_size_dict[commName]['questionCount']
        aCount = return_size_dict[commName]['answerCount']
        allComm_directories_sizes_sortedlist.append((commName,commDir,qCount+aCount))

    allComm_directories_sizes_sortedlist.sort(key=lambda x : x[2])

    # save allComm_directories_sizes_sortedlist
    with open('allComm_directories_sizes_sortedlist.dict', 'wb') as outputFile:
        pickle.dump(allComm_directories_sizes_sortedlist, outputFile)
    print(f"saved newMahine_allComm_directories_sizes_sortedlist in {os.getcwd()}.")
   
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('processing Posts.csv and combine Q and A Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
