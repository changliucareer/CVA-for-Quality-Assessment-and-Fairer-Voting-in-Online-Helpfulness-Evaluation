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


def myAction (question_tuple,commName,universal_timesteps_ofCurQuestion,answer_to_remove):
    qid = question_tuple[0]
    content = question_tuple[1]
    print(f"processing {commName} question {qid} on {mp.current_process().name}")
    answerList = content['answerList']
    if 'voteList' in content.keys():
        voteList = content['voteList']
    else: # without real votes
        return None
    
    lengthList = content['lengthList']
    try:
        # ans2lenTupDict = {answerList[aIndex]:len_tup_list for aIndex,len_tup_list in enumerate(lengthList)}
        ans2lenTupDict = defaultdict()
        for aIndex,len_tup_list in enumerate(lengthList):
            if len(len_tup_list) > 0:
                ans2lenTupDict[answerList[aIndex]] = len_tup_list
            else: # length list of current answer is empty, remove this answer
                answer_to_remove.append(answerList[aIndex])

        assert all([len(l)>0 for a, l in ans2lenTupDict.items()])

        lessThan5AnaswersQ = 0
        # remove answers that are accepted or deleted
        filtered_answerList = [i for i in answerList if i not in answer_to_remove]

        # if the useful answer of the current question is less than 5, filter out this question, return None
        if len(filtered_answerList)<5:
            lessThan5AnaswersQ +=1
            return (None,None,lessThan5AnaswersQ)
    except Exception as e:
        print(e)

    # remove the events related to removed answers
    event_to_remove = []

    if universal_timesteps_ofCurQuestion == None:
        return None
    
    try:
        for utIndex, ut in enumerate(universal_timesteps_ofCurQuestion):
            eventType = ut[0]
            if eventType == 1:
                if ut[1] in answer_to_remove: # answer creation event of an removed answer, remove the event
                    event_to_remove.append(utIndex)
            elif eventType == 2:
                if ut[3] in answer_to_remove: # answer edition event of an removed answer, remove the event
                    event_to_remove.append(utIndex)
            elif eventType == 3: 
                targetAnswerId = ut[3]
                if targetAnswerId in answer_to_remove: # voting event of an removed answer, remove the event
                    event_to_remove.append(utIndex)
            elif eventType==4:    # event of accepted an answer, remove the event
                event_to_remove.append(utIndex)
            elif eventType == 5 or eventType == 6: # closing or locking event
                event_to_remove.append(utIndex)
            elif eventType==7:    # event of delete a post which could be a question or an answer, remove the event
                event_to_remove.append(utIndex)
    except Exception as e:
        print(e)
        
    cur_ut = [ut for (utIndex, ut) in enumerate(universal_timesteps_ofCurQuestion) if utIndex not in event_to_remove] # only remain 3 types events (answer creation events or answer edition events or voting events)
    #clear universal_timesteps_ofCurQuestion to save memory
    universal_timesteps_ofCurQuestion.clear()

    # double check whether only 3 types events left in cur_ut
    assert sum([0 if ut[0]<=3 else 1 for ut in cur_ut]) ==0

    # create a matrix (rows are answers remain, columns are answer creation events or answer edition events or voting events)
    n_answers = len(filtered_answerList)
    n_events = len(cur_ut)
    vote_matrix = lil_matrix( (n_answers,n_events), dtype=np.int8 ) # initialize vote matrix, only store 0,1 or -1, dtype=int8
    length_matrix = lil_matrix( (n_answers,n_events), dtype=np.int32 ) # initialize lenght matrix, to store integers over thousands, dtype = int32 (only changing spot is non-zero to save memory) 
    
    try:
        for eventIndex, ut in enumerate(cur_ut):
            eventType = ut[0]
            if eventType == 1: # answer creation event 
                answerId = ut[1]
                answerIndex = filtered_answerList.index(answerId)
                len_tup_list = ans2lenTupDict[answerId]
                initial_length = len_tup_list[0][1] # the first recorded length as initial length
                # update length_matrix
                length_matrix[answerIndex,eventIndex:] = initial_length
            
            elif eventType == 2: # answer edition event 
                phId = ut[1]
                answerId = ut[3]
                answerIndex = filtered_answerList.index(answerId)
                len_tup_list = ans2lenTupDict[answerId]
                new_length = None
                for tup in len_tup_list:
                    if tup[0]==phId: # find the corresponding length to this post history id
                        new_length = tup[1]
                        break
                if new_length == None: # didn't find a length, error
                    print("Exception! didn't find a length, error")
                    return None
                # update length_matrix
                length_matrix[answerIndex,eventIndex:] = new_length
            elif eventType == 3: # voting event
                voteId = ut[1]
                targetAnswerId = ut[3]
                targetAnswerIndex = filtered_answerList.index(targetAnswerId)
                vote = None
                for tup in voteList:
                    if tup[0]==voteId: # find the corresponding vote type to this vote id
                        vote = tup[1]
                        break
                if vote == None: # didn't find a vote, error
                    print("Exception! didn't find a vote, error" )
                    return None  
                # update vote_matrix
                vote_matrix[targetAnswerIndex,eventIndex] = vote
    except Exception as e:
        print(e)

    # update content
    content['filtered_answerList'] = filtered_answerList
    content['local_timesteps'] = cur_ut # only contains 3 types of events (answer creation, answer editing, voting)
    content['vote_matrix'] = vote_matrix
    content['length_matrix'] = length_matrix
    

    print(f"{mp.current_process().name} return")
    return (qid,content,lessThan5AnaswersQ)
    

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'intermediate_data_folder')

    with open(final_directory+'/'+'universal_timesteps_afterCombineQAVH.dict', 'rb') as inputFile:
        universal_timesteps = pickle.load( inputFile)
    
    with open(final_directory+'/'+'answer2parentQLookup.dict', 'rb') as inputFile:
        answer2parentQ = pickle.load( inputFile)

    ### convert universal_timesteps to dict and qid as key
    qid2universal_timesteps = defaultdict()

    # variables for statistics
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

    # check whether already done part of universal time steps processing
    preDone_utIndex = 0
    resultFile = final_directory +'/'+ 'createMatrixIntermediateResults.dict'
   
    if os.path.exists(resultFile):
        print(f"{commName} has already done part of this step, continue from there...")
        with open(resultFile, 'rb') as inputFile:
            preDone_utIndex, qid2universal_timesteps, deletedQ,deletedA,lockedQ,lockedA,closedQ,closedA,acceptedA,answer_to_remove,answer_to_stop,question_to_remove,question_to_stop = pickle.load( inputFile)

    
    total_len_ut = len(universal_timesteps)
    for utIndex, ts in enumerate(universal_timesteps):
        if utIndex < preDone_utIndex: # skip the done steps
            continue
        
        # save results every 10000000 steps
        if utIndex % 10000000 == 0:
            with open(final_directory+'/'+'createMatrixIntermediateResults.dict', 'wb') as outputFile:
                pickle.dump((utIndex, qid2universal_timesteps, deletedQ,deletedA,lockedQ,lockedA,closedQ,closedA,acceptedA,answer_to_remove,answer_to_stop,question_to_remove,question_to_stop), outputFile)

        print(f"converting {utIndex+1}/{total_len_ut} time step of whole universal_timesteps.")
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
                print(f"Exception: answer edition before answer creation for answer {aid} in question {qid}")
                continue
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
                print(f"Exception: answer vote before answer creation for answer {aid} in question {qid}")
                continue
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

    # remove questions in question to remove
    print(f"total question count of qid2universal_timesteps before removing: {len(qid2universal_timesteps)}")
    for qid in question_to_remove:
        if qid in qid2universal_timesteps.keys():
            del qid2universal_timesteps[qid]
    print(f"total question count of qid2universal_timesteps after removing: {len(qid2universal_timesteps)}")

    # saved the qid2universal_timesteps and answer_to_remove
    with open(final_directory+'/'+'qid2universalTimeSteps_answerToRemove.dict', 'wb') as outputFile:
        pickle.dump((qid2universal_timesteps,answer_to_remove), outputFile)

    # clear universal_timesteps to save memory
    universal_timesteps.clear()
    
    with open(final_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistory.dict', 'rb') as inputFile:
        Questions = pickle.load( inputFile)

     # remove questions in Questions to remove
    print(f"total question count of Questions before removing: {len(Questions)}")
    for qid in question_to_remove:
        if qid in Questions.keys():
            del Questions[qid]
    print(f"total question count of qid2universal_timesteps after removing: {len(Questions)}")

    
    # process Questions chunk by chunk
    n_proc = mp.cpu_count()-2 # left 2 cores to do others
    all_outputs = []
    questionsWithoutRealVotesCount = 0
    universal_timesteps_forEachQuestion = [qid2universal_timesteps[qid] if qid in qid2universal_timesteps.keys() else None for qid in Questions.keys()]
    # clear qid2universal_timesteps dict
    qid2universal_timesteps.clear()

    with mp.Pool(processes=n_proc) as pool:
        args = zip(list(Questions.items()),[commName]*len(Questions),universal_timesteps_forEachQuestion,[answer_to_remove]*len(Questions))
        # issue tasks to the process pool and wait for tasks to complete
        try:
            # results = pool.starmap(myAction, args , chunksize=1)
            results = pool.starmap(myAction, args)
        except Exception as e:
            print(e)
        # process pool is closed automatically
        for res in results:
            if res != None:
                if isinstance(res, str):
                    print(res)
                else:
                    all_outputs.append(res)
            else: # res being None indicate this question doesn't have real votes
                questionsWithoutRealVotesCount +=1

    # clear the Questions
    original_QuestionCount = len(Questions)
    Questions.clear()

    # combine all_outputs
    # combined all Questions
    all_Questions = defaultdict()
    
    # for statistics
    # total counts of deleted, locked, closed cases
    total_deletedQ = deletedQ
    total_deletedA = len(deletedA)
    total_lockedQ = lockedQ
    total_lockedA = len(lockedA)
    total_closedQ = len(closedQ)
    total_closedA = len(closedA)
    total_acceptedA = acceptedA

    total_lessThan5AnaswersQ = 0

    for tup in all_outputs:
        if tup == None:
            continue
        # for combine outputs
        qid = tup[0]
        value = tup[1]
        if (qid != None) and (value != None): 
            all_Questions[qid] = value

        # for statistics
        lessThan5AnaswersQ = tup[2]
        if lessThan5AnaswersQ>0:
            total_lessThan5AnaswersQ +=1

    logfilename = 'createMatrix_Log.txt'
    logtext = f"Count of filtered Questions: {len(all_Questions)} out of total original {original_QuestionCount}\n"
    current_directory = os.getcwd()
    writeIntoLog(logtext, current_directory , logfilename)
    logtext = f"total_deletedQ :{total_deletedQ}, total_deletedA :{total_deletedA}, \n"
    logtext += f"total_lockedQ:{total_lockedQ}, total_lockedA :{total_lockedA}, \n"
    logtext += f"total_closedQ :{total_closedQ}, total_closedA :{total_closedA}, \n"
    logtext += f"total_acceptedA :{total_acceptedA},\n"
    logtext += f"total_lessThan5AnaswersQ:{total_lessThan5AnaswersQ}\n"
    logtext += f"questionsWithoutRealVotesCount:{questionsWithoutRealVotesCount}\n"
    writeIntoLog(logtext, commDir, logfilename)
  
    # check whether have intermediate data folder, create one if not
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'intermediate_data_folder')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    # save all Questions
    with open(final_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrix.dict', 'wb') as outputFile:
        pickle.dump(all_Questions, outputFile)

    
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
    # myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1]) 
    # test on comm "sitecore.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[253][0], commDir_sizes_sortedlist[253][1])
    # test on comm "sound.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[272][0], commDir_sizes_sortedlist[272][1])
    

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
    print('creating Matrix Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
