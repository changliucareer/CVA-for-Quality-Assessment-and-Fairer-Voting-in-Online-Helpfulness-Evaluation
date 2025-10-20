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

def combineLengthList(ori_ll, chunk_ll):
    assert(len(ori_ll)==len(chunk_ll)) # should has the same number of answers
    for ansIndex, length_events in enumerate(chunk_ll):
        ori_ll[ansIndex].extend(length_events)
        # sort by the post history id of length updated
        ori_ll[ansIndex].sort(key= lambda x: x[0])
    return ori_ll


def myAction (question_tuple,commName):
    qid = question_tuple[0]
    content = question_tuple[1]
    print(f"processing {commName} question {qid} on {mp.current_process().name}")
    filtered_answerList = content['filtered_answerList']
    vote_matrix = content['vote_matrix'].toarray() # convert sparse matrix to np.array
    length_matrix = content['length_matrix'].toarray() # convert sparse matrix to np.array

    n_answers = vote_matrix.shape[0]
    n_events = vote_matrix.shape[1] # only contains 3 types of events (answer creation, answer editing, voting)
    n_vote_events = 0 # number of voting events
    n_write_events = 0 # number of wrting events
    n_edit_events = 0 # number of editing events

    # keep a local event list with all raw data needed to training data
    eventList = []
    J = 0    # the number of answers till a time, starting from 0 (J should be J_i^{t-1})
    ranksOfAnswersBeforeT = None # initialized 
    # eventList is a list of dictionary in format of  
    #  {'et':eventType, 'ai':answerIndex,'J':J, 'v':vote, 'pvr':pvr,'nvr':nvr,'n_pos':n_pos,'n_neg':n_neg, 
    #   'ranks':ranksOfAnswersBeforeT,'rl':relativeLength} 
    #   eventType is a str with only 3 possible values 'v': vote or 'w': write or 'e': edit

    # with the initial pseudo-votes [1, -1].
    initial_pos = 1
    initial_neg = 1

    n_pos = None 
    n_neg = None

    curExistingAnswerIndex = 0 # initialize current existing answer index

    for t in range(n_events):
        cur_col_ofVM = vote_matrix[:,t]
        if cur_col_ofVM.any(): # current column of vote matrix has non-zero value (vote event)
            if t<1: # if a vote event occurs at the first column, it's an exception. The answer creation should be the first event
                print("Exception")
                return
            eventType = 'v'
            n_vote_events +=1
            # get the indices of the elements that are non-zero.
            out_tpl = np.nonzero(cur_col_ofVM)
            assert len(out_tpl[0])==1 # should be only 1 non-zero value
            ans_index = out_tpl[0][0] # usually the first non-zero value is the current vote
            cur_vote = vote_matrix[ans_index,t]
            cur_length = length_matrix[ans_index,t]

            # convert vote from -1/1 to 0/1
            if cur_vote ==1:
                vote = 1 # set positive vote as 1
            else:
                vote = 0 # set negative vote as 0

            # vote ratio related computations  
            votes_sofar = vote_matrix[ans_index, :t]
            flatten = votes_sofar.flatten()
            binCount = np.bincount(np.where(flatten==-1, 2, flatten)) # input array of bincount must be 1 dimension, nonnegative ints, so replace all -1 with 2
            if len(binCount)<2:
                n_pos = initial_pos
            else:
                n_pos =  binCount[1] + initial_pos
            if len(binCount)<3:
                n_neg = initial_neg
            else:
                n_neg = binCount[2] + initial_neg
            
            # for two sides parametrization
            voteTotal = n_pos+n_neg 
            pvr = n_pos / voteTotal 
            nvr = n_neg / voteTotal 

            # for one side parametrization
            if n_pos >= n_neg:
                seen_pos_votes = n_pos - n_neg +1
                seen_total_votes =  n_pos - n_neg +2
                seen_pvr = seen_pos_votes/seen_total_votes
            else: # n_neg > n_pos
                seen_neg_votes = n_neg - n_pos +1
                seen_total_votes =  n_neg - n_pos +2
                seen_pvr = - seen_neg_votes/seen_total_votes

            # length ratio related computation
            relativeLength = 1 # initialized lengthRatio as 1
            if cur_length != 0:
                nonZeroCount = np.count_nonzero(length_matrix[:,t])
                curAvgLength = np.sum(length_matrix[:,t]) / nonZeroCount
                relativeLength = cur_length/curAvgLength
            else: # current answer's length is 0 
                print("Exception!!!")

            # rank related computations
            voteSum = np.sum(vote_matrix[:curExistingAnswerIndex+1,:t],axis=1) # answer-wise sum
            aiWithVoteSum = [(ai,vc) for (ai,vc) in enumerate(voteSum)] # ai is answer index, vc is vote count
            aiWithVoteSum.sort(reverse=True, key=lambda x:(x[1],x[0])) # sort by the vote count and then by the answer index
            aiWithRanks = [(tup[0],r+1) for (r,tup) in enumerate(aiWithVoteSum)] # tup[0] is the answer index, r is the rank
            aiWithRanks.sort(key=lambda x:x[0])
            ranksOfAnswersBeforeT = [r for (ai,r) in aiWithRanks]

            eventList.append( {'et':eventType, 'ai':ans_index,'J':J, 'v':vote, 'pvr':pvr,'nvr':nvr,'seen_pvr':seen_pvr,'n_pos':n_pos,'n_neg':n_neg,
                                    'ranks':ranksOfAnswersBeforeT,'rl':relativeLength} )
            
        else: # current column of vote matrix has all 0s (new answer creation event or answer length changing event)
            cur_col_ofLM = length_matrix[:,t] # the current column of length matrix
            # get the indices of the elements that are non-zero.
            out_tpl = np.nonzero(cur_col_ofLM) # may have multiple non-zero values
            ans_index = out_tpl[0][-1] # usually the last non-zero value is the current corresponding edited answer
            curExistingAnswerIndex = ans_index
            cur_length = length_matrix[ans_index,t]
            
            # length ratio related computation
            relativeLength = 1 # initialized lengthRatio as 1
            if cur_length != 0:
                nonZeroCount = np.count_nonzero(cur_col_ofLM)
                curAvgLength = np.sum(cur_col_ofLM) / nonZeroCount
                relativeLength = cur_length/curAvgLength
            else: # current answer's length is 0 
                print("Exception!!!")

            if t==0: # the first event must be a new answer creation event
                eventType = 'w'
                eventList.append( {'et':eventType, 'ai':ans_index,'J':J, 'n_pos':0,'n_neg':0,'ranks':ranksOfAnswersBeforeT} )
                n_write_events +=1
                J +=1
            else:
                previous_length = length_matrix[ans_index,t-1]
                if previous_length == 0: # if the length at previous time step is 0, this is a new answer creation event
                    eventType = 'w'
                    eventList.append( {'et':eventType, 'ai':ans_index,'J':J, 'n_pos':0,'n_neg':0,'ranks':ranksOfAnswersBeforeT} )
                    n_write_events +=1
                    J +=1
                else:
                    eventType = 'e'
                    eventList.append( {'et':eventType, 'ai':ans_index,'J':J, 'n_pos':0,'n_neg':0,'ranks':ranksOfAnswersBeforeT} )
                    n_edit_events +=1

    # double check
    try:   
        assert ( n_vote_events+n_write_events+n_edit_events == n_events)
    except:
        print(AssertionError)
        return None

    if n_vote_events==0: # current question doesn't have any vote event, disgard
        return None
    
    # save eventList
    content['eventList']=eventList
    print(f"{mp.current_process().name} return")
    return (qid,content, n_vote_events, n_write_events, n_edit_events)


def myPart(commName, commDir, qf, split_QuestionsWithEventList_files_directory, part, logFileName):
    t1=time.time()
    with open(qf, 'rb') as inputFile:
        Questions = pickle.load( inputFile)
    
    original_QuestionCount = len(Questions)

    # process Questions chunk by chunk
    n_proc = mp.cpu_count()
    all_outputs = []
    with mp.Pool(processes=n_proc) as pool:
        args = zip(list(Questions.items()),[commName]*len(Questions))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction, args , chunksize=n_proc)
        # process pool is closed automatically
        for res in results:
            if res != None:
                if isinstance(res, str):
                    print(res)
                else:
                    all_outputs.append(res)

    Questions.clear()

    # combine all_outputs
    # combined all Questions
    all_Questions = defaultdict()

    # for statistics
    total_n_vote_events = 0 # total number of voting events
    total_n_write_events = 0 # total number of wrting events
    total_n_edit_events = 0 # total number of editing events

    for tup in all_outputs:
        # for combine outputs
        qid = tup[0]
        value = tup[1]
        all_Questions[qid] = value

        # for statistics
        n_vote_events= tup[2]
        n_write_events= tup[3]
        n_edit_events= tup[4]
        total_n_vote_events += n_vote_events 
        total_n_write_events += n_write_events 
        total_n_edit_events += n_edit_events 

    all_outputs.clear()

    elapsed1 = format_time(time.time() - t1)

    logtext = f'for part {part},\n'
    logtext += f"total_n_vote_events: {total_n_vote_events}, total_n_write_events: {total_n_write_events}, total_n_edit_events: {total_n_edit_events}\n"
    logtext += f"total questions with voting events: {len(all_Questions)} out of original {original_QuestionCount} questions\n"
    logtext += "Elapsed: {:}.\n".format(elapsed1)
    writeIntoLog(logtext, commDir , logFileName)
    print(logtext)
    
    # save all dataset
    with open(split_QuestionsWithEventList_files_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList_part_'+str(part)+'.dict', 'wb') as outputFile:
        pickle.dump(all_Questions, outputFile) 
        print(f"QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList_part_{part} saved for {commName}.")

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'preprocessing6_createEventList_forSplittedFiles_Log.txt'

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    
    # # check whether already done this step, skip
    # resultFiles = ['QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict']
    # resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
    # # if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
    # if os.path.exists(resultFiles[0]):
    #     # file modification timestamp of a file
    #     m_time = os.path.getmtime(resultFiles[0])
    #     # convert timestamp into DateTime object
    #     dt_m = datetime.datetime.fromtimestamp(m_time)
    #     # print('Modified on:', dt_m)
    #     if dt_m >= datetime.datetime(2024, 8, 1):
    #         print(f"{commName} has already done this step at {dt_m.date()}.")
    #         return

    # creat a folder if not exists to store splitted data files
    splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    if not os.path.exists(splitFolder_directory):
        print("Exception: no splitted_intermediate_data_folder")
        writeIntoLog("Exception: no splitted_intermediate_data_folder", commDir, logFileName)

    # under splitted_intermediate_data_folder, create a folder to store updated Questions part
    split_QuestionsWithEventList_files_directory = os.path.join(splitFolder_directory, r'QuestionsPartsWithEventList')
    if not os.path.exists(split_QuestionsWithEventList_files_directory):
        print("no split_QuestionsWithEventList_files_directory, create one")
        os.makedirs(split_QuestionsWithEventList_files_directory)
    
    # under splitted_intermediate_data_folder, access a folder that store Question parts
    split_QuestionsWithMatrix_files_directory = os.path.join(splitFolder_directory, r'QuestionsPartsWithAnswersWithVotesWithPostHistoryWithMatrix')
    if not os.path.exists(split_QuestionsWithMatrix_files_directory):
        print("Exception: no split_QuestionsWithMatrix_files_directory")
        writeIntoLog("Exception: no split_QuestionsWithMatrix_files_directory", commDir, logFileName)

    Questions_subfolders = [ f.path for f in os.scandir(split_QuestionsWithMatrix_files_directory) if f.path.split('/')[-1].startswith("QuestionsWithAnswersWithVotesWithPostHistoryWithMatrix_part_") ]
    Questions_subfolders.sort(key=lambda f: int(f.split('/')[-1].split('_')[-1].split('.')[0]))
    print(f"current total parts : {len(Questions_subfolders)}")
    time.sleep(5)
    
    # multiprocessing the parts
    startPart = 0 # only start with the first undone part
    processes = []
    for qf in Questions_subfolders:
        part = int(qf.split('/')[-1].split('_')[-1].split('.')[0])
        # skip the finished parts
        if part < startPart:
            print(f"skip done part {part}...")
            continue

        print(f"processing part {part}...")

        try:
            p = mp.Process(target=myPart, args=(commName, commDir, qf, split_QuestionsWithEventList_files_directory, part, logFileName))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            return

        processes.append(p)
        if len(processes)== 24:
            # make sure all p finish before main process finish
            for p in processes:
                p.join()
            # clear processes
            processes = []

    # join the last batch of processes
    if len(processes)>0:
        # make sure all p finish before main process finish
        for p in processes:
            p.join()
    
    print(f"finished processing splitted files for comm {commName}.")

def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    # test on comm "stackoverflow" to debug
    # myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1]) 
    
    
    # run on  stackoverflow
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
        if len(processes)==16:
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
    print('creating EventList for SOF Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
