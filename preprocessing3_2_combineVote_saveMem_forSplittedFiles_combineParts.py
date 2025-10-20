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

def combineUniversalTimeSteps(ut1, ut2):
    
    # segmentated the time steps by eventType of 0 and 1 which are question and answer creation events, then the rest would be vote and AA events
    # keep the question and answer creation events not change, only resort the vote events in between

    new_universal_timesteps = []

    i = 0 # index for ut1
    j = 0 # index for ut2
    votesInBetween_ut1 = []
    votesInBetween_ut2 = []
    while (i<len(ut1) and j<len(ut2)):
        print(f"Processing {i}/{len(ut1)} of ut1 and {j}/{len(ut2)} of ut2 ...")
        if ut1[i] == ut2[j]: # same time step, must be questions or answer creation event, add to new ut
            assert ut1[i][0]<=1
            # insert the vote in between in previous segment
            votesInBetween = votesInBetween_ut1 + votesInBetween_ut2
            if len(votesInBetween)>0:
                # re-sort first based on the datetime, then based on the id
                votesInBetween.sort(key=lambda t: (t[2], t[1]))
                # insert the votes in between in previous segment
                new_universal_timesteps.extend(votesInBetween)
            # add the current same ut
            new_universal_timesteps.append(ut1[i])
            i +=1
            j +=1

        else: # not the same
            if ut1[i][0]>1:
                if ut2[j][0]>1: # both vote events
                    votesInBetween_ut1.append(ut1[i])
                    votesInBetween_ut2.append(ut2[j])
                    i +=1
                    j +=1
                else: # only ut1[i] is vote event
                    votesInBetween_ut1.append(ut1[i])
                    i +=1
            else:
                if ut2[j][0]>1: # only ut2[j] is vote event
                    votesInBetween_ut2.append(ut2[j])
                    j +=1
                else: # both are not vote event, but not the same, Exception
                    print(f"Exception: {ut1[i]} and {ut2[j]}")

    if i<len(ut1) or j<len(ut2): # if there is remaining in ut1 or ut2, add to new ut
        # insert the vote in between in previous segment
        votesInBetween = votesInBetween_ut1 + votesInBetween_ut2 + ut1[i:] + ut2[j:]
        # re-sort first based on the datetime, then based on the id
        votesInBetween.sort(key=lambda t: (t[2], t[1]))
        # insert the votes in between in previous segment
        new_universal_timesteps.extend(votesInBetween)
    
    return new_universal_timesteps

def segmentateUT(utList):
    # the format of ut
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
    aid_to_VoteUTList_Dict = defaultdict()
    for ut in utList:
        if ut[0]==1: # an answer creation ut
            aid = ut[1]
            aid_to_VoteUTList_Dict[aid]=[] # initialize the vote ut list segement as empty list for this answer
        elif ut[0]==0: # an question creation ut, skip
            continue
        else: # vote ut or accepted answer ut
            aid = ut[3]
            if aid in aid_to_VoteUTList_Dict.keys():
                aid_to_VoteUTList_Dict[aid].append(ut)

    return aid_to_VoteUTList_Dict

def combineSegmentedUTList(universal_timesteps_ofQA, universalTimeStepsFiles):
    new_ut = []

    # segamentate each utList in universalTimeStepsFiles
    aid_to_VoteUTList_Dict_allParts = []
    for utList in universalTimeStepsFiles:
        aid_to_VoteUTList_Dict_allParts.append(segmentateUT(utList))

    for ut_qa in universal_timesteps_ofQA:
        if ut_qa[0]==1: # an answer creation ut
            # add this answer creation ut to new_ut
            new_ut.append(ut_qa)
            # extract corresponding vote ut list from all parts ut results
            aid = ut_qa[1]
            voteUTList = [] # initialize vote ut list for current aid
            for aid_to_VoteUTList_Dict in aid_to_VoteUTList_Dict_allParts:
                voteUTList.extend(aid_to_VoteUTList_Dict[aid])
            # re-sort the voteUTList, first based on the datetime, then based on the id
            voteUTList.sort(key=lambda t: (t[2], t[1]))
            # add new whole vote ut list to new_ut
            new_ut.extend(voteUTList)
        else: # an question creation ut, skip
            # add this question creation ut to new_ut
            new_ut.append(ut_qa)           

    return new_ut

def myFun(commName,commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    t1 = time.time()

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data part files
    # check whether have intermediate data folder, create one if not
    # go back to current parent comm data directory
    intermediate_data_folder = os.path.join(commDir, r'intermediate_data_folder')
    os.chdir(intermediate_data_folder)
    # load ut of QA
    with open(intermediate_data_folder+'/'+'universal_timesteps_afterCombineQA.dict', 'rb') as inputFile:
        universal_timesteps_ofQA = pickle.load( inputFile)

    splitted_intermediate_data_folder = os.path.join(intermediate_data_folder, r'splitted_intermediate_data_folder')
    if not os.path.exists(splitted_intermediate_data_folder):
        os.makedirs(splitted_intermediate_data_folder)
    os.chdir(splitted_intermediate_data_folder)
    final_directory = os.path.join(splitted_intermediate_data_folder, r'Votes_splitted_intermediate_data_folder')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)   
    os.chdir(final_directory)

    try:
        print("trying to load intermidiate data Question part files...")
        # load Questions files
        Questions_subfolders = [ f.path for f in os.scandir(final_directory) if f.path.split('/')[-1].startswith("QuestionsWithAnswersWithVotes_part_") ]
        # scanned part files are not guarranteed in order, need sorting
        Questions_subfolders.sort()
        print("Questions_subfolders loaded.")

        # load voteId2rowIndexOfVotesCSV files
        voteId2rowIndex_subfolders = [ f.path for f in os.scandir(final_directory) if f.path.split('/')[-1].startswith("voteId2rowIndexOfVotesCSV_part_") ]
        voteId2rowIndex_subfolders.sort()
        voteId2rowIndexFiles = []
        for i,f in enumerate(voteId2rowIndex_subfolders):
            with open(f, 'rb') as inputFile:
                voteId2rowIndexFiles.append (pickle.load( inputFile) )
                print(f"voteId2rowIndex part {i+1} loaded.")
        print("voteId2rowIndexFiles loaded.")

    except Exception as e:
        print(e)
        print(f"fail to load intermediate data part files from {commName}")
        return

    # combine all files
    Questions = None

    # combined universal_timestemps
    universal_timesteps_ofQAV=[]

    # keep a map of voteId to rowIndex of Votes.csv
    voteId2rowIndexOfVotesCSV= defaultdict()

    print("combining questions' voteLists and voteId2rowIndexOfVotesCSV")
    for i, f in enumerate(Questions_subfolders):
        print(f"loading file {f}, and combining QuestionsWithAnswersWithVotes_part {i+1} ...")
        with open(f, 'rb') as inputFile:
            QuestionsWithAnswersWithVotes_part = pickle.load( inputFile) 
        
        voteId2rowIndexOfVotesCSV_part = voteId2rowIndexFiles[i]

        if Questions == None: # when Questions is still empty, just copy the first part
            Questions = QuestionsWithAnswersWithVotes_part
            voteId2rowIndexOfVotesCSV = voteId2rowIndexOfVotesCSV_part

        else: 
            # combine current part voteList to Questions
            for qid, content in QuestionsWithAnswersWithVotes_part.items():
                if 'voteList' not in content.keys():  # when there's no vote for this question in this chunk, don't need to update
                    continue
                voteList = content['voteList']
                # combine acceptedAnswer to Questions
                if qid in Questions.keys():
                    if 'acceptedAnswer' in content.keys():
                        if 'acceptedAnswer' in Questions[qid].keys() and Questions[qid]['acceptedAnswer'] != content['acceptedAnswer']:
                            print(f"acceptedAnswer Exception : different!!!!!!")
                        Questions[qid]['acceptedAnswer']= content['acceptedAnswer'] # acceptedAnswerId

                if len(voteList)==0: # when there's no vote for this question in this chunk, don't need to update
                    continue
                
                if qid in Questions.keys():
                    if 'voteList' not in Questions[qid].keys():
                        Questions[qid]['voteList'] = []

                    Questions[qid]['voteList'] = Questions[qid]['voteList'] + voteList
                    Questions[qid]['voteList'].sort(key=lambda x: x[0]) # sort by vid

            # combine voteId2rowIndexOfPostsCSV
            for id, index in voteId2rowIndexOfVotesCSV_part.items():
                if id not in voteId2rowIndexOfVotesCSV.keys(): # only update when id is not in keys()
                    voteId2rowIndexOfVotesCSV[id] = index
    
    
    # clear voteId2rowIndexFiles to save mem
    voteId2rowIndexFiles.clear()

    # save all updated Questions
    print("saving updated Questions")
    with open(intermediate_data_folder+'/'+'QuestionsWithAnswersWithVotes.dict', 'wb') as outputFile:
        pickle.dump(Questions, outputFile)
    Questions.clear()

    # save updated voteId2rowIndexOfVotesCSV
    print("saving updated voteId2rowIndexOfVotesCSV")
    with open(intermediate_data_folder+'/'+'voteId2rowIndexOfVotesCSV.dict', 'wb') as outputFile:
        pickle.dump(voteId2rowIndexOfVotesCSV, outputFile)

    voteId2rowIndexOfVotesCSV.clear()

    try:
        print("trying to load intermidiate data universalTimeSteps part files...")
        # load universalTimeSteps files
        universalTimeSteps_subfolders = [ f.path for f in os.scandir(final_directory) if f.path.split('/')[-1].startswith("universal_timesteps_afterCombineQAV_part_") ]
        universalTimeSteps_subfolders.sort()
        universalTimeStepsFiles = []
        for i,f in enumerate(universalTimeSteps_subfolders):
            with open(f, 'rb') as inputFile:
                universalTimeStepsFiles.append (pickle.load( inputFile) )
                print(f"universalTimeSteps part {i+1} loaded.")
        print("all universalTimeStepsFiles loaded")
    except Exception as e:
        print(e)
        print(f"fail to load intermediate data universalTimeSteps part files from {commName}")
        return
    
    # combine universal_timestemps
    print("combining universal_timesteps_ofQA and universalTimeStepsFiles")
    universal_timesteps_ofQAV = combineSegmentedUTList(universal_timesteps_ofQA, universalTimeStepsFiles)

    # save universal_timesteps
    print("saving updated universal_timesteps")
    with open(intermediate_data_folder+'/'+'universal_timesteps_afterCombineQAV.dict', 'wb') as outputFile:
        pickle.dump(universal_timesteps_ofQAV, outputFile)

    elapsed = format_time(time.time() - t1)

    logfilename = f'preprocessing3_2_combineVote_saveMem_forSplittedFiles_combineParts_Log.txt'
    logtext = f"Total event timesteps: {len(universal_timesteps_ofQAV)}\n"
    logtext += f"Count of Questions: {len(Questions)}\n"
    logtext +=  'Elapsed: {:}.\n'.format(elapsed)
    writeIntoLog(logtext, commDir, logfilename)

    # Report progress.
    print(f"for {commName}, ")
    print('combining vote intermediate part files Done.    Elapsed: {:}.\n'.format(elapsed))



def main():

    t0=time.time()
    curDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # test on comm "datascience.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])

    """
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist[358:]:
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
    """

    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('processing splitted Votes.csv and combine Q and A and V Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
