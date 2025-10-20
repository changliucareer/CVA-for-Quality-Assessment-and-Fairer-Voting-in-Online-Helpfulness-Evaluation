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

def combineUniversalTimeSteps(universal_timesteps_ofQA, universal_timesteps_ofV, parentCommName, part):
    # create a dict to look up datetime of each answer's creation
    aid2time = defaultdict()
    for tup in universal_timesteps_ofQA:
        eventType = tup[0]
        aid = tup[1]
        datetime_obj= tup[2]
        if eventType==1 and (aid not in aid2time.keys()):
            aid2time[aid]=datetime_obj
        
    # create new_universal_timesteps_ofV to store updated datetime of votes
    new_universal_timesteps_ofV = []

    # itertools.groupby() groups only consecutive elements of the same value !!!
    # To group elements regardless of their order, use sorted() to sort the original list first, so that we can get groups with unique keys
    universal_timesteps_ofV.sort(key=lambda t:t[2]) # sort by aid first
    groupInd = 0
    for aid, g in groupby(universal_timesteps_ofV, operator.itemgetter(2)): # group by aid which is the third element of temstep tuple
        groupInd += 1
        print(f"start to process {groupInd} group of {parentCommName} part {part}...")
        # convert iter g into list
        vote_steps = list(g)
        # add 1 millisecond to the votes' datetime that earlier than their corresponding answer creation datetime
        answerDateTime = aid2time[aid]
        for j,t in enumerate(vote_steps):
            voteDateTime = t[-1]
            if voteDateTime <= answerDateTime:
                updated_voteDateTime = answerDateTime + datetime.timedelta(milliseconds=0.001) # 0.001 millisecond is the smallest unit of datetime object
                # update this vote's datetime
                # It isn't possible to assign to individual items of a tuple, so we have to convert the tuple to a list first.
                tupleList = list(vote_steps[j])
                tupleList[-1] = updated_voteDateTime
                vote_steps[j] = tuple(tupleList)

        # move the third element of timestep tuple (aid) to the last to match with the Q&A event tuple format
        # then the vote event tuple will look like (eventType, voteId, datetime_obj, answerId)
        vote_steps = [(g_tup[0],g_tup[1],g_tup[3],g_tup[2]) for g_tup in vote_steps]
        new_universal_timesteps_ofV.extend(vote_steps)     
    

    new_universal_timesteps = universal_timesteps_ofQA + new_universal_timesteps_ofV

    # re-sort the new_universal_timesteps, first based on the datetime, then based on the id
    print(f"start to re-sort the new_universal_timesteps after the invsertions for {parentCommName} part {part}...")
    new_universal_timesteps.sort(key=lambda t: (t[2], t[1]))
    return new_universal_timesteps


def myAction (chunk,chunk_start_index,commName,part, existingQuestionIds,answer2parentQ):
    chunk_size = len(chunk)
    chunk_index = int (chunk_start_index / chunk_size)
    print(f"{commName} part {part} chunk {chunk_index+1} with start index {chunk_start_index} running on {mp.current_process().name}")

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

    # get universal timesteps of votes
    # it's a list of tuple, in format (eventType, voteId, answerId, datetime)
    universal_timesteps = []

    # keep a map of voteId to rowIndex of Votes.csv
    voteId2rowIndexOfVotesCSV = defaultdict()
    
    # keep a light-weighted dict to map questionId to voteList
    QuestionId2VoteListDict = defaultdict()

    # keep a light-weighted dict to map questionId to voteList
    QuestionId2AcceptedAnswerIdDict = defaultdict()

    for index,row in chunk.iterrows():
        
        # convert datetime info string into datetime object
        datetime_str = row['CreationDate'] 
        date_time_obj = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")

        vid = int(row['Id'])
        aid = int(row['PostId'])

        voteId2rowIndexOfVotesCSV[vid] = chunk_start_index + index


        if aid not in answer2parentQ.keys(): # not a vote to an answer
            # print("not a vote to an answer")
            continue
            
        qid = answer2parentQ[aid]
        if qid not in existingQuestionIds: # not an existing question
            print("not an existing question")
            continue

        # add vote
        # add 'voteList' to Questions dictionary
        # the 'voteList' is a list of tuple about votes
        # the tupel consits of three items (voteId(int), vote(1 or -1), target answerId(int) )
        if qid not in QuestionId2VoteListDict.keys(): # new target question
            QuestionId2VoteListDict[qid] = []

        if row['VoteTypeId']==2: #Upvote
            QuestionId2VoteListDict[qid].append((vid,1,aid))
            universal_timesteps.append((3,vid,aid,date_time_obj))
        elif row['VoteTypeId']==3: #Downvote
            QuestionId2VoteListDict[qid].append((vid,-1,aid))
            universal_timesteps.append((3,vid,aid,date_time_obj))
        elif row['VoteTypeId']==1: ## AcceptedByOriginator
            QuestionId2AcceptedAnswerIdDict[qid]= aid 
            universal_timesteps.append((4,vid,aid,date_time_obj))


    print(f"{mp.current_process().name} return")
    return QuestionId2VoteListDict, QuestionId2AcceptedAnswerIdDict, universal_timesteps, voteId2rowIndexOfVotesCSV
    

def myFun(parentCommName,parentCommDir,part,partDir):
    print(f"comm {parentCommName} part {part} running on {mp.current_process().name}")
    t1 = time.time()

    # go to current parent comm data directory
    os.chdir(parentCommDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'intermediate_data_folder')
    
    with open(final_directory+'/'+'QuestionsWithAnswers.dict', 'rb') as inputFile:
        Questions = pickle.load( inputFile)

    with open(final_directory+'/'+'answer2parentQLookup.dict', 'rb') as inputFile:
        answer2parentQ = pickle.load( inputFile)

    with open(final_directory+'/'+'universal_timesteps_afterCombineQA.dict', 'rb') as inputFile:
        universal_timesteps_ofQA = pickle.load( inputFile)

    existingQuestionIds = list(Questions.keys())

    # go to current Votes.csv_files directory
    splitFolder_directory = os.path.join(parentCommDir, r'split_data_folder')
    split_votes_files_directory = os.path.join(splitFolder_directory, r'Votes.csv_files')
    os.chdir(split_votes_files_directory)
    # print(os.getcwd())

    #read data in chunks of 1 million rows at a time
    chunk_size = 10000
    chunksIter = pd.read_csv(partDir,chunksize=chunk_size) # return type <class 'pandas.io.parsers.readers.TextFileReader'>

    done_looping = False
    chunk_batch = []
    # n_proc = mp.cpu_count()-2 # n_proc as batch size
    n_proc = 1
    all_outputs = []
    chunk_index = 0
    while not done_looping:
        try:
            chunk = next(chunksIter) # chunk type is <class 'pandas.core.frame.DataFrame'>
        except StopIteration:
            done_looping = True
        else:
            # # # use shared variable to communicate among all comm's process
            # manager = mp.Manager()
            # Question2VoteListDict = manager.dict() # to save the question count and answer count of each community

            # when the batch is full, do the action with multiprocessing pool
            if len(chunk_batch)==n_proc:
                args = zip(chunk_batch,[chunk_index*chunk_size]*len(chunk_batch),[parentCommName]*len(chunk_batch),[part]*len(chunk_batch), [existingQuestionIds]*len(chunk_batch),[answer2parentQ]*len(chunk_batch))
                
                with mp.Pool(processes=n_proc) as pool:
                    # issue tasks to the process pool and wait for tasks to complete
                    #An iterator is returned with the result for each function call
                    results = pool.starmap(myAction, args, chunksize=100)
                    all_outputs.extend(results)
                    # process pool is closed automatically

                # increase the chunk_index
                chunk_index += 1
                print(f"finish processing {parentCommName} part {part} chunk {chunk_index}")
                # clear the chunk_batch
                chunk_batch = []
            chunk_batch.append(chunk)
    
    # process the last batch
    with mp.Pool(processes=n_proc) as pool:
        args = zip(chunk_batch,[chunk_index*chunk_size]*len(chunk_batch),[parentCommName]*len(chunk_batch),[part]*len(chunk_batch), [existingQuestionIds]*len(chunk_batch),[answer2parentQ]*len(chunk_batch))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction, args, chunksize=100)
        # process pool is closed automatically
        all_outputs.extend(results)

    # combine all_outputs
    print(f"start to combine all outputs of {parentCommName} part {part}...")

    # combined universal_timestemps
    universal_timesteps_ofV=[]

    # keep a map of voteId to rowIndex of Votes.csv
    voteId2rowIndexOfVotesCSV= defaultdict()

    # keep a set of question ids that has real votes
    questionsWithRealVotes = set()

    for tupInd, tup in enumerate(all_outputs):
        print(f"combining {tupInd+1} / {len(all_outputs)} output...")
        QuestionId2VoteListDict = tup[0]
        questionsWithRealVotes = questionsWithRealVotes.union(set(QuestionId2VoteListDict.keys()))
        QuestionId2AcceptedAnswerIdDict = tup[1]
        ut = tup[2]
        id2index = tup[3]

        # combine voteList to Questions
        for qid, voteList in QuestionId2VoteListDict.items():
            if len(voteList)==0: # when there's no vote for this question in this chunk, don't need to update
                continue
            
            if qid in Questions.keys():
                if 'voteList' not in Questions[qid].keys():
                    Questions[qid]['voteList'] = []

                Questions[qid]['voteList'] = Questions[qid]['voteList'] + voteList
                Questions[qid]['voteList'].sort(key=lambda x: x[0]) # sort by vid

        # combine acceptedAnswer to Questions
        for qid, acceptedAnswerId in QuestionId2AcceptedAnswerIdDict.items():
            if qid in Questions.keys():
                Questions[qid]['acceptedAnswer']= acceptedAnswerId 

        # combine universal_timestemps
        universal_timesteps_ofV.extend(ut)

        # combine voteId2rowIndexOfPostsCSV
        for id, index in id2index.items():
            if id not in voteId2rowIndexOfVotesCSV.keys(): # only update when id is not in keys()
                voteId2rowIndexOfVotesCSV[id] = index
  
    # check whether have intermediate data folder, create one if not
    # go back to current parent comm data directory
    os.chdir(parentCommDir)
    intermediate_data_folder = os.path.join(parentCommDir, r'intermediate_data_folder')
    os.chdir(intermediate_data_folder)
    splitted_intermediate_data_folder = os.path.join(intermediate_data_folder, r'splitted_intermediate_data_folder')
    if not os.path.exists(splitted_intermediate_data_folder):
        os.makedirs(splitted_intermediate_data_folder)
    os.chdir(splitted_intermediate_data_folder)
    final_directory = os.path.join(splitted_intermediate_data_folder, r'Votes_splitted_intermediate_data_folder')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)   
    os.chdir(final_directory)
    
    # save all Questions
    print("start to save updated Questions...")
    with open(final_directory+'/'+'QuestionsWithAnswersWithVotes_part_'+str(part)+'.dict', 'wb') as outputFile:
        pickle.dump(Questions, outputFile)
        print(f"QuestionsWithAnswersWithVotes_part_{part}.dict saved. ")

    # save updatedQuestions
    with open(final_directory+'/'+'voteId2rowIndexOfVotesCSV_part_'+str(part)+'.dict', 'wb') as outputFile:
        pickle.dump(voteId2rowIndexOfVotesCSV, outputFile)
        print(f"voteId2rowIndexOfVotesCSV_part_{part}.dict saved. ")

    logfilename = f'combineVote_part_{part}_Log.txt'
    logtext = f"Count of Questions: {len(Questions)}\n"
    logtext += f"Count of votes: {len(universal_timesteps_ofV)}\n"
    logtext += f"Count of Questions without real votes: {len(Questions)-len(questionsWithRealVotes)}\n"

    Questions.clear()
    voteId2rowIndexOfVotesCSV.clear()

    # sort universal_timesteps
    print(f"start to combine universal time steps of {parentCommName} part {part}...")
    universal_timesteps = combineUniversalTimeSteps(universal_timesteps_ofQA, universal_timesteps_ofV, parentCommName, part)

    # save universal_timesteps
    with open(final_directory+'/'+'universal_timesteps_afterCombineQAV_part_'+str(part)+'.dict', 'wb') as outputFile:
        pickle.dump(universal_timesteps, outputFile)
        print(f"universal_timesteps_afterCombineQAV_part_{part}.dict saved. ")
    
    elapsed = format_time(time.time() - t1)

    logtext += f"Total event timesteps: {len(universal_timesteps)}\n"
    logtext +=  'Elapsed: {:}.\n'.format(elapsed)
    writeIntoLog(logtext, parentCommDir, logfilename)

    # Report progress.
    print(f"for {parentCommName}, ")
    print('processing Votes.csv and combine Q and A and V Done.    Elapsed: {:}.\n'.format(elapsed))



def main():

    t0=time.time()
    curDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # test on comm "coffee.stackexchange" to debug
    large_commName_list = ['coffee.stackexchange']

    # test on comm "datascience.stackexchange" to debug
    # large_commName_list = ['datascience.stackexchange']

    # large_commName_list = ['math.stackexchange', 'stackoverflow']
    large_commDir_list = []

    # for tup in commDir_sizes_sortedlist[358:]:
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]
        for large_commName in large_commName_list:
            if commName == large_commName: 
                large_commDir_list.append((large_commName, commDir))
                
    # create new lists for splitted files
    splitted_files_list = []

    for commName, commDir in large_commDir_list:

        # go to the target splitted files folder
        splitFolder_directory = os.path.join(commDir, r'split_data_folder')
        split_votes_files_directory = os.path.join(splitFolder_directory, r'Votes.csv_files')
        csvFiles = [ f.path for f in os.scandir(split_votes_files_directory) if f.path.endswith('.csv') ]
        # sort csvFiles paths based on part number
        csvFiles.sort(key=lambda p: int(p.strip(".csv").split("_")[-1]))
        partsCount = len(csvFiles)
        print(f"there are {partsCount} splitted csv files in {commName}")

        for i, subDir in enumerate(csvFiles):
            part = i+1
            partDir = subDir
            splitted_files_list.append ( (commName, commDir, partsCount, part, partDir) )
    
    # save directiories of all splitted files as dict to folder "SE_codes_2022"
    os.chdir(curDir)
    fname = 'splittedVotesFilesDirList.dict'
    with open(fname, 'wb') as outputFile:
        pickle.dump(splitted_files_list, outputFile)
        print(f"{fname} Saved as dict. in {curDir}")

    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in splitted_files_list:
        parentCommName = tup[0]
        parentCommDir = tup[1]
        partsCount = tup [2]
        part = tup[3]
        partDir = tup[4]

        try:
            p = mp.Process(target=myFun, args=(parentCommName,parentCommDir,part,partDir))
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

    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('processing splitted Votes.csv and combine Q and A and V Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
