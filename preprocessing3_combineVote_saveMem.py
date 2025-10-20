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

def combineUniversalTimeSteps(universal_timesteps_ofQA, universal_timesteps_ofV, commName):
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
        print(f"start to process {groupInd} group of {commName}...")
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
    print(f"start to re-sort the new_universal_timesteps after the invsertions for {commName} ...")
    new_universal_timesteps.sort(key=lambda t: (t[2], t[1]))
    return new_universal_timesteps


def myAction (chunk,chunk_start_index,commName,existingQuestionIds,answer2parentQ):
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
    

def myFun(commName, commDir):
    # final_directory = os.path.join(commDir, r'intermediate_data_folder')
    # # if already done for this comm, return
    # resultFile= final_directory+'/'+'QuestionsWithAnswersWithVotes.dict'
    # if os.path.exists(resultFile):
    #     return

    print(f"comm {commName} running on {mp.current_process().name}")
    t1 = time.time()

    # go to current comm data directory
    os.chdir(commDir)
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

    #read data in chunks of 1 million rows at a time
    chunk_size = 10000
    chunksIter = pd.read_csv('Votes.csv',chunksize=chunk_size) # return type <class 'pandas.io.parsers.readers.TextFileReader'>

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
            # # # use shared variable to communicate among all comm's process
            # manager = mp.Manager()
            # Question2VoteListDict = manager.dict() # to save the question count and answer count of each community

            # when the batch is full, do the action with multiprocessing pool
            if len(chunk_batch)==n_proc:
                args = zip(chunk_batch,[chunk_index*chunk_size]*len(chunk_batch),[commName]*len(chunk_batch),[existingQuestionIds]*len(chunk_batch),[answer2parentQ]*len(chunk_batch))
                
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
    
    # process the last batch
    with mp.Pool(processes=n_proc) as pool:
        args = zip(chunk_batch,[chunk_index*chunk_size]*len(chunk_batch),[commName]*len(chunk_batch),[existingQuestionIds]*len(chunk_batch),[answer2parentQ]*len(chunk_batch))
        # issue tasks to the process pool and wait for tasks to complete
        results = pool.starmap(myAction, args, chunksize=100)
        # process pool is closed automatically
        all_outputs.extend(results)

    # combine all_outputs

    # combined universal_timestemps
    universal_timesteps_ofV=[]

    # keep a map of voteId to rowIndex of Votes.csv
    voteId2rowIndexOfVotesCSV= defaultdict()

    # keep a set of question ids that has real votes
    questionsWithRealVotes = set()

    for tup in all_outputs:
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

    # sort universal_timesteps
    universal_timesteps = combineUniversalTimeSteps(universal_timesteps_ofQA, universal_timesteps_ofV, commName)
    

    logfilename = 'combineVote_Log.txt'
    logtext = f"Total event timesteps: {len(universal_timesteps)}\n"
    logtext += f"Count of Questions: {len(Questions)}\n"
    logtext += f"Count of votes: {len(universal_timesteps_ofV)}\n"
    logtext += f"Count of Questions without real votes: {len(Questions)-len(questionsWithRealVotes)}\n"
    writeIntoLog(logtext, commDir, logfilename)
  
    # check whether have intermediate data folder, create one if not
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'intermediate_data_folder')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    # save all Questions
    with open(final_directory+'/'+'QuestionsWithAnswersWithVotes.dict', 'wb') as outputFile:
        pickle.dump(Questions, outputFile)

    # save universal_timesteps
    with open(final_directory+'/'+'universal_timesteps_afterCombineQAV.dict', 'wb') as outputFile:
        pickle.dump(universal_timesteps, outputFile)

    # save updatedQuestions
    with open(final_directory+'/'+'voteId2rowIndexOfVotesCSV.dict', 'wb') as outputFile:
        pickle.dump(voteId2rowIndexOfVotesCSV, outputFile)

    elapsed = format_time(time.time() - t1)
    # Report progress.
    print(f"for {commName}, ")
    print('processing Votes.csv and combine Q and A and V Done.    Elapsed: {:}.\n'.format(elapsed))



def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # test on comm "coffee.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # test on comm "datascience.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    
    """
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist[:340]:
        commName = tup[0]
        commDir = tup[1]
        if commName == 'stackoverflow': # skip stackoverflow to run at the last
            stackoverflow_dir = commDir
            continue
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


    # # run stackoverflow at the last separately
    # myFun('stackoverflow', stackoverflow_dir)

    """
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('processing Votes.csv and combine Q and A and V Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
