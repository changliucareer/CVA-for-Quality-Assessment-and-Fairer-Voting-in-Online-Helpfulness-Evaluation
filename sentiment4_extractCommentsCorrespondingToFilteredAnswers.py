import os

from numpy import average
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
from collections import Counter
import sys
import re
import torch
import pandas as pd
import json
from bs4 import BeautifulSoup

##########################################################################################
def replace_img_tags(data):
    p = re.compile(r'<img.*(/img)?>')
    return p.sub('image', data)

def replace_code_tags(data):
    p = re.compile(r'<code.*(/code)?>')
    return p.sub('code', data)

def replace_URL(data):
    p = re.compile(r'\S*https?:\S*')
    return p.sub('URL', data)

def cleanComment(comment_text):
    if ('<img' in comment_text) or ('<code' in comment_text) or ('http' in comment_text):
        # replace <img> and <code> segment
        cleaned_comment_text = replace_img_tags(comment_text)
        cleaned_comment_text = replace_code_tags(cleaned_comment_text)
        cleaned_comment_text = replace_URL(cleaned_comment_text)
        return BeautifulSoup(cleaned_comment_text, "lxml").text
    else: 
        return BeautifulSoup(comment_text, "lxml").text

def myAction (commName, commDir, chunkIndex, df, filtered_aid2qid, qidaid2comments):
   
    for line_count, row in df.iterrows():
        print(f"processing processing {commName} chunk {chunkIndex} line {line_count}...")
        # lines of data 
        targetPost = int(row['PostId'])
        if targetPost not in filtered_aid2qid.keys(): # skip if not filtered answer
            continue
        qid = filtered_aid2qid[targetPost]
        
        comment_text = row['Comment']
        cleaned_comment_text = cleanComment(comment_text)
        if len(cleaned_comment_text)<2: # skip the comment that is too short
            continue

        if (qid, targetPost) in qidaid2comments.keys():
            qidaid2comments[(qid, targetPost)]= qidaid2comments[(qid, targetPost)]+[cleaned_comment_text]
        else:
            qidaid2comments[(qid, targetPost)] = [cleaned_comment_text]
        

    # save results till this chunk
    # convert and save the last aid2qidAndComments dict
    aid2qidAndComments = defaultdict()
    for k,v in qidaid2comments.items():
        qid = k[0]
        aid = k[1]
        aid2qidAndComments[aid] = {'qid':qid, 'comments':v} 
    
    os.chdir(commDir) # go back to comm directory
    # Convert and write JSON object to file
    if commName == 'stackoverflow':
        # fot sampled SOF
        # with open('intermediate_data_folder/sampled1percent_filtered_aid2qidAndComments_tillCurChunk.json', "w") as outfile: 
        #     json.dump(aid2qidAndComments, outfile)
        #     writeIntoLog(f'saved sampled1percent_ aid2qidAndComments as json after chunk {chunkIndex} done\n',commDir,'sentiment1_filteringComments_Log.txt')
        # # for subcomm reactjs use reactjs subComm represent SOF
        # with open('intermediate_data_folder/filtered_aid2qidAndComments_tillCurChunk_reactjs.json', "w") as outfile: 
        #     json.dump(aid2qidAndComments, outfile)
        #     writeIntoLog(f'saved aid2qidAndComments of reacjs as json after chunk {chunkIndex} done\n',commDir,'sentiment1_filteringComments_Log.txt')

        # for whole SOF
         with open('intermediate_data_folder/filtered_aid2qidAndComments_tillCurChunk.json', "w") as outfile: 
            json.dump(aid2qidAndComments, outfile)
            writeIntoLog(f'saved aid2qidAndComments as json after chunk {chunkIndex} done\n',commDir,'sentiment1_filteringComments_Log.txt')

    else: # otherComm
        with open('intermediate_data_folder/filtered_aid2qidAndComments_tillCurChunk.json', "w") as outfile: 
            json.dump(aid2qidAndComments, outfile)
            writeIntoLog(f'saved aid2qidAndComments as json after chunk {chunkIndex} done\n',commDir,'sentiment1_filteringComments_Log.txt')


# Attributions of comment CSV
# 'Comments':['Id','PostId','Score','Text','CreationDate','UserId']
def myFun(commIndex,commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load filtered_answerWithVotes_ids_and_scores
    # filtered_answerWithVotes_ids_and_scores is a tuple list of (ids, learned_q, voteDiff, voteCount, sentimentScores,helpful_sentimentList[i],correct_sentimentList[i])
    # ids is a tuple (qid , aid )
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    if commName != 'stackoverflow':
        # # get filtered_aid2qid by using filtered_answerWithVotes_ids_and_scores
        # print(f"loading filtered_answerWithVotes_ids_and_scores... for {commName}")
        # try:
        #     with open(intermediate_directory+'/'+'verifyQualities.dict', 'rb') as inputFile:
        #         filtered_answerWithVotes_ids_and_scores = pickle.load( inputFile)
        # except Exception as e:
        #     print(f"for {commName} error when load the filtered_answerWithVotes_ids_and_scores: {e}")
        #     return

        # filtered_aid2qid = defaultdict()
        # for tup in filtered_answerWithVotes_ids_and_scores:
        #     qid = tup[0][0]
        #     aid = tup[0][1]
        #     filtered_aid2qid[aid]=qid

        # filtered_answerWithVotes_ids_and_scores.clear()

        # # get filtered_aid2qid by scanning Questions
        with open(intermediate_directory+'/'+f'whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
            total_answersWithVotes_indice = pickle.load( inputFile)
        with open(intermediate_directory+'/'+f'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
            Questions = pickle.load( inputFile)
            
        filtered_aid2qid = defaultdict()
        for i,(qid, ai) in enumerate(total_answersWithVotes_indice):
            filtered_answerList = Questions[qid]['filtered_answerList']
            aid = filtered_answerList[ai]
            filtered_aid2qid[aid]=qid
    
    else: # commName == 'stackoverflow'
        # for sampled stackoverflow
        # print(f"loading Sampled1percent_qidai2aid_voteDiff_voteCount.dict... for {commName}")
        # try:
        #     with open(final_directory+'/'+'Sampled1percent_qidai2aid_voteDiff_voteCount.dict', 'rb') as inputFile:
        #         qidai2aid_voteDiff_voteCount = pickle.load( inputFile)
        # except Exception as e:
        #     print(f"for {commName} error when load the qidai2aid_voteDiff_voteCount: {e}")
        #     return

        # for subcomm reactjs
        # print(f"loading whole_answersWithVotes_indice_removeFirstRealVote.dict for {commName}'s subComm reactjs'")
        # subComms_data_folder = os.path.join(commDir, f'subCommunities_folder')
        # ## Load all sub community direcotries 
        # with open(subComms_data_folder+'/'+f'subCommName2subCommDir.dict', 'rb') as inputFile:
        #     subCommName2commDir = pickle.load( inputFile)
        # subCommDir = subCommName2commDir['reactjs']
        # subComm_intermediate_directory = os.path.join(subCommDir, r'intermediate_data_folder')

        # subComm_QuestionsWithEventList_directory = subCommDir+'/'+f'QuestionsWithEventList_tag_reactjs.dict'
        # with open(subComm_QuestionsWithEventList_directory, 'rb') as inputFile:
        #     Questions = pickle.load( inputFile)

        # with open(subComm_intermediate_directory+'/'+f'whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
        #     total_answersWithVotes_indice = pickle.load( inputFile)

        # filtered_aid2qid = defaultdict()
        # for i,(qid, ai) in enumerate(total_answersWithVotes_indice):
        #     filtered_answerList = Questions[qid]['filtered_answerList']
        #     aid = filtered_answerList[ai]
        #     filtered_aid2qid[aid]=qid

        # Questions.clear()

        # for whole SOF
        print(f"loading whole_answersWithVotes_indice_removeFirstRealVote.dict for {commName}")
        with open(intermediate_directory+'/'+f'whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
            total_answersWithVotes_indice = pickle.load( inputFile)
        
        print(f"loading Questions for {commName}")
        splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
        split_QuestionsWithEventList_files_directory = os.path.join(splitFolder_directory, r'QuestionsPartsWithEventList')
        if not os.path.exists(split_QuestionsWithEventList_files_directory): # didn't find the parts files
            print("Exception: no split_QuestionsWithEventList_files_directory!")

        partFiles = [ f.path for f in os.scandir(split_QuestionsWithEventList_files_directory) if f.path.endswith('.dict') ]
        # sort csvFiles paths based on part number
        partFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
        partsCount = len(partFiles)

        assert partsCount == int(partFiles[-1].strip(".dict").split("_")[-1]) # last part file's part number should equal to the parts count

        filtered_qid2aiList = defaultdict()
        for qid, ai in total_answersWithVotes_indice:
            if qid in filtered_qid2aiList.keys():
                filtered_qid2aiList[qid].append(ai)
            else:
                filtered_qid2aiList[qid] = [ai]
        
        filtered_aid2qid = defaultdict()
        for i, partDir in enumerate(partFiles):
            print(f"scan part {i+1} out of {partsCount} parts of SOF questions...")
            with open(partDir, 'rb') as inputFile:
                Questions_part = pickle.load( inputFile)
                qidsInPart = set(Questions_part.keys()).intersection(set(filtered_qid2aiList.keys()))
                for qid in qidsInPart:
                    filtered_answerList = Questions_part[qid]['filtered_answerList']
                    filtered_aiList = filtered_qid2aiList[qid]
                    for ai in filtered_aiList:
                        aid = filtered_answerList[ai]
                        filtered_aid2qid[aid]=qid
                Questions_part.clear()

        assert len(total_answersWithVotes_indice) == len(filtered_aid2qid)

        # save filtered_aid2qid for SOF whole
        with open(intermediate_directory+'/'+'filtered_aid2qid.dict', 'wb') as outputFile:
            pickle.dump(filtered_aid2qid, outputFile)
            print(f"saved filtered_aid2qid for {commName} whole")
        
        total_answersWithVotes_indice.clear()
        
        # # load filteredfiltered_aid2qid for SOF whole
        # with open(intermediate_directory+'/'+'filtered_aid2qid.dict', 'rb') as inputFile:
        #     filtered_aid2qid = pickle.load( inputFile)

    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    qidaid2comments = manager.dict() # to save the dict of post id to comments text

    chunk_size = 1000000
    processes = []
    chunkIndex = 0
    finishedCount =0
    # load sentiment training data
    try:
        for df in pd.read_csv('intermediate_data_folder/sentiment_training_data.csv', chunksize=chunk_size, engine='python',sep=','):
            
            try:
                p = mp.Process(target=myAction, args=(commName, commDir, chunkIndex, df, filtered_aid2qid, qidaid2comments))
                p.start()
                chunkIndex += 1
            except Exception as e:
                print(e)
                sys.exit()
                return

            processes.append(p)
            if len(processes)==4:
                # make sure all p finish before main process finish
                for p in processes:
                    p.join()
                    finishedCount +=1
                    print(f"finished {finishedCount} chunks for {commName}.")
                # clear processes
                processes = []
        
        # join the last batch of processes
        if len(processes)>0:
            # make sure all p finish before main process finish
            for p in processes:
                p.join()
                finishedCount +=1
                print(f"finished {finishedCount} chunks for {commName}.")


        # convert and save the last aid2qidAndComments dict
        aid2qidAndComments = defaultdict()
        for k,v in qidaid2comments.items():
            qid = k[0]
            aid = k[1]
            aid2qidAndComments[aid] = {'qid':qid, 'comments':v} 
        
        os.chdir(commDir) # go back to comm directory
        # Convert and write JSON object to file
        if commName == 'stackoverflow':
            # for subcomm reactjs
            # with open('intermediate_data_folder/filtered_aid2qidAndComments_tillCurChunk_reactjs.json', "w") as outfile: 
            #     json.dump(aid2qidAndComments, outfile)
            #     writeIntoLog(f'saved aid2qidAndComments  of reactjs as json after chunk {chunkIndex} done. current number of filtered answer is {len(aid2qidAndComments)}\n',commDir,'sentiment1_filteringComments_Log.txt')
            with open('intermediate_data_folder/filtered_aid2qidAndComments_tillCurChunk.json', "w") as outfile: 
                json.dump(aid2qidAndComments, outfile)
                writeIntoLog(f'saved aid2qidAndComments of StackOverflow whole as json after chunk {chunkIndex} done. current number of filtered answer is {len(aid2qidAndComments)}\n',commDir,'sentiment1_filteringComments_Log.txt')
        else:
            with open('intermediate_data_folder/filtered_aid2qidAndComments_tillCurChunk.json', "w") as outfile: 
                json.dump(aid2qidAndComments, outfile)
                writeIntoLog(f'saved aid2qidAndComments as json after chunk {chunkIndex} done. current number of filtered answer is {len(aid2qidAndComments)}\n',commDir,'sentiment1_filteringComments_Log.txt')

    except Exception as e:
        logfile = 'sentiment4_Log.txt'
        logtext = f"fail for {commName}\n"
        logtext += str(e)+'\n'
        writeIntoLog(logtext, commDir, logfile)


def main():
    t0 = time.time()
    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")
    
    # test on comm "coffee.stackexchange" to debug
    # myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # test on comm "datascience.stackexchange" to debug
    # myFun(0,commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    # myFun(3,commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    # test on comm "math.stackexchange" to debug
    # myFun(358,commDir_sizes_sortedlist[358][0], commDir_sizes_sortedlist[358][1])
    # test on comm "stackoverflow" to debug
    # myFun(359,commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1])
    # test on comm "askubuntu" to debug
    myFun(356,commDir_sizes_sortedlist[356][0], commDir_sizes_sortedlist[356][1])
    
    """
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        try:
            p = mp.Process(target=myFun, args=(commIndex,commName,commDir))
            p.start()
        except Exception as e:
            print(e)
            return

        processes.append(p)
        if len(processes)==1:
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
    print('get sentiment score Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
