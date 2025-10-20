import os
import sys
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################
from toolFunctions import format_time, writeIntoLog, writeIntoResult,saveModel,savePlot
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
import copy
from itertools import groupby
import re
import psutil
# from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import torch
from tqdm import tqdm
from statistics import mean
from sklearn.model_selection import train_test_split
import sklearn
from CustomizedNN import LRNN_1layer, LRNN_1layer_bias, LRNN_1layer_bias_specify,LRNN_1layer_bias_withoutRankTerm
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix 
from sklearn.preprocessing import normalize
import random
import math
import scipy.stats
import statsmodels.api as sm
from scipy import stats
import json
from bs4 import BeautifulSoup
from openai import OpenAI
import tiktoken



def generate_typical(aid2Moderates, qid2Moderates, postHistoryId2text_total,  qid2aidsAndVoteDiff, commName, qid2tags):
    qid2typical3Answers = defaultdict()


    for qid, acList in qid2aidsAndVoteDiff.items(): 
        # if len(acList)<5: # skip questions that have less than 5 answers
        # if len(acList)<3: # skip questions that have less than 3 answers
        if len(acList)<2: # skip questions that have less than 2 answers
            continue
        # pick up best, median and worst answers as few shots based on their vote difference score
        sortedList = copy.deepcopy(acList)
        sortedList.sort(key=lambda x:x[1], reverse=True) 
        bestVD = sortedList[0][1]
        worstVD = sortedList[-1][1]
        # meanVD = (bestVD + worstVD)/2

        bestVD_aid = sortedList[0][0]
        worstVD_aid = sortedList[-1][0]
        
        # # find the answer whose voteDiff is the closest to meanVD
        # sortedList_distanceTomeanVD = [(tup[0], abs(tup[1]-meanVD), tup[1]) for tup in sortedList]
        # sortedList_distanceTomeanVD.sort(key=lambda x:x[1])
        # meanVD_aid = sortedList_distanceTomeanVD[0][0]
        # meanVD = sortedList_distanceTomeanVD[0][2]

        try:
            # assert meanVD_aid != bestVD_aid
            # assert meanVD_aid != worstVD_aid
            # assert meanVD != bestVD
            # assert meanVD != worstVD
            assert bestVD_aid != worstVD_aid
            assert bestVD != worstVD
        except Exception as e:
            print(f"assertion violated {e}. for question {qid} of {commName}")
            continue

        # extract question Title and Body
        if str(qid) not in qid2Moderates.keys():
            print(f"Exception! question {qid} not found in qid2Moderates for {commName}")
            continue
        QmDict = qid2Moderates[str(qid)]
        QmList = QmDict['moderateList']

        # Dict aid2Moderates has answerId as key, (qid2Moderates has questionId as key)
        # and the value is a dict, keys are "initialUserId" and "moderateList",
        # and the moderateList is a list of tuple (phId, phType, userId, comment, closedReasonId, votedUserIds, migrationDetails)

        questionTitlePhId = None
        questionBodyPhId = None

        for tup in QmList:
            phType = tup[1]
            if phType in [1,4,7]: # Initial Title, or Edit Title, or Rollback Title
                questionTitlePhId = tup[0]

            elif phType in [2,5,8]: # Initial Body, or Edit Body, or Rollback Body
                questionBodyPhId = tup[0]


        # extract answer Body
        BestAmDict = aid2Moderates[str(bestVD_aid)]
        BestAmList = BestAmDict['moderateList']
        BestAnswerBodyPhId = None

        WorstAmDict = aid2Moderates[str(worstVD_aid)]
        WorstAmList = WorstAmDict['moderateList']
        WorstAnswerBodyPhId = None

        # MediumAmDict = aid2Moderates[str(meanVD_aid)]
        # MediumAmList = MediumAmDict['moderateList']
        # MediumAnswerBodyPhId = None


        for tup in BestAmList:
            phType = tup[1]
            if phType in [2,5,8]: # Initial Body, or Edit Body, or Rollback Body
                BestAnswerBodyPhId = tup[0]
        
        for tup in WorstAmList:
            phType = tup[1]
            if phType in [2,5,8]: # Initial Body, or Edit Body, or Rollback Body
                WorstAnswerBodyPhId = tup[0]
        
        # for tup in MediumAmList:
        #     phType = tup[1]
        #     if phType in [2,5,8]: # Initial Body, or Edit Body, or Rollback Body
        #         MediumAnswerBodyPhId = tup[0]

        # update qid2typical3Answers
        qid2typical3Answers[qid]={'answerCount': len(acList),
                                  'tags': qid2tags[qid],
                                  'questionTitle': postHistoryId2text_total[str(questionTitlePhId).replace('"',"'")],
                                  'questionBody': postHistoryId2text_total[str(questionBodyPhId).replace('"',"'")],
                                  'BestAnswerId': bestVD_aid,
                                  'BestAnswerBody': postHistoryId2text_total[str(BestAnswerBodyPhId).replace('"',"'")],
                                  'BestAnswerVoteDiff':bestVD,
                                #   'MediumAnswerId': meanVD_aid,
                                #   'MediumAnswerBody': postHistoryId2text_total[str(MediumAnswerBodyPhId).replace('"',"'")],
                                #   'MediumAnswerVoteDiff':meanVD,
                                  'WorstAnswerId': worstVD_aid,
                                  'WorstAnswerBody': postHistoryId2text_total[str(WorstAnswerBodyPhId).replace('"',"'")],
                                  'WorstAnswerVoteDiff':worstVD}
    
    return qid2typical3Answers
            

def myFun(commName, commDir,root_dir, logFileName):
   
    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())
    print(f"processing {commName}")

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # # check whether already done this step, skip
    # resultFiles = [f'qid2typical3Answers(moreThan3Answers).json']
    # resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
    # if os.path.exists(resultFiles[0]):
    #     # target date
    #     target_date = datetime.datetime(2023, 8, 28)
    #     # file last modification time
    #     timestamp = os.path.getmtime(resultFiles[0])
    #     # convert timestamp into DateTime object
    #     datestamp = datetime.datetime.fromtimestamp(timestamp)
    #     print(f'{commName} Modified Date/Time:{datestamp}')
    #     if datestamp >= target_date:
    #         print(f"{commName} has already done this step.")
    #         return

    try:
        # extract full text body of posts
        with open(intermediate_directory+'/'+'aid2Moderates.json') as json_file:
            aid2Moderates = json.load(json_file)
        with open(intermediate_directory+'/'+'qid2Moderates.json') as json_file:
            qid2Moderates = json.load(json_file)
    except Exception as e:
        writeIntoLog(f"{commName} has No aid2Moderates.json and qid2Moderates.json\n", root_dir, logFileName)
        return

    saveChunkDir = os.path.join(intermediate_directory, r'phId2Text_chunks_folder')

    partFiles = [ f.path for f in os.scandir(saveChunkDir) if f.path.endswith('.json') ]
    if len(partFiles)==0:
        writeIntoLog(f"{commName} has No phId2Text_chunks_folder\n", root_dir, logFileName)

    # sort csvFiles paths based on part number
    partFiles.sort(key=lambda p: int(p.strip(".json").split("_")[-1]))
    partsCount = len(partFiles)                                
    print(f"there are {partsCount} splitted event list files in {commName}")

    try:
        postHistoryId2text_total = defaultdict()
        for i, subDir in enumerate(partFiles):
            part = i+1
            partDir = subDir
            with open(partDir) as json_file:
                postHistoryId2text_chunk = json.load(json_file)
                postHistoryId2text_total.update(postHistoryId2text_chunk)
                postHistoryId2text_chunk.clear()
    except Exception as e:
        writeIntoLog(f"{commName} failed to concatenate all postHistoryId2text_chunks {e}\n", root_dir, logFileName)
        return


    # get vote diff scores
    qid2aidsAndVoteDiff = defaultdict()
    try:
        # load filtered_answerWithVotes_ids_and_scores
        with open(intermediate_directory+'/'+'verifyQualities_newModelAndCVP.dict', 'rb') as inputFile:
            filtered_answerWithVotes_ids_and_scores = pickle.load( inputFile)
            # al ist of tup (ids, learned_q, voteDiff, voteCount, sentimentScores,helpful_sentimentList[i],correct_sentimentList[i], learned_q_CVP)
            
            for t in filtered_answerWithVotes_ids_and_scores:
                qid, aid = t[0]
                voteDiff = t[2]
                if qid not in qid2aidsAndVoteDiff.keys():
                    qid2aidsAndVoteDiff[qid] = [(aid,voteDiff)]
                else:
                    qid2aidsAndVoteDiff[qid].append((aid,voteDiff))
            
            filtered_answerWithVotes_ids_and_scores.clear()
    except Exception as e:
        writeIntoLog(f"{commName} has No verifyQualities_newModelAndCVP.dict\n", root_dir, logFileName)
        try:
            with open(intermediate_directory+'/'+'whole_answersWithVotes_indice_removeFirstRealVote.dict', 'rb') as inputFile:
                total_answersWithVotes_indice = pickle.load( inputFile)

                try:
                    with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
                        Questions = pickle.load( inputFile)

                        total_answersWithVotes_ids = []
                        for i,(qid, ai) in enumerate(total_answersWithVotes_indice):
                            filtered_answerList = Questions[qid]['filtered_answerList']
                            aid = filtered_answerList[ai]
                            total_answersWithVotes_ids.append((qid,aid))
                            voteMatrix = Questions[qid]['vote_matrix'].todense()
                            voteDiff = np.sum(voteMatrix[ai,:])
                            if qid not in qid2aidsAndVoteDiff.keys():
                                qid2aidsAndVoteDiff[qid] = [(aid,voteDiff)]
                            else:
                                qid2aidsAndVoteDiff[qid].append((aid,voteDiff))
                        Questions.clear()
                        total_answersWithVotes_indice.clear()

                        # save qid2aidsAndVoteDiff for filtered answersWithVotes
                        with open(intermediate_directory+'/'+'filtered_answersWithVotes_voteDiffs.dict', 'wb') as outputFile:
                            pickle.dump(qid2aidsAndVoteDiff, outputFile)
                        writeIntoLog(f"{commName} has extracted filtered_answersWithVotes_voteDiffs.dict\n", root_dir, logFileName)
                except Exception as e:
                    writeIntoLog(f"{commName} has No QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict\n", root_dir, logFileName)
                    return
        except Exception as e:
            writeIntoLog(f"{commName} has No whole_answersWithVotes_indice_removeFirstRealVote.dict\n", root_dir, logFileName)
            return

    try:
        # load qid2tags
        with open(intermediate_directory+'/'+'tqid2tags.dict', 'rb') as inputFile:
            qid2tags = pickle.load( inputFile)
    except Exception as e:
        writeIntoLog(f"{commName} has No tqid2tags.dict\n", root_dir, logFileName)
        return

    qid2typical3Answers = generate_typical(aid2Moderates, qid2Moderates, postHistoryId2text_total,  qid2aidsAndVoteDiff, commName, qid2tags)

    postHistoryId2text_total.clear()

    # with open('intermediate_data_folder/qid2typical3Answers(moreThan3Answers).json', "w") as outfile: 
    with open('intermediate_data_folder/qid2typical2Answers(moreThan2Answers).json', "w") as outfile: 
        json.dump(qid2typical3Answers, outfile, default=str)
        # print(f"saved qid2typical3Answers (length: {len(qid2typical3Answers)}) for {commName}")
        print(f"saved qid2typical2Answers (length: {len(qid2typical3Answers)}) for {commName}")

    # csvfile = open(root_dir+'/allComm_qid2typical3Answers(moreThan3Answers)_count.csv', 'a', newline='')
    csvfile = open(root_dir+'/allComm_qid2typical2Answers(moreThan2Answers)_count.csv', 'a', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([commName, len(qid2Moderates), len(qid2typical3Answers)])
    csvfile.close()

    # # save csv
    # with open('qid2typical3Answers.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=';',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow( ["question Id","question Title", "question Body Text", "Best Answer Body Text", "Best Answer Vote Score", "Medium Answer Body Text", "Medium Answer Vote Score" , "Worst Answer Body Text", "Worst Answer Vote Score"])
    #     for qid, d in qid2typical3Answers.items():
    #         writer.writerow((qid, d['questionTitle'], d['questionBody'], d['BestAnswerBody'], d['BestAnswerVoteDiff'], d['MediumAnswerBody'], d['MediumAnswerVoteDiff'], d['WorstAnswerBody'], d['WorstAnswerVoteDiff']))

    # print(f"saved {len(qid2typical3Answers)} question samples for {commName}")
    
    
def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    logFileName = "preprocessing13_extractQid2TypicalAnswers_Log.txt"

    # # save results of all comms
    # import csv
    # print(f"start to save the results as csv...")
    # csvfile = open('allComm_prompt1_tokenCounts.csv', 'w', newline='')
    # writer = csv.writer(csvfile, delimiter=',',
    #                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # writer.writerow( ["commName","total token count", "avg token count"])
    # csvfile.close()

    # test on comm "korean.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[175][0], commDir_sizes_sortedlist[175][1], root_dir)

    # test on comm "parenting.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[264][0], commDir_sizes_sortedlist[264][1], root_dir)

    # test on comm "graphicdesign.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[315][0], commDir_sizes_sortedlist[315][1], root_dir)

    # test on comm "drupal.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[329][0], commDir_sizes_sortedlist[329][1], root_dir)

    # test on comm "stats.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[347][0], commDir_sizes_sortedlist[347][1], root_dir)


    # csvfile = open(root_dir+'/allComm_qid2typical3Answers(moreThan3Answers)_count.csv', 'w', newline='')
    csvfile = open(root_dir+'/allComm_qid2typical2Answers(moreThan2Answers)_count.csv', 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["commName", "original Question Count ", "selected Question Count"])
    csvfile.close()
    
    # splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    
    # selected communities
    
    # selected_comms = ["scifi.stackexchange","academia.stackexchange","freelancing.stackexchange",
    #                   "cooking.stackexchange","writers.stackexchange","photo.stackexchange",
    #                   "unix.stackexchange","apple.stackexchange","askubuntu",
    #                   "android.stackexchange","diy.stackexchange","travel.stackexchange",
    #                   "music.stackexchange","coffee.stackexchange","matheducators.stackexchange",
    #                   "cseducators.stackexchange","homebrew.stackexchange","opensource.stackexchange",
    #                   "skeptics.stackexchange","politics.stackexchange","history.stackexchange",
    #                   "devops.stackexchange","webapps.stackexchange","moderators.stackexchange",
    #                   "french.stackexchange","russian.stackexchange","spanish.stackexchange",
    #                   "italian.stackexchange","japanese.stackexchange","portuguese.stackexchange",
    #                   "hinduism.stackexchange","judaism.stackexchange","buddhism.stackexchange",
    #                   "chinese.stackexchange","islam.stackexchange","hardwarerecs.stackexchange",
    #                   "softwarerecs.stackexchange","ai.stackexchange"]
    
    # added_comms = ["bicycles.stackexchange","workplace.stackexchange","graphicdesign.stackexchange",
    #                "softwareengineering.stackexchange","datascience.stackexchange","networkengineering.stackexchange",
    #                "pm.stackexchange","ux.stackexchange","christianity.stackexchange",
    #                "interpersonal.stackexchange","money.stackexchange","parenting.stackexchange",
    #                "pets.stackexchange","expatriates.stackexchange",
    #                "lifehacks.stackexchange","sustainability.stackexchange","dba.stackexchange",
    #                "dsp.stackexchange","gamedev.stackexchange","gis.stackexchange","german.stackexchange"]

    # selected_comms = []
    # added_comms = ['cstheory.stackexchange', 'law.stackexchange', 'mathoverflow.net']


    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist[:359]): # except stackoverflow
        commName = tup[0]
        commDir = tup[1]

        # if commName not in set(selected_comms).union(set(added_comms)):
        #     print(f"{commName} is not selected, skip")
        #     continue

        try:
            p = mp.Process(target=myFun, args=(commName,commDir,root_dir, logFileName))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()

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

    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('preprocessing13_extractQid2TypicalAnswers  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
