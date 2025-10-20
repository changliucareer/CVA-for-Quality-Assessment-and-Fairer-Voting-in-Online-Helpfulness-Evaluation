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



def generate_typical(aid2Moderates, qid2Moderates, postHistoryId2text_total,  qid2aidsAndQuality, aid2newModel_sklearn_q_rank, commName, qid2tags):
    qid2typical3Answers = defaultdict()


    for qid, acList in qid2aidsAndQuality.items(): 
        # if len(acList)<5: # skip questions that have less than 5 answers
        if len(acList)<3: # skip questions that have less than 3 answers
        # if len(acList)<2: # skip questions that have less than 2 answers
            continue
        # pick up best, median and worst answers as few shots based on their vote difference score
        sortedList = copy.deepcopy(acList)
        sortedList.sort(key=lambda x:x[1], reverse=True) 
        bestQ = sortedList[0][1]
        worstQ = sortedList[-1][1]
        meanQ = (bestQ + worstQ)/2

        bestQ_aid = sortedList[0][0]
        worstQ_aid = sortedList[-1][0]
        
        # find the answer whose quality is the closest to meanQ
        sortedList_distanceTomeanQ = [(tup[0], abs(tup[1]-meanQ), tup[1]) for tup in sortedList]
        sortedList_distanceTomeanQ.sort(key=lambda x:x[1])
        meanQ_aid = sortedList_distanceTomeanQ[0][0]
        meanQ = sortedList_distanceTomeanQ[0][2]

        try:
            assert meanQ_aid != bestQ_aid
            assert meanQ_aid != worstQ_aid
            assert meanQ != bestQ
            assert meanQ != worstQ
            assert bestQ_aid != worstQ_aid
            assert bestQ != worstQ
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
        BestAmDict = aid2Moderates[str(bestQ_aid)]
        BestAmList = BestAmDict['moderateList']
        BestAnswerBodyPhId = None

        WorstAmDict = aid2Moderates[str(worstQ_aid)]
        WorstAmList = WorstAmDict['moderateList']
        WorstAnswerBodyPhId = None

        MediumAmDict = aid2Moderates[str(meanQ_aid)]
        MediumAmList = MediumAmDict['moderateList']
        MediumAnswerBodyPhId = None


        for tup in BestAmList:
            phType = tup[1]
            if phType in [2,5,8]: # Initial Body, or Edit Body, or Rollback Body
                BestAnswerBodyPhId = tup[0]
        
        for tup in WorstAmList:
            phType = tup[1]
            if phType in [2,5,8]: # Initial Body, or Edit Body, or Rollback Body
                WorstAnswerBodyPhId = tup[0]
        
        for tup in MediumAmList:
            phType = tup[1]
            if phType in [2,5,8]: # Initial Body, or Edit Body, or Rollback Body
                MediumAnswerBodyPhId = tup[0]

        # update qid2typical3Answers
        qid2typical3Answers[qid]={'answerCount': len(acList),
                                  'tags': qid2tags[qid],
                                  'questionTitle': postHistoryId2text_total[str(questionTitlePhId).replace('"',"'")],
                                  'questionBody': postHistoryId2text_total[str(questionBodyPhId).replace('"',"'")],
                                  'BestAnswerId': bestQ_aid,
                                  'BestAnswerBody': postHistoryId2text_total[str(BestAnswerBodyPhId).replace('"',"'")],
                                  'BestAnswerQuality':bestQ,
                                  'MediumAnswerId': meanQ_aid,
                                  'MediumAnswerBody': postHistoryId2text_total[str(MediumAnswerBodyPhId).replace('"',"'")],
                                  'MediumAnswerQuality':meanQ,
                                  'WorstAnswerId': worstQ_aid,
                                  'WorstAnswerBody': postHistoryId2text_total[str(WorstAnswerBodyPhId).replace('"',"'")],
                                  'WorstAnswerQuality':worstQ}
    
    return qid2typical3Answers
            

def myFun(commName,commDir,root_dir, logFileName, roundIndex, variation, reg_alpha_newModelInteraction, reg_alpha_newModel, reg_alpha_CVP, sampleCount):
   
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


    # get quality scores
    with open(intermediate_directory+f"/temperalOrderTraining15_verifyingQualities_outputs_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_newModelInteraction})_newModelRegAlpha({reg_alpha_newModel})_CVPRegAlpha({reg_alpha_CVP}).dict", 'rb') as inputFile:
        tup = pickle.load( inputFile)
        aid2sentiment_rank = tup[0]
        aid2helpfulScore_rank = tup[1]
        aid2VoteDiff_rank = tup[2]
        aid2CVP_sklearn_q_rank = tup[3]
        aid2newModel_sklearn_q_rank = tup[4]
        aid2newModelInteraction_sklearn_q_rank = tup[5]
        aid2sentiment_rankZscore = tup[6]
        aid2helpfulScore_rankZscore = tup[7]
        aid2VoteDiff_rankZscore = tup[8]
        aid2CVP_sklearn_q_rankZscore = tup[9]
        aid2newModel_sklearn_q_rankZscore = tup[10]
        aid2newModelInteraction_sklearn_q_rankZscore = tup[11]
        disagreeAnswersIdList = tup[12]
        disagreeGaps = tup[13]
        total_answersWithVotes_ids = tup[14]
        topAnswersIdList = tup[15]
        underEstimatedAnswersIdList_forSentiment = tup[16]
        overEstimatedAnswersIdList_forSentiment = tup[17]
        underEstimatedAnswersIdList_forHelpful = tup[18]
        overEstimatedAnswersIdList_forHelpful = tup[19]
        disagreeAnswersIdList = tup[20]
        aid2quality_CVP = tup[21]
        aid2quality_newModel = tup[22]
        aid2quality_newModelInteration = tup[23]
        print( f"loaded intermediate outputs for {commName}.")

    qid2aidsAndQuality = defaultdict()
    

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
                        try:
                            quality = aid2quality_newModel[aid]
                        except:
                            print(f"answer {aid} not found quality for {commName}")
                            continue
                        if qid not in qid2aidsAndQuality.keys():
                            qid2aidsAndQuality[qid] = [(aid,quality)]
                        else:
                            qid2aidsAndQuality[qid].append((aid,quality))
                    Questions.clear()
                    total_answersWithVotes_indice.clear()

                    # save qid2aidsAndVoteDiff for filtered answersWithVotes
                    with open(intermediate_directory+'/'+'filtered_answersWithVotes_qualities.dict', 'wb') as outputFile:
                        pickle.dump(qid2aidsAndQuality, outputFile)
                    writeIntoLog(f"{commName} has extracted filtered_answersWithVotes_qualities.dict\n", root_dir, logFileName)
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
    
    

    qid2typical3Answers = generate_typical(aid2Moderates, qid2Moderates, postHistoryId2text_total,  qid2aidsAndQuality, aid2newModel_sklearn_q_rank, commName, qid2tags)

    postHistoryId2text_total.clear()

    with open('intermediate_data_folder/qid2typical3Answers(moreThan3Answers)_basedOnQuality.json', "w") as outfile: 
    # with open('intermediate_data_folder/qid2typical2Answers(moreThan2Answers)_basedOnQuality.json', "w") as outfile: 
        json.dump(qid2typical3Answers, outfile, default=str)
        print(f"saved qid2typical3Answers_basedOnQuality (length: {len(qid2typical3Answers)}) for {commName}")
        # print(f"saved qid2typical2Answer_basedOnQuality (length: {len(qid2typical3Answers)}) for {commName}")

    csvfile = open(root_dir+'/allComm_qid2typical3Answers(moreThan3Answers)_basedOnQuality_count.csv', 'a', newline='')
    # csvfile = open(root_dir+'/allComm_qid2typical2Answers(moreThan2Answers)_basedOnQuality_count.csv', 'a', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([commName, len(qid2Moderates), len(qid2aidsAndQuality),len(qid2typical3Answers)])
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

    roundIndex = 2 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    variation = '_fixedTau'

    # load commName2selected_reg_strengthList (extracted from temperalOrderTraining15_verifyingQualities_GPT.py)
    with open(f'allComm_bestRegAlphas_fixedTau.dict', 'rb') as inputFile:
        commName2selected_reg_strengthList = pickle.load( inputFile)

    logFileName = "preprocessing13_extractQid2TypicalAnswers_basedOnQuality_Log.txt"

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


    csvfile = open(root_dir+'/allComm_qid2typical3Answers(moreThan3Answers)_basedOnQuality_count.csv', 'w', newline='')
    # csvfile = open(root_dir+'/allComm_qid2typical2Answers(moreThan2Answers)_basedOnQuality_count.csv', 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["commName", "original Question Count ", "Question with Quality Count","selected Question Count"])
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


    # prepare args
    argsList = []
    for commName, tup in commName2selected_reg_strengthList.items():
        # if commName not in set(selected_comms).union(set(added_comms)):
        #     continue

        reg_alpha_newModelInteraction, reg_alpha_newModel, reg_alpha_CVP, sampleCount = tup
        for comm in commDir_sizes_sortedlist:
            if comm[0] == commName:
                commDir = comm[1]
                break
        argsList.append((commName,commDir,root_dir, logFileName, roundIndex, variation, reg_alpha_newModelInteraction, reg_alpha_newModel, reg_alpha_CVP, sampleCount))


    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for myargs in argsList:
 
        try:
            p = mp.Process(target=myFun, args=myargs)
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
