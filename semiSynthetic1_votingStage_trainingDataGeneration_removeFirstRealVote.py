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
import sys


def myAction (question_tuple,commName,questionCount,questionIndex):
    qid = question_tuple[0]
    print(f"processing {commName} question {qid} on {mp.current_process().name}")
    eventList = question_tuple[1]['eventList']
    
    # raw x and y from event list
    raw_x = []
    raw_y = []
    targetQandAOfvotes = [] # keep a recored of all vote events' corresponding questionId and answerIndex, a list of tuple (qid, ai)
    # the answer index (ai) here are corresponding to filtered_answerList, not the original answerList of a question
    
    answersWithVotes = [] # keep a list of answer index corresponding to filtered_answerList that with More than 1 vote

    # extract universal time tick as sorting base
    sortingBaseList = [] # should correspond to each data sample

    firstVoteRemovedFlagForEachAnswer =  []

    # extract all voting events
    for event in eventList:
        if event['et']=='v': # event type is voting
            if event['ai'] in firstVoteRemovedFlagForEachAnswer: # current vote's target answer already has the first vote removed
                raw_y.append(event['v'])
                targetQandAOfvotes.append((qid,event['ai']))
                if event['ai'] not in answersWithVotes:
                    answersWithVotes.append(event['ai'])
                ranksOfAnswersBeforeT = event['ranks']
                curAnswerRank = ranksOfAnswersBeforeT[event['ai']]
                IPWrank = 1/(1+curAnswerRank)
                raw_x.append((event['pvr'],event['nvr'],event['seen_pvr'],IPWrank, event['rl']))
                t = event['universalTimeTick']
                sortingBaseList.append(t)
            else: # current vote is its target answer's first vote, don't use as sample
                firstVoteRemovedFlagForEachAnswer.append(event['ai'])
        # else: # event['et']=='w' or 'e'
                
    assert len(sortingBaseList) == len(raw_y)

    answerCount = len(answersWithVotes)
    if answerCount <=0: # current question has no answer that has more than one vote
        return None
    
    # sort the answerWithVotes list
    answersWithVotes.sort()

    # create data X_part1, X_part2 and y from raw_x and raw_y
    X_part1 = []
    # Data X_part1 is a nd.array in shape of (number_of_votes, number_of_first_part_features)
    # number_of_features = number_of_coefficients + number_of_nus
    # number_of_coefficients = 4 (pos_vote_ratio, neg_vote_ratio, seen_pos_vote_ratio, inverse_displayed_rank)
    # number_of_nus : corresponding to question-level parameter relative-length
    X_part2 = []
    # Data X_part2 is a nd.array in shape of (number_of_votes, number_of_qs) 
    # number_of_qs : corresponding to answer-level parameter intrinsic quality

    y = raw_y
    # label y is a nd.array in shape of (number_of_vote,)

    # genrate data for real votes
    totalVoteEvents = len(targetQandAOfvotes)
    for i,(qid,ai) in enumerate(targetQandAOfvotes):
        print(f"Processing voting event {i+1}/{totalVoteEvents} of question {qid} running on {mp.current_process().name} ")
        temp_x = [raw_x[i][0], raw_x[i][1], raw_x[i][2], raw_x[i][3]]
        RLsOfQuestions = [0]*(questionCount) # initialize all relative length as 0  
        # change the corresponding current rl
        RLsOfQuestions[questionIndex] = raw_x[i][4]

        QualitiesOfAnswers = [0]*(answerCount) # initialize all qualities as 0
        # change the corresponding current quality
        qualityIndex = answersWithVotes.index(ai)
        QualitiesOfAnswers[qualityIndex] = 1
        
        # combine x sample
        try:
            X_part1.append(temp_x + RLsOfQuestions)
            X_part2.append(QualitiesOfAnswers )
        except Exception as e:
            sys.exit(e)

    trainingDataDict = defaultdict()
    trainingDataDict['X_part1'] = X_part1 
    trainingDataDict['X_part2'] = X_part2
    trainingDataDict['y'] = y
    trainingDataDict['answersWithVotes'] = answersWithVotes

    print(f"{mp.current_process().name} return")

    # return processed question's qid, content, the number of answers with votes, the number of real votes as samples
    return (qid, trainingDataDict, len(answersWithVotes), len(targetQandAOfvotes), sortingBaseList)
    

def myFun(commName, commDir, roundIndex, selected_reg_strengthList, variation, CVPgenerated, Interaction, Quadratic):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    current_directory = os.getcwd()
    intermediate_directory = os.path.join(current_directory, r'intermediate_data_folder')

    for reg_alpha in selected_reg_strengthList:

        # check whether already done this step, skip
        if Interaction:
            resultFiles = [f'semiSynthetic_newModelInteraction_{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_forEachQuestion_removeFirstRealVote.dict',f'semiSynthetic_newModelInteraction_{variation}_round{roundIndex}_regAlpha({reg_alpha})_qid2sortingBaseList_removeFirstRealVote.dict']
        elif Quadratic:
            resultFiles = [f'semiSynthetic_newModelQuadratic_{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_forEachQuestion_removeFirstRealVote.dict',f'semiSynthetic_newModelQuadratic_{variation}_round{roundIndex}_regAlpha({reg_alpha})_qid2sortingBaseList_removeFirstRealVote.dict']
        resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
        if os.path.exists(resultFiles[0]) and os.path.exists(resultFiles[1]):
            # target date
            target_date = datetime.datetime(2025, 3, 27)
            # file last modification time
            timestamp = os.path.getmtime(resultFiles[0])
            # convert timestamp into DateTime object
            datestamp = datetime.datetime.fromtimestamp(timestamp)
            print(f'{commName} Modified Date/Time:{datestamp}')
            if datestamp >= target_date:
                print(f"{commName} has already done this step for {reg_alpha}.")
                continue

        if Interaction:
            with open(intermediate_directory+'/'+f'simulated_data_byNewModelInteraction{variation}_round{roundIndex}_regAlpha({reg_alpha}).dict', 'rb') as inputFile:
                loadedFile = pickle.load( inputFile)
        elif Quadratic:
            with open(intermediate_directory+'/'+f'simulated_data_byNewModelQuadratic{variation}_round{roundIndex}_regAlpha({reg_alpha}).dict', 'rb') as inputFile:
                loadedFile = pickle.load( inputFile)
        elif CVPgenerated:
            with open(intermediate_directory+'/'+f'simulated_data_byCVP{variation}_round{roundIndex}_regAlpha({reg_alpha}).dict', 'rb') as inputFile:
                loadedFile = pickle.load( inputFile)
        else: # new model generated
            with open(intermediate_directory+'/'+f'simulated_data_byNewModel{variation}_round{roundIndex}_regAlpha({reg_alpha}).dict', 'rb') as inputFile:
                loadedFile = pickle.load( inputFile)

        simulatedQuestions = loadedFile[0]

        questionCount = len(simulatedQuestions)
        
        # process Questions chunk by chunk
        n_proc = mp.cpu_count()-2 # left 2 cores to do others
        all_outputs = []
        with mp.Pool(processes=n_proc) as pool:
            args = zip(list(simulatedQuestions.items()),[commName]*questionCount,[questionCount]*questionCount, list(range(questionCount)))
            # issue tasks to the process pool and wait for tasks to complete
            results = pool.starmap(myAction, args , chunksize=n_proc)
            # process pool is closed automatically
            for res in results:
                if res != None:
                    if isinstance(res, str):
                        print(res)
                    else:
                        all_outputs.append(res)
                else:
                    print(f"None")

        # clear Questions to save memory
        simulatedQuestions.clear()
        
        print("combining the outputs...")
        # combine all_outputs
        # combined all Questions
        all_Questions = defaultdict()
        answersWithVotesCountList = []

        qid2sortingBaseList = defaultdict()

        max_votesCountOfComm = 0 # max vote count of a question in entire community
        for tup in all_outputs:
            # for combine outputs
            qid = tup[0]
            value = tup[1]
            all_Questions[qid] = value

            answersWithVotesCount = tup[2] # the number of answers that involved in current question
            answersWithVotesCountList.append(answersWithVotesCount)

            votesCount = tup[3] # the number of real voting events that involved in current question
            if votesCount > max_votesCountOfComm:
                max_votesCountOfComm = votesCount

            sortingBaseList = tup[4]
            qid2sortingBaseList[qid] = sortingBaseList

        total_answersWithVotesCount = sum(answersWithVotesCountList) # the total number of answers whose qualities will be learned for the community
        total_answersWithVotes_indice = [] # keep a (qid, ai) index for each learned quality

        all_outputs.clear()

        print("updating and combining X_part1 and X_part2...")
        # combine X_part1 and X_part2
        questions_Data = defaultdict()
        dataSampleCount = 0
        part2StartIndex = 0


        for qind, (qid, content) in enumerate(all_Questions.items()):
            new_X = []
            part2EndIndex = part2StartIndex + len(content['answersWithVotes'])
            for cur_i, cur_row in enumerate(content['X_part1']):
                QualitiesOfVotes = [0]*(total_answersWithVotesCount) # initialize all qualities as 0
                # change the corresponding current quality
                QualitiesOfVotes[part2StartIndex:part2EndIndex] = content['X_part2'][cur_i]
                new_X.append(cur_row + QualitiesOfVotes)
            part2StartIndex = part2EndIndex
            
            new_X_matrix = lil_matrix( np.array(new_X), dtype=np.float16 ) 
            questions_Data[qid] = (new_X_matrix, content['y'], max_votesCountOfComm)
            dataSampleCount += len(new_X)
            total_answersWithVotes_indice.extend([(qid,ai) for ai in content['answersWithVotes']])

        assert len(total_answersWithVotes_indice) == total_answersWithVotesCount

        all_Questions.clear()

        logfilename = 'semiSynthetic1_votingStage_trainingDataGeneration_removeFirstRealVote_Log.txt'
        logtext = f"CVPgeneratedFlag:{CVPgenerated}, Interaction:{Interaction}, Quadratic:{Quadratic}, Round {roundIndex} Variation{variation}:\n"
        logtext += f"reg_alpha : {reg_alpha}\n"
        logtext += f"Count of data samples: {dataSampleCount} out of total {len(questions_Data)} questions and total {total_answersWithVotesCount} answers\n"
        writeIntoLog(logtext, commDir , logfilename)

        all_outputs.clear()

        if dataSampleCount==0:
            print(f"{commName} has no data sample!!!!")
            return
        
        # check whether have intermediate data folder, create one if not

        final_directory = os.path.join(commDir, r'intermediate_data_folder')
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        
        print('Start to save the result files')
        if Interaction:
            try:
                # save all dataset
                with open(final_directory+'/'+f'semiSynthetic_newModelInteraction_{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(questions_Data, outputFile)
                # save the questionId and answerIndex correspondingn to dataset
                with open(final_directory+'/'+f'semiSynthetic_newModelInteraction_{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(total_answersWithVotes_indice, outputFile)
                # save the qid2sortingBaseList correspondingn to dataset
                with open(final_directory+'/'+f'semiSynthetic_newModelInteraction_{variation}_round{roundIndex}_regAlpha({reg_alpha})_qid2sortingBaseList_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(qid2sortingBaseList, outputFile)
                print(f"saving result files for reg_alpha({reg_alpha}) of {commName} successfully!")
            except Exception as e:
                print(f"for {commName}, error when saving the result files: {e}")
                sys.exit()
            
        
        elif Quadratic:
            try:
                # save all dataset
                with open(final_directory+'/'+f'semiSynthetic_newModelQuadratic_{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(questions_Data, outputFile)
                # save the questionId and answerIndex correspondingn to dataset
                with open(final_directory+'/'+f'semiSynthetic_newModelQuadratic_{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(total_answersWithVotes_indice, outputFile)
                # save the qid2sortingBaseList correspondingn to dataset
                with open(final_directory+'/'+f'semiSynthetic_newModelQuadratic_{variation}_round{roundIndex}_regAlpha({reg_alpha})_qid2sortingBaseList_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(qid2sortingBaseList, outputFile)
                print(f"saving result files for reg_alpha({reg_alpha}) of {commName} successfully!")
            except Exception as e:
                print(f"for {commName}, error when saving the result files: {e}")
                sys.exit()
            
        
        elif CVPgenerated:
            try:
                # save all dataset
                with open(final_directory+'/'+f'semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(questions_Data, outputFile)
                # save the questionId and answerIndex correspondingn to dataset
                with open(final_directory+'/'+f'semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(total_answersWithVotes_indice, outputFile)
                # save the qid2sortingBaseList correspondingn to dataset
                with open(final_directory+'/'+f'semiSynthetic{variation}_round{roundIndex}_regAlpha({reg_alpha})_qid2sortingBaseList_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(qid2sortingBaseList, outputFile)
                print(f"saving result files for reg_alpha({reg_alpha}) of {commName} successfully!")
            except Exception as e:
                print(f"for {commName}, error when saving the result files: {e}")
                sys.exit()
        else: # New Model generated
            try:
                # save all dataset
                with open(final_directory+'/'+f'semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_datasets_forEachQuestion_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(questions_Data, outputFile)
                # save the questionId and answerIndex correspondingn to dataset
                with open(final_directory+'/'+f'semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha})_whole_answersWithVotes_indice_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(total_answersWithVotes_indice, outputFile)
                # save the qid2sortingBaseList correspondingn to dataset
                with open(final_directory+'/'+f'semiSynthetic_newModel_{variation}_round{roundIndex}_regAlpha({reg_alpha})_qid2sortingBaseList_removeFirstRealVote.dict', 'wb') as outputFile:
                    pickle.dump(qid2sortingBaseList, outputFile)
                print(f"saving result files for newModel_{variation}_round{roundIndex}_reg_alpha({reg_alpha}) of {commName} successfully!")
            except Exception as e:
                print(f"for {commName}, error when saving the result files: {e}")
                sys.exit()



def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    ### using CVP to generate semisynthetic dataset
    # CVPgenerated = True
    # Interaction = False
    # Quadratic = False
    # roundIndex = 1 # sampling round 
    # roundIndex = 2  # one question one answer
    # roundIndex = 3  # one question one answer, 100 times vote count per answer
    # roundIndex = 4  # one question one answer, 1000 times vote count per answer, q_std = 1
    # roundIndex = 5 # one question one answer, 1000 times vote count per answer, q_std = 1, fix lambda
    # roundIndex = 6 # one question one answer for toy example +++---, generate 1000 votes, fix lambda = -0.14189369976520538  prior_beta: -0.07276562601327896 prior_q_0: -0.14553125202655792 
    # roundIndex = 7 # one question one answer for toy example +++---, generate 1000 votes, fix lambda: -0.2684609591960907, beta: -0.058872684836387634, q_0: -0.11774536967277527
    # variation = '_fixedTau_noRL'
    # roundIndex = 8 # one question one answer, 1000 times vote count per answer, q_std = 1, fix lambda (for different regularization strength)
    # try_reg_strengthList = [0.005, 0.05, 0.5, 5, 50, 500, 5000]
    
    # roundIndex = 9 # one question one answer, 1000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    # roundIndex = 10 # one question one answer, 5000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    # roundIndex = 11 # one question one answer, 10000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    #roundIndex = 12 # one question one answer, 50000 votes, q_std = 1, fix lambda (for different regularization strength) selected_reg_strengthList = [1000, 3000, 5000]
    
    # roundIndex = 13 # one question one answer, 1000 votes, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    # roundIndex = 14 # one question one answer, 5000 votes, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    # roundIndex = 15 # one question one answer, 10000 votes, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    
    # roundIndex = 16 # one question multiple answer, 10000 events, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]

    # roundIndex = 17 # multiple question multiple answer, amplified 10 times of original total event count, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]

    # roundIndex = 18 # multiple question multiple answer, amplified 10 times of original total event count, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    # if roundIndex in [18]:
    #     variation = '_noRL'

    # selected_reg_strengthList = [300, 500, 700]

    # roundIndex = 19 ## multiple question multiple answer, original total event count, fix tau = 1, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    # roundIndex = 20 ## multiple question multiple answer, original total event count, learn tau, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    # if roundIndex in [19, 20]:
    #     variation = ''

    # selected_reg_strengthList = [500, 700]

    # roundIndex = 21 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # commName2selected_reg_strengthList = {'cstheory.stackexchange':[800, 900, 1000],
    #                                       'stackoverflow':[1000],
    #                                       'unix.meta.stackexchange':[60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
    #                                       'politics.stackexchange':[900,1000]}
    # commName2selected_reg_strengthList = {'3dprinting.stackexchange':[30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700],
    #                                       'latin.stackexchange':[50, 60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
    #                                       'meta.askubuntu':[70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
    #                                       'lifehacks.stackexchange':[400, 500, 600, 700, 800,900,1000]}
    # variation = ''

    ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and TRIPLED beta (for different regularization strength) selected_reg_strengthList of each comm
    
    # ### using new Model to generate semi-synthetic dataset statistics
    # CVPgenerated = False
    # Interaction = False
    # Quadratic = False
    # # roundIndex = 1 ## multiple question multiple answer, original total event count, fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # # roundIndex = 2 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # # roundIndex = 3 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and DOUBLED beta (for different regularization strength) selected_reg_strengthList of each comm
    # # roundIndex = 4 

    ### using new Model+ Interaction or + Quadratic rank term to generate semi-synthetic dataset statistics
    CVPgenerated = False
    Interaction = False
    Quadratic = True
    roundIndex = 2 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    # roundIndex = 3 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and DOUBLED beta (for different regularization strength) selected_reg_strengthList of each comm
    # roundIndex = 4 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and TRIPLED beta (for different regularization strength) selected_reg_strengthList of each comm

    if Interaction:
        commName2selected_reg_strengthList = {
                                          'cstheory.stackexchange':[700, 800, 900, 1000],
                                          'stackoverflow':[900,1000],
                                          'politics.stackexchange':[200, 300, 400, 500, 600, 700, 800,900,1000],
                                          'math.meta.stackexchange':[100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                          'mathoverflow.net':[400,500],
                                          'askubuntu':[400,500],
                                          'philosophy.stackexchange':[50, 60, 70,80,90,100, 200, 300, 400, 500],
                                          'codegolf.meta.stackexchange':[200, 300, 400, 500, 600, 700, 800, 900,1000]
                                          }
    elif Quadratic:
        commName2selected_reg_strengthList = {
                                          'cstheory.stackexchange':[400, 500, 600, 700],
                                          'stackoverflow':[900,1000],
                                          'politics.stackexchange':[50,60,70,80,90,100],
                                          'math.meta.stackexchange':[70, 90, 100],
                                          'mathoverflow.net':[400,500],
                                          'askubuntu':[500,600],
                                          'philosophy.stackexchange':[3,4,5,6,7,8,9,10,20],
                                          'codegolf.meta.stackexchange':[100, 200, 300, 400, 500, 600, 700, 800, 900,1000]
                                          }
    else: # CVP or NewModel generated
        commName2selected_reg_strengthList = {
                                            #   '3dprinting.stackexchange':[20, 30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700],
                                            #   'latin.stackexchange':[40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                            #   'meta.askubuntu':[20, 30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                            #   'lifehacks.stackexchange':[300, 400, 500, 600, 700, 800, 900],
                                            #   'cstheory.stackexchange':[700, 800, 900, 1000],
                                            #   'stackoverflow':[1000],
                                            #   'unix.meta.stackexchange':[30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000],
                                            #   'politics.stackexchange':[200, 300, 400, 500, 600, 700, 800,900,1000],
                                            #   'math.meta.stackexchange':[30,40,50,60, 70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900,1000]
                                            #   'mathoverflow.net':[500,600],
                                            #   'mathematica.stackexchange':[80,90,100],
                                            #   'askubuntu':[300,400,500,600],
                                            #   'philosophy.stackexchange':[60, 70,80,90,100, 200, 300, 400, 500, 600],
                                                'codegolf.meta.stackexchange':[100, 200, 300, 400, 500, 600, 700, 800, 900,1000]
                                            }
    variation = '_fixedTau'


    """
    # test on comm "3dprinting.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[227][0], commDir_sizes_sortedlist[227][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[227][0]], variation, CVPgenerated)
    # test on comm "latin.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[229][0], commDir_sizes_sortedlist[229][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[229][0]], variation, CVPgenerated)
    # test on comm "lifehacks.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[233][0], commDir_sizes_sortedlist[233][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[233][0]], variation, CVPgenerated)
    # test on comm "askubuntu.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[231][0], commDir_sizes_sortedlist[231][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[231][0]], variation, CVPgenerated)

    # test on comm "cstheory.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[256][0], commDir_sizes_sortedlist[256][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[256][0]], variation, CVPgenerated)
    # test on comm "stackoverflow" to debug
    myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[359][0]], variation, CVPgenerated)
    # test on comm "unix.meta.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[173][0], commDir_sizes_sortedlist[173][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[173][0]], variation, CVPgenerated)
    # # test on comm "politics.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[283][0], commDir_sizes_sortedlist[283][1], roundIndex, commName2selected_reg_strengthList[commDir_sizes_sortedlist[283][0]], variation, CVPgenerated)
    
    """
    # skip splitted communities
    splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']

    selected_comms = [
                    "cstheory.stackexchange", 
                     "stackoverflow", 
                     "politics.stackexchange",
                     "codegolf.meta.stackexchange",
                     "math.meta.stackexchange",
                     "mathoverflow.net",
                     "askubuntu",
                     "philosophy.stackexchange",
                     ]
    # # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]
        if commName not in selected_comms: # skip 
            continue

        # if commName in splitted_comms: # skip splitted communities 
        #     print(f"{commName} was split.")
        #     continue

        selected_reg_strengthList = commName2selected_reg_strengthList[commName]

        try:
            p = mp.Process(target=myFun, args=(commName,commDir, roundIndex, selected_reg_strengthList, variation, CVPgenerated, Interaction, Quadratic))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
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
    
    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('semiSynthetic1_votingStage_trainingDataGeneration_removeFirstRealVote Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
