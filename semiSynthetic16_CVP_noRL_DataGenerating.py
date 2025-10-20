import os
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################
from toolFunctions import format_time, writeIntoLog, insertEventInUniversalTimeStep, savePlot
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
from statistics import mean
from scipy.stats import norm, bernoulli
import numpy as np
from scipy.optimize import fsolve
from scipy.special import psi # digamma
from scipy.special import kl_div

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def getRanks (voteLists, J):
    voteSum = [] # answer-wise sum
    for i in range(J): # each i represents an answer
        voteSum.append(sum([tup[0] for tup in voteLists[i]]))
    aiWithVoteSum = [(ai,vc) for (ai,vc) in enumerate(voteSum)] # ai is answer index, vc is vote count
    aiWithVoteSum.sort(reverse=True, key=lambda x:(x[1],x[0])) # sort by the vote count and then by the answer index
    aiWithRanks = [(tup[0],r+1) for (r,tup) in enumerate(aiWithVoteSum)] # tup[0] is the answer index, r is the rank
    aiWithRanks.sort(key=lambda x:x[0])
    ranksOfAnswersBeforeT = [r for (ai,r) in aiWithRanks]
    return ranksOfAnswersBeforeT

def getAnswerIndex(answerList, aid):
    for i, tup in enumerate(answerList):
        if tup[0] == aid:
            return i
    return None

def mySolver (answerCount, eventCount ):
    def myFunction(theta):
        F = np.empty((1))
        F[0] = theta * (psi(theta + eventCount) - psi(theta)) - answerCount
        return F
    
    # def myFunction2(theta):  # equivalent to above
    #     F = np.empty((1))
    #     F[0] = - answerCount
    #     for k in range(1, eventCount+1):
    #         F[0] += theta / (theta + k -1)
    #     return F

    zGuess = np.array([1])
    z = fsolve(myFunction,zGuess)
    # z2 = fsolve(myFunction2,zGuess)

    return z[0]

##########################################################################
def mySimulation(commName, n, qid2theta, q_std, lamb, tau, qid2lifeTime, roundIndex):
    timeTick = 0 # univeral timeTick for this comm
    aid = 0 # virtual answer Id starts from 0

    if roundIndex in [2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16]: # one question
        Questions = defaultdict()
        Q = defaultdict()
        Q['answerList'] = [] # the 'answerList' is a list of tuple (aid, timeTick)
        Q['answerQualityList'] = [] # a list of quality corresponding to each answer in 'answerList'
        Q['lengthList'] = [] # the 'lengthList' is a list of lengths of each answer
        Q['voteLists'] = [] # the 'voteList' is a list of vote lists for each answer, a vote list is a list of tuple (vote, timeTick)
        # the tupel consits of three items (voteId(int), vote(1 or -1), target answerId(int) )
        Q['eventList'] = []
        qid = list(qid2lifeTime.keys())[0]
        Questions[qid] = Q
    else:
        # simulate for exist questions
        Questions = defaultdict()
        for qid in qid2lifeTime.keys():
            Q = defaultdict()
            Q['answerList'] = [] # the 'answerList' is a list of tuple (aid, timeTick)
            Q['answerQualityList'] = [] # a list of quality corresponding to each answer in 'answerList'
            Q['lengthList'] = [] # the 'lengthList' is a list of lengths of each answer
            Q['voteLists'] = [] # the 'voteList' is a list of vote lists for each answer, a vote list is a list of tuple (vote, timeTick)
            # the tupel consits of three items (voteId(int), vote(1 or -1), target answerId(int) )
            Q['eventList'] = []

            Questions[qid] = Q
    
    print(f"start to generate data for {commName}...")
    while( timeTick <= n):
        if roundIndex in [2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16]: # one question
            targetQid = list(qid2lifeTime.keys())[0] # choose the first question
        else:
            # Step 1: choose target question randomly with prior distribution
            totalLifeTime = sum(qid2lifeTime.values())
            questionLifeTimeDistribution = [t/totalLifeTime for t in qid2lifeTime.values()]
            targetQid = np.random.choice(list(qid2lifeTime.keys()),1, p=questionLifeTimeDistribution)[0] # choose 1 question Id

        # Step 2: generate one event for target questions
        if len(Questions[targetQid]['answerList']) == 0: # current question hasn't got any answer, must write a new answer first
            # sample an intrinsic quality from N(0,q_std)
            q = np.random.normal(0, q_std)
            if roundIndex  in [6]:
                q = -0.14553125202655792  #lambda = -0.14189369976520538  prior_beta: -0.07276562601327896 prior_q_0: -0.14553125202655792 
            elif roundIndex in [7]:
                q = -0.11774536967277527
            # fix the length as 500
            l = 500
            J = 0 # no previous answer before the first answer is create. J here is J_i^{t-1}
            ranksOfAnswersBeforeT = None # no ranks before the first answer is created
            Questions[targetQid]['answerList'].append((aid, timeTick))
            Questions[targetQid]['answerQualityList'].append(q)
            Questions[targetQid]['lengthList'].append(l)
            Questions[targetQid]['voteLists'].append([]) # empty list represents NO vote
            event = {'et':'w', 'ai' : getAnswerIndex(Questions[targetQid]['answerList'],aid), 'J':0, 'n_pos':0,'n_neg':0,'ranks':ranksOfAnswersBeforeT, 'universalTimeTick':timeTick}
            Questions[targetQid]['eventList'].append(event)
            aid +=1 # update aid for next new answer

        else: # generate next event which could be writing event or voting event
            if roundIndex in [2,3,4,5,6,7,8,9,10,11,12,13,14,15]: # one question one answer
                toWrite = 0
            else:
                # write a new response with prob of alpha, otherwise, choose an existing answer to vote
                theta = qid2theta[targetQid]
                existingEventCountOfTargetQuestion = len(Questions[targetQid]['eventList'])
                alpha = theta / (existingEventCountOfTargetQuestion + theta)
                toWrite = bernoulli.rvs(alpha, size=1)[0]

            if toWrite == 1:
                # sample an intrinsic quality from N(0,q_std)
                q = np.random.normal(0, q_std)
                # fix the length as 500
                l = 500
                J = len(Questions[targetQid]['answerList']) # the number of answer before current event
                ranksOfAnswersBeforeT = getRanks(Questions[targetQid]['voteLists'],J)
                Questions[targetQid]['answerList'].append((aid,timeTick))
                Questions[targetQid]['answerQualityList'].append(q)
                Questions[targetQid]['lengthList'].append(l)
                Questions[targetQid]['voteLists'].append([]) # empty list represents NO vote
                event = {'et':'w', 'ai' :  getAnswerIndex(Questions[targetQid]['answerList'],aid), 'J':0, 'n_pos':0,'n_neg':0,'ranks':ranksOfAnswersBeforeT, 'universalTimeTick':timeTick}
                Questions[targetQid]['eventList'].append(event)
                aid +=1 # update aid for next new answer
            
            else: # to vote
                J = len(Questions[targetQid]['answerList']) # the number of answer before current event
                ranksOfAnswersBeforeT = getRanks(Questions[targetQid]['voteLists'],J)
                assert( len(ranksOfAnswersBeforeT) == len(Questions[targetQid]['answerList']))
                
                # choose which answer?
                probsOfAnswers = [(1/(1+r))**tau for r in ranksOfAnswersBeforeT]
                # scale probs to sum-up to 1
                if len(probsOfAnswers)==1:
                    probsOfAnswers = [1]
                else:
                    probsOfAnswers = [p/sum(probsOfAnswers) for p in probsOfAnswers]
                
                aidList = [tup[0] for tup in Questions[targetQid]['answerList']]
                chosen_aid = np.random.choice(aidList,1, p=probsOfAnswers)[0] # choose 1 answer Id
                chosen_ai =  getAnswerIndex(Questions[targetQid]['answerList'],chosen_aid)

                # decide what to vote?
                cur_full_vl = Questions[targetQid]['voteLists'][chosen_ai]
                cur_vl = np.array([tup[0] for tup in cur_full_vl])
                initial_pos = 1
                initial_neg = 1

                if len(cur_vl) ==0: # no previous votes
                    n_pos = initial_pos
                    n_neg = initial_neg
                else: # have previous votes
                    flatten = cur_vl.flatten()
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
                
                q = Questions[targetQid]['answerQualityList'][chosen_ai]
                rl = 1
                
                z = q + lamb*seen_pvr 
                if roundIndex in [6]:
                    beta = -0.07276562601327896
                    curAnswerRank = ranksOfAnswersBeforeT[0]
                    IPWrank = 1/(1+curAnswerRank)
                    z = q + lamb*seen_pvr + beta*IPWrank
                elif roundIndex in [7]:
                    beta = -0.058872684836387634
                    curAnswerRank = ranksOfAnswersBeforeT[0]
                    IPWrank = 1/(1+curAnswerRank)
                    z = q + lamb*seen_pvr + beta*IPWrank

                logit = 1.0 / (1 + np.exp(-z))

                vote = bernoulli.rvs(logit, size=1)[0]
                if vote == 0: # convert negative vote to -1
                    vote = -1

                # add current vote
                Questions[targetQid]['voteLists'][chosen_ai].append((vote, timeTick))
            
                event = {'et':'v', 'ai':chosen_ai,'J':J, 'v':vote, 'pvr':pvr,'nvr':nvr,'seen_pvr':seen_pvr,'n_pos':n_pos,'n_neg':n_neg,'ranks':ranksOfAnswersBeforeT,'rl':rl, 'universalTimeTick':timeTick}
                Questions[targetQid]['eventList'].append(event)

        timeTick += 1
        if (timeTick % 1000 ==0):
            print(f"generated {timeTick} events for {commName}")


    # double check the q_mean and q_std of generated qs
    generated_qs = []
    generated_pos_vote_count = 0
    generated_neg_vote_count = 0
    generated_answer_count = 0
    generated_qid2answerCount = defaultdict()
    generated_qid2lifetime = defaultdict()
    for qid, d in Questions.items():
        generated_qs.extend(d['answerQualityList'])
        eventList = d['eventList']
        for e in eventList:
            if e['et'] == 'v':
                if e['v'] == 1:
                    generated_pos_vote_count +=1
                else:
                    generated_neg_vote_count +=1
            else:
                generated_answer_count +=1
        generated_qid2answerCount[qid] = len(d['answerQualityList'])
        generated_qid2lifetime[qid] = len(d['eventList'])

    generated_q_mean, generated_q_std = norm.fit(generated_qs)
    print(f"generated_q_mean:{generated_q_mean} vs q_mean:{0}, generated_q_std:{generated_q_std} vs q_std:{q_std} for {commName}")
    print(f"generated pos votes: {generated_pos_vote_count}, neg votes: {generated_neg_vote_count}, answer count: {generated_answer_count} for {commName}")
    
    return Questions, generated_q_mean, generated_q_std, generated_neg_vote_count , generated_pos_vote_count, generated_answer_count, generated_qid2answerCount, generated_qid2lifetime

def myFun(commName, commDir, rootDir, roundIndex, selected_reg_strengthList):

    print(f"comm {commName} running on {mp.current_process().name}")

    logFileName = "semiSynthetic16_CVP_noRL_DataGenerating_Log.txt"

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    for reg_alpha in selected_reg_strengthList:

        # # check whether already done this step, skip
        # resultFiles = [f'simulated_data_byCVP_noRL_round{roundIndex}.dict']
        # resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
        # if os.path.exists(resultFiles[0]):
        #     # target date
        #     target_date = datetime.datetime(2023, 8, 27)
        #     # file last modification time
        #     timestamp = os.path.getmtime(resultFiles[0])
        #     # convert timestamp into DateTime object
        #     datestamp = datetime.datetime.fromtimestamp(timestamp)
        #     print(f'{commName} Modified Date/Time:{datestamp}')
        #     if datestamp >= target_date:
        #         print(f"{commName} has already done this step.")
        #         return

        # load CVP one-side temperal training result
        if roundIndex in [18]:
            with open(intermediate_directory+'/'+f"temperalOrderTraining10_CVP_fixedTau_noRL_regAlpha({reg_alpha})_return.dict", 'rb') as inputFile:
                return_trainSuccess_dict_CVP = pickle.load( inputFile)

        # # get original question count
        # if commName != 'stackoverflow':
        #     with open(intermediate_directory+'/'+'QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList.dict', 'rb') as inputFile:
        #         ori_Questions = pickle.load( inputFile)
        #     ori_questionCount = len(ori_Questions)  # corresponding to the number of nus

        #     # get questions life time (event count), and relative lengths
        #     qid2lifeTime = defaultdict()
        #     qid2answerCount = defaultdict()

        #     for qid, d in ori_Questions.items():
        #         writingAndVotingEventCount = 0
        #         for e in d['eventList']:
        #             eventType = e['et']
        #             if eventType == 'w' or eventType =='v':
        #                 writingAndVotingEventCount +=1
        #         qid2lifeTime[qid] = writingAndVotingEventCount
        #         qid2answerCount[qid] = len(d['filtered_answerList'])

        #     ori_Questions.clear()
        
        # else: # stackoverflow
        #     splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
        #     split_QuestionsWithEventList_files_directory = os.path.join(splitFolder_directory, r'QuestionsPartsWithEventList')

        #     # to get the origianl question count when generating nus, load eventlist file
        #     ori_questionCount = 0
        #     qid2lifeTime = defaultdict()
        #     qid2answerCount = defaultdict()

        #     partFiles = [ f.path for f in os.scandir(split_QuestionsWithEventList_files_directory) if f.path.startswith("QuestionsWithAnswersWithVotesWithPostHistoryWithMatrixWithEventList_part_") ]
        #     for i, partDir in enumerate(partFiles):
        #         print(f"scanning part {i+1} eventlist file of {commName} for original question count...")
        #         # get question count of each part
        #         with open(partDir, 'rb') as inputFile:
        #             Questions_part = pickle.load( inputFile)
        #         ori_questionCount += len(Questions_part)
        #         for qid, d in Questions_part.items():
        #             writingAndVotingEventCount = 0
        #             for e in d['eventList']:
        #                 eventType = e['et']
        #                 if eventType == 'w' or eventType =='v':
        #                     writingAndVotingEventCount +=1
        #             qid2lifeTime[qid] = writingAndVotingEventCount
        #             qid2answerCount[qid] = len(d['filtered_answerList'])

        #         Questions_part.clear() # clear this to same memory

        # assert len(qid2lifeTime) == ori_questionCount

        # # save
        # with open(intermediate_directory+'/'+f'semiSynthetic11_CVP_fixedTau_noRL_DataGenerating_round{roundIndex}_outputs.dict', 'wb') as outputFile:
        #     pickle.dump((qid2lifeTime,qid2answerCount), outputFile)
        # print(f"semiSynthetic11_CVP_fixedTau_noRL_DataGenerating_outputs Saved for {commName}.")   
        
        # load 
        with open(intermediate_directory+'/'+'semiSynthetic11_CVP_fixedTau_noRL_DataGenerating_round1_outputs.dict', 'rb') as inputFile:
            qid2lifeTime,qid2answerCount = pickle.load( inputFile)
        
        # get prior alpha
        with open(intermediate_directory+'/'+'CVP_selectionPhaseTrainingData.dict', 'rb') as inputFile:
            loadedFiles = pickle.load( inputFile)
        
        # naive community-level alpha in CVP
        # writingEventCount_total = loadedFiles[5]
        # votingEventCount_total = loadedFiles[6]
        # alpha = writingEventCount_total/(writingEventCount_total + votingEventCount_total)
        # loadedFiles=[]

        # question-level alpha == theta in CRP wiki 
        # get the number of events and the number of answers for each question
        qid2EventCountAndAnswerCount = defaultdict()
        ET = loadedFiles[1]
        J = loadedFiles[2]
        qidList = loadedFiles[4]
        for i, qid in enumerate(qidList):
            cur_ET = ET[i] # event type list of current question qid
            cur_J = J[i] # J list of current question qid, J is # the number of answers till a time, starting from 0 (J should be J_i^{t-1})
            
            if cur_ET[-1] == 'w':
                answerCount = cur_J[-1] + 1
            else:
                answerCount = cur_J[-1]
            
            eventCount = len(cur_ET)

            qid2EventCountAndAnswerCount[qid] = (answerCount, eventCount)
        loadedFiles = []
        
        # solve the theta for each question
        qid2theta = defaultdict()
        for qid, tup in qid2EventCountAndAnswerCount.items():
            answerCount, eventCount = tup
            try:
                theta = mySolver (answerCount, eventCount )
            except Exception as e:
                print(e)

            qid2theta[qid] = theta

        # check
        try:
            assert set(qid2lifeTime.keys()).issubset(set(qid2theta.keys()))
        except Exception as e:
            writeIntoLog(f"{commName} exception: {e}",commDir,logFileName)
            return

        # get prior coefs
    
        try:
            coefs = return_trainSuccess_dict_CVP[commName]['coefs_sklearn']
            lamb = coefs[0] # for one side training
            qs = return_trainSuccess_dict_CVP[commName]['qs_sklearn']
        except:
            print(f"No CVP_fixedTau_noRL voting stage sklearn training results for {commName}")
            return

        if roundIndex in [6]:
            lamb = -0.14189369976520538
        if roundIndex in [7]:
            lamb = -0.2684609591960907
        
        # get prior tau
        result_directory = os.path.join(commDir, r'result_folder')
        with open(result_directory+'/'+'CVP1_selectionPhaseTrainingResults.dict', 'rb') as inputFile:
            tau, tau_record, ll_record, convergeFlag, convergeIter = pickle.load( inputFile)
        
        if not convergeFlag: # not converge, can't use this tau
            print(f"selection phase training of CVP for {commName} didn't converge, can't use the learned tau!")
            return

        # get prior sigma of qualities
        # # Fit a normal distribution to the data:
        # q_mean, q_std = norm.fit(qs)
        # if (q_mean > 0.011) or (q_mean < -0.011):  # q_mean must be around 0
        #     print(f"Exception: q_mean is {q_mean}, not around 0. for {commName}")
        #     writeIntoLog(f"Exception: q_mean is {q_mean}, not around 0. ", commDir, logFileName)
        #     return

        # arbitrary given q_std
        q_std = 1


        # n = sum(qid2lifeTime.values()) * 1 # total sample size for round 1
        if roundIndex in [18]:
            n = sum(qid2lifeTime.values()) * 10 # amplified 10 times of original total event count

        simulatedQuestions, generated_q_mean, generated_q_std, generated_neg_vote_count , generated_pos_vote_count, generated_answer_count, generated_qid2answerCount, generated_qid2lifetime = mySimulation(commName, n, qid2theta, q_std, lamb, tau, qid2lifeTime, roundIndex)

        # save simulated Questions

        with open(intermediate_directory+'/'+f'simulated_data_byCVP_noRL_round{roundIndex}_regAlpha({reg_alpha}).dict', 'wb') as outputFile:
            pickle.dump((simulatedQuestions, qid2theta, generated_q_mean, generated_q_std, generated_neg_vote_count , generated_pos_vote_count, generated_answer_count, generated_qid2answerCount, generated_qid2lifetime), outputFile)
        print(f"simulatedQuestions using reg_alpha = {reg_alpha} Saved as dict for {commName}.")

        """
        # sanity check the generated counts
        answerCounts = []
        eventCounts = []
        generated_answerCounts = []
        generated_eventCounts = []
        for qid, answerCount in qid2answerCount.items():
            answerCounts.append(answerCount)
            eventCounts.append(qid2lifeTime[qid])
            generated_answerCounts.append(generated_qid2answerCount[qid])
            generated_eventCounts.append(generated_qid2lifetime[qid])


        # Quantitatively compute a distance: KL divergence as a KL (semi-synthetic || original)
        minAnswerCount = min([min(answerCounts), min(generated_answerCounts)])
        maxAnswerCount = max([max(answerCounts), max(generated_answerCounts)])
        binStepOfAnswerCount = math.ceil((maxAnswerCount - minAnswerCount+1)/5) # the width of bin, make sure there are 5 bins for diff comms
        if binStepOfAnswerCount < 2:
            binStepOfAnswerCount == 2
        binsForAnswerCounts = list(range(minAnswerCount,maxAnswerCount+2, binStepOfAnswerCount)) 

        minEventCount = min([min(eventCounts), min(generated_eventCounts)])
        maxEventCount = max([max(eventCounts), max(generated_eventCounts)])
        binStepOfEventCount = math.ceil((maxEventCount - minEventCount+1)/20) # the width of bin, make sure there are 20 bins for diff comms
        if binStepOfEventCount <3:
            binStepOfEventCount ==3
        binsForEventCounts = range(minEventCount,maxEventCount+2, binStepOfEventCount)  

        answerCounts_Probs = list(np.histogram(answerCounts, density=True, bins=binsForAnswerCounts)[0])
        generated_answerCounts_Probs = list(np.histogram(generated_answerCounts, density=True, bins=binsForAnswerCounts)[0])
        eventCounts_Probs = list(np.histogram(eventCounts, density=True, bins=binsForEventCounts)[0])
        generated_eventCounts_Probs = list(np.histogram(generated_eventCounts, density=True, bins=binsForEventCounts)[0])
        
        KLsynToOri_answerCount = sum([0 if math.isinf(kl) else kl for kl in kl_div(answerCounts_Probs, generated_answerCounts_Probs)])
        KLsynToOri_eventCount = sum([0 if math.isinf(kl) else kl for kl in kl_div(eventCounts_Probs, generated_eventCounts_Probs)])

        # plot histo
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

        # We can set the number of bins with the *bins* keyword argument.
        axs[0].hist(answerCounts, bins=binsForAnswerCounts, histtype='step', rwidth=0.8, label= 'prior distribution')
        axs[0].hist(generated_answerCounts, bins=binsForAnswerCounts, histtype='step', rwidth=0.8, label= 'generated distribution')
        axs[0].set_title(f'answerCount per question Histo\nKL:{round(KLsynToOri_answerCount,4)}')
        axs[0].set_xticks(binsForAnswerCounts)
        axs[0].legend(fontsize = 'x-small')
        axs[1].hist(eventCounts, bins=binsForEventCounts, histtype='step', rwidth=0.8, label= 'prior distribution')
        axs[1].hist(generated_eventCounts, bins=binsForEventCounts, histtype='step', rwidth=0.8, label= 'generated distribution')
        axs[1].set_title(f'eventCount per question Histo\nKL:{round(KLsynToOri_eventCount,4)}')
        axs[1].set_xticks(binsForEventCounts)
        xlabels = [f'{b}' if (i%5==0) else "" for i, b in enumerate(binsForEventCounts) ]
        axs[1].set_xticklabels(xlabels)
        axs[1].legend(fontsize = 'x-small')

        fig.suptitle(f"{commName}")
        savePlot(fig, f"semiSynthetic11_CVP_no_dataGenerating_Histo_round{roundIndex}.png")
        """

        if roundIndex != 1:
            KLsynToOri_answerCount  = None
            KLsynToOri_eventCount = None
        
        # save csv statistics
        with open(rootDir+'/'+f'allComm_CVP_noRL_semiSyntheticDataGeneration_round{roundIndex}_statistics.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow( [commName,len(simulatedQuestions), generated_answer_count,reg_alpha,lamb,tau, generated_pos_vote_count, generated_neg_vote_count, generated_q_mean, generated_q_std, KLsynToOri_answerCount, KLsynToOri_eventCount])
        


def main():
    rootDir = os.getcwd()
    t0=time.time()
    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # save csv for CVP generating semi-synthetic dataset statistics
    roundIndex = 18 # multiple question multiple answer, amplified 10 times of original total event count, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    
    selected_reg_strengthList = [300, 500, 700]

    with open(rootDir+'/'+f'allComm_CVP_noRL_semiSyntheticDataGeneration_round{roundIndex}_statistics.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","total question Count", "generated Answer count", "reg_strength","prior_lambda", "prior_tau", "generated Pos Vote Count", "generated Neg Vote Count", "generated Quality mean", "generated Quality STD", "KL (generated Answer Count Per Question Distribution to prior)", "KL (generated Event Count Per Question Distribution to prior)"])
    

    """
    # test on comm "3dprinting.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[227][0], commDir_sizes_sortedlist[227][1], rootDir, roundIndex, selected_reg_strengthList)
    # test on comm "latin.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[229][0], commDir_sizes_sortedlist[229][1], rootDir, roundIndex, selected_reg_strengthList)
    # test on comm "lifehacks.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[233][0], commDir_sizes_sortedlist[233][1], rootDir, roundIndex, selected_reg_strengthList)
    # test on comm "askubuntu.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[231][0], commDir_sizes_sortedlist[231][1], rootDir, roundIndex, selected_reg_strengthList)
    
    """
    selectedComms = ['3dprinting.stackexchange','latin.stackexchange','meta.askubuntu','lifehacks.stackexchange']
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        if commName not in selectedComms: # skip the comm that is not selected
            continue
 
        try:
            p = mp.Process(target=myFun, args=(commName,commDir, rootDir, roundIndex, selected_reg_strengthList))
            p.start()
        except Exception as e:
            print(e)
            return

        processes.append(p)
        if len(processes)==4: # must do this one comm by one comm, or the sampled qs are the same
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
    print('semiSynthetic16_CVP_noRL_DataGenerating Done completely.    Elapsed: {:}.\n'.format(elapsed))
    
      
if __name__ == "__main__":
  
    # calling main function
    main()


##########################################################################