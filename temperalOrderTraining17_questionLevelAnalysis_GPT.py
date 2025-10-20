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
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
from scipy import stats
import json
from openai import OpenAI

def get_zScore(rankList):
    rankList = list(rankList)
    mean = np.mean(rankList)
    # standard deviation
    std = np.std(rankList, dtype=np.float64)
    zScores = []
    for r in rankList:
        if std==0:
            zScores.append(0)
        else:
            zScores.append(-(r-mean)/std) # flip the sign of z, since top ranked z is smaller and lower ranked z is bigger
    return zScores

# get Probability Less Than a Certain Z-Score
def probFromZscore(z):
    return scipy.stats.norm.cdf(z)

#######################################################################################################
def generatePrompt(sucessfulQid2Aids, postId2text, aid2VoteDiff_rank,aid2CVP_sklearn_q_rank,aid2newModel_sklearn_q_rank, promptType):
    prompt_Dict = defaultdict() # key: qid, value: prompt

    if len(sucessfulQid2Aids)>5:
        selectedQids = random.sample(list(sucessfulQid2Aids.keys()), 5)
    else:
        selectedQids = list(sucessfulQid2Aids.keys())
    
    for qid, aids in sucessfulQid2Aids.items():
        if qid not in selectedQids: # skip 
            continue
        
        filteredAids = []
        for i, aid in enumerate(aids): 
            # adding answers
            if isinstance(list(postId2text.keys())[0], str):
                aid = str(aid)
            if aid not in postId2text.keys():
                print(f"{aid} not in postId2text.keys()")
                continue
            if int(aid) not in aid2VoteDiff_rank.keys() or int(aid) not in aid2CVP_sklearn_q_rank.keys() or int(aid) not in aid2newModel_sklearn_q_rank.keys():
                print(f"{aid} not in aid2rank.keys()")
                continue
            filteredAids.append(aid)

        # adding question
        if isinstance(list(postId2text.keys())[0], str):
                qid = str(qid)
        questionText = postId2text[qid].strip()

        if promptType == 2:
            prompt = f"For a QUESTION and its {len(filteredAids)} ANSWERS, we have 3 algorithms to evaluate the helpfulness of the ANSWERS to the QUESTION and rank the ANSWERS from the most helpful to the least helpful. The 3 algorithms are named 'voteDiff', 'CVP' and 'newModel'. \n"
        elif promptType == 3:
            prompt = f"For a QUESTION and its {len(filteredAids)} ANSWERS, we have 3 algorithms to evaluate the helpfulness of the ANSWERS to the QUESTION and rank the ANSWERS from the most helpful to the least helpful. The 3 algorithms are named 'algorithm_1', 'algorithm_2' and 'algorithm_3'. \n"
        
        prompt += f"The question is followed by '[QUESTION]:'. Then the ANSWERS are given from [ANSWER-1] to [ANSWER-{len(filteredAids)}]. At last, the rank orders of ANSWERS based on the 3 algorithms mentioned above are also provided. \n"
        prompt += f"Please consider all the given information, and do the following 2 tasks:\n"
        prompt += f"(1) Guess the evaluating metrics used by the 3 different algorithms when they rank the answers. Explain what metrics are valued importantly for each algorithm. \n"
        prompt += f"(2) Do a 3-way comparison among 3 algorithms about their evaluating metrics and write them as a table.\n"

        prompt += "-------------------------------------------------------------------------------"
        prompt += f'\n[QUESTION]:\n'+ questionText + '\n'
        
        voteDiff_ranks = []
        CVP_ranks = []
        newModel_ranks = []

        for i, aid in enumerate(filteredAids): 
            answerSerial = i+1
            # adding answers
            if isinstance(list(postId2text.keys())[0], str):
                aid = str(aid)
            if aid not in postId2text.keys():
                print(f"debug")
                continue
            if int(aid) not in aid2VoteDiff_rank.keys() or int(aid) not in aid2CVP_sklearn_q_rank.keys() or int(aid) not in aid2newModel_sklearn_q_rank.keys():
                print(f"debug")
                continue

            answerText = postId2text[aid].strip()
            prompt += "-------------------------------------------------------------------------------"
            prompt += f'\n[ANSWER-{answerSerial}]:\n' + answerText + '\n'

            # adding ranks
            voteDiff_rank = aid2VoteDiff_rank[int(aid)]
            CVP_rank = aid2CVP_sklearn_q_rank[int(aid)]
            newModel_rank = aid2newModel_sklearn_q_rank[int(aid)]

            voteDiff_ranks.append((answerSerial, voteDiff_rank))
            CVP_ranks.append((answerSerial, CVP_rank))
            newModel_ranks.append((answerSerial, newModel_rank))
        
        voteDiff_ranks.sort(key=lambda x: x[1])
        CVP_ranks.sort(key=lambda x: x[1])
        newModel_ranks.sort(key=lambda x: x[1])

        prompt += "-------------------------------------------------------------------------------"
        if promptType == 2:
            prompt += f"\n < voteDiff > rank orders:\n"
        elif promptType == 3:
            prompt += f"\n < algorithm_1 > rank orders:\n"

        for r, (answerSerial, rank) in enumerate(voteDiff_ranks):
            prompt += f"Rank-{r+1}: [ANSWER-{answerSerial}]\n"

        if promptType == 2:
            prompt += f"\n < CVP > rank orders:\n"
        elif promptType == 3:
            prompt += f"\n < algorithm_2 > rank orders:\n"

        for r, (answerSerial, rank) in enumerate(CVP_ranks):
            prompt += f"Rank-{r+1}: [ANSWER-{answerSerial}]\n"

        if promptType == 2:
            prompt += f"\n < newModel > rank orders:\n"
        elif promptType == 3:
            prompt += f"\n < algorithm_3 > rank orders:\n"

        for r, (answerSerial, rank) in enumerate(newModel_ranks):
            prompt += f"Rank-{r+1}: [ANSWER-{answerSerial}]\n"

        prompt_Dict[qid] = prompt
    
    return prompt_Dict


def askGPT (prompts_Dict, my_model):
    response_Dict = defaultdict()

    for qid, my_prompt in prompts_Dict.items(): 
        my_prompt = my_prompt.replace('"',"'") # replace " with ' to avoid "" break in script

        client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            # api_key="your-key",
            api_key="your_key",
        )

        response = client.chat.completions.create(
            model=my_model,
            messages=[{"role": "user", "content": my_prompt}],
            temperature=0, # this is the degree of randomness of the model's output
            )
        
        res_text = response.choices[0].message.content.strip()
        # print(f"question {qid}:\n{res_text}")

        response_Dict[qid] = res_text
    
    return response_Dict
#####################################################################################################################

def myFun(commName, commDir, rootDir, roundIndex, variation, reg_alpha_NewModelInteraction, reg_alpha_NewModel, reg_alpha_CVP, sampled_comms):
   
    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())
    print(f"processing {commName}")

    # load intermediate_data filesdd
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    if commName == 'stackoverflow': # using subcomm to represent stackoverflow
        subComms_data_folder = os.path.join(commDir, f'subCommunities_folder')
        ## Load all sub community direcotries 
        with open(subComms_data_folder+'/'+f'subCommName2subCommDir.dict', 'rb') as inputFile:
            subCommName2commDir = pickle.load( inputFile)
        subCommDir = subCommName2commDir['reactjs']
        subComm_intermediate_directory = os.path.join(subCommDir, r'intermediate_data_folder')

    """
    # load intermediate outputs
    with open(intermediate_directory+f"/temperalOrderTraining15_verifyingQualities_outputs_newModelGenereated{variation}_round{roundIndex}_newModelInteractionRegAlpha({reg_alpha_NewModelInteraction})_newModelRegAlpha({reg_alpha_NewModel})_CVPRegAlpha({reg_alpha_CVP}).dict", 'rb') as inputFile:
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
        print( f"loaded intermediate outputs for {commName}.")
    ############################################################################################
    
    # get qid2Aids
    qid2Aids = defaultdict()
    for tup in total_answersWithVotes_ids:
        qid, aid = tup
        if qid not in qid2Aids.keys():
            qid2Aids[qid] = [aid]
        else:
            qid2Aids[qid].append(aid)

    # find successful qids
    sucessfulQid2Aids_forSentiment = defaultdict()
    sucessfulQid2Aids_forHelpful = defaultdict()
    
    sucessfulAids_forSentiment = []
    sucessfulAids_forHelpful = []

    # using the prob difference between zscores to filter successful aids
    for qid, aids in qid2Aids.items():
        voteDiff_rankZscores = []
        CVP_sklearn_q_rankZscores = [] 
        newModel_sklearn_q_rankZscores = []
        newModelInteraction_sklearn_q_rankZscores = []
        sentiment_rankZscores = []
        helpfulScore_rankZscores = []
        
        filteredAids = []
        for aid in aids:
            if (aid not in aid2sentiment_rankZscore.keys()) or (aid not in aid2helpfulScore_rankZscore.keys()) or (aid not in aid2VoteDiff_rankZscore.keys()) or (aid not in aid2CVP_sklearn_q_rankZscore.keys()) or (aid not in aid2newModel_sklearn_q_rankZscore.keys()) or (aid not in aid2newModelInteraction_sklearn_q_rankZscore.keys()):
                continue
            filteredAids.append(aid)
            voteDiff_rankZscores.append(aid2VoteDiff_rankZscore[aid])
            CVP_sklearn_q_rankZscores.append(aid2CVP_sklearn_q_rankZscore[aid])
            newModel_sklearn_q_rankZscores.append(aid2newModel_sklearn_q_rankZscore[aid])
            newModelInteraction_sklearn_q_rankZscores.append(aid2newModelInteraction_sklearn_q_rankZscore[aid])
            sentiment_rankZscores.append(aid2sentiment_rankZscore[aid])
            helpfulScore_rankZscores.append(aid2helpfulScore_rankZscore[aid])

        for i in range(len(filteredAids)):
            aid = filteredAids[i]
            voteDiff_rankZscore = voteDiff_rankZscores[i]
            CVP_sklearn_q_rankZscore = CVP_sklearn_q_rankZscores[i]
            newModel_sklearn_q_rankZscore = newModel_sklearn_q_rankZscores[i]
            newModelInteraction_sklearn_q_rankZscore = newModelInteraction_sklearn_q_rankZscores[i]
            sentiment_rankZscore = aid2sentiment_rankZscore[aid]
            helpfulScore_rankZscore = aid2helpfulScore_rankZscore[aid]

            voteDiff_rankZscore_diffProbToSentiment = probFromZscore(voteDiff_rankZscore) - probFromZscore(sentiment_rankZscore)
            CVP_sklearn_q_rankZscore_diffProbToSentiment = probFromZscore(CVP_sklearn_q_rankZscore) - probFromZscore(sentiment_rankZscore)
            newModel_sklearn_q_rankZscore_diffProbToSentiment = probFromZscore(newModel_sklearn_q_rankZscore) - probFromZscore(sentiment_rankZscore)
            newModelInteraction_sklearn_q_rankZscore_diffProbToSentiment = probFromZscore(newModelInteraction_sklearn_q_rankZscore) - probFromZscore(sentiment_rankZscore)
            
            voteDiff_rankZscore_diffProbToHelpfulScore = probFromZscore(voteDiff_rankZscore) - probFromZscore(helpfulScore_rankZscore)
            CVP_sklearn_q_rankZscore_diffProbToHelpfulScore = probFromZscore(CVP_sklearn_q_rankZscore) - probFromZscore(helpfulScore_rankZscore)
            newModel_sklearn_q_rankZscore_diffProbToHelpfulScore = probFromZscore(newModel_sklearn_q_rankZscore) - probFromZscore(helpfulScore_rankZscore)
            newModelInteraction_sklearn_q_rankZscore_diffProbToHelpfulScore = probFromZscore(newModelInteraction_sklearn_q_rankZscore) - probFromZscore(helpfulScore_rankZscore)

            # filter successful aids
            if abs(voteDiff_rankZscore_diffProbToSentiment) > abs(CVP_sklearn_q_rankZscore_diffProbToSentiment) and abs(CVP_sklearn_q_rankZscore_diffProbToSentiment) > abs(newModel_sklearn_q_rankZscore_diffProbToSentiment):
                sucessfulAids_forSentiment.append(aid)
            
            if abs(voteDiff_rankZscore_diffProbToHelpfulScore) > abs(CVP_sklearn_q_rankZscore_diffProbToHelpfulScore) and abs(CVP_sklearn_q_rankZscore_diffProbToHelpfulScore) > abs(newModel_sklearn_q_rankZscore_diffProbToHelpfulScore):
                sucessfulAids_forHelpful.append(aid)
        
        # if len(sucessfulAids_forSentiment) == len(filteredAids) and len(filteredAids)>2:
        #     sucessfulQid2Aids_forSentiment[qid] = sucessfulAids_forSentiment
        # if len(sucessfulAids_forHelpful)  == len(filteredAids) and len(filteredAids)>2:
        #     sucessfulQid2Aids_forHelpful[qid] = sucessfulAids_forHelpful

    # using the kendalTau to filter successful qids
    # # compare prior q and learned q ranks (question-level aggregate)
    # compute question-level kendalltau distance of ranks
    for qid, aidList in qid2Aids.items():
        if len(aidList) <5:
            continue

        voteDiffRanks = [aid2VoteDiff_rank[aid] for aid in aidList if aid in aid2VoteDiff_rank.keys()]
        CVPqualityRanks = [aid2CVP_sklearn_q_rank[aid] for aid in aidList if aid in aid2VoteDiff_rank.keys()]
        newModelqualityRanks = [aid2newModel_sklearn_q_rank[aid] for aid in aidList if aid in aid2VoteDiff_rank.keys()]
        newModelInteractionqualityRanks = [aid2newModelInteraction_sklearn_q_rank[aid] for aid in aidList if aid in aid2VoteDiff_rank.keys()]
        
        priorQualityRanks = [aid2sentiment_rank[aid] for aid in aidList if aid in aid2VoteDiff_rank.keys()]
        kendalltau_voteDiff = stats.kendalltau(priorQualityRanks, voteDiffRanks).statistic
        kendalltau_CVP = stats.kendalltau(priorQualityRanks, CVPqualityRanks).statistic
        kendalltau_newModel = stats.kendalltau(priorQualityRanks, newModelqualityRanks).statistic
        kendalltau_newModelInteraction = stats.kendalltau(priorQualityRanks, newModelInteractionqualityRanks).statistic

        if kendalltau_voteDiff < kendalltau_CVP and kendalltau_CVP < kendalltau_newModel:
            if kendalltau_voteDiff >0 and kendalltau_CVP >0 and kendalltau_newModel >0:
                sucessfulQid2Aids_forSentiment[qid] = aidList
        
        priorQualityRanks = [aid2helpfulScore_rank[aid] for aid in aidList if aid in aid2VoteDiff_rank.keys()]
        kendalltau_voteDiff = stats.kendalltau(priorQualityRanks, voteDiffRanks).statistic
        kendalltau_CVP = stats.kendalltau(priorQualityRanks, CVPqualityRanks).statistic
        kendalltau_newModel = stats.kendalltau(priorQualityRanks, newModelqualityRanks).statistic
        kendalltau_newModelInteraction = stats.kendalltau(priorQualityRanks, newModelInteractionqualityRanks).statistic

        if kendalltau_voteDiff < kendalltau_CVP and kendalltau_CVP < kendalltau_newModel:
            if kendalltau_voteDiff >0 and kendalltau_CVP >0 and kendalltau_newModel >0:
                sucessfulQid2Aids_forHelpful[qid] = aidList

    print(f"sucessfulQid2Aids_forSentiment: {len(sucessfulQid2Aids_forSentiment)}, sucessfulQid2Aids_forHelpful: {len(sucessfulQid2Aids_forHelpful)}")

    # save sucessful qid2Aids into csv
    with open(rootDir +'/'+'allComm_temperalOrderTraining17_successfulQid2Aids.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for qid, aids in sucessfulQid2Aids_forSentiment.items():
            successAids = set(aids).intersection(set(sucessfulAids_forSentiment))
            writer.writerow( [commName, 'sentiment', qid, len(aids), len(successAids)])
        for qid, aids in sucessfulQid2Aids_forHelpful.items():
            successAids = set(aids).intersection(set(sucessfulAids_forHelpful))
            writer.writerow( [commName, 'helpful', qid, len(aids), len(successAids)])



    # generate GPT prompt
    promptType = 3
    # load post texts
    if commName == 'stackoverflow':
        with open(intermediate_directory+'/'+'postId2text_reactjs.json') as json_file:
            postId2text = json.load(json_file)
    else:
        with open(intermediate_directory+'/'+'postId2text.json') as json_file:
            postId2text = json.load(json_file)

    if len(sucessfulQid2Aids_forSentiment)>0:
        prompt_forSentiment = generatePrompt(sucessfulQid2Aids_forSentiment, postId2text, 
                                            aid2VoteDiff_rank,aid2CVP_sklearn_q_rank,aid2newModel_sklearn_q_rank, promptType)
    else:
        prompt_forSentiment = None
    
    if len(sucessfulQid2Aids_forHelpful)>0:
        prompt_forHelpful = generatePrompt(sucessfulQid2Aids_forHelpful, postId2text,
                                        aid2VoteDiff_rank,aid2CVP_sklearn_q_rank,aid2newModel_sklearn_q_rank, promptType)
    else:
        prompt_forHelpful = None    

    # save prompt into json
    promptType = 3
    promptJson_files_directory = os.path.join(commDir, r'promptJson_folder')

    if prompt_forSentiment is not None:
        if commName == 'stackoverflow':
            with open(f'promptJson_folder/prompt_template_{promptType}_forSentiment_reactjs.json', "w") as outfile: 
                json.dump(prompt_forSentiment, outfile) 
                print(f"prompt saved for {commName}_forSentiment length: {len(prompt_forSentiment)}")
        else:
            with open(f'promptJson_folder/prompt_template_{promptType}_forSentiment.json', "w") as outfile: 
                json.dump(prompt_forSentiment, outfile) 
                print(f"prompt saved for {commName}_forSentiment length: {len(prompt_forSentiment)}")
    
    if prompt_forHelpful is not None:
        if commName == 'stackoverflow':
            with open(f'promptJson_folder/prompt_template_{promptType}_forHelpful_reactjs.json', "w") as outfile: 
                json.dump(prompt_forHelpful, outfile) 
                print(f"prompt saved for {commName}_forHelpful length: {len(prompt_forHelpful)}")
        else:
            with open(f'promptJson_folder/prompt_template_{promptType}_forHelpful.json', "w") as outfile: 
                json.dump(prompt_forHelpful, outfile) 
                print(f"prompt saved for {commName}_forHelpful length: {len(prompt_forHelpful)}")

    """

    # load promptJson files
    promptType = 3
    if commName == 'stackoverflow':
        with open(f'promptJson_folder/prompt_template_{promptType}_forSentiment_reactjs.json') as jsonfile: 
            prompt_forSentiment = json.load(jsonfile) 
            print(f"prompt loaded for {commName}_forSentiment length: {len(prompt_forSentiment)}")
    else:
        with open(f'promptJson_folder/prompt_template_{promptType}_forSentiment.json') as jsonfile: 
            prompt_forSentiment = json.load(jsonfile) 
            print(f"prompt loaded for {commName}_forSentiment length: {len(prompt_forSentiment)}")

    if commName == 'stackoverflow':
        with open(f'promptJson_folder/prompt_template_{promptType}_forHelpful_reactjs.json') as jsonfile: 
            prompt_forHelpful = json.load(jsonfile) 
            print(f"prompt loaded for {commName}_forHelpful length: {len(prompt_forHelpful)}")
    else:
        with open(f'promptJson_folder/prompt_template_{promptType}_forHelpful.json') as jsonfile: 
            prompt_forHelpful = json.load(jsonfile) 
            print(f"prompt loaded for {commName}_forHelpful length: {len(prompt_forHelpful)}")

    # ask GPT
    my_model = "gpt-4o"
    response_forSentiment = askGPT(prompt_forSentiment, my_model)
    response_forHelpful = askGPT(prompt_forHelpful, my_model)

    # save results
    if commName == 'stackoverflow':
        with open(f'GPTresponse_folder/responseTo_prompts_forSentiment_{promptType}_{my_model}_reactjs.json', "w") as outfile: 
            json.dump( response_forSentiment, outfile)     
            print(f"saved GPT response forSentiment for {commName}")
        with open(f'GPTresponse_folder/responseTo_prompts_forHelpful_{promptType}_{my_model}_reactjs.json', "w") as outfile: 
            json.dump( response_forHelpful, outfile)     
            print(f"saved GPT response forHelpful for {commName}")
    else:
        with open(f'GPTresponse_folder/responseTo_prompts_forSentiment_{promptType}_{my_model}.json', "w") as outfile: 
            json.dump( response_forSentiment, outfile)     
            print(f"saved GPT response forSentiment for {commName}")
        with open(f'GPTresponse_folder/responseTo_prompts_forHelpful_{promptType}_{my_model}.json', "w") as outfile: 
            json.dump( response_forHelpful, outfile)     
            print(f"saved GPT response forHelpful for {commName}")

    # print all
    log_txt = ""
    for qid, prompt in prompt_forSentiment.items():
        log_txt += f"===> For question {qid}:\n\n"
        log_txt += f"Prompt:\n{prompt}\n\n"
        log_txt += f"===================================================================================\n\n"
        log_txt += f"GPT-4-o Response:\n{response_forSentiment[qid]}\n\n"

    writeIntoLog(log_txt,commDir, f'temperalOrderTraining17_{commName.strip(".stackexchange")}_promptType_{promptType}_sentimentAsGroundTruth.txt')

    log_txt = ""
    for qid, prompt in prompt_forHelpful.items():
        log_txt += f"===> For question {qid}:\n\n"
        log_txt += f"Prompt:\n{prompt}\n\n"
        log_txt += f"===================================================================================\n\n"
        log_txt += f"GPT-4-o Response:\n{response_forHelpful[qid]}\n\n"

    writeIntoLog(log_txt,commDir, f'temperalOrderTraining17_{commName.strip(".stackexchange")}_promptType_{promptType}_helpfulAsGroundTruth.txt')
    
    
    

   
def main():

    t0=time.time()
    rootDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    # # # save into csv
    # with open('allComm_temperalOrderTraining17_successfulQid2Aids.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow( ["commName","groud truth", "qid", "answerCount","successful answerCount"])

    # roundIndex = 1 ## multiple question multiple answer, original total event count, fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    roundIndex = 2 ## multiple question multiple answer, original total event count, use real answer length fix tau =1, with RL,  q_std = 1, use lambda and beta (for different regularization strength) selected_reg_strengthList of each comm
    
    commName2selected_reg_strengthList = {
                                        'cstheory.stackexchange':(400,500,500),
                                          'unix.meta.stackexchange':(300,300,300),
                                          'stackoverflow':(0.1,0.1,0.1),
                                          'politics.stackexchange':(0.2,0.1,0.2),
                                        #   '3dprinting.stackexchange':(40,20,80),
                                        #   'latin.stackexchange':(0.3,0.3,0.3),
                                        #   'meta.askubuntu':(700,700,500),
                                        #   'lifehacks.stackexchange':(0.2,0.2,600)
                                          'math.meta.stackexchange':(0.4,0.4,0.2),
                                        'mathoverflow.net':(600,600,0.1),
                                            # 'mathematica.stackexchange':(90,80,100),
                                            'askubuntu':(0.1,600,0.1),
                                            'philosophy.stackexchange':(700,700,600),
                                            
                                          }
    variation = '_fixedTau'
    
    # for sampled comms
    sampled_comms = ['academia.stackexchange','askubuntu',
                      'english.stackexchange','math.stackexchange','mathoverflow.net',
                      'meta.stackexchange','meta.stackoverflow','serverfault',
                      'softwareengineering.stackexchange','superuser','unix.stackexchange',
                      'worldbuilding.stackexchange','physics.stackexchange','electronics.stackexchange',
                      'codegolf.stackexchange','workplace.stackexchange']
    
    # prepare args
    argsList = []
    for commName, tup in commName2selected_reg_strengthList.items():
        reg_alpha_newModelInteraction, reg_alpha_newModel, reg_alpha_CVP = tup
        for comm in commDir_sizes_sortedlist:
            if comm[0] == commName:
                commDir = comm[1]
                break
        argsList.append((commName, commDir, rootDir, roundIndex, variation, reg_alpha_newModelInteraction, reg_alpha_newModel, reg_alpha_CVP, sampled_comms))


    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for myargs in argsList:
 
        try:
            p = mp.Process(target=myFun, args=myargs)
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

    # Report progress.
    elapsed = format_time(time.time() - t0)
    print('verify qualities  Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
