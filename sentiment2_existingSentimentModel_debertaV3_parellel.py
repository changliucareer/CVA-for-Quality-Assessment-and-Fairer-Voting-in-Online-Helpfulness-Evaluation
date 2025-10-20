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
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import pandas as pd
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

def myAction (commName, commDir, chunkIndex, df, model, tokenizer, aspects, post2sentiment):
    # check gpu count
    cuda_count = torch.cuda.device_count()
    # assign one of gpu as device
    d = (chunkIndex+1) % cuda_count
    device = torch.device('cuda:'+str(d) if torch.cuda.is_available() else 'cpu')
    print(f"comm {commName} chunk {chunkIndex}, Start to train NN model... on device: {device}")
    classifier_1 = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

    for line_count, row in df.iterrows():
        print(f"processing processing {commName} chunk {chunkIndex} line {line_count}...")
        # lines of data 
        comment_text = row['Comment']

        cleaned_comment_text = cleanComment(comment_text)
        if len(cleaned_comment_text)<2: # skip the comment that is too short
            continue
        targetPost = row['PostId']

        # get sentiment score of cleaned_comment
        absa_scores_1 = defaultdict()

        for aspect in aspects:
            absa_scores_1[aspect] = classifier_1(cleaned_comment_text,  text_pair=aspect)

        # labels = [dictList[0]['label'] for dictList in list(absa_scores_1.values())]
        # if 'Positive' in labels:
        #     print("Positive")
        # if 'Negative' in labels:
        #     print("Negative")
        # if 'Neutral' in labels:
        #     print("Neutral")
        if targetPost not in post2sentiment.keys():
            post2sentiment[targetPost]={'sentimentScores':[absa_scores_1]}
        else:
            post2sentiment[targetPost]={'sentimentScores': post2sentiment[targetPost]['sentimentScores']+[absa_scores_1]}

    # save results till this chunk
    # convert and save the last post2sentiment dict
    post2sentiment_normalDict = defaultdict()
    for pid, d in post2sentiment.items():
        post2sentiment_normalDict[pid] = {'sentimentScores':d['sentimentScores']}
    
    os.chdir(commDir) # go back to comm directory
    with open('intermediate_data_folder/debertaV3_large_SentimentScores_of_posts_replaceTags_tillCurChunk.dict', 'wb') as outputFile:
        pickle.dump(post2sentiment_normalDict, outputFile)
        print(f"saved sentiment scores of posts for {commName} chunk {chunkIndex}")
        writeIntoLog(f'saved post2sentiment after chunk {chunkIndex} done\n',commDir,'debertaV3_large_SentimentScores_of_posts_replaceTags_Log.txt')


# Attributions of comment CSV
# 'Comments':['Id','PostId','Score','Text','CreationDate','UserId']
def myFun(commIndex,commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # check whether already done this step, skip
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    if os.path.exists(intermediate_directory): 
        resultFiles = ['debertaV3_large_SentimentScores_of_posts_replaceTags.dict']
        resultFiles = [intermediate_directory+'/'+f for f in resultFiles]
        if os.path.exists(resultFiles[0]):
            with open(resultFiles[0], 'rb') as inputFile:
                post2sentiment = pickle.load( inputFile)
                if len(post2sentiment) > 0:
                    print(f"{commName} has already done this step.")
                    return

    # prepare for sentiment analysis
    # Load the ABSA model and tokenizer
    try:
        # model_name = "yangheng/deberta-v3-base-absa-v1.1"
        model_name = "yangheng/deberta-v3-large-absa-v1.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=4096, return_tensors="pt")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)   
    except Exception as eee:
        print("preparing sentiment analysis model failed.")
        print(eee)

    # selected aspects
    aspects = ['useful','helpful','correct']

    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    post2sentiment = manager.dict() # to save the used train mode (wholebatch or minibatch) of each community

    chunk_size = 1000000
    processes = []
    chunkIndex = 0
    finishedCount =0
    # load sentiment training data
    try:
        for df in pd.read_csv('intermediate_data_folder/sentiment_training_data.csv', chunksize=chunk_size, engine='python',sep=','):
            
            try:
                p = mp.Process(target=myAction, args=(commName, commDir, chunkIndex, df,  model, tokenizer,  aspects, post2sentiment))
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


        # convert and save the last post2sentiment dict
        post2sentiment_normalDict = defaultdict()
        for pid, d in post2sentiment.items():
            post2sentiment_normalDict[pid] = {'sentimentScores':d['sentimentScores']}
        
        os.chdir(commDir) # go back to comm directory
        with open('intermediate_data_folder/debertaV3_large_SentimentScores_of_posts_replaceTags.dict', 'wb') as outputFile:
            pickle.dump(post2sentiment_normalDict, outputFile)
            print(f"saved sentiment scores of posts till 2015 for {commName}")
            writeIntoLog(f'saved post2sentiment for comments till 2015',commDir,'debertaV3_large_SentimentScores_of_posts_replaceTags_Log.txt')

    except Exception as e:
        logfile = 'debertaV3_large_Log.txt'
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
    myFun(166,commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # test on comm "datascience.stackexchange" to debug
    # myFun(0,commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    # myFun(3,commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    # test on comm "math.stackexchange" to debug
    # myFun(358,commDir_sizes_sortedlist[358][0], commDir_sizes_sortedlist[358][1])
    # test on comm "stackoverflow" to debug
    # myFun(359,commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1])
    
    """
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist[359:]):
        commName = tup[0]
        commDir = tup[1]

        try:
            p = mp.Process(target=myFun, args=(commIndex,commName,commDir))
            p.start()
        except Exception as e:
            print(e)
            return

        processes.append(p)
        if len(processes)==60:
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
