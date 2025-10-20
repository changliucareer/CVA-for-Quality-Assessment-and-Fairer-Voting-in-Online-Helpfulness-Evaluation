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
# import SentimentIntensityAnalyzer class
# from vaderSentiment.vaderSentiment module.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

##########################################################################################
def remove_img_tags(data):
    p = re.compile(r'<img.*(/img)?>')
    return p.sub('', data)

def remove_code_tags(data):
    p = re.compile(r'<code.*(/code)?>')
    return p.sub('', data)

def cleanComment(comment_text):
    if ('<img' in comment_text) or ('<code' in comment_text):
        # remove <img> and <code> segment
        cleaned_comment_text = remove_img_tags(comment_text)
        cleaned_comment_text = remove_code_tags(cleaned_comment_text)
        return cleaned_comment_text
    else: 
        return comment_text


# Attributions of comment CSV
# 'Comments':['Id','PostId','Score','Text','CreationDate','UserId']
def myFun(commName, commDir):
    # targetCommNames = ["english.stackexchange","bicycles.stackexchange","webapps.stackexchange","mathoverflow.net"]
    # # test on target communities
    # if commName not in targetCommNames: # skip all comm except those in target
    #     return

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    post2sentiment = defaultdict()

    # load sentiment training data
    try:
        with open('intermediate_data_folder/sentiment_training_data.csv',) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            print(f"read sentiment_training_data.csv successfully for {commName}")
            line_count = 0
            columnNames = None
            colindex_of_score= None
            colindex_of_comment = None
            colindex_of_post = None

            # prepare for sentiment analysis
            # Create a SentimentIntensityAnalyzer object.
            sid_obj = SentimentIntensityAnalyzer()

            for row in csv_reader:
                if line_count == 0: # header line
                    # print(f'Column names are {", ".join(row)}')
                    columnNames = row
                    colindex_of_score = columnNames.index('Score')
                    colindex_of_comment= columnNames.index('Comment')
                    colindex_of_post = columnNames.index('PostId')
                    
                    line_count += 1
                else: # lines of data 
                    comment_text = row[colindex_of_comment]
                    cleaned_comment_text = cleanComment(comment_text)
                    targetPost = row[colindex_of_post]
                    score = row[colindex_of_score]
                    # do sentiment analysis
                    # polarity_scores method of SentimentIntensityAnalyzer
                    # object gives a sentiment dictionary.
                    # which contains pos, neg, neu, and compound scores.
                    sentiment_dict = sid_obj.polarity_scores(cleaned_comment_text)
                    sentiment_score = sentiment_dict['compound']
                    if targetPost not in post2sentiment.keys():
                        post2sentiment[targetPost]={'sentimentScores':[sentiment_score],'voteScore':score}
                    else:
                        post2sentiment[targetPost]['sentimentScores'].append(sentiment_score)
            
            # save total post counts of all communities for improving batch splition later 
            with open('intermediate_data_folder/vadarSentimentScores_of_posts.dict', 'wb') as outputFile:
                pickle.dump(post2sentiment, outputFile)
                print(f"saved sentiment scores of posts for {commName}")

    except Exception as e:
        logfile = 'extractTrainingData.py_Log.txt'
        logtext = f"fail for {commName}\n"
        logtext += str(e)+'\n'
        writeIntoLog(logtext, commDir, logfile)


def main():
    t0 = time.time()
    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")
    """
    # test on comm "coffee.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    
    """
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
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

    
    # run stackoverflow at the last separately
    # myFun('stackoverflow', stackoverflow_dir)
   
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('get sentiment score Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
