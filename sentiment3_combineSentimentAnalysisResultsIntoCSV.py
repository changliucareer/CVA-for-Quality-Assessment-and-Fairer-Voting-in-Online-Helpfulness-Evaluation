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



def myFun(commName, commDir):
    
    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    print(f'start to combine sentiment analysis results data for {commName}')

    print("loading Questions...")
    final_directory = os.path.join(commDir, r'intermediate_data_folder')
    with open(final_directory+'/'+'QuestionsWithAnswersWithVotes.dict', 'rb') as inputFile:
        Questions = pickle.load( inputFile)

    with open(final_directory+'/'+'answer2parentQLookup.dict', 'rb') as inputFile:
        answer2parentQ = pickle.load( inputFile)
    
    with open(final_directory+'/'+'debertaV3SentimentScores_of_posts.dict', 'rb') as inputFile:
        post2sentiment_base = pickle.load( inputFile)
    
    with open(final_directory+'/'+'debertaV3_large_SentimentScores_of_posts_replaceTags.dict', 'rb') as inputFile:
        post2sentiment_large = pickle.load( inputFile)

    # make sure csv reader has big enough field size
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    # write into a CSV file
    print(f'start to write into training data for {commName}')
    csvfile = open('intermediate_data_folder/sentiment_results_data.csv', 'w', newline='')
    fieldnames = ['PostId','PostType','voteDiff','voteCount','useful-absa-base-label','useful-absa-base-score','useful-absa-large-label','useful-absa-large-score','helpful-absa-base-label','helpful-absa-base-score','helpful-absa-large-label','helpful-absa-large-score','correct-absa-base-label','correct-absa-base-score','correct-absa-large-label','correct-absa-large-score','Comment']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
            
    # extract comment text
    try:
        with open('intermediate_data_folder/sentiment_training_data.csv',) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            print(f"read sentiment_training_data.csv successfully for {commName}")
            line_count = 0
            columnNames = None
            colindex_of_score= None
            colindex_of_comment = None
            colindex_of_post = None

            for row in csv_reader:
                print(f"processing line {line_count}...")
                if line_count == 0: # header line
                    # print(f'Column names are {", ".join(row)}')
                    columnNames = row
                    colindex_of_score = columnNames.index('Score')
                    colindex_of_comment= columnNames.index('Comment')
                    colindex_of_post = columnNames.index('PostId')
                    
                else: # lines of data 
                    comment_text = row[colindex_of_comment]
                    targetPost = int(row[colindex_of_post])
                    voteDiff = int(row[colindex_of_score])
                    
                    voteCount = None
                    # get whether targetPost is a question or an answer
                    if targetPost in Questions.keys():
                        postType = 'q'
                    elif targetPost in answer2parentQ.keys():
                        postType = 'a'
                        if 'voteList' in Questions[answer2parentQ[targetPost]].keys():
                            voteCount = len(Questions[answer2parentQ[targetPost]]['voteList'])
                        else:
                            voteCount = 0
                        sentimentScores_base = post2sentiment_base[str(targetPost)]['sentimentScores']
                        aspact2sentimentScores_base = {'useful':None, 'helpful':None,'correct':None}
                        for sentDict in sentimentScores_base:
                            for aspact, value in sentDict.items():
                                label = value[0]['label']
                                score = value[0]['score']
                                aspact2sentimentScores_base[aspact]= (label,score)
                        sentimentScores_large = post2sentiment_large[str(targetPost)]['sentimentScores']
                        aspact2sentimentScores_large = {'useful':None, 'helpful':None,'correct':None}
                        for sentDict in sentimentScores_large:
                            for aspact, value in sentDict.items():
                                label = value[0]['label']
                                score = value[0]['score']
                                aspact2sentimentScores_large[aspact]= (label,score)
                    
                        # write row
                        needed = {'PostId':targetPost,'PostType':postType, 'voteDiff':voteDiff, 'voteCount':voteCount, 
                                'useful-absa-base-label':aspact2sentimentScores_base['useful'][0],'useful-absa-base-score':aspact2sentimentScores_base['useful'][1],'useful-absa-large-label':aspact2sentimentScores_large['useful'][0],'useful-absa-large-score':aspact2sentimentScores_large['useful'][1],
                                'helpful-absa-base-label':aspact2sentimentScores_base['helpful'][0],'helpful-absa-base-score':aspact2sentimentScores_base['helpful'][1],'helpful-absa-large-label':aspact2sentimentScores_large['helpful'][0],'helpful-absa-large-score':aspact2sentimentScores_large['helpful'][1],
                                'correct-absa-base-label':aspact2sentimentScores_base['correct'][0],'correct-absa-base-score':aspact2sentimentScores_base['correct'][1],'correct-absa-large-label':aspact2sentimentScores_large['correct'][0],'correct-absa-large-score':aspact2sentimentScores_large['correct'][1],
                                'Comment':comment_text}
                        writer.writerow(needed)  
                    else: # not a stored answer
                        continue         
                line_count += 1
            print(f"wrote {line_count} lines. for {commName}.")
            csvfile.close()
    
    except Exception as e:
        logfile = 'combineSentimentAnalysisResultsIntoCSV_Log.txt'
        logtext = f"fail for {commName}\n"
        logtext += str(e)+'\n'
        writeIntoLog(logtext,commDir, logfile)


def main():
    
    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    
    """
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

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

    """
   
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('combining sentiment analysis results into csv Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
