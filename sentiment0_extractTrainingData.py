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

    """
    # load list of score
    try:
        pFile = 'intermediate_data_folder/statistics_posts.dict'
        with open(pFile, 'rb') as inpusFile:
            totalPostCount,maxScore, minScore, listOfScores= pickle.load(inpusFile)
            print("successfully load scores")
    except:
        print(f"no post statistics found for {commName}")
        return
    
    # convert listofScore as a dict
    post2score = dict(listOfScores)
    """
    endCommentId = 56847720 # till 2015

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
    # csvfile = open('intermediate_data_folder/sentiment_training_data.csv', 'w', newline='') # for all comments
    csvfile = open('intermediate_data_folder/sentiment_training_data_till2015.csv', 'w', newline='') # comments till 2015
    # fieldnames = ['Score','Comment','PostId']
    fieldnames = ['Comment','PostId']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
            
    # extract comment text
    try:
        with open('Comments.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            print(f"read comments.csv successfully for {commName}")
            line_count = 0
            columnNames = None
            colindex_of_Id = None
            colindex_of_Post = None
            colindex_of_Text = None

            for row in csv_reader:
                if line_count == 0: # header line
                    # print(f'Column names are {", ".join(row)}')
                    columnNames = row
                    colindex_of_Id = columnNames.index('Id')
                    colindex_of_Post = columnNames.index('PostId')
                    colindex_of_Text= columnNames.index('Text')
                    line_count += 1
                else: # lines of data 
                    comment_text = row[colindex_of_Text]
                    targetPost = row[colindex_of_Post]
                    # score = post2score[targetPost]
                    # write row
                    # needed = {'Score':score, 'Comment':comment_text, 'PostId':targetPost}
                    comment_Id = int(row[colindex_of_Id])
                    print(f"processing line {line_count} commentId {comment_Id}")
                    if comment_Id < endCommentId:
                        needed = {'Comment':comment_text, 'PostId':targetPost}
                        writer.writerow(needed)           
                        line_count += 1
                    else: # reach the end comment, stop
                        break

            csvfile.close()
    
    except Exception as e:
        logfile = 'extractTrainingData.py_Log.txt'
        logtext = f"fail for {commName}\n"
        logtext += str(e)+'\n'
        writeIntoLog(logtext, commDir,logfile)


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
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1])
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist[359:]:
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

   
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('construct sentiment training datasets Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
