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
import psutil

def combineLengthList(ori_ll, chunk_ll):
    assert(len(ori_ll)==len(chunk_ll)) # should has the same number of answers
    for ansIndex, length_events in enumerate(chunk_ll):
        ori_ll[ansIndex].extend(length_events)
        # sort by the post history id of length updated
        ori_ll[ansIndex].sort(key= lambda x: x[0])
    return ori_ll


def myFun(commName,commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    t1 = time.time()

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data part files
    # check whether have intermediate data folder, create one if not
    # go back to current parent comm data directory
    intermediate_data_folder = os.path.join(commDir, r'intermediate_data_folder')
    os.chdir(intermediate_data_folder)

    splitted_intermediate_data_folder = os.path.join(intermediate_data_folder, r'splitted_intermediate_data_folder')
    if not os.path.exists(splitted_intermediate_data_folder):
        os.makedirs(splitted_intermediate_data_folder)
    os.chdir(splitted_intermediate_data_folder)
    final_directory = os.path.join(splitted_intermediate_data_folder, r'PostHistory_splitted_intermediate_data_folder')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)   
    os.chdir(final_directory)

    try:
        print("trying to load intermidiate data Question part files...")
        # load Questions files
        Questions_subfolders = [ f.path for f in os.scandir(final_directory) if f.path.split('/')[-1].startswith("QuestionsWithAnswersWithVotesWithPostHistory_part_") ]
        # scanned part files are not guarranteed in order, need sorting
        Questions_subfolders.sort()
        print("Questions_subfolders loaded.")

        # load phId2rowIndexOfPostHistoryCSV files
        phId2rowIndex_subfolders = [ f.path for f in os.scandir(final_directory) if f.path.split('/')[-1].startswith("phId2rowIndexOfPostHistoryCSV_part_") ]
        phId2rowIndex_subfolders.sort()
        phId2rowIndexFiles = []
        for i,f in enumerate(phId2rowIndex_subfolders):
            with open(f, 'rb') as inputFile:
                phId2rowIndexFiles.append (pickle.load( inputFile) )
                print(f"phId2rowIndex part {i+1} loaded.")
        print("phId2rowIndexFiles loaded.")

    except Exception as e:
        print(e)
        print(f"fail to load intermediate data part files from {commName}")
        return

    # combine all questions 
    Questions = None
    # keep a map of phId to rowIndex of PostHistory.csv
    phId2rowIndexOfPostHistoryCSV= defaultdict()

    print("combining questions' lengthLists and phId2rowIndexOfPostHistoryCSV")
    for i, f in enumerate(Questions_subfolders):
        print(f"loading file {f}, and combining QuestionsWithAnswersWithVotesWithPostHistory_part {i+1} ...")
        with open(f, 'rb') as inputFile:
            QuestionsWithAnswersWithVotesWithPostHistory_part = pickle.load( inputFile) 
        
        phId2rowIndexOfPostHistoryCSV_part = phId2rowIndexFiles[i]

        if Questions == None: # when Questions is still empty, just copy the first part
            Questions = QuestionsWithAnswersWithVotesWithPostHistory_part
            phId2rowIndexOfPostHistoryCSV = phId2rowIndexOfPostHistoryCSV_part

        else: 
            # combine current part lengthList to Questions
            for qid, content in QuestionsWithAnswersWithVotesWithPostHistory_part.items():
                if 'lengthList' not in content.keys():  # when there's no vote for this question in this chunk, don't need to update
                    continue
                lengthList = content['lengthList']

                if len(lengthList)==0: # when there's no vote for this question in this chunk, don't need to update
                    continue
                
                if qid in Questions.keys():
                    if 'lengthList' not in Questions[qid].keys(): # when there's no lengthList for this question, initialized it
                        Questions[qid]['lengthList'] = lengthList
                    else:
                        Questions[qid]['lengthList'] = combineLengthList(Questions[qid]['lengthList'], lengthList)
                else:
                    print("Exception: qid not in Questions.keys()")

            # combine phId2rowIndexOfPostHistoryCSV
            for id, index in phId2rowIndexOfPostHistoryCSV_part.items():
                if id not in phId2rowIndexOfPostHistoryCSV.keys(): # only update when id is not in keys()
                    phId2rowIndexOfPostHistoryCSV[id] = index
    
    
    # clear phId2rowIndexFiles to save mem
    phId2rowIndexFiles.clear()

    # save all updated Questions
    print("saving updated Questions")
    with open(intermediate_data_folder+'/'+'QuestionsWithAnswersWithVotesWithPostHistory.dict', 'wb') as outputFile:
        pickle.dump(Questions, outputFile)

    logfilename = f'preprocessing4_2_combinePostHistory_saveMem_forSplittedFiles_combineParts_Log.txt'
    logtext = f"Total Questions: {len(Questions)}\n"
    Questions.clear()

    # save phId2rowIndexOfPostHistoryCSV
    with open(intermediate_data_folder+'/'+'phId2rowIndexOfPostHistoryCSV.dict', 'wb') as outputFile:
        pickle.dump(phId2rowIndexOfPostHistoryCSV, outputFile)

    phId2rowIndexOfPostHistoryCSV.clear()
    
    # combined universal_timestemps
    try:
        print("trying to load intermidiate data universalTimeStepsPH part files...")
        # load universalTimeSteps files
        universalTimeStepsPH_subfolders = [ f.path for f in os.scandir(final_directory) if f.path.split('/')[-1].startswith("universal_timesteps_ofPH_part_") ]
        universalTimeStepsPH_subfolders.sort()
        print("all universalTimeStepsPHFiles loaded")
    except Exception as e:
        print(e)
        print(f"fail to load intermediate data universalTimeSteps part files from {commName}")
        return
    
    print("loading universal time steps of QAV...")
    with open(intermediate_data_folder+'/'+'universal_timesteps_afterCombineQAV.dict', 'rb') as inputFile:
        universal_timesteps_ofQAV = pickle.load( inputFile)

    # combine and sort universal_timesteps
    print("combining universal time steps...")
    universal_timesteps = universal_timesteps_ofQAV
    for i,f in enumerate(universalTimeStepsPH_subfolders):
        with open(f, 'rb') as inputFile:
            universal_timesteps.extend(pickle.load( inputFile) )
            print(f"universalTimeStepsPH part {i+1} loaded.")
    universal_timesteps.sort(key=lambda x: (x[2],x[1])) # sort all, first based on the datetime, then based on the id
    
    # save universal_timesteps
    print("saving updated universal_timesteps")
    with open(intermediate_data_folder+'/'+'universal_timesteps_afterCombineQAVH.dict', 'wb') as outputFile:
        pickle.dump(universal_timesteps, outputFile)
        print("universal time steps of QAVH saved.")
    logtext += f"Total event timesteps: {len(universal_timesteps)}\n"
    universal_timesteps.clear()
    
    # combine aid2phId_img and answerId2phId_code
    print("start to combine aid2phId_img and aid2phId_code ...")
    # load aid2phId_img part files
    aid2phId_img_subfolders = [ f.path for f in os.scandir(final_directory) if f.path.split('/')[-1].startswith("answerId2phId_img_part_") ]
    aid2phId_img_subfolders.sort()
    aid2phId_img = defaultdict() # keep a map of answerId to posthistoryId that contains img 
    for i,f in enumerate(aid2phId_img_subfolders):
        with open(f, 'rb') as inputFile:
            aid2phId_img_part = pickle.load( inputFile) 
            print(f"aid2phId_img part {i+1} loaded.")
            for id, index in aid2phId_img_part.items():
                if id not in aid2phId_img.keys(): # only update when id is not in keys()
                    aid2phId_img[id] = index

    # save aid2phId_img whole file
    with open(final_directory+'/'+'answerId2phId_img.dict', 'wb') as outputFile:
        pickle.dump(aid2phId_img, outputFile)
    logtext += f"Total answers with image: {len(aid2phId_img)}\n"
    aid2phId_img.clear()
        
    # load aid2phId_code part files
    aid2phId_code_subfolders = [ f.path for f in os.scandir(final_directory) if f.path.split('/')[-1].startswith("answerId2phId_code_part_") ]
    aid2phId_code_subfolders.sort()
    aid2phId_code = defaultdict() # keep a map of answerId to posthistoryId that contains code 
    for i,f in enumerate(aid2phId_code_subfolders):
        with open(f, 'rb') as inputFile:
            aid2phId_code_part = pickle.load( inputFile) 
            print(f"aid2phId_code part {i+1} loaded.")
            for id, index in aid2phId_code_part.items():
                if id not in aid2phId_code.keys(): # only update when id is not in keys()
                    aid2phId_code[id] = index

    # save aid2phId_code whole file
    with open(final_directory+'/'+'answerId2phId_code.dict', 'wb') as outputFile:
        pickle.dump(aid2phId_code, outputFile)
    logtext += f"Total answers with codes: {len(aid2phId_code)}\n"
    aid2phId_code.clear()

    elapsed = format_time(time.time() - t1)

    
    logtext +=  'Elapsed: {:}.\n'.format(elapsed)
    writeIntoLog(logtext, commDir, logfilename)

    # Report progress.
    print(f"for {commName}, ")
    print('combining post history intermediate part files Done.    Elapsed: {:}.\n'.format(elapsed))


def main():

    t0=time.time()
    curDir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])

    
    # run on only the last two communities: math.stackexchange and stackoverflow
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
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            return

        processes.append(p)
        if len(processes)==20:
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
    print('processing splitted PostHistory.csv and combine Parts Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
