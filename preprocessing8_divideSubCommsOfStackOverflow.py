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
import copy

def find_between(s, start,end):
    subs = s.split(start)[1:]
    return [ss.split(end)[0] for ss in subs]

def myAction (question_tuple, part, target_tag):
    target_tag = target_tag.lower()
    qid = question_tuple[0]
    content = question_tuple[1]
    print(f"processing question {qid} of {part} for tag {target_tag} on {mp.current_process().name}...")
    if 'tagList' not in content.keys():
        return None
    
    curTags = [t.lower() for t in content['tagList']]
    
    curSubTags = []
    for tag in curTags:
        subTags = tag.split('-')
        curSubTags.extend(subTags)

    if (target_tag in curTags) or (target_tag in curSubTags):
        return question_tuple
    else:
        return None

def my_divide(parentCommName, parentCommDir, partFiles, subCommName, subCommDir, logFileName):
    selected_question_tuples = []
    for i, subDir in enumerate(partFiles):
        part = i+1
        partDir = subDir
        # get question count of each part
        with open(partDir, 'rb') as inputFile:
            Questions_part = pickle.load( inputFile)
            print(f"part {part} of {parentCommName} is loaded.")
        
        # process Questions chunk by chunk
        n_proc = mp.cpu_count()-2 # left 2 cores to do others
        with mp.Pool(processes=n_proc) as pool:
            args = zip(list(Questions_part.items()), len(Questions_part)*[part], len(Questions_part)*[subCommName])
            # issue tasks to the process pool and wait for tasks to complete
            results = pool.starmap(myAction, args , chunksize=n_proc)
            # process pool is closed automatically
            for res in results:
                if res != None:
                    selected_question_tuples.append(res)
        
        logtext = f"part {part} of stackoverflow result added to selected_question_tuples, current total length {len(selected_question_tuples)}.\n"
        writeIntoLog(logtext, parentCommDir, logFileName)
        print(logtext)
        Questions_part.clear()
        results.clear()

    # combine results
    print(f"start to combine selected_question_tuples for target tag {subCommName}...")
    sub_Questions = dict(selected_question_tuples)
    selected_question_tuples.clear()
            
    # save sub comm for target rag Questions
    with open(subCommDir+'/'+f'QuestionsWithEventList_tag_{subCommName}.dict', 'wb') as outputFile:
        pickle.dump(sub_Questions, outputFile) 
        logtext =f"saved QuestionsWithEventList_tag_{subCommName}.dict\n"
        writeIntoLog(logtext, parentCommDir, logFileName)
        print(logtext)

def myFun_SOF(parentCommName, parentCommDir):
    print(f"comm {parentCommName} running on {mp.current_process().name}")
    logFileName = 'preprocessing8_divideSubCommOfStackOverflow_Log.txt'

    # go to current comm data directory
    os.chdir(parentCommDir)
    print(os.getcwd())

    selectedTags = ['c','c#','c++','reactjs',
                    'java','python','r','node.js','php',
                    'javascript','css','html','sql','pandas',
                    'android','Rust', 'Elixir', 'Clojure', 'Typescript','Julia',
                    'ruby','perl','xml','ios','tensorflow','pytorch']
    newly_addedTags = ['ruby','perl','xml','ios','tensorflow','pytorch']
    
    # create a sub comm name and dir dict
    # under SOF folder, create a folder to store data of sub communities 
    subComms_data_folder = os.path.join(parentCommDir, f'subCommunities_folder')
    if not os.path.exists( subComms_data_folder):
        print("no subComms_data_folder, create one")
        os.makedirs(subComms_data_folder)
    subCommName2commDir = defaultdict()
    for subCommName in selectedTags:
        # under SOF folder, create a folder to store data of sub communities 
        cur_subComm_folder = os.path.join(subComms_data_folder, f'subComm_{subCommName}_folder')
        if not os.path.exists(cur_subComm_folder):
            print("no cur_subComm_folder, create one")
            os.makedirs(cur_subComm_folder)
        subCommName2commDir[subCommName] = cur_subComm_folder
    
    # save subCommName2commDir
    with open(subComms_data_folder+'/'+f'subCommName2subCommDir.dict', 'wb') as outputFile:
        pickle.dump(subCommName2commDir, outputFile) 
        print("subCommName2commDir saved.")

    # go to the target splitted files folder
    intermediate_directory = os.path.join(parentCommDir, r'intermediate_data_folder')

    splitted_intermediate_data_folder = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    split_QuestionsWithEventList_files_directory = os.path.join(splitted_intermediate_data_folder, r'QuestionsPartsWithEventList')
    if not os.path.exists(split_QuestionsWithEventList_files_directory): # didn't find the parts files
        print("Exception: no split_QuestionsWithEventList_files_directory!")

    partFiles = [ f.path for f in os.scandir(split_QuestionsWithEventList_files_directory) if f.path.endswith('.dict') ]
    # sort csvFiles paths based on part number
    partFiles.sort(key=lambda p: int(p.strip(".dict").split("_")[-1]))
    partsCount = len(partFiles)

    assert partsCount == int(partFiles[-1].strip(".dict").split("_")[-1]) # last part file's part number should equal to the parts count
                                
    print(f"there are {partsCount} splitted event list files in {parentCommName}")
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for subCommName, subCommDir in subCommName2commDir.items():
        if subCommName not in newly_addedTags: # skip 
            continue
        try:
            # p = mp.Process(target=myFun, args=(commIndex, commName,commDir, return_trainSuccess_dict, root_dir))
            p = mp.Process(target=my_divide, args=(parentCommName, parentCommDir, partFiles, subCommName, subCommDir, logFileName))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()
            return

        processes.append(p)
        if len(processes)==6:
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
            print(f"finished {finishedCount} sub comm.")


def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # test on comm "stackoverflow" to debug
    myFun_SOF(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1])
            
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('preproceesing 8 divide sub communities of SOF Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
