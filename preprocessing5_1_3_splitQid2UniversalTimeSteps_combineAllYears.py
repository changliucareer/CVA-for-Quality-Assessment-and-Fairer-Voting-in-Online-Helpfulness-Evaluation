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
import copy

def myAction(f , commName,  commDir, year,yearly_qid2uts, qidsListOfCurrentPart,logFileName):
    # input f is the directory of current qid2uts part
    part = int(f.split('/')[-1].split('_')[-1].split('.')[0])
    print(f"combining year {year} for part {part} of {commName} on {mp.current_process().name}...")

    # load current part
    with open(f, 'rb') as inputFile:
        cur_qid2uts_part = pickle.load( inputFile)
        print(f"part {part} of {commName} loaded.")
    
    # extract subset of current year qid2uts corresponding to current part qids
    yearly_qid2uts = dict((qid, yearly_qid2uts[qid]) for qid in qidsListOfCurrentPart if qid in yearly_qid2uts.keys())
    
    for qid, cur_uts in yearly_qid2uts.items():
        if qid in cur_qid2uts_part.keys():
            cur_qid2uts_part[qid] = cur_qid2uts_part[qid] + cur_uts
        else: # a new question
            cur_qid2uts_part[qid] = cur_uts

    
    print(f"processed year {year} qid2uts for part {part}.")
    
    # save updated current part
    with open(f, 'wb') as outputFile:
        pickle.dump(cur_qid2uts_part, outputFile)
        print(f"saved part {part} qid2uts for year {year}.")
        writeIntoLog(f"saved part {part} qid2uts for year {year}.\n", commDir, logFileName)
    

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")
    logFileName = 'preprocessing5_1_3_splitQid2UniversalTimeSteps_combineAllYears_Log.txt'
    logtext = ""

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # creat a folder if not exists to store splitted data files
    splitFolder_directory = os.path.join(intermediate_directory, r'splitted_intermediate_data_folder')
    if not os.path.exists(splitFolder_directory):
        print("Exception: no splitted_intermediate_data_folder")
        writeIntoLog("Exception: no splitted_intermediate_data_folder\n", commDir, logFileName)

    
    # under splitted_intermediate_data_folder, create a folder to store yearly saved qid2universal_timesteps
    split_qid2uts_files_directory = os.path.join(splitFolder_directory, r'qid2universalTimeSteps_Yearly')
    if not os.path.exists(split_qid2uts_files_directory):
        print("Exception: no split_qid2uts_files_directory")
    else: 
        accumulated_yearlyQid2uts_subfolders = [ f.path for f in os.scandir(split_qid2uts_files_directory) if f.path.split('/')[-1].startswith("qid2universalTimeSteps_tillYear_") ]
        yearlyQid2uts_subfolders = [ f.path for f in os.scandir(split_qid2uts_files_directory) if f.path.split('/')[-1].startswith("qid2universalTimeSteps_year_") ]

        # accumulated year's qid2universal_timestemps files
        if len(accumulated_yearlyQid2uts_subfolders)>0: # has some years done
            accumulated_yearlyQid2uts_subfolders.sort()

        # single year's qid2universal_timestemps files
        if len(yearlyQid2uts_subfolders)>0: # has some years done
            yearlyQid2uts_subfolders.sort()

    """
    # load the last accumulated file
    print(f"loading the last accumulated qid2uts file...")
    with open(accumulated_yearlyQid2uts_subfolders[-1], 'rb') as inputFile:
        try:
            accumulated_qid2universal_timesteps, deletedQ,deletedA,lockedQ,lockedA,closedQ,closedA,acceptedA,answer_to_remove,answer_to_stop,question_to_remove,question_to_stop = pickle.load( inputFile)
        except Exception as e:
            print(f"fail to load pre done qid2universal_timesteps for {commName}: {e}")
            return
    """

    # load the splitted question ids
    print(f"loading partOfQuestionsWithAVH2qids.dict ...")
    part2qids_dir = intermediate_directory+'/'+'partOfQuestionsWithAVH2qids.dict'
    with open(part2qids_dir, 'rb') as inputFile:
        part2qids =  pickle.load( inputFile)

    # under splitted_intermediate_data_folder, create a folder to store splitted qid2UniversalTimeSteps files
    split_qid2utsPart_files_directory = os.path.join(splitFolder_directory, r'qid2UniversalTimeSteps_Parts')
    if not os.path.exists(split_qid2utsPart_files_directory):
        print("no qid2UniversalTimeSteps_Parts folder, create one")
        os.makedirs(split_qid2utsPart_files_directory)
    
    """
    # split the last accumultaed file
    for part, qidList in part2qids.items():
        # print(f"extracting part {part} from accumulated_qid2universal_timesteps...")
        # cur_qid2uts_part = {qid: value for qid, value in accumulated_qid2universal_timesteps.items() if qid in qidList} # run very slow
        cur_qid2uts_part = dict((qid, accumulated_qid2universal_timesteps[qid]) for qid in qidList if qid in accumulated_qid2universal_timesteps.keys())
        if len(cur_qid2uts_part)==0: # extract None, to make sure save as a dictionary
            cur_qid2uts_part = defaultdict()
        # save the current part
        out_file_dir = split_qid2utsPart_files_directory+'/'+'qid2UniversalTimeSteps_part_'+ str(part) +'.dict'
        with open(out_file_dir, 'wb') as outputFile:
            pickle.dump(cur_qid2uts_part, outputFile)
            logtext += f"saved part {part} qid2uts, length: {len(cur_qid2uts_part)}\n"
            print(f"saved part {part} qid2uts, length: {len(cur_qid2uts_part)}")
    
    writeIntoLog(logtext, commDir, logFileName)

    # clear accumulated file to save memory
    accumulated_qid2universal_timesteps.clear()
    """

    # combine the single yearly qid2uts files part by part
    print(f"start to combine single yearly qid2uts to part files...")
    qid2uts_part_subfolders = [ f.path for f in os.scandir(split_qid2utsPart_files_directory) if f.path.split('/')[-1].startswith("qid2UniversalTimeSteps_part_") ]
    # sort by part
    qid2uts_part_subfolders.sort(key=lambda f: int(f.split('/')[-1].split('_')[-1].split('.')[0]))

    # inorder to reduce the loading, process year by year
    # scan all single year files
    for sf in  yearlyQid2uts_subfolders:
        year = int(sf.split('/')[-1].split('_')[-1].split('.')[0])
        # load current year file
        try:
            with open(sf, 'rb') as inputFile:
                fileTuple = pickle.load( inputFile)
                yearly_qid2uts = fileTuple[0]
        except Exception as e:
            print(e)

        # process parts parallelly
        n_proc = mp.cpu_count()
        finishedCount = 0
        processes = []
        for i,f in enumerate(qid2uts_part_subfolders):
            # if i > 0 : # for debugging
            #     break
            try:
                p = mp.Process(target=myAction, args=(f, commName,commDir, year,yearly_qid2uts, list(part2qids.values())[i], logFileName))
                p.start()
            except Exception as e:
                print(e)
                pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
                print(f"current python3 processes count {pscount}.")
                return

            processes.append(p)
            if len(processes)==n_proc:
            # if len(processes)==1: # uncomment this line just for debugging
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
        
    
def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # # save the all comm sorted list into a csv file
    # import csv
    # with open('allComm_directories_sizes_sortedlist.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow( ["commName","totalPostCount"])
    #     for tup in commDir_sizes_sortedlist:
    #         writer.writerow([tup[0],tup[2]])

    # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1])
    # test on comm "datascience.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[301][0], commDir_sizes_sortedlist[301][1])
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1]) 
    # test on comm "stackoverflow" to debug
    myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1]) 
    

    # # run on all communities other than stackoverflow
    # finishedCount = 0
    # processes = []
    # for tup in commDir_sizes_sortedlist:
    #     commName = tup[0]
    #     commDir = tup[1]
    #     if commName == 'stackoverflow': # skip stackoverflow to run at the last
    #         stackoverflow_dir = commDir
    #         continue
    #     try:
    #         p = mp.Process(target=myFun, args=(commName,commDir))
    #         p.start()
    #     except Exception as e:
    #         print(e)
    #         pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
    #         print(f"current python3 processes count {pscount}.")
    #         return

    #     processes.append(p)
    #     if len(processes)==20:
    #         # make sure all p finish before main process finish
    #         for p in processes:
    #             p.join()
    #             finishedCount +=1
    #             print(f"finished {finishedCount} comm.")
    #         # clear processes
    #         processes = []
    
    # # join the last batch of processes
    # if len(processes)>0:
    #     # make sure all p finish before main process finish
    #     for p in processes:
    #         p.join()
    #         finishedCount +=1
    #         print(f"finished {finishedCount} comm.")


    # # run stackoverflow at the last separately
    # myFun('stackoverflow', stackoverflow_dir)

   
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('splitQid2UniversalTimeSteps_combineAllYears Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
