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


def myAction (question_tuple,commName):
    qid = question_tuple[0]
    content = question_tuple[1]
    print(f"processing {commName} question {qid} on {mp.current_process().name}")
    eventList = content['eventList']

    writingEventCount = 0
    votingEventCount = 0
    
    for t, event in enumerate(eventList):
        if t == 0: # the first event (must be a writing event)
            try:
                assert (event['et']=='w')
            except:
                print(AssertionError)
                return
            
            cur_z = [ event['ai'] ]
            cur_ET=[ event['et']]
            cur_J =[event['J']]
            cur_ranks_of_a_at_each_time=[event['ranks']]
            writingEventCount +=1
            continue
        
        if event['et'] == 'e': # ignore editing event
            continue

        cur_z.append(event['ai'])
        cur_ET.append(event['et'])
        cur_J.append(event['J'])

        if event['et']=='v':
            try:
                assert ( event['J'] == len(event['ranks']))
            except:
                print(AssertionError)
                return
            votingEventCount +=1
        else: # writing event
            writingEventCount +=1
        
        cur_ranks_of_a_at_each_time.append(event['ranks'])
    
    print(f"{mp.current_process().name} return")

    # return processed question's qid, content, the number of answers with votes, the number of real votes as samples
    if writingEventCount == 0 and votingEventCount == 0:
        return None
    return cur_z,cur_ET,cur_J,cur_ranks_of_a_at_each_time, qid, writingEventCount,votingEventCount
    

def myFun(commName, commDir):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # roundIndex = 1
    # variation = '_fixedTau_noRL'

    # roundIndex = 18 # multiple question multiple answer, amplified 10 times of original total event count, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    # if roundIndex in [18]:
    #     variation = '_noRL'

    # selected_reg_strengthList = [300, 500, 700]

    roundIndex = 19 ## multiple question multiple answer, original total event count, fix tau = 1, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    # roundIndex = 20 ## multiple question multiple answer, original total event count, learn tau, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    if roundIndex in [19, 20]:
        variation = ''

    selected_reg_strengthList = [500, 700]

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    for reg_alpha in selected_reg_strengthList:

        # # check whether already done this step, skip
        # resultFiles = [f'semiSynthetic_CVP{variation}_round{roundIndex}_selectionPhaseTrainingData.dict']
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
        
        with open(intermediate_directory+'/'+f'simulated_data_byCVP{variation}_round{roundIndex}_regAlpha({reg_alpha}).dict', 'rb') as inputFile:
            loadedFile = pickle.load( inputFile)
        simulatedQuestions = loadedFile[0]

        questionCount = len(simulatedQuestions)
        
        # process Questions chunk by chunk
        n_proc = mp.cpu_count()-2 # left 2 cores to do others
        all_outputs = []
        with mp.Pool(processes=n_proc) as pool:
            args = zip(list(simulatedQuestions.items()),[commName]*questionCount)
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
        qidList = []
        z = []
        ET = [] # event type
        J = [] # answer count
        ranks_of_a_at_each_time = [] # displayed ranks of each answer at each time
        dataSampleCount = 0
        writingEventCount_total = 0
        votingEventCount_total = 0

        for tup in all_outputs:
            # for combine outputs
            cur_z,cur_ET,cur_J,cur_ranks_of_a_at_each_time, cur_qid, writingEventCount,votingEventCount = tup
            z.append(cur_z)
            ET.append(cur_ET)
            J.append(cur_J)
            ranks_of_a_at_each_time.append(cur_ranks_of_a_at_each_time)
            qidList.append(cur_qid)
            dataSampleCount += len(cur_ET)
            writingEventCount_total += writingEventCount
            votingEventCount_total += votingEventCount
        
        all_outputs.clear()

        logfilename = 'semiSynthetic4_CVP_selectionStage_trainingDataGeneration_Log.txt'
        logtext = f"for {commName},\n"
        logtext = f"Round {roundIndex} Variation{variation} reg_alpha : {reg_alpha}:\n"

        logtext += f"Count of data samples: {dataSampleCount} out of total {len(qidList)} questions. writingEventRatio: {writingEventCount_total/(dataSampleCount)}\n"
        writeIntoLog(logtext, commDir , logfilename)
        print(logtext)

        if dataSampleCount==0:
            print(f"{commName} has no data sample!!!!")
            return
        
        print('Start to save the result files')
        try:
            # save all dataset
            with open(intermediate_directory+'/'+f'semiSynthetic_CVP{variation}_round{roundIndex}_regAlpha({reg_alpha})_selectionPhaseTrainingData.dict', 'wb') as outputFile:
                pickle.dump((z,ET,J,ranks_of_a_at_each_time, qidList, writingEventCount_total, votingEventCount_total), outputFile)
            print(f"saving semiSynthetic CVP{variation}_round{roundIndex}_regAlpha({reg_alpha}) selection phase training data for {commName} successfully!")
        except Exception as e:
            print(f"for {commName}, error when saving the CVP{variation}_round{roundIndex}_regAlpha({reg_alpha}) selection phase training data: {e}")
            sys.exit()


def main():

    t0=time.time()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    """
    # test on comm "3dprinting.stackexchange" to debug
    myFun(commDir_sizes_sortedlist[227][0], commDir_sizes_sortedlist[227][1])
    # test on comm "latin.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[229][0], commDir_sizes_sortedlist[229][1])
    # test on comm "lifehacks.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[233][0], commDir_sizes_sortedlist[233][1])
    # test on comm "askubuntu.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[231][0], commDir_sizes_sortedlist[231][1])
    """
    selected_comms = ['3dprinting.stackexchange','latin.stackexchange','meta.askubuntu','lifehacks.stackexchange']
    
    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        if commName not in selected_comms: # skip 
            continue
        
        try:
            p = mp.Process(target=myFun, args=(commName,commDir))
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
    print('semiSynthetic4_CVP_selectionStage_trainingDataGeneration Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
