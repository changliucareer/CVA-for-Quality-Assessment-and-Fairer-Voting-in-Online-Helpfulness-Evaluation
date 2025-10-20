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
from statistics import mean
from scipy.optimize import fsolve
from scipy.special import psi # digamma

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


# define f
def f(r,tau):
    if r != None:
        return (1/(1+r)) ** tau
    else:
        return 0

def sgd( args ):
    commName, n_iter, thetaList, lr, start_tau, tolerance, z, ET, J, ranks_of_a_at_each_time,commDir,log_filename  = args 

    tau_gradhist = 0.

    # these are for recording!
    tau_record = [start_tau]
    tau = start_tau
    convergeFlag = False

    # initialize log-likelihood
    ll_record = []
    questionCount = len(ET)

    # check 
    # eventLengthOfQuestions = [len(el) for el in ET]
    # print(Counter(eventLengthOfQuestions))

    batchSize = 2000000
    batchedGradients = []
    
    for iter in range(n_iter + 1):
        print(f"training iteration: {iter+1} for {commName}")
        for qi, et_traj in enumerate(ET):
            grad_tau = 0.
            # Gradient for tau  
            for t in range(len(z[qi])):
                theta = thetaList[qi]
                existingEventCountOfTargetQuestion = t
                alpha = theta / (existingEventCountOfTargetQuestion + theta)

                exceptionFlag = False
                if t == 0 or ranks_of_a_at_each_time[qi][t]==None:
                    sum_J_f = 0.
                    sum_J_flog = 0.
                    exceptionFlag = True
                else:
                    sum_J_f = sum([f(r,tau) for r in ranks_of_a_at_each_time[qi][t]])
                    sum_J_flog = sum([f(r,tau)*math.log(1/(1+r)) for r in ranks_of_a_at_each_time[qi][t]])
                
                second_term = - sum_J_flog / (alpha + sum_J_f)
                
                if et_traj[t]=='v':
                    z_it = z[qi][t]
                    z_it_rank = ranks_of_a_at_each_time[qi][t][z_it]
                    first_term =  math.log(1/(1+z_it_rank))
                else:
                    first_term = 0.
                
                 
                grad_tau = first_term + second_term
                
                if grad_tau != 0:
                    batchedGradients.append(grad_tau)

                # check whether batchGradients is full to be ready to update tau
                if len(batchedGradients) == batchSize:
                    avgGrad_tau = mean(batchedGradients)
                    diff = lr * avgGrad_tau
                    if not exceptionFlag:
                        if abs(diff) <= tolerance:
                            convergeFlag = True
                            if iter >=1 : # OK to stop training when more than one iteration after converge
                                break
                    tau += diff
                    print(f"grad_tau :{grad_tau}\t updated_tau:{tau}, for {commName}")
                    # clear batchGradients
                    batchedGradients = []

        if len(batchedGradients) > 0: # update at least once for one iter
            avgGrad_tau = mean(batchedGradients)
            diff = lr * avgGrad_tau
            if not exceptionFlag:
                if abs(diff) <= tolerance:
                    convergeFlag = True
                    if iter >=1 : # OK to stop training when more than one iteration after converge
                        break
            tau += diff
            print(f"grad_tau :{grad_tau}\t updated_tau:{tau}, for {commName}")
            # clear batchGradients
            batchedGradients = []

        ll = log_likelihood(alpha, tau, z, ET, J, ranks_of_a_at_each_time)
        tau_record.append(tau)
        ll_record.append(ll)
        # total_ll = sum(ll_record)
        # # shrink learning rate every 100 epoch
        # if (iter+1)%500 ==0:
        #     lr = lr * 0.1 
        # print out every iter
        print(f"after iter:{iter+1}, updated tau:{tau}, log_likelihood:{ll} for {commName}")
    # write in log every epoch
    writeIntoLog(f"convergeFlag: {convergeFlag} iter:{iter+1}\n, learned tau:{tau}, ll:{ll} for {commName}\n",commDir, log_filename)

    if convergeFlag:
        convergeIter = iter +1
    else:
        convergeIter = None

    maxLL = max(ll_record)
    index_maxLL = ll_record.index(maxLL)
    tau_maxLL = tau_record[index_maxLL]
          
    return tau_maxLL, tau_record, ll_record, convergeFlag, convergeIter



def log_likelihood(alpha, tau, z, ET, J, ranks_of_a_at_each_time):

    if alpha < 0.:
        return float('nan')

    total_ll = 0.

    for qi, et_traj in enumerate(ET): 
        ll = 0.

        for t in range(len(z[qi])):
            z_it = z[qi][t]
            et_it = et_traj[t]
            sum_J_f =0 # clear sum_J_f

            if t ==0 or ranks_of_a_at_each_time[qi][t]==None:
                sum_J_f =0
            else:
                try:
                    sum_J_f = sum([f(r,tau) for r in ranks_of_a_at_each_time[qi][t]])
                except:
                    print("Exception!")
            
            ll += - math.log(alpha + sum_J_f)

            if et_it == 'w': # writing
                ll += math.log(alpha)

            elif et_it == 'v': # voting
                try:
                    ll += math.log(f(ranks_of_a_at_each_time[qi][t][z_it],tau))
                except:
                    print("why?") # IndexError('list index out of range')
            else:
                # print(f"current event is not writing nor voting, but {et_it}, skip")
                continue           

        total_ll += ll

    return total_ll
    

def myFun(commName, commDir, root_dir, roundIndex, variation, selected_reg_strengthList):
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')


    # train using the data and priors from selected reg_alpha options
    for reg_alpha in selected_reg_strengthList:

        # # check whether already done this step, skip
        # result_directory = os.path.join(commDir, r'result_folder')
        # resultFiles = ['semiSynthetic_CVP1_selectionPhaseTrainingResults.dict']
        # resultFiles = [result_directory+'/'+f for f in resultFiles]
        # if os.path.exists(resultFiles[0]):
        #     # target date
        #     target_date = datetime.datetime(2023, 8, 23)
        #     # file last modification time
        #     timestamp = os.path.getmtime(resultFiles[0])
        #     # convert timestamp into DateTime object
        #     datestamp = datetime.datetime.fromtimestamp(timestamp)
        #     print(f'{commName} Modified Date/Time:{datestamp}')
        #     if datestamp >= target_date:
        #         print(f"{commName} has already done this step.")
        #         return
        
        with open(intermediate_directory+'/'+f'semiSynthetic_CVP{variation}_round{roundIndex}_regAlpha({reg_alpha})_selectionPhaseTrainingData.dict', 'rb') as inputFile:
            loadedFiles = pickle.load( inputFile)

        z,ET,J,ranks_of_a_at_each_time, qidList, writingEventCount_total, votingEventCount_total = loadedFiles
        
        
        #  # save csv for CVP selection phase training data statistics 
        # with open(rootDir+'/'+'allComm_CVP_selectionPhaseTrainingData_statistics.csv', 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',',
        #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #     writer.writerow( [commName,writingEventCount_total+votingEventCount_total, writingEventCount_total,votingEventCount_total, writingEventCount_total/(writingEventCount_total+votingEventCount_total)])
        

        ### training selection phase
        # set arguments
        n_iter = 2000
        lr = 0.1 # initialized learning rate
        start_tau = 1 # initialized tau
        tolerance = 1e-5  # the default of sklearn LR model is 1e-4
        log_filename = 'semiSynthetic5_CVP1_selectionPhase_training_Log.txt'

        # question-level alpha == theta in CRP wiki 
        # get the number of events and the number of answers for each question
        qid2EventCountAndAnswerCount = defaultdict()
        for i, qid in enumerate(qidList):
            cur_ET = ET[i] # event type list of current question qid
            cur_J = J[i] # J list of current question qid, J is # the number of answers till a time, starting from 0 (J should be J_i^{t-1})
            
            if cur_ET[-1] == 'w':
                answerCount = cur_J[-1] + 1
            else:
                answerCount = cur_J[-1]
            
            eventCount = len(cur_ET)

            qid2EventCountAndAnswerCount[qid] = (answerCount, eventCount)
        
        # solve the theta for each question
        qid2theta = defaultdict()
        for qid, tup in qid2EventCountAndAnswerCount.items():
            answerCount, eventCount = tup
            try:
                theta = mySolver (answerCount, eventCount )
            except Exception as e:
                print(e)

            qid2theta[qid] = theta

        thetaList = [qid2theta[qid] for qid in qidList]


        arg = (commName, n_iter, thetaList, lr, start_tau, tolerance, z, ET, J, ranks_of_a_at_each_time ,commDir,log_filename)

        writeIntoLog(f"Start training...\n, n_iter:{n_iter}, lr:{lr}, start_tau:{start_tau}, tolerance:{tolerance}\n",commDir, log_filename)

        learned_tau, tau_record, ll_record, convergeFlag, convergeIter = sgd(arg)
        
        writeIntoLog(f"learned tau: {learned_tau}, convergeFlag:{convergeFlag}, convergeIter:{convergeIter}, maxLL:{max(ll_record)}\n",commDir, log_filename)


        print(f"for {commName}, learned tau: {learned_tau}, convergeFlag:{convergeFlag}, convergeIter:{convergeIter}")

        # save csv for CVP selection phase training results 
        with open(root_dir+'/'+f'allComm_semiSynthetic5_CVP{variation}_round{roundIndex}_selectionPhaseTrainingResults.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow( [commName,writingEventCount_total+votingEventCount_total, reg_alpha, learned_tau, convergeFlag, convergeIter])
        

        ### save new data
        print("Start to save....")
        result_directory = os.path.join(commDir, r'result_folder')
        new_fname = f'semiSynthetic_CVP{variation}_round{roundIndex}_regAlpha({reg_alpha})_selectionPhaseTrainingResults.dict'  # for sanityCheck round
        with open(result_directory+'/'+new_fname, 'wb') as outputFile:
            pickle.dump((learned_tau, tau_record, ll_record, convergeFlag, convergeIter), outputFile)
        print(f"{new_fname} Saved as dict.")

    
    

def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("other sorted CommDir loaded.")

    # # save csv for CVP selection phase training data statistics
    # with open(rootDir+'/'+'allComm_CVP_selectionPhaseTrainingData_statistics.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',
    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow( ["commName","total event count", "writing event count","voting event count", "alpha (writing event proportion)"])

    # roundIndex = 18 # multiple question multiple answer, amplified 1000 times of original total event count, q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [300, 500, 700]
    # if roundIndex in [18]:
    #     variation = '_noRL'
    
    # selected_reg_strengthList = [300, 500, 700]

    # roundIndex = 19 ## multiple question multiple answer, original total event count, fix tau = 1, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    roundIndex = 20 ## multiple question multiple answer, original total event count, learn tau, with RL,  q_std = 1, use lambda (for different regularization strength) selected_reg_strengthList = [500, 700]
    if roundIndex in [19, 20]:
        variation = ''

    selected_reg_strengthList = [500, 700]

    # try:
    #     # test on comm "3dprinting.stackexchange" to debug
    #     myFun(0,commDir_sizes_sortedlist[227][0], commDir_sizes_sortedlist[227][1], root_dir, roundIndex, variation, selected_reg_strengthList)
    #     # test on comm "latin.stackexchange" to debug
    #     myFun(1,commDir_sizes_sortedlist[229][0], commDir_sizes_sortedlist[229][1], root_dir, roundIndex, variation, selected_reg_strengthList)
    #     # test on comm "lifehacks.stackexchange" to debug
    #     myFun(2,commDir_sizes_sortedlist[233][0], commDir_sizes_sortedlist[233][1], root_dir, roundIndex, variation, selected_reg_strengthList)
    #     # test on comm "askubuntu.stackexchange" to debug
    #     myFun(3,commDir_sizes_sortedlist[231][0], commDir_sizes_sortedlist[231][1], root_dir, roundIndex, variation, selected_reg_strengthList)
    # except Exception as e:
    #     print(e)
    #     sys.exit()

    selected_comms = ['3dprinting.stackexchange','latin.stackexchange','meta.askubuntu','lifehacks.stackexchange']

    # save csv for CVP selection phase training results
    with open(root_dir+'/'+f'allComm_semiSynthetic5_CVP{variation}_round{roundIndex}_selectionPhaseTrainingResults.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","total event count", "reg_alpha", "learned_tau", "convergeFlag", "convergeIter"])

    
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        if commName not in selected_comms: # skip 
            continue

        try:
            p = mp.Process(target=myFun, args=(commName,commDir, root_dir, roundIndex, variation, selected_reg_strengthList))
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
    print('semiSythetic 5 CVP1_selection stage training Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
