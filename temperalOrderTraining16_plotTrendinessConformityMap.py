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
from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import torch
from tqdm import tqdm
from statistics import mean, median
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
from adjustText import adjust_text
import csv
import matplotlib as mpl

typical_comms = ['cstheory.stackexchange','unix.meta.stackexchange','stackoverflow','politics.stackexchange',
                 'math.meta.stackexchange','mathoverflow.net','askubuntu','philosophy.stackexchange']

try_reg_strengthList = [0.1,0.2, 0.3,0.4, 0.5,0.6, 0.7,0.8,0.9,
                            1, 2, 3,4,5, 6, 7,8,9,
                            10,20, 30,40,50,60, 70,80,90,
                            100, 200, 300, 400, 500, 600, 700, 800, 900,
                            1000]

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def plotMap(return_trainResult_dict,root_dir, sampled_comms, modelName):
    ylim_max = 60

    combinedList = []  # a list of tuple (commName, sampleCount, trendiness, conformity, trainMode)

    for commName, reg_alphaAndResultList in return_trainResult_dict.items():
        
        # choose from those with the best testAccuracy, use the middle
        sorted_by_testAccuracy_descending = sorted(reg_alphaAndResultList, key=lambda item: item[5], reverse=True)
        sklearnOnly = [tup for tup in sorted_by_testAccuracy_descending if tup[3]=='sklearn']
        if (len(sklearnOnly)>0):
            sorted_by_testAccuracy_descending = sklearnOnly
        else: # skip the comm has no sklearn results
            continue

        maxTestAccuracy = sorted_by_testAccuracy_descending[0][5]
        pool = []
        for tup in sorted_by_testAccuracy_descending:
            curTestAccuracy = tup[5]
            if curTestAccuracy == maxTestAccuracy:
                pool.append(tup)
        chosenIndexOfPool = 0
        if commName == 'mathoverflow.net':
            chosenIndexOfPool = 3
        sampleCount = pool[chosenIndexOfPool][4]
        trendiness = pool[chosenIndexOfPool][1]
        conformity = pool[chosenIndexOfPool][2]
        trainMode = pool[chosenIndexOfPool][3]
        combinedList.append((commName, sampleCount, trendiness, conformity, trainMode))
    
    combinedList.sort(key=lambda item:item[1], reverse=True) # sort by sampleCount

    selectCommCount = 120
    topSelectedCommNames = [tup[0] for tup in combinedList[:selectCommCount]]

    # save topSelectedCommNames
    with open(f'topSelectedCommNames_{modelName}_fixedTau.dict', 'wb') as outputFile:
        pickle.dump(topSelectedCommNames, outputFile)

    #only plot top selected communities
    topSelected_trendiness=[]
    topSelected_conformity=[]
    topSelected_sizes = []
    topSelected_cn =[]
    topSelected_tm = []
    sgdCount = 0
    topSelected_trendiness_sklearnOnly=[]
    topSelected_conformity_sklearnOnly=[]
    topSelected_linewidth = []
    for i, commName in enumerate(topSelectedCommNames):
        if commName in typical_comms:
            linewidth = 8
        else:
            linewidth = 1
        topSelected_linewidth.append(linewidth)

        cn = commName.replace('.stackexchange','')
        if cn =='meta':
            cn = 'meta.stackexchange'
        elif cn =='stackoverflow':
            cn = 'reactjs(SOF)'
        # elif cn in sampled_comms:
        #     cn = 'sampled_'+cn

        topSelected_cn.append(cn)
        topSelected_trendiness.append(combinedList[i][2])
        if combinedList[i][3] > ylim_max:
            topSelected_conformity.append(ylim_max)
        else:
            topSelected_conformity.append(combinedList[i][3])
        
        topSelected_sizes.append(combinedList[i][1])
        topSelected_tm.append(combinedList[i][4])
        if combinedList[i][4] == 'sgd':
            sgdCount +=1
        else:
            topSelected_trendiness_sklearnOnly.append(combinedList[i][2])
            topSelected_conformity_sklearnOnly.append(combinedList[i][3])

    
    # medianTrendiness = median(sorted(topSelected_trendiness_sklearnOnly))
    medianTrendiness = 2.52
    # medianConformity = median(sorted(topSelected_conformity_sklearnOnly))
    medianConformity = 25
    # medianTrendiness = median([t[2] for t in combinedList[:180]])
    # medianConformity = median([t[3] for t in combinedList[:180]])

    minSize = sorted(topSelected_sizes[:120])[0]
    # minSize = sorted(topSelected_sizes)[len(topSelected_sizes)-120]
    # norm_sizes =  [(float(i)/minSize)*15 for i in topSelected_sizes]
    norm_sizes =  [np.log2(i/minSize +1)**2*100 for i in topSelected_sizes]
    print(f"Filtered commu count {len(topSelected_trendiness)}")   


    minColor = 'green'
    maxTrendinessColor = 'red'
    maxConformityColor = 'blue'   

    facecolors = []
    edgecolors = []
    for i in range(len(topSelected_trendiness)):
        mixTrendiness = (topSelected_trendiness[i]-min(topSelected_trendiness))/(4-min(topSelected_trendiness))
        if mixTrendiness >1:
            mixTrendiness = 1
        mixConformity = (topSelected_conformity[i]-min(topSelected_conformity))/(60-min(topSelected_conformity))
        if mixConformity >1:
            mixConformity = 1
        curTrendinessColor = colorFader(minColor,maxTrendinessColor, mixTrendiness)
        curConformityColor = colorFader(minColor,maxConformityColor, mixConformity)
        curColor = colorFader(curTrendinessColor,curConformityColor,0.5)
        facecolors.append(curColor)

        if topSelected_trendiness[i] >= medianTrendiness and topSelected_conformity[i] >= medianConformity:
            curColor = 'purple'
        elif topSelected_trendiness[i] >= medianTrendiness and topSelected_conformity[i] < medianConformity:
            curColor = 'red'
        elif topSelected_trendiness[i] < medianTrendiness and topSelected_conformity[i] >= medianConformity:
            curColor = 'blue'
        else:
            curColor = 'green'
        edgecolors.append(curColor)
        
    plt.figure(figsize=(50,30))

    plt.scatter(topSelected_trendiness, topSelected_conformity, s=norm_sizes, edgecolor=edgecolors, linewidth=topSelected_linewidth ,facecolor=facecolors, cmap='viridis', alpha=0.5)
    plt.xlabel('Sensitivity to Position Bias',fontsize=40)
    plt.ylabel('Degree of Herding Bias',fontsize=40)

    secondMaxOfTrendiness = sorted(topSelected_trendiness,reverse=True)[1]
    secondMaxOfConformity = sorted(topSelected_conformity,reverse=True)[1]
    plt.xlim(xmin=min(topSelected_trendiness),xmax=secondMaxOfTrendiness+0.1)
    # plt.ylim(ymin=min(topSelected_conformity)-0.1,ymax=secondMaxOfConformity+0.1)
    plt.ylim(ymin=min(topSelected_conformity)-0.1,ymax=ylim_max)

    plt.text(medianTrendiness,medianConformity-1,'MEDIAN',fontsize=25, color='black')
    plt.axvline(x=medianTrendiness,color='g', lw=0.8, ls='--')
    plt.axhline(y=medianConformity,color='g', lw=0.8, ls='--')

    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    
    
    texts = [plt.text(topSelected_trendiness[i],topSelected_conformity[i],topSelected_cn[i],fontsize=25) for i in range(len(topSelected_trendiness))]
    adjust_text(texts) 

    # go back to root dir
    os.chdir(root_dir)
    savePlot(plt, f"{modelName}_fixedTau_betaAsTrendiness_map.pdf")
    print(f"saved {modelName}_fixedTau_betaAsTrendiness_map.pdf, sgdCount:{sgdCount}")

    # # save csv
    with open(root_dir +'/'+f'allComm_{modelName}_fixedTau_TrendinessAndConformity.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for tup in combinedList:
            commName, sampleCount, trendiness, conformity, trainMode = tup
            selectedToTop = None
            if commName in topSelectedCommNames:
                selectedToTop = True
            else:
                selectedToTop = False
            
            quadrant = None
            if trendiness >= medianTrendiness and conformity >= medianConformity:
                quadrant = 'experience'
            elif trendiness >= medianTrendiness and conformity < medianConformity:
                quadrant = 'opinion'
            elif trendiness < medianTrendiness and conformity >= medianConformity:
                quadrant = 'knowledge'
            else:
                quadrant = 'belief'

            if commName == 'stackoverflow':
                commName = 'reactjs_SOF'
            else:
                if commName in sampled_comms:
                    commName = 'sampled_' + commName.replace('.stackexchange','')
                else:
                    commName = commName.replace('.stackexchange','')

            writer.writerow( [commName, sampleCount, trendiness, conformity, trainMode, selectedToTop, quadrant])
   
def myFun(commIndex, commName, commDir, return_trainResult_dict, sampled_comms, modelName, commName2selected_reg_strengthList):
    t0=time.time()
    print(f"comm {commName} running on {mp.current_process().name}")

    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')
    
    reg_alphaAndResultList = [] # a list of tuple (reg_alpha, beta, conformity, trainMode, sampleCount, testAccuracy)
    for reg_alpha in try_reg_strengthList:
        if modelName == 'newModel':
            resultFiles = [intermediate_directory+f"/temperalOrderTraining12_newModel_fixedTau_regAlpha({reg_alpha})_return.dict",intermediate_directory+f"/temperalOrderTraining12_newModel_fixedTau_regAlpha({reg_alpha})_forSampledQuestion_return.dict"]
        elif modelName == 'newModel_interaction':
            resultFiles = [intermediate_directory+f"/temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha})_return.dict",intermediate_directory+f"/temperalOrderTraining13_newModel_interaction_fixedTau_regAlpha({reg_alpha})_forSampledQuestion_return.dict"]
        
        if (commName not in sampled_comms) and os.path.exists(resultFiles[0]):
            # target date
            target_date = datetime.datetime(2024, 10, 1)
            # file last modification time
            timestamp = os.path.getmtime(resultFiles[0])
            # convert timestamp into DateTime object
            datestamp = datetime.datetime.fromtimestamp(timestamp)
            print(f'{commName} Modified Date/Time:{datestamp}')
            if datestamp >= target_date: # the final result file exists
                # print(f"{commName} has already done this step for reg_alpha({reg_alpha}).")
                # load result file
                with open(resultFiles[0], 'rb') as inputFile:
                    return_trainSuccess_dict = pickle.load( inputFile)
        elif (commName in sampled_comms): # try to load sampledQuestion result
            try:
                with open(resultFiles[1], 'rb') as inputFile:
                    return_trainSuccess_dict = pickle.load( inputFile)
            except:
                with open(resultFiles[0], 'rb') as inputFile:
                    return_trainSuccess_dict = pickle.load( inputFile)
        else:
            print(f"{commName} has no trainning result for reg_alpha {reg_alpha}.")
            continue

        if len(return_trainSuccess_dict)==1:
            simplifiedCommName = list(return_trainSuccess_dict.keys())[0]
            curCommResult = return_trainSuccess_dict[simplifiedCommName]

        cur_beta = None
        cur_conformity = None
        cur_trainMode = None
        cur_testAccuracy = None

        if 'coefs_sklearn' in curCommResult.keys():
            if curCommResult['coefs_sklearn'] != None: # has sklearn results
                cur_beta = curCommResult['coefs_sklearn'][1]
                cur_conformity = curCommResult['conformity_sklearn']
                cur_trainMode = 'sklearn'
                cur_testAccuracy = curCommResult['CV_scores'][try_reg_strengthList.index(reg_alpha)]

        # if cur_beta == None and ('coefs' in curCommResult.keys()):
        #     if curCommResult['coefs'] != None: # has sgd results
        #         cur_beta = curCommResult['coefs'][1]
        #         cur_conformity = curCommResult['conformity']
        #         cur_trainMode = 'sgd'
        #         cur_testAccuracy = curCommResult['testAcc']
        if cur_beta == None:
            print(f"{commName} has no training results for reg_alpha({reg_alpha})")
            continue
        
        cur_sampleCount = curCommResult['dataShape'][0]
        
        if cur_testAccuracy == None:
            continue
        if cur_conformity == None or math.isinf(cur_conformity):
            continue

        reg_alphaAndResultList.append((reg_alpha,cur_beta,cur_conformity,cur_trainMode,cur_sampleCount,cur_testAccuracy))
        
    if len(reg_alphaAndResultList)>0:
        return_trainResult_dict[commName] = reg_alphaAndResultList
        print(f"saved reg_alphaAndResultList for {commName}")



def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    modelName = 'newModel'
    # modelName = 'newModel_interaction'
    

    # for sampled comms
    sampled_comms = ['academia.stackexchange','askubuntu',
                      'english.stackexchange','math.stackexchange','mathoverflow.net',
                      'meta.stackexchange','meta.stackoverflow','serverfault',
                      'softwareengineering.stackexchange','superuser','unix.stackexchange',
                      'worldbuilding.stackexchange','physics.stackexchange','electronics.stackexchange',
                      'codegolf.stackexchange','workplace.stackexchange']
    
    
    # # save csv
    with open(root_dir +'/'+f'allComm_{modelName}_fixedTau_TrendinessAndConformity.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow( ["commName","totalSampleCount","Trendiness","Conformity", "trainMode", "selectedToPlot","quadrant"])

    """
    # load Trendiness fitting results of all comm
    # use shared variable to communicate among all comm's process
    manager = mp.Manager()
    return_trainResult_dict = manager.dict() # to save the train result and mode (wholebatch or minibatch) of each community
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    for commIndex, tup in enumerate(commDir_sizes_sortedlist):
        commName = tup[0]
        commDir = tup[1]

        try:
            p = mp.Process(target=myFun, args=(commIndex, commName,commDir, return_trainResult_dict, sampled_comms, modelName, commName2selected_reg_strengthList))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()
            return

        processes.append(p)
        if len(processes)==16:
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

    # convert to normal dict
    normal_return_trainResult_dict = defaultdict()
    for commName, d in return_trainResult_dict.items():
        normal_return_trainResult_dict[commName] = d
    # save return_trainResult_dict
    os.chdir(root_dir) # go back to root directory
    with open(f'allComm_{modelName}_fixedTau_TrendinessAndConformity.dict', 'wb') as outputFile:
        pickle.dump(normal_return_trainResult_dict, outputFile)
        print(f"saved allComm_{modelName}_fixedTau_TrendinessAndConformity for {len(normal_return_trainResult_dict)} comms.")
    """
    
    # load return_trainResult_dict
    with open(f'allComm_{modelName}_fixedTau_TrendinessAndConformity.dict', 'rb') as inputFile:
        normal_return_trainResult_dict = pickle.load( inputFile)

    plotMap(normal_return_trainResult_dict,root_dir, sampled_comms, modelName)
    
    elapsed = format_time(time.time() - t0)
    # Report progress.
    print('plot trendiness and conformity Done completely.    Elapsed: {:}.\n'.format(elapsed))
    
if __name__ == "__main__":
  
    # calling main function
    main()
