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
# from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
import torch
from tqdm import tqdm
from statistics import mean
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

from scipy import stats
import json
from bs4 import BeautifulSoup
from openai import OpenAI
import tiktoken


def generate_prompt(qid2aidsAndComments,postId2text, promptType):
    prompts_Dict = defaultdict()

    # zero=shot
    for qid, acList in qid2aidsAndComments.items(): 
        aid2Dict = defaultdict()
        for i, acTuple in enumerate(acList): 
            aid, comments = acTuple     
            commentCount = len(comments)  
            commentSerialListStartedWithAt = []

            if promptType == 1:
                prompt = f"Given the following [QUESTION] and [ANSWER] with {commentCount} COMMENTS from [COMMENT-1] to [COMMENT-{commentCount}], "
                prompt += f'Do the following 3 tasks. \n'
                prompt += f'(1) Estimate a sentiment score from -1 to 1 for each comment. 1 as the most positive and -1 as the most negative. Response in a new line only with the sentiment scores and separate them with commas, such as "0.1,-0.3,0.9".\n'
                prompt += f'(2) Find out which comments are about other comments. Response in a new line and only with the serial numbers of comments and separate them with commas, such as "1,3,5".\n'
                prompt += f'(3) Estimate a score from -1 to 1 for the [ANSWER] about how helpful it is to the [QUESTION] considering all the COMMENTS. 1 as the most helpful and -1 as the most non-helpful. Response in a new line only with the helpfulness score, such as "0.5".\n'


            if isinstance(list(postId2text.keys())[0], str):
                qid = str(qid)
            questionText = postId2text[qid].strip()
            prompt += f'\n[QUESTION]:\n'+ questionText + '\n'

            # adding answers
            if isinstance(list(postId2text.keys())[0], str):
                aid = str(aid)
            if aid not in postId2text.keys():
                print(f"debug")
                continue
            answerText = postId2text[aid].strip()
            prompt += f'\n[ANSWER]:\n' + answerText + '\n'
            # adding comments
            for j, c in enumerate(comments):
                prompt += f'\n[COMMENT-{j+1}]:\n'
                prompt += c.strip() + '\n'
                if c.strip().startswith('@'):
                    commentSerialListStartedWithAt.append(j+1)

            aid2Dict[aid] = {'commentCount':commentCount,'prompt':prompt, 'commentSerialListStartedWithAt':commentSerialListStartedWithAt}
        
        prompts_Dict[qid] = aid2Dict
    
    return prompts_Dict

def parseResponseText(res_text, commentCount, commentSerialListStartedWithAt):
    resDict = defaultdict()
    lines = res_text.split('\n')

    segments = []
    try:
        for line in lines:
            line = line.strip()
            if len(line)>0: # not empty line
                if ":" in line: # consider whether in the same line
                    if len(line.split(":")) > 0:
                        line = line.split(":")[1]
                        if len(line)>0: # not empty line
                            segments.append(line)
                else:
                    segments.append(line)
        
        if len(segments) == 2:
            segments = [segments[0], 'None', segments[1] ]    
        
        assert len(segments) == 3

        
        for i, l in enumerate(segments):
            # remove (*).
            if ').' in l:
                if len(l.split(")."))>0:
                    l = l.split(").")[1]
                else:
                    l = ""
            
            # remove (*)
            if ')' in l:
                if len(l.split(")")) >0:
                    l = l.split(")")[1]
                else:
                    l = ""
            
            # remove ***:
            if ':' in l:
                if len(l.split(":")) > 0:
                    l = l.split(":")[1]
                else:
                    l = ""

            splitted_l = l.strip().split(",")

            if i == 0: # res for the first task
                sentiments = [float(t.strip()) for t in splitted_l if t != '']
                assert len(sentiments) == commentCount

            elif i == 1: # res for the second task
                if ('none' in l) or ('None' in l):
                    commentsAboutOther = []
                else:
                    commentsAboutOther = [int(s) for s in re.findall(r'\b\d+\b', l)]
                    assert len(commentsAboutOther) <= commentCount


            elif i == 2: # res for the third task
                if len(splitted_l) > 1:
                    print("the res_text for task 3 is more than one score!")
                    helpfulScore = None
                else:
                    helpfulScore = float(splitted_l[0].strip())
            
        resDict = {'sentiments':sentiments, 'commentsAboutOther':commentsAboutOther, 'helpfulScore':helpfulScore}

        return resDict
    except:
        return None

def askGPT (prompts_Dict, my_model):
    response_Dict = defaultdict()

    for qid,aid2Dict in prompts_Dict.items(): 
        aid2response = defaultdict()

        for aid, myDict in aid2Dict.items(): 
       
            my_prompt = myDict['prompt']
            my_prompt = my_prompt.replace('"',"'") # replace " with ' to avoid "" break in script

            client = OpenAI(
                # defaults to os.environ.get("OPENAI_API_KEY")
                # api_key="your-key",
                api_key="yourkey",
            )

            response = client.chat.completions.create(
                model=my_model,
                messages=[{"role": "user", "content": my_prompt}],
                temperature=0, # this is the degree of randomness of the model's output
                )
            
            res_text = response.choices[0].message.content.strip()
            print(f"question {qid}, answer {aid}:\n{res_text}")

            aid2response[aid] = res_text
        
        response_Dict[qid] = aid2response
    
    return response_Dict

def askGPTBacth(requestsFile, my_model, commName):
    # you must first upload your input file so that you can reference it correctly when kicking off batches. Upload your .jsonl file using the Files API.
    client = OpenAI(
            api_key="Your_key",
            )
    # uploading the requests file
    batch_input_file = client.files.create(
        file=open(requestsFile, "rb"),
        purpose="batch"
    )

    #Creating the Batch
    batch_input_file_id = batch_input_file.id
    batch_object = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"eval job for {commName}"
        }
    )
    
    print(batch_object)

    return batch_object

def checkBatchStatus(batch_object_id):
    client = OpenAI(
            api_key="your_key",
    )
    batch_object = client.batches.retrieve(batch_object_id)
    print(f"{batch_object.metadata['description']} batch is {batch_object.status}, {batch_object.request_counts}") 
    return batch_object

def retrieveBatchResults(batch_object_id):
    client = OpenAI(
            api_key="your-key",
    )
    batch_object = client.batches.retrieve(batch_object_id)
    if batch_object.status ==  "completed":
        output_file_id = batch_object.output_file_id
        if output_file_id:
            file_response = client.files.content(output_file_id).content
            return file_response
        else:
            print(f"{batch_object.metadata['description']} batch has no output file!")
            return None
    else:
        print(f"{batch_object.metadata['description']} batch is not completed yet!")
        return None

def generateRequests(prompts_Dict, my_model, promptType, commName, commDir, requestJsonl_files_directory):
    requests = []
    for qid,aid2Dict in prompts_Dict.items(): 
        for aid, myDict in aid2Dict.items(): 
    
            my_prompt = myDict['prompt']
            my_prompt = my_prompt.replace('"',"'") # replace " with ' to avoid "" break in script

            my_request = {
                            "custom_id": aid, # An ID for your request. This is important and must be unique because it will be used to match outputs to inputs.
                            "method": "POST",                       # HTTP method of the request. Only POST is allowed for now, but it is a mandatory field.
                            "url": "/v1/chat/completions",          # The OpenAI endpoint you want to hit with the request. This can be /v1/chat/completions.
                            # body is The request itself. The structure of this object changes depending on the endpoint you want to hit.
                            "body": {
                                    "model": my_model, 
                                    "messages": [{"role": "user", "content": my_prompt}],
                                    "temperature":0, # this is the degree of randomness of the model's output
                                    }
                        }
            requests.append(my_request)
    
    # save all requests in a jsonl file
    # restriction for openai Batch API:
    #   (1) The file can contain up to 50,000 requests.
    #   (2)The file cannot be more than 100 MB in size.
    MAX_REQUESTS = 50000
    MAX_FILE_SIZE_MB = 100
    
    if len(requests) > MAX_REQUESTS:
        print(f"requests number {len(requests)} exceeds the limit {MAX_REQUESTS}! for {commName}")
        return None
    else:
        requestsFile = f'{requestJsonl_files_directory}/requests_prompt_template_{promptType}.jsonl'
        with open(requestsFile, 'w') as outfile:
            for my_request in requests:
                json.dump(my_request, outfile)
                outfile.write('\n')
        # Get file size in MB
        file_size_mb = os.path.getsize(requestsFile) / (1024 * 1024)
        # Check file size
        if file_size_mb > MAX_FILE_SIZE_MB: 
            print(f"file size {file_size_mb} exceeds the limit {MAX_FILE_SIZE_MB}! for {commName}")
            return None
        
    return requestsFile

def parseBatchResponse(response_file_name, commName,aid2qid):  
    response_Dict = defaultdict()

    # Loading data from saved file
    results = []
    with open(response_file_name, 'r') as file:
        for line in file:
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip())
            results.append(json_object)
    
    for res in results:
        aid = res['custom_id']
        res_text = res['response']['body']['choices'][0]['message']['content'].strip()

        qid = aid2qid[aid]
        if qid not in response_Dict.keys():
            response_Dict[qid] = {aid:res_text}
        else:
            response_Dict[qid].update({aid:res_text})

    return response_Dict

def myFun(commName, commDir,root_dir, promptType):
   
    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())
    print(f"processing {commName}")
    
    
    # load intermediate_data files
    intermediate_directory = os.path.join(commDir, r'intermediate_data_folder')

    # # check whether already done this step, skip
    # my_model = "gpt-4o"
    # resultFiles = [commDir+ f'/GPTresponse_folder/qid2resDict_prompt{promptType}_{my_model}.json']
    # batchObject_files_directory = os.path.join(commDir, r'batchObject_folder')
    # batchObject_file = f"{batchObject_files_directory}/batchObject_prompt_template_{promptType}_{my_model}.dict"
    # if os.path.exists(resultFiles[0]):
    #     # target date
    #     target_date = datetime.datetime(2024, 11, 30)
    #     # file last modification time
    #     timestamp = os.path.getmtime(resultFiles[0])
    #     # convert timestamp into DateTime object
    #     datestamp = datetime.datetime.fromtimestamp(timestamp)
    #     print(f'{commName} Modified Date/Time:{datestamp}')
    #     if datestamp >= target_date: # the final result file exists
    #         # csvfile = open(root_dir+f'/allComm_prompt_{promptType}_batch_object_status.csv', 'a', newline='')
    #         # writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #         if os.path.exists(batchObject_file): # did by batch GPT
    #             with open(batchObject_file, 'rb') as inputFile:
    #                 tup = pickle.load( inputFile)
    #                 batch_object_id = tup[0]
    #             print(f"{commName} has already done by GPT batch.")
    #             # writer.writerow([commName, batch_object_id, "retrieved"])
    #             # csvfile.close()
    #             return
    #         else: # did by instant GPT
    #             print(f"{commName} has already done by GPT.")
    #             # writer.writerow([commName, None, "completed"])
    #             # csvfile.close()
    #             return
    
    
    # load question, answer and comment text 
    print(f"loading filtered_aid2qidAndComments_tillCurChunk.json... for {commName}")
    extractCommemntComplete = False
    try:
        if commName == 'stackoverflow':
            # in case using subComm reactjs
            # with open(intermediate_directory+'/'+'filtered_aid2qidAndComments_tillCurChunk_reactjs.json') as json_file:
            #     filtered_aid2qidAndComments = json.load(json_file)

            # in case using the whole SOF
            with open(intermediate_directory+'/'+'filtered_aid2qidAndComments_tillCurChunk.json') as json_file:
                filtered_aid2qidAndComments = json.load(json_file)
        else:
            with open(intermediate_directory+'/'+'filtered_aid2qidAndComments_tillCurChunk.json') as json_file:
                filtered_aid2qidAndComments = json.load(json_file)
        # check whether completed by checking the Log
        log_file = open('log_folder'+'/'+'sentiment1_filteringComments_Log.txt', 'r')
        last_line = log_file.readlines()[-1]
        if "current number of filtered answer is" in last_line: # this comm completed
            extractCommemntComplete = True
        else:
            print(f"{commName} did NOT complete the comment extraction!")
            return
        log_file.close()
              
    except Exception as e:
        print(f"for {commName} error when load filtered_aid2qidAndComments_tillCurChunk.json: {e}")
        return

    
    # load post text
    selectedAids = list(filtered_aid2qidAndComments.keys())
    selectedAids = [ int(aid) for aid in selectedAids]
    selectedQids =[]
    for aid, d in filtered_aid2qidAndComments.items():
        selectedQids.append(d['qid'])
    selectedQids = list(set(selectedQids))
    # filtered_aid2qidAndComments.clear() # to save memory
    try:
        with open(intermediate_directory+'/'+'postId2text.json') as json_file:
            postId2text = json.load(json_file)
    except:
        print(f"{commName} did NOT complete post text extraction! Doing it now...")
        #######################################
        # targetDir = intermediate_directory
        # if commName == 'stackoverflow':
        #     subComms_data_folder = os.path.join(commDir, f'subCommunities_folder')
        #     with open(subComms_data_folder+'/'+f'subCommName2subCommDir.dict', 'rb') as inputFile:
        #         subCommName2commDir = pickle.load( inputFile)
        #     subCommDir = subCommName2commDir['reactjs'] # use reactjs as replacement of SOF
        #     subComm_intermediate_directory = os.path.join(subCommDir, r'intermediate_data_folder')
        #     targetDir = subComm_intermediate_directory
        
        # # extract full text body of posts
        # with open(targetDir+'/'+'aid2Moderates.json') as json_file:
        #     aid2Moderates = json.load(json_file)
        # with open(targetDir+'/'+'qid2Moderates.json') as json_file:
        #     qid2Moderates = json.load(json_file)
        
        # # get postId2phId
        # postId2phId = defaultdict()
        # for qid, mDict in  qid2Moderates.items():
        #     if int(qid) not in selectedQids:
        #         continue
        #     QmList = mDict['moderateList']
        #     questionBodyPhId = None

        #     for tup in QmList:
        #         phType = tup[1]
        #         if phType in [1,4,7]: # Initial Title, or Edit Title, or Rollback Title
        #             questionTitlePhId = tup[0]

        #         elif phType in [2,5,8]: # Initial Body, or Edit Body, or Rollback Body
        #             questionBodyPhId = tup[0]
        #     postId2phId[int(qid)] =  questionBodyPhId
        
        # for aid, mDict in  aid2Moderates.items():
        #     if aid not in selectedAids:
        #         continue
        #     AmList = mDict['moderateList']
        #     answerBodyPhId = None
        #     for tup in AmList:
        #         phType = tup[1]
        #         if phType in [2,5,8]: # Initial Body, or Edit Body, or Rollback Body
        #             answerBodyPhId = tup[0]
        #     postId2phId[int(aid)] =  answerBodyPhId    
        
        # aid2Moderates.clear()
        # qid2Moderates.clear()

        # # convert postId2phId to phId2postId
        # phId2postId = defaultdict()
        # for postId, phId in postId2phId.items():    
        #     phId2postId[phId] = postId          
        

        # # get postId2text
        # saveChunkDir = os.path.join(targetDir, r'phId2Text_chunks_folder')
        # partFiles = [ f.path for f in os.scandir(saveChunkDir) if f.path.endswith('.json') ]
        # if len(partFiles)==0:
        #     print(f"{commName} has No phId2Text_chunks_folder\n")

        # # sort csvFiles paths based on part number
        # partFiles.sort(key=lambda p: int(p.strip(".json").split("_")[-1]))
        # partsCount = len(partFiles)                                
        # print(f"there are {partsCount} splitted event list files in {commName}")

        # try:
        #     print(f"concatenating all postHistoryId2text_chunks for sub comm {commName}...")
        #     postId2text = defaultdict()
        #     for i, subDir in enumerate(partFiles):
        #         part = i+1
        #         print(f"scanning part {part} of postHistoryId2text_chunks {commName}...")
        #         partDir = subDir
        #         with open(partDir) as json_file:
        #             postHistoryId2text_chunk = json.load(json_file)
        #             for phId, text in postHistoryId2text_chunk.items():
        #                 if int(phId) in postId2phId.values():
        #                     postId = phId2postId[int(phId)]
        #                     postId2text[postId] = text
        #             postHistoryId2text_chunk.clear()
            
        #     # save postId2text
        #     with open(intermediate_directory+'/postId2text_reactjs.json', "w") as outfile: 
        #         json.dump(postId2text, outfile, default=str)
        # except Exception as e:
        #     print(f"{commName} failed to concatenate postId2text {e}\n")
        #     return
        ########################################        
        # extract question and answer post text
        postId2text = defaultdict()
        chunk_size = 1000000
        chunkIndex = 0
        for df in pd.read_csv('Posts.csv', chunksize=chunk_size, engine='python',sep=','):
            for line_count, row in df.iterrows():
                print(f"processing processing {commName} chunk {chunkIndex} line {line_count}...")
                targetPost = int(row['Id'])
                if targetPost in selectedQids+selectedAids: # found a post
                    cur_text = row['Body']
                    postId2text[targetPost]=cur_text
                
        # Convert and write JSON object to file
        with open('intermediate_data_folder/postId2text.json', "w") as outfile: 
            json.dump(postId2text, outfile)
            print(f"saved postId2text.json for {commName}, len{len(postId2text)}")
        
        # load post text
        with open(intermediate_directory+'/'+'postId2text.json') as json_file:
            postId2text = json.load(json_file)

      
    """
    # convert filtered_aid2qidAndComments to qid2aidsAndComments
    qid2aidsAndComments = defaultdict()
    for aid, d in filtered_aid2qidAndComments.items():
        aid = int(aid)
        qid = d['qid']
        comments = d['comments']
        if qid not in qid2aidsAndComments.keys():
            qid2aidsAndComments[qid] = [(aid, comments)]
        else:
            qid2aidsAndComments[qid].append((aid,comments))
    
    # sort answers in each question
    for qid, tupList in qid2aidsAndComments.items():
        sortedList = copy.deepcopy(tupList)
        sortedList.sort(key=lambda x:x[0])
        qid2aidsAndComments[qid] = sortedList

    # generate prompts
    prompts_Dict = generate_prompt(qid2aidsAndComments,postId2text, promptType)

    # save prompts
    promptJson_files_directory = os.path.join(commDir, r'promptJson_folder')
    if not os.path.exists(promptJson_files_directory):
        print("no promptJson_files_directory, create one")
        os.makedirs(promptJson_files_directory)
    if commName == 'stackoverflow':
        with open(f'promptJson_folder/prompt_template_{promptType}_reactjs.json', "w") as outfile: 
            json.dump(prompts_Dict, outfile) 
            print(f"prompt saved for {commName} length: {len(prompts_Dict)}")
    else:
        with open(f'promptJson_folder/prompt_template_{promptType}.json', "w") as outfile: 
            json.dump(prompts_Dict, outfile) 
            print(f"prompt saved for {commName} length: {len(prompts_Dict)}")

    filtered_aid2qidAndComments.clear()
    postId2text.clear()
    
    
    # # # count tokens
    # enc = tiktoken.get_encoding("cl100k_base")
    # # assert enc.decode(enc.encode("hello world")) == "hello world"
    # # To get the tokeniser corresponding to a specific model in the OpenAI API:
    # # try:
    # #     enc = tiktoken.encoding_for_model("gpt-4o")
    # # except Exception as e:
    # #     print(e)
    # tokenCounts = []
    # for qid,aid2Dict in prompts_Dict.items(): 
    #     for aid, myDict in aid2Dict.items():
    #         my_prompt = myDict['prompt']
    #         my_prompt = my_prompt.replace('"',"'") # replace " with ' to avoid "" break in script
    #         tokenCounts.append(len(enc.encode(my_prompt)))
        
    # avg_tokenCount = int(mean(tokenCounts))
    # total_tokenCount = sum(tokenCounts)
    
    # csvfile = open(root_dir+f'/allComm_prompt_tokenCounts_{promptType}.csv', 'a', newline='')
    # writer = csv.writer(csvfile, delimiter=',',
    #                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # writer.writerow([commName, total_tokenCount, avg_tokenCount])
    # csvfile.close()
    
    
    # load  prepared prompts
    with open(f'promptJson_folder/prompt_template_{promptType}.json') as json_file:
        prompts_Dict = json.load(json_file)
    
    # extract aid2qid
    aid2qid = defaultdict()
    for qid,aid2Dict in prompts_Dict.items():
        for aid, myDict in aid2Dict.items():
            aid2qid[aid] = qid

    
    # ask GPT
    # my_model = "gpt-3.5-turbo"
    # my_model = "gpt-4-turbo-preview"
    my_model = "gpt-4o"
    # response_Dict = askGPT(prompts_Dict, my_model)

    # generate request jsonl file
    requestJsonl_files_directory = os.path.join(commDir, r'requestJsonl_folder')
    if not os.path.exists(requestJsonl_files_directory):
        print(f"no requestJsonl_files_directory for {commName}, create one")
        os.makedirs(requestJsonl_files_directory)

    requestsFile = generateRequests(prompts_Dict, my_model, promptType, commName, commDir, requestJsonl_files_directory)
    print(f"{commName} done requests generation.")

    # ask GPT in batch
    print(f"uploading batch requests for {commName}...")
    batch_object = askGPTBacth(requestsFile, my_model, commName)
    batch_object_id = batch_object.id
    input_file_id = batch_object.input_file_id
    
    # save batch_object
    batchObject_files_directory = os.path.join(commDir, r'batchObject_folder')
    if not os.path.exists(batchObject_files_directory):
        print(f"no batchObject_files_directory for {commName}, create one")
        os.makedirs(batchObject_files_directory)
    
    batchObject_file = f"{batchObject_files_directory}/batchObject_prompt_template_{promptType}_{my_model}.dict"
    with open(batchObject_file, 'wb') as outputFile:
        pickle.dump((batch_object_id,input_file_id), outputFile)
        print(f"batch object ids saved for {commName}")

    csvfile = open(root_dir+f'/allComm_prompt_{promptType}_batch_object_id.csv', 'a', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([commName, batch_object_id, batch_object.status])
    csvfile.close()
    """
    
    # load batch_object_id
    my_model = "gpt-4o"
    batchObject_files_directory = os.path.join(commDir, r'batchObject_folder')
    batchObject_file = f"{batchObject_files_directory}/batchObject_prompt_template_{promptType}_{my_model}.dict"
    with open(batchObject_file, 'rb') as inputFile:
        tup = pickle.load( inputFile)
        batch_object_id = tup[0]

    # check the batch status
    batch_object = checkBatchStatus(batch_object_id)

    csvfile = open(root_dir+f'/allComm_prompt_{promptType}_batch_object_status.csv', 'a', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([commName, batch_object_id, batch_object.status])
    csvfile.close()

    """
    # retrieve the batch results
    GPTresponse_files_directory = os.path.join(commDir, r'GPTresponse_folder')
    if not os.path.exists(GPTresponse_files_directory):
        print(f"no GPTresponse_files_directory for {commName}, create one")
        os.makedirs(GPTresponse_files_directory)

    file_response = retrieveBatchResults(batch_object_id)

    if file_response:
        response_file_name = f'GPTresponse_folder/rawResponseTo_prompts_{promptType}_{my_model}.jsonl'
        with open(response_file_name, 'wb') as file:
            file.write(file_response)
            print(f"raw response file saved for {commName}")
    
    # parse the batch response file
    response_Dict = parseBatchResponse(response_file_name, commName,aid2qid)
    
    # save results
    with open(f'GPTresponse_folder/responseTo_prompts_{promptType}_{my_model}.json', "w") as outfile: 
        json.dump( response_Dict, outfile)     
        print(f"saved GPT response for {commName}")
    
    # # Once we finish processing the results, we can delete the files from the OpenAI storage if we want to, 
    # # taking into account that the storage limit is 100GB
    # client = OpenAI(
    #             # defaults to os.environ.get("OPENAI_API_KEY")
    #             api_key="your-key",
    #         )
    # client.files.delete(batch_object.input_file_id)
    # client.files.delete(batch_object.output_file_id)
    # if batch_object.error_file_id: # if there is an error file
    #     client.files.delete(batch_object.error_file_id)
    
    
    # parse the response
    # my_model = "gpt-4-turbo-preview"
    my_model = "gpt-4o"

    with open(commDir+ f'/promptJson_folder/prompt_template_{promptType}.json') as json_file:
        prompts_Dict = json.load(json_file)

    with open(commDir+ f'/GPTresponse_folder/responseTo_prompts_{promptType}_{my_model}.json') as json_file:
        response_Dict = json.load(json_file)

    qid2resDict = defaultdict()

    commentStartedWithAtFlags = []
    commentAboutOtherByGPTFlags = []
    
    countNone = 0
    answerCount = 0
    for qid, aid2res in response_Dict.items():
        answerCount += len(aid2res)
        aid2result = defaultdict()
        for aid, res_text in aid2res.items():
            commentCount = prompts_Dict[qid][aid]['commentCount']
            commentSerialListStartedWithAt = prompts_Dict[qid][aid]['commentSerialListStartedWithAt']
            parseReturn = parseResponseText(res_text, commentCount, commentSerialListStartedWithAt)
            if parseReturn == None:
                countNone +=1
                continue
            else:
                sentiments = parseReturn['sentiments']
                commentsAboutOtherByGPT= parseReturn['commentsAboutOther']
                assert len(sentiments) == commentCount
                # compare GPT filtered comments that started with @
                cur_commentStartedWithAtFlags = [1 if i+1 in commentSerialListStartedWithAt else 0 for i in range(commentCount)]
                cur_commentAboutOtherByGPTFlags = [1 if i+1 in commentsAboutOtherByGPT else 0 for i in range(commentCount)]

                commentStartedWithAtFlags.extend(cur_commentStartedWithAtFlags)
                commentAboutOtherByGPTFlags.extend(cur_commentAboutOtherByGPTFlags)

                aid2result[aid]= parseReturn

        if len(aid2result) > 0:
            qid2resDict[qid] = aid2result

    print(f"{countNone} answers out of {answerCount} return None for {commName}")
    # save qid2resDict
    with open(f'GPTresponse_folder/qid2resDict_prompt{promptType}_{my_model}.json', "w") as outfile: 
        json.dump( qid2resDict, outfile)

    # compute precision and recall
    tp = 0
    for i in range(len(commentStartedWithAtFlags)):
        if (commentStartedWithAtFlags[i] == commentAboutOtherByGPTFlags[i]) and (commentStartedWithAtFlags[i] ==1):
            tp += 1
    allRetrivedCount = sum(commentAboutOtherByGPTFlags)
    allReleventCount = sum(commentStartedWithAtFlags)

    precision = tp / allRetrivedCount
    recall = tp / allReleventCount

    csvfile = open(root_dir+f'/allComm_prompt_precisionAndRecall_{promptType}.csv', 'a', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([commName, answerCount, countNone, len(commentAboutOtherByGPTFlags), allReleventCount, allRetrivedCount, precision, recall])
    csvfile.close()
    """
    
    

    
def main():

    t0=time.time()
    root_dir = os.getcwd()

    ## Load all other community direcotries sorted by sizes .dict files
    with open('allComm_directories_sizes_sortedlist.dict', 'rb') as inputFile:
        commDir_sizes_sortedlist = pickle.load( inputFile)
    print("sorted CommDir loaded.")

    promptType = 1 # one prompt for each answer, ask GPT for 3 tasks (task1: sentiment score of each comment; task2: which comment is about other comments; task3: helpfulness score of the answer)

    
    # save token counts of all comms
    # import csv
    # print(f"start to save the results as csv...")
    # csvfile = open(f'allComm_prompt_tokenCounts_{promptType}.csv', 'w', newline='')
    # writer = csv.writer(csvfile, delimiter=',',
    #                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # writer.writerow( ["commName","total token count", "avg token count"])
    # csvfile.close()

    # # save precision and recall  of all comms
    # import csv
    # print(f"start to save the results as csv...")
    # csvfile = open(f'allComm_prompt_precisionAndRecall_{promptType}.csv', 'w', newline='')
    # writer = csv.writer(csvfile, delimiter=',',
    #                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # writer.writerow( ["commName","total answer count","answer Count with None res","total comment count", "comment Started with at count", "comment filtered by GPT count","precision","recall"])
    # csvfile.close()

    # # save batch_object_id of all comms
    # import csv
    # print(f"start to save the batch_object_id as csv...")
    # csvfile = open(f'allComm_prompt_{promptType}_batch_object_id.csv', 'w', newline='')
    # writer = csv.writer(csvfile, delimiter=',',
    #                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # writer.writerow( ["commName","batch_object_id","batch_status"])
    # csvfile.close()

    # import csv
    # print(f"start to save the batch_object_status as csv...")
    # csvfile = open(f'allComm_prompt_{promptType}_batch_object_status.csv', 'w', newline='')
    # writer = csv.writer(csvfile, delimiter=',',
    #                     quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # writer.writerow( ["commName","batch_object_id","batch_status"])
    # csvfile.close()
    
    
     # test on comm "coffee.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[166][0], commDir_sizes_sortedlist[166][1],root_dir, promptType)
    # test on comm "webapps.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[305][0], commDir_sizes_sortedlist[305][1],root_dir, promptType)
    # test on comm "3dprinting.stackexchange" to debugd
    # myFun(commDir_sizes_sortedlist[227][0], commDir_sizes_sortedlist[227][1],root_dir, promptType)
    # # # test on comm "latin.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[229][0], commDir_sizes_sortedlist[229][1],root_dir, promptType)
    # # # test on comm "meta.askubuntu" to debug
    # myFun(commDir_sizes_sortedlist[231][0], commDir_sizes_sortedlist[231][1],root_dir, promptType)
    # # # test on comm "lifehacks.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[233][0], commDir_sizes_sortedlist[233][1],root_dir, promptType)
    #  # # test on comm "unix.meta.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[173][0], commDir_sizes_sortedlist[173][1],root_dir, promptType)
    # # # test on comm "cstheory.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[256][0], commDir_sizes_sortedlist[256][1],root_dir, promptType)
    # # # # test on comm "politics.stackexchange" to debug
    # myFun(commDir_sizes_sortedlist[283][0], commDir_sizes_sortedlist[283][1],root_dir, promptType)
    # # test on comm "stackoverflow" to debug
    # myFun(commDir_sizes_sortedlist[359][0], commDir_sizes_sortedlist[359][1],root_dir, promptType)
    # test on comm "math.meta" to debug
    # myFun(commDir_sizes_sortedlist[250][0], commDir_sizes_sortedlist[250][1],root_dir, promptType)
    # test on comm "mathoverflow" to debug
    # myFun(commDir_sizes_sortedlist[343][0], commDir_sizes_sortedlist[343][1],root_dir, promptType)
    # test on comm "askubuntu" to debug
    myFun(commDir_sizes_sortedlist[356][0], commDir_sizes_sortedlist[356][1],root_dir, promptType)

    
    # splitted communities
    # splitted_comms = ['tex.stackexchange','meta.stackexchange','softwareengineering.stackexchange','superuser','math.stackexchange','worldbuilding.stackexchange','codegolf.stackexchange','stackoverflow']
    # selected_comms = ['3dprinting.stackexchange','latin.stackexchange','meta.askubuntu','lifehacks.stackexchange',
    #                   'cstheory.stackexchange','unix.meta.stackexchange','politics.stackexchange','stackoverflow']

    """
    # run on all communities other than stackoverflow
    finishedCount = 0
    processes = []
    commIndex = 0
    for tup in commDir_sizes_sortedlist:
        commName = tup[0]
        commDir = tup[1]

        # if commName not in selected_comms:
        #     # print(f"{commName} is not selected, skip")
        #     continue

        try:
            p = mp.Process(target=myFun, args=(commName,commDir,root_dir, promptType))
            p.start()
        except Exception as e:
            print(e)
            pscount = sum(1 for proc in psutil.process_iter() if proc.name() == 'python3')
            print(f"current python3 processes count {pscount}.")
            sys.exit()

        processes.append(p)
        commIndex += 1
        if len(processes)==10:
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
    print('sentiment6 commentFilteringAndSentimentAnalysisByGPT Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
