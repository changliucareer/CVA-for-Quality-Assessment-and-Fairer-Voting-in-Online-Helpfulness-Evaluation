import os
#First print the current working directory
Current_FOLDER = os.getcwd()
#First print the current working directory
print("Current Working Directory", os.getcwd())
###############################################################

import toolFunctions
import time
import pickle
import glob
import multiprocessing as mp
import math

###############################################################
#Python code to illustrate parsing of XML files
# importing the required modules
import csv
import requests
import xml.etree.ElementTree as ET

AttribDict ={'Badges':['Id','UserId','Name','Date','Class','TagBased'],
             'Posts':['Id','PostTypeId','ParentId','AcceptedAnswerId','CreationDate','Score','ViewCount',
                      'Body','OwnerUserId','LastEditorUserId','LastEditorDisplayName','LastEditDate',
                      'LastActivityDate', 'CommunityOwnedDate','ClosedDate','Title','Tags','AnswerCount',
                      'CommentCount','FavoriteCount'],
             'Comments':['Id','PostId','Score','Text','CreationDate','UserId'],
             'Votes':['Id','PostId','VoteTypeId','CreationDate','UserId','BountyAmount'],
             'PostHistory':['Id','PostHistoryTypeId','PostId','RevisionGUID','CreationDate','UserId',
                            'Comment','Text','CloseReasonId'],
             'PostLinks':['Id','CreationDate','PostId','RelatedPostId','PostLinkTypedId'],
             'Users':['Id','Reputation','CreationDate','DisplayName','EmailHash','LastAccessDate','WebsiteUrl',
                      'Location','Age','AboutMe','Views','UpVotes','DownVotes'],
             'Tags':['Id','TagName','Count','IsRequired','IsModeratorOnly','ExcerptPostId','WikiPostId']}


class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for count,element in enumerate(parent_element):
            if count%100==0:
                print(f"processing {count} element")
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself 
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a 
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                if element.tag == 'row':
                    key = 'row_'+ str(len(self)+1)
                self.update({key: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})         
  

def parse_xml(fn,file_name, writer):
    events = ("start", "end")
    context = ET.iterparse(file_name, events=events)

    return pt(fn,context, writer)

def pt(fn,context, writer, cur_elem=None):

    if cur_elem!=None:
        needed = {a:cur_elem.attrib[a] for a in AttribDict[fn] if a in cur_elem.attrib.keys()}
        if 'Text' in needed.keys():
            # if '\r\n' in needed['Text']:
            #     needed['Text'] = needed['Text'].replace('\r\n','\t')
            if '\r' in needed['Text']:
                needed['Text'] = needed['Text'].replace('\r','\t')
            if '\n' in needed['Text']:
                needed['Text'] = needed['Text'].replace('\n','\t')
        if 'Comment' in needed.keys():    
            if '\r' in needed['Comment']:
                needed['Comment'] = needed['Comment'].replace('\r','\t')
            if '\n' in needed['Comment']:
                needed['Comment'] = needed['Comment'].replace('\n','\t')
            
        if not needed['Id'].isdigit(): # id is not digit, invalid
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!{needed['Id']} not digit, but {type(needed['Id'])}")
        
        writer.writerow(needed)
        print('Added row_'+needed['Id'])

    text = ""

    for action, elem in context:
        # print("{0:>6} : {1:20} {2:20} '{3}'".format(action, elem.tag, elem.attrib, str(elem.text).strip()))

        if action == "start":
            if elem.tag=='row':
                pt(fn,context,writer,elem)
        elif action == "end":
            text = elem.text.strip() if elem.text else ""
            elem.clear()
            break

def myFun(commName, commDir):
    # skip stackoverflow
    if commName == 'stackoverflow':
        print(f"{commName} Done, skip.")
        return
    # go to current comm data directory
    os.chdir(commDir)
    print(os.getcwd())

    # find out all the XML files in the current community folder
    extension = 'xml'
    result_filelist = glob.glob('*.{}'.format(extension))
    # assert len(result_filelist) ==8 # Posts.xml, Users.xml, Tags.xml, Badges.xml, Votes.xml, Comments.xml, PostLinks.xml, PostHistory.xml

    for file in result_filelist:
        filename = file.split('.')[0]

        # # only reprocess PostHistory
        # if filename != 'PostHistory':
        #     print(f"filename is not PostHistory. skip")
        #     continue
        # only reprocess Comments
        if filename != 'Comments':
            print(f"filename is not Comments. skip")
            continue

        # write to a csv file
        csvfile = open(filename+'.csv', 'w',encoding='utf-8')

        if filename not in AttribDict.keys(): # cannot find the file
            print(f"unable to find {file} in {commName}")
            quit()
        else:
            fieldnames = AttribDict[filename]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='', extrasaction='raise', dialect='excel')
        writer.writeheader()
    
        # parse xml file
        parse_xml(filename,filename+'.xml', writer)
    
        csvfile.close()

        # Report progress.
        print(f'Done XML2CSV of {file} for {commName}.')



def main():
    
    #"Number of processors: 
    n_proc = mp.cpu_count()-2 # use 22 cores for this, and leave 2 cores for other task

    t0=time.time()

    ## Load community direcotries .dict files
    with open('commDirectories.dict', 'rb') as inputFile:
        commDirDict = pickle.load( inputFile)
    print(" all CommDir loaded.")

    with mp.Pool(processes=n_proc) as pool:
        # issue tasks to the process pool and wait for tasks to complete
        pool.starmap(myFun, list(commDirDict.items()))
    # process pool is closed automatically

   
    elapsed = toolFunctions.format_time(time.time() - t0)
    # Report progress.
    print('XML to CVS Done completely.    Elapsed: {:}.\n'.format(elapsed))

if __name__ == "__main__":
  
    # calling main function
    main()
