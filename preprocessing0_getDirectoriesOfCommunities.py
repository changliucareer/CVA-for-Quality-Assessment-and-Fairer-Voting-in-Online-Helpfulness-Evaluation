import os
#First print the current working directory
print("Current Working Directory", os.getcwd())
#Find out this file's dirname
ROOT_FOLDER = os.path.dirname(os.path.abspath('../../'))
print("Current file is in folder:", ROOT_FOLDER)
#Change the current working directory to this folder
os.chdir(ROOT_FOLDER+'data/chang/SE_data_2022')
print("Current working directory is changed to ", os.getcwd())
###############################################################

from collections import defaultdict
import pickle


directoriesOfCommu = defaultdict()
curDir = os.getcwd()
subfolders = [ f.path for f in os.scandir(curDir) if f.is_dir() ]
for subDir in subfolders:
  if '.com' in subDir:
    commName = subDir.split('/')[-1].split('.com')[0]
  else:
    print(f'No .com in {subDir}')
    commName = subDir.split('/')[-1]
  directoriesOfCommu[commName]=subDir

# save directiories of all communities as dict to folder "SE_codes_2022"
os.chdir(ROOT_FOLDER+'home/chang/SE_codes_2022')
print("Current working directory is changed to ", os.getcwd())
fname = 'commDirectories.dict'
with open(fname, 'wb') as outputFile:
    pickle.dump(directoriesOfCommu, outputFile)
    print(f"{fname} Saved as dict.")