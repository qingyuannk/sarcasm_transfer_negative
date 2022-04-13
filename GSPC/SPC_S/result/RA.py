import numpy as np
import pandas  as pd
import os 
dir='.'
sub_dir='3/'
def csv_filter_senti(file):
    return file[4] not in ['s','n']
def csv_filter_sar(file):
    return file[4]=='s'
def csv_filter_new(file):
    return file[4]=='n'
path=sub_dir+'result/'
files = os.listdir(path)
filenames_senti=list(filter(csv_filter_senti,files))
filenames_sar=list(filter(csv_filter_sar,files))
filenames_new=list(filter(csv_filter_new,files))
print(filenames_senti)
print(filenames_sar)
print(filenames_new)
def result(filenames,str):
    res=[]
    print('----------------{}-----------------'.format(str))
    for file in filenames:
        filename = os.path.join(path,file)
        res.append(pd.read_csv(filename)['f1-score'].to_list())
        # print(pd.read_csv(filename)['f1-score'])
    print(np.mean(res,axis=0))
    print(np.std(res,axis=0))  
result(filenames_senti,"senti")
result(filenames_sar,"sar")
result(filenames_new,"new")