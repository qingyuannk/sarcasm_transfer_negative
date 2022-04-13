import numpy as np
import pandas  as pd
import os 
import matplotlib.pyplot as plt
import re
import random
import matplotlib
from matplotlib import rcParams
print(matplotlib.matplotlib_fname())
dir='.'
sub_dirs=['0/','1/','2/','3/','4/','5/','6/','7/','8/','9/','10/']
params=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
random_num=[32,64,4096,8192,10240,20480,102400,102400]

def csv_filter_senti(file):
    return file[8]=='o'
def csv_filter_sar(file):
    return file[7]=='s' 
def csv_filter_new(file):
    return file[8]=='e' 
def result(path,filenames,str):
    res=[]
    # print('----------------{}-----------------'.format(str))
    for file in filenames:
        filename = os.path.join(path,file)
        res.append(pd.read_csv(filename)['f1-score'].to_list())
        # print(len(res[0]))
        # print(pd.read_csv(filename)['f1-score'])
    return np.mean(res,axis=0),np.std(res,axis=0)
senti=[]
senti_std=[]
sar=[]
sar_std=[]
new=[]
new_std=[]
def senti_file(s,para):
    files=[]
    for num in random_num:
       file="{}_{}_{}_32_2.csv".format(s,para,num)
       files.append(file)
    return files
def sar_file(para):
    files=[]
    for num in random_num:
       file="sp_spc_sar_no_{}_32_2.csv".format(num)
       files.append(file)
    return files
for dir in params:
    path='{}/result/'.format(int(dir*10))
    files = os.listdir(path)
    filenames_senti=senti_file("sp_spc_no",dir)
    filenames_sar=sar_file("sp_spc_sar_no")
    filenames_new=senti_file("sp_spc_new_no",dir)

    # print(filenames_sar)
    mean_res,std_res=result(path,filenames_senti,"senti")
    mean_res_sar,std_res_sar=result(path,filenames_sar,"sar")
    mean_res_new,std_res_new=result(path,filenames_new,"new")
    senti.append(mean_res)
    senti_std.append(std_res)
    sar.append(mean_res_sar)
    sar_std.append(std_res_sar)
    new.append(mean_res_new)
    new_std.append(std_res_new)
print(senti[0])
print(sar[0])
print(new[0])
print(senti[3])
print(sar[3])
print(new[3])
senti = np.array(senti).T
sar = np.array(sar)
sar = sar.T
new = np.array(new).T
# print(senti)
# print(sar)
# print(new)

# print(senti[3])
# print(sar[3])
# print(new[4])
pic_name='weight_2.jpg'
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
weight=np.arange(0.0,1.1,0.1)
plt.figure(figsize=(4,3.5))
matplotlib.rcParams['mathtext.default'] = 'regular'
rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
fontsize=12
ax = plt.gca()
# plt.plot(weight,senti_sar_F1,marker='s',markersize=8)
plt.plot(weight,senti[3],marker='s',markersize=4,label=r" Acc. on $Test_{ST}$")
plt.plot(weight,sar[3],marker='*',markersize=4,label=r"S$\rightarrow$N on $Test_{SC}$ ")
plt.plot(weight,new[3],marker='^',markersize=4,label=r"Acc. on $Test_{ST\cup SC}$")
plt.grid()

my_x_ticks = np.arange(0, 1.1, 0.2)
my_y_ticks = np.arange(0.5,0.8,0.1)
plt.xticks(my_x_ticks,fontsize=18)
plt.yticks(my_y_ticks,fontsize=18)
plt.xlabel(r"$\alpha$",fontsize=18)
# plt.legend(loc='lower right',fontsize=12,borderpad=0.1,handletextpad=0.1)  
ax.tick_params(which='major',length=8,labelsize=18)
# ax.tick_params(which='minor',length=4)
plt.legend(loc='lower right', bbox_to_anchor=(1.04, -0.04),fontsize=14,frameon=False,columnspacing=0.5,handletextpad=0,markerscale=0.5)

plt.savefig(pic_name,dpi=300,bbox_inches='tight')