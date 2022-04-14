import os
import sys

sys.path.append('../')

import torch
import sys
import warnings
import torch
from preprocess.preprocess import NewsDataset,DataProcess
from model.model import Sp_SPC,SPC_S,SPC,MTLnet
warnings.filterwarnings("ignore")
import yaml
from sklearn.model_selection import train_test_split
import pandas as pd    # to load dataset
import numpy as np     # for mathematic equation
from sklearn.manifold import TSNE
import seaborn as sns
import random
import os 
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoModel,AutoConfig
import matplotlib.pyplot as plt
param=yaml.load(open("param.yaml","r",encoding="utf-8").read(),Loader=yaml.FullLoader)
categorys= 2
UNCASED=param["UNCASED"]
dataset = param["dataset"]
batch=32
RANDOM_SEED = 41234
print(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Sp-SPC model
model_output=2
Sp_model = Sp_SPC(UNCASED,output=model_output).to(device)
Sp_model.load_state_dict(torch.load("../model/sp_spc_0.4_1024_32_2.bin"))  

#SPC model
model_output=2
spc_model = SPC(UNCASED,output=model_output).to(device)
spc_model.load_state_dict(torch.load("../model/spc_288789_32_2.bin"))  

#SPC-S model
model_output=2
spc_s_model = SPC_S(UNCASED,output=model_output).to(device)
spc_s_model.load_state_dict(torch.load("../model/spc_s_237713_32_2.bin"))
#MTL model
model_output=2
mlt_2 = MTLnet(UNCASED,output=model_output).to(device)
mlt_2.load_state_dict(torch.load("../model/mlt_214887_32_2.bin"))

pdata=DataProcess(dataset,UNCASED) 
# sms_test_data=pdata.read_data(pdata.sms_test)
sms_test_data=pdata.read_data(pdata.senti_test,["neutral"])

print("\033[0;35;46m sms_train_data read successfully! %d  \033[0m" % (len(sms_test_data[0])))
sms_test_loader=pdata.get_loader(sms_test_data,shuffle=False,batch_size=batch)

def test(data_loader,model,flag=True):
        model.eval()
        predictions_senti = []
        labels=[]
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                index = d["tweet"].numpy()
                label = d["label"]
                if flag:         
                    out,_,emb,_ = model(input_ids, attention_mask)
                else:
                    out,emb= model(input_ids, attention_mask)

                _, senti_preds = torch.max(out, dim=1)
                predictions_senti.extend(senti_preds)
                labels.extend(label)
            predictions_senti = torch.stack(predictions_senti).cpu()
            labels = torch.stack(labels).cpu()

        return predictions_senti,labels

## Sp-SPC 2 3 test
print('--------senti_eval----------')
sp_spc_p2,label=test(sms_test_loader,Sp_model)
mlt_p2,label=test(sms_test_loader,mlt_2)
spc2,label=test(sms_test_loader,spc_model,False)
spc_s_2,label=test(sms_test_loader,spc_s_model,False)

def write(file,sentence):
    fo = open(file,"a")
    fo.write(sentence+'\n')
    fo.close()
print(len(sp_spc_p2))
print(len(mlt_p2))
print(len(spc2))
print(len(spc_s_2))
print(len(label))
result={
    "sentence":sms_test_data[0],
    "spc":spc2,
    "spc_s":spc_s_2,
    "mlt":mlt_p2,
    "sp_spc":sp_spc_p2,
    "label":label,
}
df = pd.DataFrame(result)
df.to_csv("result_senti.csv")