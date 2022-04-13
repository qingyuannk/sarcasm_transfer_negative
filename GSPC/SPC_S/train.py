import os
import torch
import sys
import warnings
import torch
warnings.filterwarnings("ignore")
import torch.nn as nn
sys.path.append('../')

from torch.autograd import Variable, grad
from transformers import AdamW, get_linear_schedule_with_warmup
from preprocess.preprocess import NewsDataset,DataProcess
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, recall_score, f1_score, accuracy_score, average_precision_score
from sklearn.manifold import TSNE
import numpy as np
import random 
import pandas as pd
from model.model import SPC_S
from torch.utils.data import Dataset, DataLoader
import  matplotlib.pyplot as plt
import yaml

def RANDOM_SEED_FIX(RANDOM_SEED):
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
RANDOM_SEED = np.random.randint(0,1024000)
RANDOM_SEED_FIX(RANDOM_SEED)


config_file=sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[2]



#load params
param=yaml.load(open(config_file,"r",encoding="utf-8").read(),Loader=yaml.FullLoader)

r=param['lr']
batch=param['batch']
model_output=param["model_output"]
clean_tag=param["clean_tag"]
epochs=param["epoch"]
categorys=param["class"]
UNCASED=param["UNCASED"]
start_dir=param['start_dir']
dataset = param["dataset"]
date=param["path"]
dir="{}/{}/".format(date,model_output)
model=dir+'model'
result=dir+'result'
if not os.path.exists(model):
    os.makedirs(model)
if not os.path.exists(result):
    os.makedirs(result)
model_name="{}/spc_s".format(model)+'_'+str(RANDOM_SEED)+'_'+str(batch)+'_'+str(model_output)+'.bin'
file_name="{}/spc_s_senti".format(result)+'_'+str(RANDOM_SEED)+'_'+str(batch)+'_'+str(model_output)+'.csv'
sfile_name="{}/spc_s_sar".format(result)+'_'+str(RANDOM_SEED)+'_'+str(batch)+'_'+str(model_output)+'.csv'
nfile_name="{}/spc_s_new".format(result)+'_'+str(RANDOM_SEED)+'_'+str(batch)+'_'+str(model_output)+'.csv'
print(model_name)
print(file_name)
print(nfile_name)
data_dir=os.listdir(start_dir)
print(data_dir)
for file in data_dir:
    dataset['senti_train'].append(os.path.join(start_dir,file))


 # your path for model and vocab 
UNCASED=os.path.abspath(UNCASED)
VOCAB=param["VOCAB"]
VOCAB=os.path.join(UNCASED,VOCAB)
train_clean_tag=[clean_tag]
dev_clean_tag=[clean_tag]
test_clean_tag=[clean_tag]
valid_result=[]
test_result=[]




pdata=DataProcess(dataset,UNCASED) 


#senti_trian_data 
senti_train_data=pdata.read_data(pdata.senti_train,train_clean_tag)
print("\033[0;35;46m senti_train_data read successfully! %d \033[0m" % (len(senti_train_data[0])))
print(len(senti_train_data[0]),len(senti_train_data[1]))
print("senti_train 0:1:2 %s %s %s" % (senti_train_data[1].count(0),senti_train_data[1].count(1),senti_train_data[1].count(2)))

train_data,dev_data,train_y,dev_y= train_test_split(senti_train_data[0],senti_train_data[1], test_size=0.2, random_state=5678)
senti_dev_data=[dev_data,dev_y]
senti_train_data=[train_data,train_y]
print("positive:negative:neural = %d : %d :%d " % (senti_train_data[1].count(1),senti_train_data[1].count(0),senti_train_data[1].count(2)))
print("positive:negative:neural= %d ：%d ：%d " % (senti_dev_data[1].count(1),senti_dev_data[1].count(0),senti_dev_data[1].count(2)))

#senti_test_data
senti_test_data=pdata.read_data(pdata.senti_test,test_clean_tag)
print("\033[0;35;46m senti_test_data read successfully! %d \033[0m" % (len(senti_test_data[0])))

# sms_train
sms_train_data=pdata.read_data(pdata.sms_train)
#combine two dataset
senti_train_data[0] += sms_train_data[0]
senti_train_data[1] += sms_train_data[1]
print(len(senti_train_data[0]),len(senti_train_data[1]))
print("\033[0;35;46m sms_train_data read successfully! %d  \033[0m" % (len(sms_train_data[0])))
print(sms_train_data[1].count(0))
print("0:1:2 %s %s %s" % (senti_train_data[1].count(0),senti_train_data[1].count(1),senti_train_data[1].count(2)))




#sms_test
sms_test_data=pdata.read_data(pdata.sms_test)
print("\033[0;35;46m sms_test_data read successfully! %d \033[0m" % (len(sms_test_data[0])))
print(len(sms_test_data[0]),sms_test_data[1].count(0))

#new test_data_set
new_test_data=[[],[]]
new_test_data[0]=sms_test_data[0]+senti_test_data[0]
new_test_data[1]+=sms_test_data[1]+senti_test_data[1]
print("\033[0;35;46m  new_test_data read successfully! %d \033[0m" % (len(new_test_data[0])))

#train_loader
senti_train_loader=pdata.get_loader(senti_train_data)
print("\n\033[0;35;46m train_data loader successfully: %s \033[0m" )


print(len(senti_train_loader))
#dev_loader
senti_dev_loader=pdata.get_loader(senti_dev_data)
print("\n\033[0;35;46m dev_data loader successfully: %s \033[0m")
#test_loader
sms_test_loader=pdata.get_loader(sms_test_data)
senti_test_loader=pdata.get_loader(senti_test_data)
#new test_data_set loader
new_test_loader=pdata.get_loader(new_test_data)
print("\n\033[0;35;46m test_data loader successfully: %s \033[0m")
#new test_data_set loader

#model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n\033[0;35;46m %s 能用 \033[0m" % device)

class Model_utils(object):
    def __init__(self,UNCASED,model_output):
        self.model = SPC_S(UNCASED,output=model_output).to(device)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optim = AdamW(self.model.parameters(), lr=1e-5)
  
    


    def train(self,train_data_loader):
        self.model.train()
        total_train_loss = 0
        iter_num = 0
        total_iter = len(train_data_loader)
        process_num = total_iter//50
        for batch in train_data_loader:
            p_adv=None
            self.optim.zero_grad()
            input_ids= batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            out1,emb1=self.model(input_ids,mask)
            loss1 = self.criterion(out1, labels)          
            loss = loss1
            total_train_loss += loss.item()
            loss.backward()
            self.optim.step()
            iter_num += 1
            a = iter_num//process_num *  "▋"
            b = ((total_iter-iter_num)//process_num) * '.'
            c = iter_num / total_iter *100
            
            print("\r{:^3.0f}%[{}->{}]".format(c,a,b),end='\r',flush=True)
        return total_train_loss/len(train_data_loader)
    

    def validation(self,data_loader,n):
        self.model.eval()
        tower1_precision = 0 
        total_valid_loss= 0
        tower1_f1=0
        iter_num = 0
        total_iter = len(data_loader)
        process_num =max(1,total_iter//50)
        with torch.no_grad():
            for batch in data_loader:
                # 正常传播
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                out1,emb1 = self.model(input_ids, attention_mask)                 
                loss1 = self.criterion(out1, labels)
                total_valid_loss += loss1.item()
                _,out1 =torch.max(out1,1)
                tower1_precision += torch.sum(out1==labels).cpu().numpy()
        return tower1_precision/n,total_valid_loss/len(data_loader)
        


    def test(self,data_loader):
        self.model.eval()
        predictions_senti = []
        predictions_senti_probs=[]
        real_values = []
        tweets_index = []
        embedding = []
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                index = d["tweet"]
                label = d["label"]   
                out,emb= self.model(input_ids, attention_mask)
                _, senti_preds = torch.max(out, dim=1)
                senti_probs = nn.functional.softmax(out, dim=1).cpu().numpy()
                tweets_index.extend(index)
                predictions_senti.extend(senti_preds)
                predictions_senti_probs.extend(senti_probs)
                real_values.extend(label)
                emb = emb.cpu().numpy()
                embedding.extend(emb)

            predictions_senti = torch.stack(predictions_senti).cpu()
            real_values = torch.stack(real_values).numpy()
        return predictions_senti,predictions_senti_probs,real_values,embedding


best_accuracy=0
early_stop=0
for i in range(0,1):
    model_utils=Model_utils(UNCASED,model_output)
    best_dev_acc = 0
    for epoch in range(1,epochs+1):
        print("\n\033[0;35;46m Epoch/epochs: %d/%d process start.\033[0m" % (epoch,epochs+1))
        senti_train_loss = model_utils.train(senti_train_loader)
        tower1_dev_acc,loss1 =  model_utils.validation(senti_dev_loader,len(senti_dev_data[0]))
        print(f' tower1_senti_acc {tower1_dev_acc:.4f}         senti_train_loss {senti_train_loss:.4f}         valid_loss {loss1:.4f}                    ')  
        if  tower1_dev_acc > best_accuracy:
            torch.save(model_utils.model.state_dict(), model_name)
            best_accuracy = tower1_dev_acc
            early_stop = 0
        else:
            early_stop = early_stop + 1
            if early_stop == 5:
                break
    print("\033[0;35;46m Epoch: %d process end.\033[0m" % 10)
    model_utils.model.load_state_dict(torch.load(model_name))  
    print('--------senti_eval----------')
    eval_senti_acc,loss1=model_utils.validation(senti_test_loader,len(senti_test_data[0]))
    predictions_senti,predictions_senti_probs,label_senti,emb_senti=model_utils.test(senti_test_loader)
    print(classification_report(label_senti,predictions_senti,digits=4))
    report=classification_report(label_senti,predictions_senti,digits=4,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(file_name, index= True)
    print(report)
    print('----------sms_eval--------')
    eval_sms_acc,loss1=model_utils.validation(sms_test_loader,len(sms_test_data[0]))
    prediction_sms,predictions_sms_probs,label_sms,emb_sms=model_utils.test(sms_test_loader)#MLT时要修改一下
    print(classification_report(label_sms,prediction_sms,digits=4))
    report=classification_report(label_sms,prediction_sms,digits=4,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(sfile_name, index= True)
    print()
    print('----------new_eval--------')
    eval_new_acc,loss1=model_utils.validation(new_test_loader,len(new_test_data[0]))
    predictions_new,predictions_new_probs,label_new,emb_new=model_utils.test(new_test_loader)#MLT时要修改一下
    print(classification_report(label_new,predictions_new,digits=4))
    report=classification_report(label_new,predictions_new,digits=4,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(nfile_name, index= True)

