
import os
from selectors import EpollSelector
from tracemalloc import start
from unicodedata import category
import torch
import sys
sys.path.append('../')

import warnings
import torch
warnings.filterwarnings("ignore")
import torch.nn as nn
from torch.autograd import Variable, grad
from transformers import AdamW
from preprocess.preprocess import DataProcess
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report, recall_score, f1_score, accuracy_score, average_precision_score
from sklearn.manifold import TSNE
import numpy as np
import random 
import pandas as pd
from model.model import MTLnet
from torch.utils.data import Dataset, DataLoader
import  matplotlib.pyplot as plt
import yaml


def RANDOM_SEED_FIX(RANDOM_SEED):
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

config_file=sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[2]
RANDOM_SEED = np.random.randint(0,1024000)
RANDOM_SEED_FIX(RANDOM_SEED)




#load params
param=yaml.load(open(config_file,"r",encoding="utf-8").read(),Loader=yaml.FullLoader)

lr=param['lr']
batch=param['batch']
model_output=param["model_output"]
clean_tag=param["clean_tag"]
epochs=param["epoch"]
categorys=param["class"]
UNCASED=param["UNCASED"]
dataset = param["dataset"]
start_dir=param['start_dir']
date=param["path"]
dir="{}/{}/".format(date,model_output)
model=dir+'model'
result=dir+'result'
if not os.path.exists(model):
    os.makedirs(model)
if not os.path.exists(result):
    os.makedirs(result)
model_name="{}/mlt".format(model)+'_'+str(RANDOM_SEED)+'_'+str(batch)+'.bin'
file_name="{}/mlt_senti".format(result)+'_'+str(RANDOM_SEED)+'_'+str(batch)+'_'+str(model_output)+'.csv'
sfile_name="{}/mlt_sar".format(result)+'_'+str(RANDOM_SEED)+'_'+str(batch)+'_'+str(model_output)+'.csv'
nfile_name="{}/mlt_new".format(result)+'_'+str(RANDOM_SEED)+'_'+str(batch)+'_'+str(model_output)+'.csv'
tfile_name="{}/trans_sar".format(result)+'_'+str(RANDOM_SEED)+'_'+str(batch)+'_'+str(model_output)+'.csv'
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


#senti_test_data
senti_test_data=pdata.read_data(pdata.senti_test,test_clean_tag)
print("\033[0;35;46m senti_test_data read successfully! %d \033[0m" % (len(senti_test_data[0])))


#sms_train
sms_train_data=pdata.read_data(pdata.sms_train)
sms_train_data_ns=pdata.read_data("../data/SemEval2018-Task3/train_data/SemEval2018-T3-train-taskA_emoji.txt","1")
sms_train_data[1]=[1]*len(sms_train_data[1])
sms_train_data[1]+=sms_train_data_ns[1]
sms_train_data[0]+=sms_train_data_ns[0]
print(len(sms_train_data[0]),len(sms_train_data[1]))
print("\033[0;35;46m sms_train_data read successfully! %d  \033[0m" % (len(sms_train_data[0])))
print(sms_train_data[1].count(1))


sms_test_data=pdata.read_data(pdata.sms_test)
print("\033[0;35;46m sms_test_data read successfully! %d \033[0m" % (len(sms_test_data[0])))
print(len(sms_test_data[0]),sms_test_data[1].count(0))


#new test_data_set
new_test_data=[[],[]]
new_test_data[0]=sms_test_data[0]+senti_test_data[0]
new_test_data[1]+=sms_test_data[1]+senti_test_data[1]
print("\033[0;35;46m  new_test_data read successfully! %d \033[0m" % (len(new_test_data[0])))

#train_loader
senti_train_loader=pdata.get_loader(senti_train_data,batch_size=batch)
sms_train_loader=pdata.get_loader(sms_train_data,batch_size=batch)
sms_train_dict=dict(enumerate(sms_train_loader))
print("\n\033[0;35;46m train_data loader successfully: %s \033[0m" )


sms_test_loader=pdata.get_loader(sms_test_data,batch_size=batch)
senti_test_loader=pdata.get_loader(senti_test_data,batch_size=batch)
print("\n\033[0;35;46m test_data loader successfully: %s \033[0m")
#new test_data_set loader
new_test_loader=pdata.get_loader(new_test_data,batch_size=batch)
#model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n\033[0;35;46m %s 能用 \033[0m" % device)

class Model_utils(object):
    def __init__(self,UNCASED,model_output,loss_type):
        self.model = MTLnet(UNCASED,output=model_output).to(device)
        self.criterion=nn.CrossEntropyLoss().to(device)
        self.criterion_sms = nn.CrossEntropyLoss().to(device)
        self.optim = AdamW(self.model.parameters(), lr=1e-5)
        self.loss_type = loss_type
    def FSGM_LOSS(self,loss,emb,input_ids,mask,labels):
        emb_grad = grad(loss, emb, retain_graph=True)
        p_adv = torch.nn.functional.normalize(emb_grad[0],p=2.0,dim=1,eps=1e-12).to(device)
        p_adv = Variable(p_mult*p_adv)
        adv_loss =self.criterion(self.model(input_ids,mask,perturbation=p_adv)[0], labels)
        return adv_loss
    def sarcastic_loss(self,index):
        input_ids = sms_train_dict[index]['input_ids'].to(device)
        mask = sms_train_dict[index]['attention_mask'].to(device)
        labels = sms_train_dict[index]['label'].to(device)
        out,emb=self.model(input_ids,mask,None)
        adv_loss=self.criterion(out, labels)
        return adv_loss

    


    def train(self,train_data_loader):
        self.model.train()
        total_train_loss = 0
        senti_loss =0
        sar_loss=0
        iter_num = 0
        total_iter = len(train_data_loader)
        process_num = total_iter//50
        for batch in train_data_loader:
            p_adv=None
            self.optim.zero_grad()
            input_ids= batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            out1,out2,emb1_senti,emb2_senti=self.model(input_ids,mask)
            loss1 = self.criterion(out1, labels)
            index = iter_num % (len(sms_train_loader))
            input_sms=sms_train_dict[index]["input_ids"].to(device)
            mask_sms=sms_train_dict[index]["attention_mask"].to(device)
            labels_sms=sms_train_dict[index]["label"].to(device)            
            out1,out2,emb1_sms,emb2_sms = self.model(input_sms,mask_sms)
            loss2 = self.criterion_sms(out2,labels_sms)
            loss = loss1 + loss2           
            total_train_loss += loss.item()
            sar_loss += loss2.item()
            senti_loss += loss1.item()
            loss.backward()
            self.optim.step()
            iter_num += 1
            a = iter_num//process_num *  "▋"
            b = ((total_iter-iter_num)//process_num) * '.'
            c = iter_num / total_iter *100
            
            print("\r{:^3.0f}%[{}->{}]".format(c,a,b),end='\r',flush=True)
        return senti_loss/len(train_data_loader),sar_loss/len(train_data_loader)
    

    def validation(self,data_loader,n):
        self.model.eval()
        tower1_precision = 0 
        tower2_precision = 0     
        iter_num = 0
        total_iter = len(data_loader)
        process_num =max(1,total_iter//50)
        with torch.no_grad():
            for batch in data_loader:
                # 正常传播
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                out1,out2,emb1,emb2 = self.model(input_ids, attention_mask)         
                _,out1 =torch.max(out1,1)
                _,out2 =torch.max(out2,1)
                tower1_precision += torch.sum(out1==labels).cpu().numpy()
                tower2_precision += torch.sum(out2==labels).cpu().numpy()
        return tower1_precision/n,tower2_precision/n
        


    def test(self,data_loader,flag=True):
        self.model.eval()
        predictions_senti = []
        predictions_sms = []
        predictions_senti_probs=[]
        real_values = []
        tweets_index = []
        embedding = []
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                index = d["tweet"].numpy()
                label = d["label"]         
                out1,out2,emb,emb2 = self.model(input_ids, attention_mask)
                _, senti_preds = torch.max(out1, dim=1)
                _, sms_preds = torch.max(out2, dim=1)
                tweets_index.extend(index)
                predictions_senti.extend(senti_preds)
                predictions_sms.extend(sms_preds)
                real_values.extend(label)
                emb = emb.cpu().numpy()
                embedding.extend(emb)

            predictions_senti = torch.stack(predictions_senti).cpu().numpy()
            real_values = torch.stack(real_values).numpy()
        return predictions_senti,predictions_sms,real_values,embedding



best_dev_acc=0
early_stop = 0
for alpha in range(0,1):
    model_utils=Model_utils(UNCASED,model_output)
    best_dev_acc = 0
    for epoch in range(1,epochs+1):
        print("\n\033[0;35;46m Epoch/epochs: %d/%d process start.\033[0m" % (epoch,epochs+1))
        senti_loss,sar_loss= model_utils.train(senti_train_loader)
        tower1_dev_acc,_ =  model_utils.validation(senti_test_loader,len(senti_test_data[0]))
        tower2_test,tower2_dev_acc =  model_utils.validation(sms_test_loader,len(sms_test_data[0]))
        print(f' tower1_senti_acc {tower1_dev_acc:.4}     senti_train_loss {senti_loss:.4f}     sar_train_loss {sar_loss:.4f}                             ')  
        if best_dev_acc <tower1_dev_acc :
            print('Get the best model, save it.')
            best_dev_acc = tower1_dev_acc
            torch.save(model_utils.model.state_dict(),model_name)
            print('Best dev acc till now is {:.3f}'.format(best_dev_acc))
            early_stop = 0
        else:
            early_stop = early_stop + 1
            if early_stop == 5:
                break
        print("\033[0;35;46m Epoch: %d process end.\033[0m" % epoch)
        
    model_utils.model.load_state_dict(torch.load(model_name))
    print(RANDOM_SEED)
    print('--------senti_eval----------')
    eval_senti_acc,_=model_utils.validation(senti_test_loader,len(senti_test_data[0]))
    predictions_senti,predictions_sms_senti,label_senti,emb_senti=model_utils.test(senti_test_loader)
    print(classification_report(label_senti,predictions_senti,target_names=categorys,digits=4))
    report=classification_report(label_senti,predictions_senti,digits=4,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(file_name, index= True)
    print('----------sms_eval--------')
    predictions_senti,predictions_sms,label_sms,emb_senti=model_utils.test(sms_test_loader)
    print(classification_report(label_sms,predictions_senti,digits=4))
    report=classification_report(label_sms,predictions_senti,digits=4,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(sfile_name, index= True)
    print('----------new_eval--------')
    eval_new_acc,_=model_utils.validation(new_test_loader,len(new_test_data[0]))
    predictions_new,predictions_sms,label_new,emb_new=model_utils.test(new_test_loader)#MLT时要修改一下
    report=classification_report(label_new,predictions_new,digits=4,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(nfile_name, index= True)
    print(classification_report(label_new,predictions_new,target_names=categorys,digits=4))    
    count=0
    count_nt=0
    count_n = 0
    for senti,sar,label in zip(predictions_new,predictions_sms,label_new):
        if sar == 1:
            if(label == 1):
                count=count+1;
            elif(label==2):
                count_nt=count_nt + 1
            else:
                count_n = count_n + 1
    sarcasm_trans={
        "P_to_S":count,
        "NT_to_S":count_nt,
        "N_to_S":count_n
    }
    df=pd.DataFrame(sarcasm_trans,index=["P_to_S","NT_to_S","N_to_S"])
    df.to_csv(tfile_name,index=True)



  

