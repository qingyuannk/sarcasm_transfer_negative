from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from transformers import AutoTokenizer

import os 
SENTI_TAG={
    "positive":1,
    "negative":0,
    "neutral":2,
    "0":0,
    "1":1,
    "2":2
}

class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.index = list(range(len(labels)))
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(int(self.labels[idx]))
        item['tweet']=torch.tensor(self.index[idx])
        return item

    def __len__(self):
        return len(self.labels)


class DataProcess(object):
    def __init__(self,data:dict,path):
        for key,value in data.items():
            setattr(self,key,value)
        self.tokenizer=AutoTokenizer.from_pretrained(path,normalization = True, use_fast=False,)

       
    def read_data(self,datapaths,TAG=None):
        samples=[]
        y=[]
        if isinstance(datapaths,str):
            datapaths=[datapaths]
        for filepath in datapaths:
            filepath=os.path.abspath(filepath)
            result=self.read_file(filepath,2)
            sample,tag=self.parse_data(result,TAG)
            samples+=sample
            y+=tag
        return [samples,y]
    
    def parse_data(self,data,clean_tag):
        samples=[]
        tag=[]
        for d in data:
            if self.clean_tag_file(d[1],clean_tag):
                continue
            tag.append(SENTI_TAG[d[1]])
            samples.append(d[2])
        return [samples,tag]


    
    def read_file(self,filepath,num,split='\t'):
        result=[]
        fid = open(filepath,'r')
        sentences = fid.readlines()
        for data in sentences:
            test = data.strip().split(split,num)
            result.append(test)
        fid.close()
        return result

    def clean_tag_file(self,tag,clean_tag):
        if(clean_tag and tag in clean_tag):
            return True
        return False
        
    
    def unit_test(self,string:str):
        return self.get_encode(string)

    def get_encode(self,data):
        encode = self.tokenizer(data,padding='max_length', add_special_tokens=True, return_token_type_ids=False,return_attention_mask=True,max_length=128,return_tensors='pt')
        
        return encode

    def get_loader(self,data,shuffle=True,batch_size=32):
        data_encoding=self.get_encode(data[0])
        print(data_encoding.keys())
        dataset=NewsDataset(data_encoding, data[1])
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader





if __name__ == '__main__':
    data={
    "train":["../../../../../data/train_data/twitter-2016train-A.txt","../../../data/train_data/twitter-2015train-A.txt","../../../data/train_data/twitter-2013train-A.txt"],
    "test":["../data/test_data/SemEval2018-T3-train-taskA.txt"],
    "dev":["../data/vaild_data/twitter-2016devtest-A.txt"]
}
    pdata=DataProcess(data)
    train_data=pdata.get_train_data(["neutral"])
    print(len(train_data[1] ))
    
