import os 
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoModel,AutoConfig


class Sp_SPC(nn.Module):
    def __init__(self,UNCASED='../bert-base-uncased',freeze_bert=False, hidden_size=768,output=2):
        super(Sp_SPC,self).__init__()
        config = AutoConfig.from_pretrained(UNCASED)
        config.update({'output_hidden_states':True})
        self.shared_layer=AutoModel.from_pretrained(UNCASED,config=config)
        self.tower1 = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output)
        )
        self.tower2 = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,2)

        )

    def forward(self, input_ids, attention_mask):
        tower1_emb = self.shared_layer(input_ids, attention_mask)
        tower1_hidden= torch.stack(tower1_emb[2])  #因为输出的是所有层的输出，是元组保存的，所以转成矩阵                                 
        tower1_cls_concat=tower1_hidden[-1][:,0,:]  #取 [CLS] 这个token对应的经过最后4层concat后的输出 
        out1 = self.tower1(tower1_cls_concat)
        tower2_emb = self.shared_layer(input_ids, attention_mask)
        tower2_hidden= torch.stack(tower2_emb[2])  #因为输出的是所有层的输出，是元组保存的，所以转成矩阵                   
        tower2_cls_concat=tower2_hidden[-1][:,0,:]  #取 [CLS] 这个token对应的经过最后4层concat后的输出 
        out2 = self.tower2(tower2_cls_concat)        
        return out1,out2,tower1_cls_concat,tower2_cls_concat

class MTLnet(nn.Module):
    def __init__(self,UNCASED='../bert-base-uncased',freeze_bert=False, hidden_size=768,output=2):
        super(MTLnet,self).__init__()
        config = AutoConfig.from_pretrained(UNCASED)
        config.update({'output_hidden_states':True})
        self.shared_layer=AutoModel.from_pretrained(UNCASED,config=config)
        self.tower1 = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output)

        )
        self.tower2 = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,2),

        )

    def forward(self, input_ids, attention_mask):
        tower1_emb = self.shared_layer(input_ids, attention_mask)
        tower1_hidden= torch.stack(tower1_emb[2])  #因为输出的是所有层的输出，是元组保存的，所以转成矩阵                                 
        tower1_cls_concat=tower1_hidden[-1][:,0,:]  #取 [CLS] 这个token对应的经过最后4层concat后的输出 
        out1 = self.tower1(tower1_cls_concat)
        tower2_emb = self.shared_layer(input_ids, attention_mask)
        tower2_hidden= torch.stack(tower2_emb[2])  #因为输出的是所有层的输出，是元组保存的，所以转成矩阵                   
        tower2_cls_concat=tower2_hidden[-1][:,0,:]  #取 [CLS] 这个token对应的经过最后4层concat后的输出 
        out2 = self.tower2(tower2_cls_concat)        
        return out1,out2,tower1_cls_concat,tower2_cls_concat
class SPC(nn.Module):
    def __init__(self,UNCASED='../bert-base-uncased',freeze_bert=False, hidden_size=768,output=2):
        super(SPC,self).__init__()
        config = AutoConfig.from_pretrained(UNCASED)
        config.update({'output_hidden_states':True})
        self.shared_layer=AutoModel.from_pretrained(UNCASED,config=config)
        self.tower1 = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output)
        )    

    def forward(self, input_ids, attention_mask):
        tower1_emb = self.shared_layer(input_ids, attention_mask)
        tower1_hidden= torch.stack(tower1_emb[2])  #因为输出的是所有层的输出，是元组保存的，所以转成矩阵                                 
        tower1_cls_concat=tower1_hidden[-1][:,0,:]  #取 [CLS] 这个token对应的经过最后4层concat后的输出 
        out1 = self.tower1(tower1_cls_concat)
        return out1,tower1_cls_concat
class SPC_S(nn.Module):
    def __init__(self,UNCASED='../bert-base-uncased',freeze_bert=False, hidden_size=768,output=2):
        super(SPC_S,self).__init__()
        config = AutoConfig.from_pretrained(UNCASED)
        config.update({'output_hidden_states':True})
        self.shared_layer=AutoModel.from_pretrained(UNCASED,config=config)
        self.tower1 = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output)
        )    

    def forward(self, input_ids, attention_mask):
        tower1_emb = self.shared_layer(input_ids, attention_mask)
        tower1_hidden= torch.stack(tower1_emb[2])  #因为输出的是所有层的输出，是元组保存的，所以转成矩阵                                 
        tower1_cls_concat=tower1_hidden[-1][:,0,:]  #取 [CLS] 这个token对应的经过最后4层concat后的输出 
        out1 = self.tower1(tower1_cls_concat)
        return out1,tower1_cls_concat