r'''
Author       : PiKaChu_wcg
Date         : 2021-08-16 06:58:37
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-17 02:09:17
FilePath     : \ifly_bert\module\Net.py
'''
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertConfig
import torch
class Net(nn.Module):
    def __init__(self,num_hidden_layers,out_features):
        super(Net,self).__init__()
        self.l1=BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.l1.resize_token_embeddings(21131)
        config=BertConfig(num_hidden_layers=num_hidden_layers)
        self.l2=BertModel(config=config)
        self.l3_1=nn.Linear(768*4,out_features[0])
        self.l3_2=nn.Linear(768*4,out_features[1])
        self.l3_3=nn.Linear(768*4,out_features[2])
        self.l3_4=nn.Linear(768*4,out_features[3])
    def forward(self,x):
        o=torch.cat([self.l1(i)[0][...,0:1,:] for i in x[0]],dim=-2)
        o=self.l2(inputs_embeds=o)[0]
        o=o.view(o.shape[0],-1)
        o1=self.l3_1(o)
        o2=self.l3_2(o)
        o3=self.l3_3(o)
        o4=self.l3_4(o)
        return [o1,o2,o3,o4]
