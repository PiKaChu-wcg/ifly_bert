r'''
Author       : PiKaChu_wcg
Date         : 2021-08-16 06:56:07
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-18 00:22:54
FilePath     : \ifly_bert\preprocess.py
'''
from transformers import BertTokenizer
import torch
from IPython.display import clear_output
import pandas as pd
import re
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from utils.KnowledgeDict import KnowledgeDict as KD
from utils.Qestion import question
clear_output()
class Data:
    def __init__(self,tokenizer,data_path="data/train_data.csv",k_level=3,q_level=5,batch_size=2,train=True):
        self.df=pd.read_csv(data_path)
        self.tokenizer=tokenizer
        self.tokenizer.add_tokens("[single]")
        self.tokenizer.add_tokens("[mutli]")
        self.tokenizer.add_tokens("[selection]")
        self.k_level=k_level
        self.q_level=q_level
        self.batch_size=batch_size
        self.train=train
        self.data={}
        if self.train:
            self.getKD()
        self.preprocess_data()
        self.get_dataloader()
    def getKD(self):
        self.kd=KD(self.k_level,self.df)
        self.weights=self.kd.weights
    def preprocess_data(self):
        data={}
        self.tokenizer.add_tokens("[single]")
        self.tokenizer.add_tokens("[mutli]")
        self.tokenizer.add_tokens("[selection]")
        cls_tk=self.tokenizer.cls_token
        f=lambda x:self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(cls_tk+str(x)))
        if self.train:
            for _,item in self.df.iterrows():
                if not item.TestQuestionID in data.keys():
                    data[item.TestQuestionID]={}
                    if item.type=="单选题":
                        data[item.TestQuestionID]["type"]="[single]"
                    else:
                        data[item.TestQuestionID]["type"]="[mutli]"
                    data[item.TestQuestionID]["type"]=f(data[item.TestQuestionID]["type"])
                    data[item.TestQuestionID]["content"]=f(re.sub("（\s*）","[selection]",item.Content))[:512]
                    data[item.TestQuestionID]["analysis"]=f(item.Analysis)[:512]
                    data[item.TestQuestionID]["options"]=f(item.options)[:512]
                    data[item.TestQuestionID]["kid"]={}
                    data[item.TestQuestionID]["q_level"]=item.q_Level-1
                data[item.TestQuestionID]["kid"][item.k_Level]=self.kd.check_k(item.k_Level,item.KnowledgeID)
        else:
           for _,item in self.df.iterrows():
                if not item["index"] in data.keys():
                    data[item["index"]]={}
                    if item.type=="单选题":
                        data[item["index"]]["type"]="[single]"
                    else:
                        data[item["index"]]["type"]="[mutli]"
                    data[item["index"]]["type"]=f(data[item["index"]]["type"])
                    data[item["index"]]["content"]=f(re.sub("（\s*）","[selection]",item.Content))[:512]
                    data[item["index"]]["analysis"]=f(item.Analysis)[:512]
                    data[item["index"]]["options"]=f(item.options)[:512]
                    data[item["index"]]["k_level"]=item.k_Level-1
                    data[item["index"]]["tid"]=item.TestQuestionID
        self.data=data

    def get_dataloader(self,batch_size=None):
        dataset=question(self.data,3,self.train)
        def collate_fn(batch):
            pad=lambda x:rnn_utils.pad_sequence(x, batch_first=True, padding_value=0)
            input=[]
            for i in range(4):
                input.append(pad([torch.tensor(line[0][i]) for line in batch]))
            output=[]
            if(self.train):
                for i in range(self.k_level):
                    output.append(torch.tensor([line[1][i] for line in batch]))
                output.append(torch.tensor([line[1][-1] for line in batch]))
                return [input,output]
            else:
                output=[[line[1][i] for line in batch] for i in range(2)]
                return [input,output]
        if batch_size:
            dataloader=DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    drop_last=True,
                    collate_fn=collate_fn
                ) 
            self.dataloader=dataloader
            return self.dataloader
        else :
            dataloader=DataLoader(
                    dataset, 
                    batch_size=self.batch_size, 
                    shuffle=self.train, 
                    drop_last=True,
                    collate_fn=collate_fn
                ) 
            self.dataloader=dataloader
    def features_nums(self):
        res=self.kd.feature_size()
        res.append(self.q_level)
        return res
if __name__=="__main__":
    bert_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    data=Data(bert_tokenizer)