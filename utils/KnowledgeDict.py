r'''
Author       : PiKaChu_wcg
Date         : 2021-08-07 03:32:07
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-18 10:05:01
FilePath     : \ifly_bert\utils\KnowledgeDict.py
'''
from typing import List
import pandas as pd
import torch
class KnowledgeDict:
    def __init__(self,k_Level,df=None):
        self.kl=[]
        self.k_level=k_Level
        for i in range(k_Level+1):
            self.kl.append({"k2l":{},"l2k":[]})
        if  isinstance(df,pd.DataFrame):
            self.load_df(df)
    def add_k(self,kid:int,k_level:int):
        if kid not in self.kl[k_level]["k2l"].keys():
            self.kl[k_level]['l2k'].append(kid)
            self.kl[k_level]['k2l'][kid]=len(self.kl[k_level]['l2k'])-1
    def check_l(self,k_level:int,id:int)->int:
        return self.kl[k_level]['l2k'][id]
    def check_k(self,k_level:int,kid:int)->int:
        return self.kl[k_level]['k2l'][kid]
    def feature_size(self)->List[int]:
        return [len(self.kl[i]['l2k']) for i in range(1,self.k_level+1)]
    def load_df(self,df):
        for _,item in df.iterrows():
            self.add_k(item.KnowledgeID,item.k_Level)
        self.init_weight()
        for _,item in df.iterrows():
            self.weights[item.k_Level-1][self.check_k(item.k_Level,item.KnowledgeID)]+=1
            self.weights[-1][item.q_Level-1]+=1
        for i in range(len(self.weights)):
            self.weights[i]=self.weights[i].sum()/self.weights[i]
            self.weights[i]=self.weights[i]/self.weights[i].sum()
    def init_weight(self):
        self.weights=[]
        for i in range(self.k_level):
            t=torch.zeros(self.feature_size()[i])
            self.weights.append(t)
        self.weights.append(torch.zeros(5))
if __name__=="__main__":
    df=pd.read_csv("data/train_data.csv")
    KD=KnowledgeDict(3,df)
    print(KD.weights)