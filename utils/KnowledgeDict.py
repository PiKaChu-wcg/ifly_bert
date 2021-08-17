r'''
Author       : PiKaChu_wcg
Date         : 2021-08-07 03:32:07
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-16 06:57:21
FilePath     : \ifly_bert\utils\KnowledgeDict.py
'''
from typing import List
import pandas as pd
class KnowledgeDict:
    """一个记录知识点和(知识等级,序号)对应关系的类,有feature_size能返回各个知识等级有的知识点的数目
    """
    def __init__(self,k_Level,df=None):
        self.kl=[]
        self.k_level=k_Level
        for i in range(k_Level+1):
            self.kl.append({"k2l":{},"l2k":[]})
        if  isinstance(df,pd.DataFrame):
            self.load_df(df)
    def add_k(self,kid:int,k_level:int):
        """增加一个新的知识点

        Args:
            kid ([int]): 知识点id
            k_level ([int]): 知识点等级
        """
        if kid not in self.kl[k_level]["k2l"].keys():
            self.kl[k_level]['l2k'].append(kid)
            self.kl[k_level]['k2l'][kid]=len(self.kl[k_level]['l2k'])-1
    def check_l(self,k_level:int,id:int)->int:
        """查看一个指定k_level的第id个元素的knowledge id

        Args:
            k_level (int): [description]
            id (int): [description]

        Returns:
            int: [description]
        """
        return self.kl[k_level]['l2k'][id]
    def check_k(self,k_level:int,kid:int)->int:
        """查看一个knowledgeid对应的k_level和序号

        Args:
            kid (int): [description]

        Returns:
            int: [description]
        """
        return self.kl[k_level]['k2l'][kid]
    def feature_size(self)->List[int]:
        """等到各个k_level的元素与个数

        Returns:
            List[int]: [description]
        """ 
        return [len(self.kl[i]['l2k']) for i in range(1,self.k_level+1)]
    def load_df(self,df):
        for _,item in df.iterrows():
            self.add_k(item.KnowledgeID,item.k_Level)

if __name__=="__main__":
    df=pd.read_csv("data/train_data.csv")
    KD=KnowledgeDict(3,df)
    print(KD.feature_size())