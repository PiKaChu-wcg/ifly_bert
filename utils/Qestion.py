r'''
Author       : PiKaChu_wcg
Date         : 2021-08-16 00:42:39
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-17 22:23:50
FilePath     : \ifly_bert\utils\Qestion.py
'''
from torch.utils.data import Dataset

class question(Dataset):
    def __init__(self,data,k_level,train):
        self.data=data
        self.k_level=k_level
        self.keys=list(self.data.keys())
        self.train=train
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index) :
        idx=self.keys[index]
        input=[
            self.data[idx]['type'],
            self.data[idx]['content'],
            self.data[idx]['analysis'],
            self.data[idx]['options']
            ]
        if self.train:    
            output=[
                self.data[idx]['kid'][i] for i in range(1,self.k_level+1)
            ]  
            output.append(self.data[idx]["q_level"])
            return input,output
        else :
            output=[self.data[idx]['k_level'],self.data[idx]['tid']]
            return input,output
        