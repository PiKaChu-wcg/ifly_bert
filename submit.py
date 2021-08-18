r'''
Author       : PiKaChu_wcg
Date         : 2021-08-17 21:25:43
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-18 10:54:42
FilePath     : \ifly_bert\submit.py
'''
import torch
import pandas as pd
from preprocess import Data
from tqdm import tqdm
import numpy as np
import argparse
from transformers import BertTokenizer

class Submit:
    def __init__(self,data_path,model_path,data):
        data_test=Data(data.tokenizer,data_path,train=False,batch_size=1)
        self.model=torch.load(model_path)
        self.model.eval()
        self.kd=data.kd
        self.dataloader=data_test.dataloader
        self.result=pd.read_csv(data_path)[['index','TestQuestionID']]
    def mkres(self,to):
        turn_to_cuda=lambda x: [turn_to_cuda(y) for y in x ] if type(x)==list else  x.cuda()
        for _,item in tqdm(enumerate(self.dataloader)):
            item[0]=turn_to_cuda(item[0])
            # item=turn_to_cuda(item)
            try:
                output=self.model(item)
            except:
                print(item)
                continue
            k_level=item[1][0][0]+1
            testid=item[1][1][0]
            kid=output[k_level-1][0].argmax(dim=-1).item()
            # print(f"klevel:{k_level}\nkid:{kid}")
            kid=self.kd.check_l(k_level,kid)
            self.result.loc[self.result['TestQuestionID']==testid,'KnowledgeID']=kid
            self.result.loc[self.result['TestQuestionID']==testid,'q_Level']=output[-1][0].argmax(dim=-1).item()+1
        # self.result=self.result.fillna(0)
        self.result['KnowledgeID']=self.result['KnowledgeID'].astype(np.int64)
        self.result['q_Level']=self.result['q_Level'].astype(np.int64)
        self.result.to_csv(to,index=False)
        return self.result
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default="data/train_data.csv", type=str, required=False, help='训练数据')
    parser.add_argument('--test_data', default="data/test_data.csv", type=str, required=False, help='测试数据')
    parser.add_argument('--model', default="model/model_9.pth", type=str, required=False, help='模型')
    parser.add_argument('--to', default="res.csv", type=str, required=False, help='保存')
    args = parser.parse_args()
    bert_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    data=Data(bert_tokenizer,args.train_data)
    args = parser.parse_args()
    s=Submit(args.test_data,args.model,data)
    s.mkres(args.to)