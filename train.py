r'''
Author       : PiKaChu_wcg
Date         : 2021-08-17 09:06:22
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-17 09:19:46
FilePath     : \ifly_bert\train.py
'''
from preprocess import Data
from transformers import BertTokenizer,BertModel,BertConfig
from IPython.display import clear_output
import torchmetrics
import torch
from module.Net import Net
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='runs/exp1', type=str, required=False, help='训练曲线')
    parser.add_argument('--data', default='data/train_data.csv', type=str, required=False, help='数据地址')
    parser.add_argument('--model', default='', type=str, required=False, help='初始模型')
    parser.add_argument('--epoch', default=10, type=int, required=False, help='训练周期')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch_size')
    args = parser.parse_args()
    return args

def main(args):
    epoch=args.epoch
    bert_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    data=Data(bert_tokenizer,args.data)
    data.get_dataloader(args.batch_size)

    model=Net(3,data.features_nums()) if not args.model else torch.load(args.model)
    clear_output()
    writer=SummaryWriter(args.log)
    use_gpu=torch.cuda.is_available()
    model=model.cuda() if use_gpu else model
    for param in model.l1.parameters():
        param.requires_grad=False
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn=torch.nn.CrossEntropyLoss()
    metric = torchmetrics.Accuracy()
    turn_to_cuda=lambda x: [turn_to_cuda(y) for y in x ] if type(x)==list else  x.cuda()
    for e in range(epoch):
        print(f"epoch{e+1}ing:")
        err=[]
        for batch,item in tqdm(enumerate(data.dataloader)):
            model.train()
            item=turn_to_cuda(item) if use_gpu else item
            try:
                output=model(item)
            except:
                print(item[0])
                assert()
            loss=torch.tensor(0).cuda() if use_gpu else torch.tensor(0)
            for i in range(4):
                loss=loss_fn(output[i],item[1][i])+loss
                metric(output[i],item[1][i])
            err.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalars("train",{"loss":sum(err)/len(err),"F1":metric.compute()},e)
        print(f"epoch{e+1} has been done!\nthe the loss is {sum(err)/len(err):.2f} now!\nAnd the F1 value is {metric.compute():.3f} now!")
        if e%5==4:
            torch.save(model,"model/model_"+str(e)+".pth")
            
if __name__=="__main__":
    args=get_args()
    main(args)
