r'''
Author       : PiKaChu_wcg
Date         : 2021-08-17 09:06:22
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-18 11:32:02
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
import os
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='runs/exp2', type=str, required=False, help='训练曲线')
    parser.add_argument('--data', default='data/train_data.csv', type=str, required=False, help='数据地址')
    parser.add_argument('--model', default='', type=str, required=False, help='初始模型')
    parser.add_argument('--epoch', default=10, type=int, required=False, help='训练周期')
    parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch_size')
    parser.add_argument('--train_all', default=False, action="store_true",help="训练完整的网络")
    args = parser.parse_args()
    return args

def main(args):
    epoch=args.epoch
    print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
    bert_tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    data=Data(bert_tokenizer,args.data)
    data.get_dataloader(args.batch_size)
    print("data has been prepared!")
    model=Net(3,data.features_nums()) if not args.model else torch.load(args.model)
    print("model has been prepared!")
    writer=SummaryWriter(args.log)
    use_gpu=torch.cuda.is_available()
    model=model.cuda() if use_gpu else model
    for param in model.parameters():
        param.requires_grad=True
    if not args.train_all:
        for param in model.l1.parameters():
            param.requires_grad=False
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    metrics=[]
    for i in range(4):
        metrics.append({"loss":torch.nn.CrossEntropyLoss(data.weights[i].cuda()),"F":torchmetrics.FBeta().cuda()})
    turn_to_cuda=lambda x: [turn_to_cuda(y) for y in x ] if type(x)==list else  x.cuda()
    for e in range(epoch):
        print(f"epoch{e+1}ing:")
        err=[[],[],[],[]]
        for _,item in tqdm(enumerate(data.dataloader)):
            model.train()
            item=turn_to_cuda(item) if use_gpu else item
            try:
                output=model(item)
            except:
                print(item[0])
                assert()
            loss=torch.tensor(0,dtype=torch.float32).cuda() if use_gpu else torch.tensor(0,dtype=torch.float32)
            for i in range(4):
                loss_t=metrics[i]['loss'](output[i],item[1][i])
                err[i].append(loss_t.item())
                loss+=loss_t if not i == 3 else 3*loss_t
                metrics[i]["F"](output[i],item[1][i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalars(
            "loss",
            {
                "loss1":sum(err[0])/len(err[0]),
                "loss2":sum(err[1])/len(err[1]),
                "loss3":sum(err[2])/len(err[2]),
                "loss4":sum(err[3])/len(err[3])
                },
                e
            )
        writer.add_scalars(
            "F value",
            {
                "F1":metrics[0]["F"].compute(),
                "F2":metrics[1]["F"].compute(),
                "F3":metrics[2]["F"].compute(),
                "F4":metrics[3]["F"].compute(),
            },
            e
        )
        if not os.path.exists("model"):
            os.mkdir("model")
        if e%5==4:
            torch.save(model,"model/model_"+str(e)+".pth")
            
if __name__=="__main__":
    args=get_args()
    main(args)
