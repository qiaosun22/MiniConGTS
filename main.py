import json
from transformers import RobertaTokenizer, RobertaModel
# from transformers import AutoTokenizer, RobertaModel
import numpy as np
import argparse
import math
import torch
import torch.nn.functional as F
from tqdm import trange
import datetime
import os, random

from utils.common_utils import Logging

from utils.data_utils import load_data_instances
from data.data_preparing import DataIterator


from modules.models.roberta import Model
from modules.f_loss import FocalLoss

from tools.trainer import Trainer

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sequence_len', type=int, default=100, help='max length of the tagging matrix')
    parser.add_argument('--sentiment2id', type=dict, default={'negative': 2, 'neutral': 3, 'positive': 4}, help='mapping sentiments to ids')
    parser.add_argument('--model_cache_dir', type=str, default='./modules/models/', help='model cache path')
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base', help='reberta model path')
    parser.add_argument('--batch_size', type=int, default=32, help='json data path')
    parser.add_argument('--device', type=str, default="cuda:0", help='gpu or cpu')
    parser.add_argument('--prefix', type=str, default="./data/", help='dataset and embedding path prefix')

    parser.add_argument('--data_version', type=str, default="D2", choices=["D1", "D2"], help='dataset and embedding path prefix')
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"], help='dataset')

    parser.add_argument('--bert_feature_dim', type=int, default=768, help='dimension of pretrained bert feature')
    parser.add_argument('--epochs', type=int, default=2000, help='training epoch number')
    parser.add_argument('--class_num', type=int, default=5, help='label number')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"], help='option: pair, triplet')
    parser.add_argument('--model_save_dir', type=str, default="./modules/models/saved_models/", help='model path prefix')
    parser.add_argument('--log_path', type=str, default=None, help='log path')
    # parser.add_argument('--beta_1', type=float, default=1e-1, help='loss beta_1')
    # parser.add_argument('--beta_2', type=float, default=1e-1, help='loss beta_2')

   
    
    args = parser.parse_known_args()[0]
    if args.log_path is None:
        args.log_path = './logs/log_{}_{}_{}.log'.format(args.data_version, args.dataset, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    if args.device == 'cuda':
        torch.cuda.set_device(1)

    #加载预训练字典和分词方法
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path,
        cache_dir=args.model_cache_dir,  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
        force_download=False,  # 是否强制下载
    )

    logging = Logging(file_name=args.log_path).logging


    def seed_torch(seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        
    seed = 666
    seed_torch(seed)
    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Load Dataset
    train_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/train.json')))
    # random.shuffle(train_sentence_packs)
    dev_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/dev.json')))
    test_sentence_packs = json.load(open(os.path.abspath(args.prefix + args.data_version + '/' + args.dataset + '/test.json')))

    train_instances = load_data_instances(tokenizer, train_sentence_packs, args)
    dev_instances = load_data_instances(tokenizer, dev_sentence_packs, args)
    test_instances = load_data_instances(tokenizer, test_sentence_packs, args)

    trainset = DataIterator(train_instances, args)
    devset = DataIterator(dev_instances, args)
    testset = DataIterator(test_instances, args)

    model = Model(args).to(args.device)
    optimizer = torch.optim.AdamW([
                    {'params': model.bert.parameters(), 'lr': 1e-5},
                    {'params': model.linear1.parameters(), 'lr': 1e-4},
                    {'params': model.cls_linear.parameters(), 'lr': 1e-3},
                    {'params': model.cls_linear1.parameters(), 'lr': 1e-3}
                ], lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 250, 300, 400, 450, 500, 600], gamma=0.8)

    # label = ['N', 'CTD', 'POS', 'NEU', 'NEG']
    weight = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0]).float().to(args.device)
    f_loss = FocalLoss(weight, ignore_index=-1)#.forwardf_loss(preds,labels)

    weight1 = torch.tensor([1.0, 2.0]).float().to(args.device)
    f_loss1 = FocalLoss(weight1, ignore_index=-1)

    beta_1 = 1e-0
    beta_2 = 1e-1
    bear = 0
    bear_max = 10

    trainer = Trainer(model, trainset, devset, testset, optimizer, (f_loss, f_loss1), lr_scheduler, args, logging, beta_1, beta_2, bear_max=10, last=10, plot=True)


    logging(
f"""
\n\n
========= - * - =========
date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
seed: {seed}
beta_1: {beta_1}
beta_2: {beta_2}
weight: {weight.tolist()}   
weight1: {weight1.tolist()}
lr_scheduler: {lr_scheduler}
lr_scheduler milestones: {lr_scheduler.milestones}
lr_scheduler gamma: {lr_scheduler.gamma}
lr_bert: {optimizer.param_groups[0]['lr']}
lr_linear1: {optimizer.param_groups[1]['lr']}
lr_cls_linear: {optimizer.param_groups[2]['lr']}
lr_cls_linear1: {optimizer.param_groups[3]['lr']}

========= - * - =========
"""
        )

    trainer.train()

