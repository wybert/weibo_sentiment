# -*- coding: utf-8 -*-
# 使用base 环境 nerc

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import logging
import argparse
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import os
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from train import convert_examples_to_features
from train import MyPro
from flask import Flask
from flask import request
from tqdm.notebook import tqdm
import json
from datetime import date

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask("sentiment_parser")  # 生成app实例

ph = None
return_text = True


def init_model(args):
    # 对模型输入进行处理的processor，git上可能都是针对英文的processor
    processors = {'mypro': MyPro}
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        print("using cuda",torch.cuda.get_device_name(0))
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                              args.local_rank), num_labels=len(label_list))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_save_pth, map_location='cpu')['state_dict'])
    else:
        model.load_state_dict(torch.load(args.model_save_pth)['state_dict'])

    return model, processor, args, label_list, tokenizer, device


class parse_handler:
    def __init__(self, model, processor, args, label_list, tokenizer, device):
        self.model = model
        self.processor = processor
        self.args = args
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()

# FIXME: 这里对数据进行了预测
    def parse(self, text_list):
        result = []
        print("load samples")
        test_examples = self.processor.get_ifrn_examples(text_list)
        print("load fetures")
        test_features = convert_examples_to_features(
            test_examples, self.label_list, self.args.max_seq_length, self.tokenizer, show_exp=False)
        print("load train data")
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        print("load tensor")
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        print("Run prediction for full data")
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.args.eval_batch_size)

        for idx, (input_ids, input_mask, segment_ids) in tqdm(enumerate(test_dataloader),total=len(test_dataloader),
                                                                        desc="predicting"):
            item = {}
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            text = test_examples[idx].text_a
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                logits = F.softmax(logits, dim=1)
                pred = logits.max(1)[1]
                logits = logits.detach().cpu().numpy()[0].tolist()
                if return_text:
                    item['text_y'] = text
                item['sentiment_label'] = pred.item()
                item['sentiment_scores_0'] = logits[0]
                item['sentiment_scores_1'] = logits[1]
                result +=[item]
        return result

# FIXME: 这是一个APP可以支持调用web api来使用训练好的模型
# TODO: 对这个模型进行修改以支持自己的数据
# 可以对单个的文件进行测试，然后看看用了多久
@app.route('/sa', methods=['POST'])
def recive_text():
    text = request.get_data().decode("utf-8")
    text = json.dumps(["好好好", "不好不好", "大家好才是真的好", "你好嗄"])
    print('接收到请求！' + text)
    text_list = json.loads(text)
    if len(text_list) > 32:
        return json.dumps({"error,list length must less than 32"})
    text_list = [text[:120] for text in text_list]
    out = ph.parse(text_list)
    return out

def predict_text(text):
    # text = request.get_data().decode("utf-8")
    # text = json.dumps(["好好好", "不好不好", "大家好才是真的好", "你好嗄"])
    # print('接收到请求！' + text)
    # text_list = json.loads(text)
    assert len(text_list) > 32

    # if len(text_list) > 32:
    #     print("err")
    text_list = [text[:120] for text in text_list]
    out = ph.parse(text_list)
    return out

def extrac_date(t):
    t = t.strip("China_").strip(".parquet").split("-")[0]

    day = int(t.split('_')[-1])
    y = int('20'+t.split('_')[0][:2])
    m = int(t.split('_')[0][-2:])
    # print(t,y,m,day)
    return date(y,m,day)


if __name__ == "__main__":
    # ArgumentParser对象保存了所有必要的信息，用以将命令行参数解析为相应的python数据类型
    parser = argparse.ArgumentParser()
    # required parameters
    # 调用add_argument()向ArgumentParser对象添加命令行参数信息，这些信息告诉ArgumentParser对象如何处理命令行参数
    parser.add_argument("--data_dir",
                        default='./data',
                        type=str,
                        # required = True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--input_path",
                        default='../../data/raw/',
                        type=str,
                        # required = True,
                        help="The input data dir. Should contain the .parquet files (or other data files) for sentiment                             imputer.")
    parser.add_argument("--output_path",
                        default='../../data/processed/sentiment_bert/',
                        type=str,
                        # required = True,
                        help="The output imputed sentimnet results path")
    parser.add_argument("--first_n",
                        default=0,
                        type=int,
                        # required = True,
                        help="The first n to be processed")
    parser.add_argument("--last_n",
                        default=0,
                        type=int,
                        # required = True,
                        help="The last n files to be processd")
    parser.add_argument("--between_mn",
                        default='',
                        type=str,
                        # required = True,
                        help="The m:n data will be processed")    
    
    parser.add_argument("--bert_model",
                        default='bert-base-chinese',
                        type=str,
                        # required = True,
                        help="choose [bert-base-chinese] mode.")
    parser.add_argument("--task_name",
                        default='MyPro',
                        type=str,
                        # required = True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='checkpoints/',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")
    parser.add_argument("--model_save_pth",
                        default='checkpoints/bert_classification.pth',
                        type=str,
                        # required = True,
                        help="The output directory where the model checkpoints will be written")

    # other parameters
    parser.add_argument("--max_seq_length",
                        default=22,
                        type=int,
                        help="字符串最大长度")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="训练模式")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="验证模式")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="英文字符的大小写转换，对于中文来说没啥用")
    parser.add_argument("--train_batch_size",
                        default=128,
                        type=int,
                        help="训练时batch大小")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="验证时batch大小")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="Adam初始学习步长")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="训练的epochs次数")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for."
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="用不用CUDA")
    parser.add_argument("--local_rank",
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus.")
    parser.add_argument("--seed",
                        default=777,
                        type=int,
                        help="初始化时的随机数种子")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--optimize_on_cpu",
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU.")
    parser.add_argument("--fp16",
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.")
    parser.add_argument("--loss_scale",
                        default=128,
                        type=float,
                        help="Loss scaling, positive power of 2 values can improve fp16 convergence.")

    args = parser.parse_args()
    print('[INFO]Init model started.')
    model, processor, args, label_list, tokenizer, device = init_model(args)
    print('[INFO]Init model finished.')
    ph = parse_handler(model, processor, args, label_list, tokenizer, device)


    # TODO: 
    # input_path = "/media/kang/T7 Touch/work/weibo_process/weibo_output/weibo_text_clean_parquet/"
    # output_path = "/media/kang/T7 Touch/work/weibo_sentiment/data/processed/sentiment_bert/"
    # input_path = "../../data/raw/"
    # output_path = "../../data/processed/sentiment_bert/"
    # need_processed = [item for item in os.listdir(input_path
    #                         ) if item not in os.listdir(output_path) and ".parquet" in item]
    # temp = [extrac_date(item) for item in need_processed]
    # files = pd.DataFrame(columns=["file_name","date"],data=zip(need_processed,temp))
    # files = files.sort_values("date")
    # files = files[:10]

    
    # TODO: 
    need_processed = [item for item in os.listdir(args.input_path
                            ) if ".parquet" in item]
    # temp = [extrac_date(item) for item in need_processed]
    # files = pd.DataFrame(columns=["file_name","date"],data=zip(need_processed,temp))
    # files = files.sort_values("date")
    files = pd.DataFrame(columns=["file_name"],data=need_processed)
    if args.first_n != 0:
        files = files[:args.first_n]
    elif args.last_n !=0:
        files = files[files.shape[0]-args.last_n:]
    elif args.between_mn !="":
        [m,n] = [int(item) for item in args.between_mn.split(":")]
        files = files[m:n]
    else:
        pass
    already = [item for item in os.listdir(args.output_path) if ".parquet" in item]
    files = files[files["file_name"].map(lambda x: x not in already)]
    print("The following %d data will be processed"%files.shape[0])
    print(files)

    for k,v in tqdm(files.iterrows(),total=files.shape[0],desc='total'):
        print(k,v['file_name']) 
        file_name = v['file_name']
        file_path = args.input_path+file_name
        data = pd.read_parquet(file_path,engine="fastparquet")
        # data = data.sample(n=100,random_state=100) 
        out = ph.parse(data["text_clean"].values)
        out = pd.DataFrame(out)
        data["text_y"] = out["text_y"].values
        data["sentiment_label"] = out["sentiment_label"].values
        data["sentiment_scores_0"] = out["sentiment_scores_0"].values
        data["sentiment_scores_1"] = out["sentiment_scores_1"].values
        # comv = np.hstack([samples.values,out.values])
        # res = pd.DataFrame(comv,columns=list(samples.columns) + list(out.columns))
        # res = pd.concat([samples, out], axis=1)
        data.to_parquet(args.output_path + file_name)
        # break
 
