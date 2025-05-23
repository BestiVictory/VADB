from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import sys
import os
sys.path.append('/video_hy2/workspace/qianqian.qqq/projects/3.3_NIPS_VideoScore/')
from IPython import embed
import torch
from scipy.stats import kendalltau  # 添加这个导入

import torch.nn as nn
import numpy as np
import random
import os
import time
import argparse
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import spearmanr, pearsonr

from util import parallel_apply, get_logger
from dataloaders.dataloaders_nohuman import DATALOADER_DICT

global logger

def get_args(description='CLIP4Clip for Visual Output Extraction'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_dim', type=int, default=512)
    
    parser.add_argument('--val_csv', type=str, default='2general_attribute_csv/val.csv', help='')
    parser.add_argument('--features_path', type=str, default='/video_hy2/workspace/qianqian.qqq/projects/3QQ-VideoScore/dataset-NOfenjing-beiying+mayi/video', help='feature path')
    
    parser.add_argument('--num_thread_reader', type=int, default=4, help='') 
    parser.add_argument('--batch_size_val', type=int, default=32, help='batch size eval')    #32
    parser.add_argument('--n_display', type=int, default=50, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_frames', type=int, default=12, help='')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default="/video_hy2/workspace/qianqian.qqq/projects/2CLIP4clip-Pretraining/output_dir_0506/pytorch_model.bin.0", type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default="", type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--coef_lr', type=float, default=1e-4, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_false', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=2, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    args = parser.parse_args()

    return args


class AestheticPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=1):
        super(AestheticPredictor, self).__init__()

        # 定义全连接层（MLP）
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))  # 逐层构建全连接
            layers.append(nn.ReLU())  # 使用 ReLU 激活函数
            in_dim = hidden_dim

        # 最后一层输出美学分数
        layers.append(nn.Linear(in_dim, output_dim))  # 输出一个标量（美学分数）
        
        self.Mlp = nn.Sequential(*layers)  # 将所有层组合成一个网络

    def forward(self, x):
        return self.Mlp(x)  # 前向传播


class IntegratedAestheticPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], num_branches=6):
        super(IntegratedAestheticPredictor, self).__init__()
        self.branches = nn.ModuleList([
            AestheticPredictor(input_dim, hidden_dims) for _ in range(num_branches)
        ])

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)


def set_seed_logger(args):     #set_seed_logger函数用于接收一个参数args
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output_dir = "2train_general_attribute/output_dir_0508-model0506"
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    return args

def init_device(args):
    global logger
    # 强制使用GPU 0，忽略分布式训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = 1  # 只使用一个GPU
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # 如果 batch_size_val 不能被 n_gpu 整除，抛出异常
    if args.batch_size_val % n_gpu != 0:
        raise ValueError("Invalid batch_size_val and n_gpu parameter: {}%{} should be == 0".format(
            n_gpu, args.batch_size_val))

    return device, n_gpu

def init_model(args, device, n_gpu):
    if args.init_model:
        # 直接加载模型权重
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        print(f"Loaded model weights from {args.init_model}")
    else:
        model_state_dict = None
        print("No initial model provided, loading default weights.")

    # 准备模型
    cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'local')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    model.to(device)
    print("Model loaded successfully.")
    return model


def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def predict_video_scores(model, aesthetic_model_path, dataloader, device):
    # 加载集成美学预测模型
    integrated_model = IntegratedAestheticPredictor(input_dim=512).to(device)
    integrated_model.load_state_dict(torch.load(aesthetic_model_path, map_location=device))
    integrated_model.eval()
    
    model = model.to(device)
    model.eval()  # 不更新 CLIP4Clip 参数

    all_video_ids = []
    all_predicted_scores = []
    all_true_scores = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            
            video, video_mask, *scores, video_id = batch
            scores = torch.stack(scores, dim=1).float()  
            
            # 计算 visual_output
            visual_output = model.get_visual_output(video, video_mask)
            visual_output = visual_output.mean(dim=1).float()  # [batch_size, 512]

            # 使用集成模型预测所有分支的分数
            predicted_scores = integrated_model(visual_output)  #
            
            # 收集结果
            all_video_ids.extend(video_id)
            all_predicted_scores.append(predicted_scores.cpu().numpy())
            all_true_scores.append(scores.cpu().numpy())

    # 合并结果
    predicted_scores = np.concatenate(all_predicted_scores, axis=0)  
    true_scores = np.concatenate(all_true_scores, axis=0)  
    
    return all_video_ids, predicted_scores, true_scores

def save_predictions_to_csv(video_ids, predicted_scores, true_scores=None, output_file="predict-NOhuman.csv"):
    attribute_names = ['画面结构', '景别', '用光', '影调', '色彩', '景深']
    
    # 创建DataFrame
    data = {'video_id': video_ids}
    
    # 添加预测分数
    for i, name in enumerate(attribute_names):
        data[f'预测_{name}'] = predicted_scores[:, i]
    
    # 如果有真实分数，也添加进去
    if true_scores is not None:
        for i, name in enumerate(attribute_names):
            data[f'真实_{name}'] = true_scores[:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"预测结果已保存到 {output_file}")

""" def evaluate_predictions(true_scores, predicted_scores):
    attribute_names = ['画面结构', '景别', '用光', '影调', '色彩', '景深']
    
    results = []
    
    for branch in range(6):
        true = true_scores[:, branch]
        pred = predicted_scores[:, branch]
        
        # 计算各项指标
        mse = mean_squared_error(true, pred)
        srocc, _ = spearmanr(true, pred)
        plcc, _ = pearsonr(true, pred)
        
        threshold = 5.0
        true_labels = np.where(true <= threshold, 0, 1)
        pred_labels = np.where(pred <= threshold, 0, 1)
        acc = accuracy_score(true_labels, pred_labels)
        
        wasd = np.mean(np.abs(true - pred))
        
        results.append({
            '属性': attribute_names[branch],
            'MSE': mse,
            'SROCC': srocc,
            'PLCC': plcc,
            'ACC': acc,
            'WASD': wasd
        })
    
    # 创建评估结果DataFrame
    eval_df = pd.DataFrame(results)
    print("\n评估结果:")
    print(eval_df.round(4))
    
    return eval_df
 """


def evaluate_predictions(true_scores, predicted_scores):
    attribute_names = ['画面结构', '景别', '用光', '影调', '色彩', '景深']
    
    results = []
    
    for branch in range(6):
        true = true_scores[:, branch]
        pred = predicted_scores[:, branch]
        
        # 计算各项指标
        mse = mean_squared_error(true, pred)
        srocc, _ = spearmanr(true, pred)
        plcc, _ = pearsonr(true, pred)
        krcc, _ = kendalltau(true, pred)  # 计算Kendall Tau相关系数
        
        threshold = 5.0
        true_labels = np.where(true <= threshold, 0, 1)
        pred_labels = np.where(pred <= threshold, 0, 1)
        acc = accuracy_score(true_labels, pred_labels)
        
        wasd = np.mean(np.abs(true - pred))
        
        results.append({
            '属性': attribute_names[branch],
            'MSE': mse,
            'SROCC': srocc,
            'PLCC': plcc,
            'KRCC': krcc,  # 新增KRCC指标
            'ACC': acc,
            'WASD': wasd
        })
    
    # 创建评估结果DataFrame
    eval_df = pd.DataFrame(results)
    print("\n评估结果:")
    print(eval_df.round(4))
    
    return eval_df

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args)
    model = init_model(args, device, n_gpu)
    
    # 验证数据类型的有效性
    if args.datatype not in DATALOADER_DICT:
        raise ValueError(f"Invalid datatype: {args.datatype}. Available options: {list(DATALOADER_DICT.keys())}")
    
    if DATALOADER_DICT[args.datatype]["val"] is None:
        raise ValueError(f"No validation dataloader defined for datatype: {args.datatype}")

    # 准备预测数据
    predict_dataloader, predict_length = DATALOADER_DICT[args.datatype]["val"](args, subset="val")

    # 输出测试信息（建议使用logger）
    logger.info("***** Running prediction *****")
    logger.info(f"  Num examples = {predict_length}")
    logger.info(f"  Using device: {device}")

    # 模型路径
    aesthetic_model_path = "2train_general_attribute/output_dir_0508-model0506/IntegratedAestheticPredictor_epoch5_avg_test_loss_0.3734_best.pth" 
    # 确保输出目录存在
    result_dir = "2train_general_attribute/val_result"
    #output_dir = os.path.dirname(aesthetic_model_path)
    os.makedirs(result_dir, exist_ok=True)

    # 进行预测
    video_ids, predicted_scores, true_scores = predict_video_scores(
        model=model,
        aesthetic_model_path=aesthetic_model_path,
        dataloader=predict_dataloader,
        device=device
    )
    
    # 保存预测结果
    output_csv = os.path.join(result_dir, "predict-NOhuman-test.csv")
    save_predictions_to_csv(
        video_ids=video_ids,
        predicted_scores=predicted_scores,
        true_scores=true_scores,
        output_file=output_csv
    )
    
    # 评估预测结果
    if true_scores is not None:
        eval_results = evaluate_predictions(true_scores, predicted_scores)
        # 保存评估结果
        eval_csv = os.path.join(result_dir, "evaluation_results_test.csv")
        eval_results.to_csv(eval_csv, index=False)
        logger.info(f"Evaluation results saved to {eval_csv}")

    logger.info("Prediction completed successfully.")
    
if __name__ == "__main__":
    main()

