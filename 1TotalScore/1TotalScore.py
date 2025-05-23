from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import sys

import os
sys.path.append('/video_hy2/workspace/qianqian.qqq/projects/3.3_NIPS_VideoScore/')

from IPython import embed
import torch
from scipy.stats import kendalltau

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
from dataloaders.dataloaders import DATALOADER_DICT


def get_args(description='CLIP4Clip for Visual Output Extraction'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--val_csv', type=str, default='1total_score_csv/val.csv', help='')
    parser.add_argument('--features_path', type=str, default='/video_hy2/workspace/qianqian.qqq/projects/3QQ-VideoScore/dataset-NOfenjing-beiying+mayi/video', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=4, help='')
    parser.add_argument('--batch_size_val', type=int, default=32, help='batch size eval')    #16
    parser.add_argument('--n_display', type=int, default=50, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--max_frames', type=int, default=12, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    #parser.add_argument("--init_model", default="/video_hy2/workspace/qianqian.qqq/projects/2CLIP4clip-Pretraining/output_dir_0501_2/pytorch_model.bin.0", type=str, required=False, help="Initial model.")
    parser.add_argument("--init_model", default="/video_hy2/workspace/qianqian.qqq/projects/2CLIP4clip-Pretraining/output_dir_0506/pytorch_model.bin.0", type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default="", type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--coef_lr', type=float, default=1e-3, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

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
    args.output_dir = "1train_MLP+classify/output_dir_0509-model0506"
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    return args

def init_device(args):
    global logger
    # 强制使用GPU 0，忽略分布式训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = 1  # 只使用一个GPU
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
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

class AestheticPredictor(nn.Module):
    #def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=1, num_classes=5):
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=1):
        super(AestheticPredictor, self).__init__()

        # 定义全连接层（MLP）
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))  # 逐层构建全连接
            layers.append(nn.ReLU())  # 使用 ReLU 激活函数
            in_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.regression_head = nn.Linear(in_dim, output_dim)
        #self.classification_head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        aesthetic_score = self.regression_head(features)
        #class_logits = self.classification_head(features)
        #return aesthetic_score, class_logits
        return aesthetic_score


def predict_video_scores(model, aesthetic_predictor, dataloader, device):
    model = model.to(device)
    aesthetic_predictor.to(device)
    predicted_scores = []
    true_scores = []
    predicted_binary = []  # 存储二分类预测结果
    true_binary = []       # 存储二分类真实标签
    video_ids = []  
    model.eval()
    aesthetic_predictor.eval()
    
    with torch.no_grad():
        for bid, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            video, video_mask, score, video_id = batch

            # 计算 visual_output
            visual_output = model.get_visual_output(video, video_mask)
            visual_output = visual_output.mean(dim=1).float()

            # 使用 MLP 预测美学分数
            predicted_score = aesthetic_predictor(visual_output)
            predicted_score = predicted_score.squeeze()

            # 生成二分类标签（0表示低质量，1表示高质量）
            binary_labels = (score >= 5.0).long()  # 假设5分以上为高质量
            binary_labels = binary_labels.to(device)

            # 记录预测分数和真实分数
            predicted_scores.append(predicted_score.cpu().numpy())
            true_scores.append(score.cpu().numpy())

            # 记录二分类预测结果和真实标签
            predicted_binary_class = (predicted_score >= 5.0).long()  # 预测分数5分以上为高质量
            predicted_binary.append(predicted_binary_class.cpu().numpy())
            true_binary.append(binary_labels.cpu().numpy())
                
            video_ids.append(video_id)

    # 拼接结果
    predicted_scores = np.concatenate(predicted_scores)
    true_scores = np.concatenate(true_scores)
    predicted_binary = np.concatenate(predicted_binary)
    true_binary = np.concatenate(true_binary)
    video_ids = np.concatenate(video_ids)

    # 计算 MSE
    mse = mean_squared_error(true_scores, predicted_scores)
    print(f"MSE: {mse:.4f}")

    # 计算 SROCC
    srocc, _ = spearmanr(true_scores, predicted_scores)
    print(f"SROCC: {srocc:.4f}")

    # 计算 PLCC
    plcc, _ = pearsonr(true_scores, predicted_scores)
    print(f"PLCC: {plcc:.4f}")

    # 计算 KRCC
    krcc, _ = kendalltau(true_scores, predicted_scores)
    print(f"KRCC: {krcc:.4f}")

    # 计算2分类准确率
    binary_accuracy = accuracy_score(true_binary, predicted_binary)
    print(f"Binary Classification Accuracy: {binary_accuracy:.4f}")

    return video_ids, predicted_scores, true_scores, predicted_binary, true_binary


def save_predictions_to_csv(video_ids, predicted_scores, true_scores, predicted_binary, true_binary, output_file):
    df = pd.DataFrame({
        'video_id': video_ids,
        'predicted_score': predicted_scores,
        'true_score': true_scores,
        'true_binary': true_binary,
        'predicted_binary': predicted_binary
    })
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")



# 主函数
def main():
    # 获取参数和初始化
    args = get_args()
    args = set_seed_logger(args)  # 设置随机种子
    device, n_gpu = init_device(args)  # 初始化设备
    
    # 初始化模型
    model = init_model(args, device, n_gpu)
    assert args.datatype in DATALOADER_DICT
    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    # 准备预测数据
    predict_dataloader, predict_length = DATALOADER_DICT[args.datatype]["val"](args, subset="val")

    # 输出测试的初始信息
    print("***** Running test *****")
    print("  Num examples =", predict_length)

    # 定义美学评分预测器
    input_dim = 512  
    aesthetic_predictor = AestheticPredictor(input_dim=input_dim)
    
    aesthetic_predictor.load_state_dict(torch.load('1train_MLP+classify/output_dir_0509-model0506/AestheticPredictor_epoch15_train_loss0.2937_test_loss0.2935_best.pth', map_location=device))  # 加载训练好的权重
    
    # 执行预测
    video_ids, predicted_scores, true_scores, predicted_classes, true_classes = predict_video_scores(model, aesthetic_predictor, predict_dataloader, device)

    # 保存预测结果
    save_predictions_to_csv(video_ids, predicted_scores, true_scores, predicted_classes, true_classes, '1train_MLP+classify/result/0510/result.csv')

# 执行主函数
if __name__ == "__main__":
    main()