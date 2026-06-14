import argparse
import os
from argparse import Namespace

import torch
import random
import numpy as np

from exp import build_exp
from utils import GlobalConfig



if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Automated Timeseries Augmentation Evaluation")
    parser.add_argument('--task', type=str, required=False, default='classification',
                        help='task name, options:[classification]')
    parser.add_argument('--is_training', type=int, required=False, default=1,
                        help='training mode or not')
    parser.add_argument('--tsa', type=str, required=False, default='AutoDA-Timeseries',
                        help='timeseries augmentation method, options:[classification]')
    parser.add_argument('--downstream', type=str, required=False, default='TCN',
                        help='downstream model for timeseries analysis')
    parser.add_argument('--feature_extractor', type=str, required=False, default='Catch22',
                        help='timeseries feature extractor')
    parser.add_argument('--downstream_config_path', type=str, required=False, default="",
                        help='config file path for downstream models')
    parser.add_argument('--tsa_config_path', type=str, required=False, default="",
                        help='config file path for tsa models')
    parser.add_argument('--socks_proxy',type=str, required=False, default="",
                        help="socks proxy for data loading. Example: 127.0.0.1:1080")
    parser.add_argument('--dataset_root',type=str, required=False, default="dataset", help="dataset root path")
    parser.add_argument('--dataset',type=str, required=True, help="dataset name (the directory name)")
    parser.add_argument('--dataset_type',type=str, required=False, default="Default", help="dataset type")
    parser.add_argument('--log_file',type=str, required=False, help="redirected log file target")
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    parser.add_argument('--gpu_id', type=int, default=-1, help='assigned gpu')
    parser.add_argument('--checkpoints', type=str, required=False, default="checkpoints",
                        help='checkpoint directory path')
    parser.add_argument('--test_result_dir', type=str, required=False, default="./test_results",
                        help='test result directory path')

    # parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='data loader num workers')
    parser.add_argument('--num_workers', type=int, default=56, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    parser.add_argument('--skip_exist', type=int, default=1,
                        help='skip training existed model checkpoints')

    # parser.add_argument('--train_prob', type=int, default=1, help='probability flag for training the augmentation model, 0 disables training, 1 enables training for all steps')
    # parser.add_argument('--train_strength', type=int, default=1, help='strength flag for training the augmentation model, 0 disables training, 1 enables training for all steps')
    
    # Long Term Forecasting
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, option:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict unvariate, MS:multivariate predict univariate')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # Anomaly Detection
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # Augmentation save (for visualization)
    parser.add_argument('--save_aug_dir', type=str, default='',
                        help='save augmentation before/after to this dir, e.g. /workspace/code_data/douzj/Sentinel/augmentation_cases')
    parser.add_argument('--save_aug_batches', type=int, default=1,
                        help='number of batches to save when save_aug_dir is set')

    


    args = parser.parse_args()

    config = GlobalConfig(args)

    if args.skip_exist:
        if os.path.exists(config.get_test_result_path()):
            print(f"[Skip] Found {config.get_test_result_path()}")
            #print('>>>>>>>start testing >>>>>>>>>>>>>>>>>>>>>>>>>>')
            #exp = build_exp(config)
            #exp.test(load_checkpoint=True)
            #torch.cuda.empty_cache()
            exit(0)

    config.display()

    if config.args.is_training:
        for i in range(config.args.itr):
            exp = build_exp(config)
            print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
            print(f'config: {config.get_keyword()}')
            exp.train()
            print('>>>>>>>start testing >>>>>>>>>>>>>>>>>>>>>>>>>>')

            exp.test(load_checkpoint=False)
            torch.cuda.empty_cache()
    else:
        print('>>>>>>>start testing >>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp = build_exp(config)
        exp.test(load_checkpoint=True)
        torch.cuda.empty_cache()





