import os.path
from argparse import Namespace
import sys
import json
import torch
import random
import toml
import numpy as np
import subprocess
from typing import Dict

def _get_gpu_memory_usage():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory


class DimensionConfig:
    def __init__(self, batch=None, n_channels=None, seq_len=None, n_features=None, pred_len=None):
        super().__init__()
        self.batch = batch
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.n_features = n_features
        self.pred_len = pred_len

    def update(self, batch=None, n_channels=None, seq_len=None, n_features=None, pred_len=None):
        if batch:
            self.batch = batch
        if n_channels:
            self.n_channels = n_channels
        if seq_len:
            self.seq_len = seq_len
        if n_features:
            self.n_features = n_features
        if pred_len:
            self.pred_len = pred_len
        print(f"[Dimension] batch={self.batch} n_channels={self.n_channels} "
              f"seq_len={self.seq_len} n_features={self.n_features} pred_len={self.pred_len}")


def _parse_config_files(path:str):
    args = dict()
    if not os.path.exists(path):
        print(f"Config path not found: {path}")
        return args
    if path.endswith(".json"):
        args.update(json.load(open(path, encoding="utf8")))
    elif path.endswith(".toml"):
        args.update(toml.load(open(path, encoding="utf8")))
    return args


class GlobalConfig:
    _instance: 'GlobalConfig' = None

    @classmethod
    def get_config(cls) -> 'GlobalConfig':
        return cls._instance


    def __init__(self, args: Namespace):
        self.__class__._instance = self
        self.args = args
        if hasattr(args, 'log_file') and args.log_file:
            self.log_file = open(args.log_file, 'a')
            sys.stdout = self.log_file  # redirect the output file
            sys.stderr = self.log_file  # redirect the output file
        else:
            self.log_file = None
        args.use_gpu &= torch.cuda.is_available()
        self.device = self._parse_device()
        self.downstream_args = _parse_config_files(self.args.downstream_config_path)
        self.tsa_args = _parse_config_files(self.args.tsa_config_path)
        print(f"tsa_args: {self.tsa_args}")
        self._construct_keyword()
        self.dimensions: DimensionConfig = DimensionConfig()

    def _parse_device(self):
        if not self.args.use_gpu:
            return "cpu"
        if 0<=self.args.gpu_id<torch.cuda.device_count():
            return f"cuda:{self.args.gpu_id}"

        gpu_usage = _get_gpu_memory_usage()[:torch.cuda.device_count()]
        gpu_id = np.argmin(gpu_usage)

        # gpu_id = random.randint(0,torch.cuda.device_count()-1)

        device = f"cuda:{gpu_id}"
        print(f"Device config={self.args.gpu_id} ({torch.cuda.device_count()} available gpu in total).\n"
              f"Assign gpu [{device}] (current gpu_usage={gpu_usage})")
        return device

    def _construct_keyword(self):
        self._keyword = "_".join([str(e) for e in [
            self.args.task,
            self.args.tsa,
            self.args.downstream,
            f"fe{self.args.feature_extractor}",
            self.args.dataset,
            f"lr{self.args.learning_rate}",
            "/DS",
            ".".join([f"{k}_{v}" for k,v in self.downstream_args.items()]),
            "/TSA",
            ".".join([f"{k}_{v}" for k,v in self.tsa_args.items()])
        ]])

    def get_keyword(self) -> str:
        return self._keyword

    def display(self):
        print("[Arguments]")
        print(json.dumps(vars(self.args), indent=2))

    def get_checkpoint_path(self):
        task = self.args.task.lower()
        if "forecast" in task:
            subdir = f"{self.get_keyword()}_pl{self.args.pred_len}"
        else:
            subdir = self.get_keyword()
        
        ckpt_dir = os.path.join(self.args.checkpoints, subdir)
        os.makedirs(ckpt_dir, exist_ok=True)
        return os.path.join(ckpt_dir, "checkpoints.pth")

    def get_test_result_path(self):
        task = self.args.task.lower()

        if "forecast" in task:
            subdir = f"{self.get_keyword()}_pl{self.args.pred_len}"
            # print(f"subdir is: {subdir}")
        else:
            subdir = self.get_keyword()

        full_path = os.path.join(self.args.test_result_dir, subdir, "test_result.txt")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        return os.path.abspath(full_path)

        

