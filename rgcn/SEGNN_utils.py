# -*- coding=utf-8 -*-
from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
import random

# global config
CONFIG = OmegaConf.create()

DATASET_STATISTICS = dict(
    ICEWS14=dict(n_ent=7128, n_rel=230, n_train=74845, n_valid=8514, n_test=7371),
)


def get_param(*shape):
    param = Parameter(torch.zeros(shape))
    xavier_normal_(param)
    return param


def set_global_config(cfg: DictConfig): # :表示传入的参数值类型
    global CONFIG
    CONFIG = cfg

def get_global_config() -> DictConfig: # ->表示建议的函数返回值类型
    global CONFIG
    return CONFIG

def remove_randomness():
    """
    remove the randomness (not completely)
    :return:
    """
    # fix the seed
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

