from torch.nn import functional as F
from torch import nn
import torch
from torch.nn.parameter import Parameter
import math
import os
import numpy as np
path_dir = os.getcwd()

class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, sequence_len = 10, use_bias=True):

        super(ConvTransE, self).__init__()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)

        # 输入信号的通道：2   （最后输出的维度（卷积核的个数））channels：卷积产生的通道50   kernel_size: 卷积核的尺寸为(3, 2)   stride: 卷积步长为1
        # 输入为：(batch_size, 2, h_dim) 输出为：(batch_size, channels, h_dim)
        # padding是指卷积前进行padding，保证输出的图像形状大小与输入相同
        # 卷积核的尺寸为3，输出的第3维度为h_dim-3+1=h_dim-2，所以padding后的输出尺寸为h_dim+2*padding-3+1=h_dim
        self.conv_list = torch.nn.ModuleList()
        self.bn0_list = torch.nn.ModuleList()
        self.bn1_list = torch.nn.ModuleList()
        self.bn2_list = torch.nn.ModuleList()
        for _ in range(sequence_len):
            self.conv_list.append(torch.nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2))))  # kernel size is odd, then padding = math.floor(kernel_size/2))
            self.bn0_list.append(torch.nn.BatchNorm1d(2))
            self.bn1_list.append(torch.nn.BatchNorm1d(channels))
            self.bn2_list.append(torch.nn.BatchNorm1d(embedding_dim))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rels, triplets, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        '''
        传入每个历史子图中的实体和关系的分布式embedding，并读取valid或者test集中记录每个事实的历史重复事实的本地文件
        :param embedding: (num_ents, h_dim) 在gpu上
        :param emb_rels: (num_rels*2, h_dim)
        :param triplets: 包含反关系的valid/test集的一个时间戳的triples二维张量 (num_triples(batch_size), 3)
        :param nodes_id:
        :param mode:
        :param negative_rate:
        :param partial_embeding:
        :return:
        '''
        score_list = []
        batch_size = len(triplets)
        for idx in range(len(embedding)):
            e1_embedded_all = F.tanh(embedding[idx])
            emb_rel = emb_rels[idx]
            e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
            rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
            stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
            stacked_inputs = self.bn0_list[idx](stacked_inputs)
            x = self.inp_drop(stacked_inputs)
            x = self.conv_list[idx](x)
            x = self.bn1_list[idx](x)
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = x.view(batch_size, -1)
            x = self.fc(x)
            x = self.hidden_drop(x)
            if batch_size > 1:
                x = self.bn2_list[idx](x)
            x = F.relu(x)
            if partial_embeding is None:
                x = torch.mm(x, e1_embedded_all.transpose(1, 0))
            else:
                x = torch.mm(x, partial_embeding.transpose(1, 0))
            score_list.append(x)
        # print(score_list)
        # print("--------------------")
        return score_list

class ConvTransR(torch.nn.Module):
    def __init__(self, num_relations, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, sequence_len = 10, use_bias=True):
        super(ConvTransR, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)

        self.conv_list = torch.nn.ModuleList()
        self.bn0_list = torch.nn.ModuleList()
        self.bn1_list = torch.nn.ModuleList()
        self.bn2_list = torch.nn.ModuleList()
        for _ in range(sequence_len):
            self.conv_list.append(torch.nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(math.floor(kernel_size / 2))))  # kernel size is odd, then padding = math.floor(kernel_size/2))
            self.bn0_list.append(torch.nn.BatchNorm1d(2))
            self.bn1_list.append(torch.nn.BatchNorm1d(channels))
            self.bn2_list.append(torch.nn.BatchNorm1d(embedding_dim))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, emb_rels, triplets, nodes_id=None, mode="train", negative_rate=0):
        '''
        :param embedding: (num_ents, h_dim)
        :param emb_rel: (num_rels*2, h_dim)
        :param triplets: 测试集一个时间戳内的所有事实（包含逆关系）
        :return:
        '''
        score_list = []
        batch_size = len(triplets)
        for idx in range(len(embedding)):
            e1_embedded_all = F.tanh(embedding[idx])
            emb_rel = emb_rels[idx]
            e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
            e2_embedded = e1_embedded_all[triplets[:, 2]].unsqueeze(1)
            stacked_inputs = torch.cat([e1_embedded, e2_embedded], 1)
            stacked_inputs = self.bn0_list[idx](stacked_inputs)
            x = self.inp_drop(stacked_inputs)
            x = self.conv_list[idx](x)
            x = self.bn1_list[idx](x)
            x = F.relu(x)
            x = self.feature_map_drop(x)
            x = x.view(batch_size, -1)
            x = self.fc(x)
            x = self.hidden_drop(x)
            if batch_size > 1:
                x = self.bn2_list[idx](x)
            x = F.relu(x)
            x = torch.mm(x, emb_rel.transpose(1, 0))
            score_list.append(x)
        return score_list
