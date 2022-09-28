import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR
from src.segnn import SE_GNN

import sys
import scipy.sparse as sp
sys.path.append("..")

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx): # 实现了BaseRGCN中的build_hidden_layer
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn": # 2层的UnionRGCNLayer
            # num_rels*2
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        # g: 当前历史子图; self.h: node嵌入 (num_ents, h_dim); [self.h_0, self.h_0]: 边的嵌入
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze() # node id
            g.ndata['h'] = init_ent_emb[node_id] # node embedding
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers): # n_layers = 2 两层的UnionRGCNLayer
                layer(g, [], r[i]) # g: 当前历史子图; r[i]: self.h_0 (num_rels*2, h_dim) 更新了两轮的g.ndata['h']
            return g.ndata.pop('h') # 返回了图中更新的node embedding
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', use_cuda=False, gpu = 0, analysis=False,
                 segnn=False, dataset='ICEWS14', kg_layer=2, bn=False, comp_op='mul', ent_drop=0.2, rel_drop=0.1,
                 num_words=0, num_static_rels=0, weight=1, discount=0, angle=0, use_static=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name # convtranse
        self.encoder_name = encoder_name # uvrgcn
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.emb_rel = None
        self.gpu = gpu

        # static parameters
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.p_rel = torch.nn.Parameter(torch.Tensor(4*2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.p_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float() # 所有实体的进化嵌入
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_e = torch.nn.CrossEntropyLoss()
        self.loss_r = torch.nn.CrossEntropyLoss()

        if segnn:
            if dataset == 'YAGO' or dataset == 'WIKI':
                self.rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis, num_hidden_layers, dropout, self_loop, skip_connect, encoder_name, self.opn, self.emb_rel, use_cuda, analysis)
                self.super_rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis, num_hidden_layers, dropout, self_loop, skip_connect, encoder_name, self.opn, self.emb_rel, use_cuda, analysis)
            elif dataset == 'ICEWS14':
                self.rgcn = SE_GNN(h_dim, dataset, kg_layer, bn, comp_op, ent_drop, rel_drop, device=gpu)
                self.super_rgcn = SE_GNN(h_dim, dataset, kg_layer, bn, comp_op, ent_drop, rel_drop, device=gpu)
        else:
            self.rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis, num_hidden_layers, dropout, self_loop, skip_connect, encoder_name, self.opn, self.emb_rel, use_cuda, analysis)
            self.super_rgcn = RGCNCell(num_ents, h_dim, h_dim, num_rels * 2, num_bases, num_basis, num_hidden_layers, dropout, self_loop, skip_connect, encoder_name, self.opn, self.emb_rel, use_cuda, analysis)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.LSTMCell(self.h_dim*2, self.h_dim)
        self.relation_cell_2 = nn.LSTMCell(self.h_dim*2, self.h_dim)
        self.relation_cell_3 = nn.GRUCell(self.h_dim, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout, sequence_len=self.sequence_len)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout, sequence_len=self.sequence_len)
        else:
            raise NotImplementedError 

    def forward(self, g_list, super_g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []


        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0) # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []
        rel_embs = []

        for i, g in enumerate(g_list): # 对于每一个历史子图
            g = g.to(self.gpu)
            super_g = super_g_list[i]
            super_g = super_g.to(self.gpu)

            # self.h: (num_ents, h_dim); g.r_to_e: 子图g中和每一条边r相关的node列表，按照g.uniq_r中记录边的顺序排列
            temp_e = self.h[g.r_to_e] # 取出r_to_e列表中node的embedding
            # x_input: (num_rels*2, h_dim) 所有边的embedding
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            # g.r_len: 记录和边r相关的node在r_to_e列表中的idx范围，也与uniq_r中边的顺序保持一致 [(0, 4), (4, 9), ...]
            # g.uniq_r: 在当前时间戳内出现的所有的边(包括反向边) [r0, r1, ..., r0', r1', ...]
            # 首先为关系聚合邻域实体信息
            for span, r_idx in zip(g.r_len, g.uniq_r): # 对于当前时间戳子图内出现的每一条关系r
                x = temp_e[span[0]:span[1],:] # 取出与关系r相关的所有node的embedding
                x_mean = torch.mean(x, dim=0, keepdim=True) # (1, h_dim), 将当前r_idx相关的所有node的embedding求均值
                x_input[r_idx] = x_mean # 更新x_input内当前时间戳内出现过的边r的嵌入为其所有相邻node的embedding的均值
            x_input_temp = x_input

            # emb_rel: (num_rels*2, h_dim)边的嵌入 x_input: (num_rels*2, h_dim)聚合边的相关node的边的嵌入
            if i == 0: # 第一个历史子图
                # emb_rel：所有边的初始化嵌入
                x_input = torch.cat((self.emb_rel, x_input), dim=1) # (num_rels*2, h_dim*2)
                self.h_0, self.c_0 = self.relation_cell_1(x_input, (self.emb_rel, x_input_temp)) # 第一个时序子图的LSTM输入与输出，h_0与emb_rel维度保持一致(num_rels*2, h_dim)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                self.c_0 = F.normalize(self.c_0) if self.layer_norm else self.c_0
                # 用邻域关系的聚合嵌入表示四个超关系
                temp_h = self.h_0[super_g.r_to_e] # 取出关系超图r_to_e列表中rel的embedding
                super_x_input = torch.zeros(4*2, self.h_dim).float().cuda() if use_cuda else torch.zeros(4*2, self.h_dim).float() # (8, h_dim)超图中p_rel的embedding
                for span, p_r_idx in zip(super_g.r_len, super_g.uniq_super_r): # 对于当前超图内出现的每一条关系p_rel
                    super_x = temp_h[span[0]:span[1],:] # 取出与超关系p_rel相关的所有rel的embedding
                    super_x_mean = torch.mean(super_x, dim=0, keepdim=True)
                    super_x_input[p_r_idx] = super_x_mean
                super_x_input_temp = super_x_input
                super_x_input = torch.cat((self.p_rel, super_x_input), dim=1) # (8, h_dim*2)
                self.p_h_0, self.p_c_0 = self.relation_cell_2(super_x_input, (self.p_rel, super_x_input_temp))
                self.p_h_0 = F.normalize(self.p_h_0) if self.layer_norm else self.p_h_0 # self.p_h_0: (8, h_dim)
                self.p_c_0 = F.normalize(self.p_c_0) if self.layer_norm else self.p_c_0
                # 将超图送入聚合关系嵌入的超图RGCN
                current_h_0 = self.super_rgcn.forward(super_g, self.h_0, [self.p_h_0, self.p_h_0]) # 返回了超图中更新的rels embedding
                current_h_0 = F.normalize(current_h_0) if self.layer_norm else current_h_0 # (num_rels*2, h_dim)
                self.h_0 = self.relation_cell_3(current_h_0, self.h_0)  # self.h: (num_ents, h_dim)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                # self.h_0 = current_h_0
                rel_embs.append(self.h_0)
            else:
                # self.emb_rel: 第一个历史子图的边嵌入
                x_input = torch.cat((self.emb_rel, x_input), dim=1) # (num_rels*2, h_dim*2)
                self.h_0, self.c_0 = self.relation_cell_1(x_input, (self.h_0, self.c_0)) # 下一个时序子图的LSTM的输入=上一个时序子图的GRU的输出
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0 # self.h_0: (num_rels*2, h_dim)
                self.c_0 = F.normalize(self.c_0) if self.layer_norm else self.c_0
                # 用邻域关系的聚合嵌入表示四个超关系
                temp_h = self.h_0[super_g.r_to_e]  # 取出关系超图r_to_e列表中rel的embedding
                super_x_input = torch.zeros(4*2, self.h_dim).float().cuda() if use_cuda else torch.zeros(4*2, self.h_dim).float() # (8, h_dim)超图中p_rel的embedding
                for span, p_r_idx in zip(super_g.r_len, super_g.uniq_super_r):  # 对于当前超图内出现的每一条关系p_rel
                    super_x = temp_h[span[0]:span[1], :]  # 取出与超关系p_rel相关的所有rel的embedding
                    super_x_mean = torch.mean(super_x, dim=0, keepdim=True)
                    super_x_input[p_r_idx] = super_x_mean
                super_x_input = torch.cat((self.p_rel, super_x_input), dim=1) # (8, h_dim*2)
                self.p_h_0, self.p_c_0 = self.relation_cell_2(super_x_input, (self.p_h_0, self.p_c_0))
                self.p_h_0 = F.normalize(self.p_h_0) if self.layer_norm else self.p_h_0
                self.p_c_0 = F.normalize(self.p_c_0) if self.layer_norm else self.p_c_0
                # 将超图送入聚合关系嵌入的超图RGCN, 更新关系的嵌入表示
                current_h_0 = self.super_rgcn.forward(super_g, self.h_0, [self.p_h_0, self.p_h_0])  # 返回了超图中更新的rels embedding
                current_h_0 = F.normalize(current_h_0) if self.layer_norm else current_h_0  # (num_rels*2, h_dim)
                self.h_0 = self.relation_cell_3(current_h_0, self.h_0)  # self.h: (num_ents, h_dim)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                # self.h_0 = current_h_0
                rel_embs.append(self.h_0)
            # g: 当前历史子图;
            # self.h: node嵌入 (num_ents, h_dim);
            # [self.h_0, self.h_0]: 边的嵌入 (num_rels*2, h_dim) 因为有2层，所以传入两个输入
            # RGCN是一个聚合邻域节点信息，更新节点表示的过程
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0]) # 返回了图中更新的node embedding
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            # time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            # self.h = time_weight * current_h + (1-time_weight) * self.h # current_h: 当前历史子图的node embedding; 贯穿历史序列的整体图谱node embedding
            self.h = self.entity_cell_1(current_h, self.h) # self.h: (num_ents, h_dim)
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            history_embs.append(self.h) # 每一个历史子图的node embedding
        return history_embs, rel_embs[-1], static_emb, gate_list, degree_list


    def predict(self, test_graph, test_super_graph, num_rels, static_graph, test_triplets, use_cuda):
        '''
        :param test_graph: 原始时序子图
        :param test_super_graph: 时序关系子超图
        :param num_rels: 原始关系数目
        :param static_graph: 静态图
        :param test_triplets: 一个时间戳内的所有事实 [[s, r, o], [], ...] (num_triples_time, 3)
        :param use_cuda:
        :return:
        '''
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets)) # (batch_size, 3)

            evolve_embeddings = []
            rel_embeddings = []
            for idx in range(len(test_graph)):
                evolve_embs, r_emb, _, _, _ = self.forward(test_graph[idx:], test_super_graph[idx:], static_graph, use_cuda)
                # evolve_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
                evolve_emb = evolve_embs[-1]
                evolve_embeddings.append(evolve_emb)
                rel_embeddings.append(r_emb)
            evolve_embeddings.reverse()
            rel_embeddings.reverse()

            score_list = self.decoder_ob.forward(evolve_embeddings, rel_embeddings, all_triples, mode="test") # all_triples: 包含反关系的三元组二维张量
            score_rel_list = self.rdecoder.forward(evolve_embeddings, rel_embeddings, all_triples, mode="test") # (batch_size, num_rel*2)

            score_list = [_.unsqueeze(2) for _ in score_list]
            score_rel_list = [_.unsqueeze(2) for _ in score_rel_list]
            scores = torch.cat(score_list, dim=2)
            scores = torch.softmax(scores, dim=1)
            scores_rel = torch.cat(score_rel_list, dim=2)
            scores_rel = torch.softmax(scores_rel, dim=1)

            scores = torch.sum(scores, dim=-1)
            scores_rel = torch.sum(scores_rel, dim=-1)

            return all_triples, scores, scores_rel # (batch_size, 3) (batch_size, num_ents)

    def get_ft_loss(self, glist, super_glist, triple_list, static_graph, use_cuda):
        glist = [g.to(self.gpu) for g in glist]
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triple_list[-1][:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triple_list[-1], inverse_triples])
        all_triples = all_triples.to(self.gpu)

        # for step, triples in enumerate(triple_list):
        evolve_embeddings = []
        rel_embeddings = []
        for idx in range(len(glist)):
            evolve_embs, r_emb, _, _, _ = self.forward(glist[idx:], super_glist[idx:], static_graph, use_cuda)
            # evolve_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            evolve_emb = evolve_embs[-1]
            evolve_embeddings.append(evolve_emb)
            rel_embeddings.append(r_emb)
        evolve_embeddings.reverse()
        rel_embeddings.reverse()
        scores_ob = self.decoder_ob.forward(evolve_embeddings, rel_embeddings, all_triples) #.view(-1, self.num_ents)
        for idx in range(len(glist)):
            loss_ent += self.loss_e(scores_ob[idx], all_triples[:, 2])
        scores_rel = self.rdecoder.forward(evolve_embeddings, rel_embeddings, all_triples)
        for idx in range(len(glist)):
            loss_rel += self.loss_r(scores_rel[idx], all_triples[:, 1])

        evolve_embs, r_emb, static_emb, _, _ = self.forward(glist, super_glist, static_graph, use_cuda)
        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    angle = 90 // len(evolve_embs)
                    # step = (self.angle * math.pi / 180) * (time_step + 1)
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static

    def get_loss(self, glist, super_glist, static_graph, triples, use_cuda):
        """
        还需传入当前时间戳下的所有事实在各个历史子图中的历史重复事实列表
        :param glist: 历史子图列表
        :param super_glist: 历史超图列表
        :param static_graph: 静态资源
        :param triplets: 当前时间戳下的所有事实，一个时间戳内的所有事实三元组
        :param use_cuda:
        :return:
        """
        # 进行关系预测和实体预测的损失统计
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embeddings = []
        rel_embeddings = []
        for idx in range(len(glist)):
            evolve_embs, r_emb, _, _, _ = self.forward(glist[idx:], super_glist[idx:], static_graph, use_cuda) # evolve_embs, static_emb, r_emb在gpu上
            # evolve_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            evolve_emb = evolve_embs[-1]
            evolve_embeddings.append(evolve_emb)
            rel_embeddings.append(r_emb)
        evolve_embeddings.reverse()
        rel_embeddings.reverse()
        scores_ob = self.decoder_ob.forward(evolve_embeddings, rel_embeddings, all_triples)
        for idx in range(len(glist)):
            loss_ent += self.loss_e(scores_ob[idx], all_triples[:, 2])
        scores_rel = self.rdecoder.forward(evolve_embeddings, rel_embeddings, all_triples)
        for idx in range(len(glist)):
            loss_rel += self.loss_r(scores_rel[idx], all_triples[:, 1])

        evolve_embs, r_emb, static_emb, _, _ = self.forward(glist, super_glist, static_graph, use_cuda)
        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    angle = 90 // len(evolve_embs)
                    # step = (self.angle * math.pi / 180) * (time_step + 1)
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static
