#!/usr/bin/python3
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import rgcn.SEGNN_utils as utils
from rgcn.SEGNN_utils import get_param

class SE_GNN(nn.Module):
    def __init__(self, h_dim, dataset, kg_layer, bn, comp_op, ent_drop, rel_drop, device):
        super().__init__()
        # self.cfg = utils.get_global_config()
        self.dataset = dataset
        self.device = device
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']

        # gnn layer
        self.kg_n_layer = kg_layer
        # relation SE layer (h_dim, dataset, bn, device)
        self.edge_layers = nn.ModuleList([EdgeLayer(h_dim, dataset, bn, device) for _ in range(self.kg_n_layer)])
        # entity SE layer (h_dim, dataset, bn, device)
        self.node_layers = nn.ModuleList([NodeLayer(h_dim, dataset, bn, device) for _ in range(self.kg_n_layer)])
        # triple SE layer (h_dim, dataset, comp_op, bn, device)
        self.comp_layers = nn.ModuleList([CompLayer(h_dim, dataset, comp_op, bn, device) for _ in range(self.kg_n_layer)])

        self.ent_drop = nn.Dropout(ent_drop)
        self.rel_drop = nn.Dropout(rel_drop)
        self.act = nn.Tanh()

    def forward(self, kg, init_ent_emb, init_rel_emb):
        """
        aggregate embedding.
        :param kg:
        :param init_ent_emb: 传入的是实体embeddings
        :param init_rel_emb: 传入的是关系embeddings列表[h_0, h_0]
        :return:
        """
        # rel_embs = nn.ParameterList(init_rel_emb)
        rel_embs = init_rel_emb
        ent_emb = init_ent_emb
        rel_emb_list = []
        for edge_layer, node_layer, comp_layer, rel_emb in zip(self.edge_layers, self.node_layers, self.comp_layers, rel_embs):
            ent_emb, rel_emb = self.ent_drop(ent_emb), self.rel_drop(rel_emb)
            edge_ent_emb = edge_layer(kg, ent_emb, rel_emb)
            node_ent_emb = node_layer(kg, ent_emb)
            comp_ent_emb = comp_layer(kg, ent_emb, rel_emb)
            ent_emb = ent_emb + edge_ent_emb + node_ent_emb + comp_ent_emb
            rel_emb_list.append(rel_emb)

        return ent_emb


class CompLayer(nn.Module):
    def __init__(self, h_dim, dataset, comp_op, bn, device):
        super().__init__()
        # self.cfg = utils.get_global_config()
        self.device = device
        dataset = dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        self.comp_op = comp_op
        assert self.comp_op in ['add', 'mul']

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        # assert kg.number_of_nodes() == ent_emb.shape[0]
        # assert rel_emb.shape[0] == 2 * self.n_rel

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]
            # neihgbor entity and relation composition
            if self.comp_op == 'add':
                kg.apply_edges(fn.u_add_e('emb', 'emb', 'comp_emb'))
            elif self.comp_op == 'mul':
                kg.apply_edges(fn.u_mul_e('emb', 'emb', 'comp_emb'))
            else:
                raise NotImplementedError

            # attention
            kg.apply_edges(fn.e_dot_v('comp_emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])
            # agg
            kg.edata['comp_emb'] = kg.edata['comp_emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('comp_emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class NodeLayer(nn.Module):
    def __init__(self, h_dim, dataset, bn, device):
        super().__init__()
        # self.cfg = utils.get_global_config()
        self.device = device
        dataset = dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.neigh_w = get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb):
        # assert kg.number_of_nodes() == ent_emb.shape[0]

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb

            # attention
            kg.apply_edges(fn.u_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.update_all(fn.u_mul_e('emb', 'norm', 'm'), fn.sum('m', 'neigh'))
            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


class EdgeLayer(nn.Module):
    def __init__(self, h_dim, dataset, bn, device):
        super().__init__()
        # self.cfg = utils.get_global_config()
        self.device = device
        dataset = dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']

        self.neigh_w = utils.get_param(h_dim, h_dim)
        self.act = nn.Tanh()
        if bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        # assert kg.number_of_nodes() == ent_emb.shape[0]
        # assert rel_emb.shape[0] == 2 * self.n_rel

        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            rel_id = kg.edata['rel_id']
            kg.edata['emb'] = rel_emb[rel_id]

            # attention
            kg.apply_edges(fn.e_dot_v('emb', 'emb', 'norm'))  # (n_edge, 1)
            kg.edata['norm'] = dgl.ops.edge_softmax(kg, kg.edata['norm'])

            # agg
            kg.edata['emb'] = kg.edata['emb'] * kg.edata['norm']
            kg.update_all(fn.copy_e('emb', 'm'), fn.sum('m', 'neigh'))

            neigh_ent_emb = kg.ndata['neigh']

            neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)

            if callable(self.bn):
                neigh_ent_emb = self.bn(neigh_ent_emb)

            neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb
