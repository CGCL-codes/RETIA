import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import random
import scipy.sparse as sp
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph, get_relhead_reltal, build_super_g
from src.rrgcn import RecurrentRGCN
import torch.nn.modules.rnn
import copy

import warnings
warnings.filterwarnings(action='ignore')

def temporal_regularization(params1, params2):
    regular = 0
    for (param1, param2) in zip(params1, params2):
        regular += torch.norm(param1 - param2, p=2)
    # print(regular)
    return regular

def continual_test(model, history_list, data_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, mode):
    """
    用于online setting
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    # load pretrained model which valid in the whole valid data
    start_idx = len(history_list)
    # if test, load the model fine tuned at the last timestamp at the valid dataset
    # else load the pretrained model.
    # if mode=="test":
    #     model_name = "{}-{}".format(model_name, start_idx-1)
    if not os.path.exists(model_name):
        print("Train the model first before continual learning...")
        sys.exit()
    else:
        if mode == "test":
            checkpoint = torch.load(
                "{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, start_idx - 1),
                map_location=torch.device(args.gpu))
            init_checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            # print(model_name)
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
            init_checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        print("Load pretrain model: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n" + "-" * 10 + "start continual learning" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])
        # save an init model for analysis
        model_initial = copy.deepcopy(model)
        model_initial.load_state_dict(init_checkpoint['state_dict'])
        model_initial.eval()
        # parameter for the temporal normalize at the first timestamp
        previous_param = [param.detach().clone() for param in model.parameters()]

        model.eval()
        epoch = checkpoint['epoch']

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.ft_lr, weight_decay=0)

    valid_input_list = [snap for snap in history_list[-args.test_history_len - 2:-2]]  # history for ft training (, tc-2)
    valid_snap = history_list[-2]  # snapshot for ft training at snapshot at tc-2
    ft_input_list = [snap for snap in history_list[-args.test_history_len - 1:-1]]  # history for ft validation (,tc-1)
    ft_snap = history_list[-1]  # snapshot for ft validation snapshot at tc-1
    test_input_list = [snap for snap in history_list[-args.test_history_len:]]  # history for testing (, tc)

    # starting continual learning
    for time_idx, test_snap in enumerate(data_list):

        if args.dataset == 'YAGO' or args.dataset == 'WIKI':
            pass
        else:
            if time_idx - 2 - args.test_history_len < 0:
                history_list_temp = [snap for snap in history_list[time_idx - 2 - args.test_history_len:]] + [snap for snap in data_list[0: time_idx]]  # history for ft training (, tc-2)
                valid_input_list = [snap for snap in history_list_temp[-args.test_history_len - 2:-2]]  # -1 0 1 2
                valid_snap = history_list_temp[-2]
                ft_input_list = [snap for snap in history_list_temp[-args.test_history_len - 1:-1]]
                ft_snap = history_list_temp[-1]
                test_input_list = [snap for snap in history_list_temp[-args.test_history_len:]]
            else:
                history_list_temp = [snap for snap in data_list[time_idx - 2 - args.test_history_len: time_idx]]
                valid_input_list = [snap for snap in history_list_temp[-args.test_history_len - 2:-2]]  # history for ft training (, tc-2)
                valid_snap = history_list_temp[-2]
                ft_input_list = [snap for snap in history_list_temp[-args.test_history_len - 1:-1]]
                ft_snap = history_list_temp[-1]
                test_input_list = [snap for snap in history_list_temp[-args.test_history_len:]]

        tc = start_idx + time_idx
        print("-----------------------{}-----------------------".format(tc))
        # step 1: get the history graphs for ft training : ft_input_list -> ft_snapshot
        ft_history_super_glist = []
        for ft_sub_g in ft_input_list:
            ft_rel_head, ft_rel_tail = get_relhead_reltal(ft_sub_g, num_nodes, num_rels)
            ft_super_sub_g = build_super_g(num_rels, ft_rel_head, ft_rel_tail, use_cuda, args.gpu)
            ft_history_super_glist.append(ft_super_sub_g)
        ft_history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in ft_input_list]
        # ft_tensor = torch.LongTensor(ft_snap).cuda() if use_cuda else torch.LongTensor(ft_snap)
        # ft_tensor = ft_tensor.to(args.gpu)
        ft_output_tensor = [torch.from_numpy(_).long().cuda() for _ in test_input_list] if use_cuda else [torch.from_numpy(_).long() for _ in test_input_list]

        # step 2: get the history graphs for ft validation : valid_input_list -> valid_snap
        valid_history_super_glist = []
        for valid_sub_g in valid_input_list:
            valid_rel_head, valid_rel_tail = get_relhead_reltal(valid_sub_g, num_nodes, num_rels)
            valid_super_sub_g = build_super_g(num_rels, valid_rel_head, valid_rel_tail, use_cuda, args.gpu)
            valid_history_super_glist.append(valid_super_sub_g)
        valid_history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in valid_input_list]
        valid_tensor = torch.LongTensor(valid_snap).cuda() if use_cuda else torch.LongTensor(valid_snap)
        valid_tensor = valid_tensor.to(args.gpu)

        # step 2: prepare inputs for test
        test_history_super_glist = []
        for test_sub_g in test_input_list:
            test_rel_head, test_rel_tail = get_relhead_reltal(test_sub_g, num_nodes, num_rels)
            test_super_sub_g = build_super_g(num_rels, test_rel_head, test_rel_tail, use_cuda, args.gpu)
            test_history_super_glist.append(test_super_sub_g)
        test_history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in test_input_list]
        test_tensor = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_tensor = test_tensor.to(args.gpu)

        # result of the pre-trained model on validation set (tc-1)
        _, final_score, final_r_score = model.predict(valid_history_glist, valid_history_super_glist, num_rels, valid_tensor, use_cuda)
        mrr_filter_valid_snap_r, mrr_valid_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(valid_tensor, final_r_score, all_ans_r_list[tc - 2], eval_bz=1000, rel_predict=1)
        mrr_filter_valid_snap, mrr_valid_snap, rank_raw, rank_filter = utils.get_total_rank(valid_tensor, final_score, all_ans_list[tc - 2], eval_bz=1000, rel_predict=0)

        # result of the pre-trained model on test set (tc)
        _, final_score, final_r_score = model_initial.predict(test_history_glist, test_history_super_glist, num_rels, test_tensor, use_cuda)
        mrr_filter_test_snap_r, mrr_test_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_tensor, final_r_score, all_ans_r_list[tc], eval_bz=1000, rel_predict=1)
        mrr_filter_test_snap, mrr_test_snap, rank_raw, rank_filter = utils.get_total_rank(test_tensor, final_score, all_ans_list[tc], eval_bz=1000, rel_predict=0)
        print("Pretrained Model : test entity mrr ", mrr_test_snap)
        print("Pretrained Model : test relation mrr ", mrr_test_snap_r)

        # result of the last step fine-tuned model on test set (tc)
        _, final_score, final_r_score = model.predict(test_history_glist, test_history_super_glist, num_rels, test_tensor, use_cuda)
        mrr_filter_test_snap_r, mrr_test_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_tensor, final_r_score, all_ans_r_list[tc], eval_bz=1000, rel_predict=1)
        mrr_filter_test_snap, mrr_test_snap, rank_raw, rank_filter = utils.get_total_rank(test_tensor, final_score, all_ans_list[tc], eval_bz=1000, rel_predict=0)
        print("Continual Model : test entity mrr before ft ", mrr_test_snap)
        print("Continual Model : test relation mrr before ft ", mrr_test_snap_r)

        # init mrr for validation
        if args.dataset == 'YAGO' or args.dataset == 'WIKI' or args.dataset == 'ICEWS14':
            best_mrr = mrr_valid_snap * args.task_weight + mrr_valid_snap_r
        else:
            best_mrr = mrr_valid_snap * args.task_weight + mrr_valid_snap_r * (1 - args.task_weight)
        # best_mrr = mrr_valid_snap_r

        ft_epoch, losses = 0, []

        while ft_epoch < args.ft_epochs:
            model.train()
            loss_ent, loss_rel = model.get_ft_loss(ft_history_glist, ft_history_super_glist, ft_output_tensor, use_cuda)
            if args.dataset == 'YAGO' or args.dataset == 'WIKI' or args.dataset == 'ICEWS14':
                loss = loss_ent * args.task_weight + loss_rel
            else:
                loss = loss_ent * args.task_weight + loss_rel * (1 - args.task_weight)
            loss_norm = temporal_regularization(model.parameters(), previous_param)

            loss += args.norm_weight * loss_norm

            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
            optimizer.step()
            optimizer.zero_grad()

            model.eval()

            # validation on tc-1 snapshot
            _, final_score, final_r_score = model.predict(valid_history_glist, valid_history_super_glist, num_rels, valid_tensor, use_cuda)
            mrr_filter_valid_snap_r, mrr_valid_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(valid_tensor, final_r_score, all_ans_r_list[tc - 2], eval_bz=1000, rel_predict=1)
            mrr_filter_valid_snap, mrr_valid_snap, rank_raw, rank_filter = utils.get_total_rank(valid_tensor, final_score, all_ans_list[tc - 2], eval_bz=1000, rel_predict=0)

            # update best_mrr
            ft_epoch += 1

            if args.dataset == 'YAGO' or args.dataset == 'WIKI' or args.dataset == 'ICEWS14':
                mrr_valid = mrr_valid_snap * args.task_weight + mrr_valid_snap_r
            else:
                mrr_valid = mrr_valid_snap * args.task_weight + mrr_valid_snap_r * (1 - args.task_weight)
            if mrr_valid >= best_mrr:
                print("model updated")
                best_mrr = mrr_valid
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, "{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, tc), _use_new_zipfile_serialization = False)
            else:
                if not os.path.exists("{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, tc)):
                    print("copy model at {}".format(tc - 1))
                    if mode == "valid" and time_idx == 0:
                        checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
                    else:
                        checkpoint = torch.load(
                            "{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, tc - 1),
                            map_location=torch.device(args.gpu))
                    model.load_state_dict(checkpoint['state_dict'])
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                               "{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, tc), _use_new_zipfile_serialization = False)
                    if ft_epoch > 3:
                        break
                else:
                    print("exist best model, skip save")
                    break

        # save the best parameter in model-tc
        previous_param = [param.detach().clone() for param in model.parameters()]
        # ---------------start evaluate test snaoshot---------------

        # step 1: load current model
        checkpoint = torch.load("{}-flr{}-norm{}-{}".format(model_name, args.ft_lr, args.norm_weight, tc), map_location=torch.device(args.gpu))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        # step 3: start test
        _, final_score, final_r_score = model.predict(test_history_glist, test_history_super_glist, num_rels, test_tensor, use_cuda)
        # step 4: evaluation
        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_tensor, final_r_score, all_ans_r_list[tc], eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_tensor, final_score, all_ans_list[tc], eval_bz=1000, rel_predict=0)
        print("Continual Model : ***test mrr*** ", mrr_filter_snap)

        # step 5: update history glist and prepare inputs
        ft_input_list.pop(0)
        ft_input_list.append(ft_snap)
        valid_input_list.pop(0)
        valid_input_list.append(valid_snap)
        test_input_list.pop(0)
        test_input_list.append(test_snap)

        valid_snap = ft_snap.copy()
        ft_snap = test_snap.copy()

        # step 6: save results
        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)

        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r

def test(model, history_len, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name, mode):
    '''
    :param model:
    :param history_len:
    :param history_list: valid传入的是train; test传入的是train+valid (按时间戳划分的事实(内部array, 外层list)：[[[s, r, o], [], ...], [], ...])
    :param test_list: valid传入的是valid; test传入的是test
    :param num_rels: 边（关系集）的数量，不包括inverse关系
    :param num_nodes: 实体集数量
    :param use_cuda:
    :param all_ans_list:
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_len: 静态图的所有时间戳数目
    :param mode:
    :return:
    '''
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # num_rel_2 = num_rels * 2 # 图中的边的数目

    # test_snap: 对于(valid或者test集中)一个时间戳内的所有事实三元组；time_idx从0开始 一个时间戳一个时间戳地处理
    for time_idx, test_snap in enumerate(tqdm(test_list)): # 对于每一个待测试的时间戳 snapshot_list，按时间戳划分的事实array：[[[s, r, o], [], ...], [], ...]
        # input_list = [snap for snap in history_list[-args.test_history_len:]]
        # 计算当前时间戳的历史子图序列
        if time_idx - history_len < 0:
            input_list = [snap for snap in history_list[time_idx - history_len:]] + [snap for snap in test_list[0: time_idx]]
        else:
            input_list = [snap for snap in test_list[time_idx - history_len: time_idx]]

        # input_list: 按时间戳划分的历史子图序列, 内层array, 外层list, [[[s, r, o], [], ...], [], ...]
        # 聚合关系的RGCN，构造关系超图的DGL对象，同样组织为DGL对象的列表
        history_super_glist = []
        for sub_g in input_list:
            rel_head, rel_tail = get_relhead_reltal(sub_g, num_nodes, num_rels)
            super_sub_g = build_super_g(num_rels, rel_head, rel_tail, use_cuda, args.gpu)
            history_super_glist.append(super_sub_g)

        # 聚合实体的RGCN，返回一个DGL对象列表
        # num_nodes：实体集大小；num_rels：关系集大小；g：array时序子图中的所有事实三元组[[s, r, o], [], ...]
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]

        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu) # 一个时间戳内的所有事实三元组

        # (tensor)all_triples: (batch_size, 3); (tensor)score: (batch_size, num_ents); (tensor)score_r: (batch_size, num_rel*2)
        test_triples, final_score, final_r_score = model.predict(history_glist, history_super_glist, num_rels, test_triples_input, use_cuda)
        # 每一个时间戳内的事实三元组再按照batch_size进行指标的计算
        # test_triples: 一个时间戳内的所有事实三元组, 包含反关系(tensor)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        input_list.pop(0)
        input_list.append(test_snap)
        idx += 1
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r

def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset) # data.train, data.valid, data.test: np.array([[s, r, o, time], []...])
    train_list = utils.split_by_time(data.train) # 列表，snapshot_list，按时间戳划分的事实array：[[[s, r, o], [], ...], [], ...]
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test) # snapshot_list，按时间戳划分的事实array：[[[s, r, o], [], ...], [], ...]

    num_nodes = data.num_nodes # 整个数据集的节点数目
    num_rels = data.num_rels # 整个数据集的关系数目

    # for time-aware filtered evaluation
    all_ans_list_test_time_filter = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False) # data.test: np.array([[s, r, o, time], []...])
    all_ans_list_valid_time_filter = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_test_time_filter = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_r_valid_time_filter = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    # for time-aware filtered evaluation in online setting
    total_data = np.concatenate((data.train, data.valid, data.test), axis=0)
    all_ans_list_online = utils.load_all_answers_for_time_filter(total_data, num_rels, num_nodes, False)
    all_ans_list_r_online = utils.load_all_answers_for_time_filter(total_data, num_rels, num_nodes, True)

    test_model_name = "{}-{}-{}-ly{}-his{}-dp{}|{}|{}|{}-gpu{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.test_history_len,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu)
    if not os.path.exists('../models/{}/'.format(args.dataset)):
        os.makedirs('../models/{}/'.format(args.dataset))
    test_state_file = '../models/{}/{}'.format(args.dataset, test_model_name)
    print("Sanity Check: stat name : {}".format(test_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    # create stat
    model = None
    if args.dataset == "ICEWS14":
        model = RecurrentRGCN(args.decoder,
                              args.encoder,
                              num_nodes,
                              num_rels,
                              args.n_hidden,
                              args.opn,
                              sequence_len=10,
                              num_bases=args.n_bases,
                              num_basis=args.n_basis,
                              num_hidden_layers=args.n_layers,
                              dropout=args.dropout,
                              self_loop=args.self_loop,
                              skip_connect=args.skip_connect,
                              layer_norm=args.layer_norm,
                              input_dropout=args.input_dropout,
                              hidden_dropout=args.hidden_dropout,
                              feat_dropout=args.feat_dropout,
                              aggregation=args.aggregation,
                              use_cuda=use_cuda,
                              gpu = args.gpu,
                              analysis=args.run_analysis)
    elif args.dataset == 'YAGO' or 'WIKI':
        model = RecurrentRGCN(args.decoder,
                              args.encoder,
                              num_nodes,
                              num_rels,
                              args.n_hidden,
                              args.opn,
                              sequence_len=3,
                              num_bases=args.n_bases,
                              num_basis=args.n_basis,
                              num_hidden_layers=args.n_layers,
                              dropout=args.dropout,
                              self_loop=args.self_loop,
                              skip_connect=args.skip_connect,
                              layer_norm=args.layer_norm,
                              input_dropout=args.input_dropout,
                              hidden_dropout=args.hidden_dropout,
                              feat_dropout=args.feat_dropout,
                              aggregation=args.aggregation,
                              use_cuda=use_cuda,
                              gpu=args.gpu,
                              analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test_valid and os.path.exists(test_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = continual_test(model,
                                                                      train_list,
                                                                      valid_list,
                                                                      num_rels,
                                                                      num_nodes,
                                                                      use_cuda,
                                                                      all_ans_list_online,
                                                                      all_ans_list_r_online,
                                                                      test_state_file,
                                                                      "valid")
    elif args.test_test and os.path.exists(test_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = continual_test(model,
                                                                      train_list+valid_list,
                                                                      test_list,
                                                                      num_rels,
                                                                      num_nodes,
                                                                      use_cuda,
                                                                      all_ans_list_online,
                                                                      all_ans_list_r_online,
                                                                      test_state_file,
                                                                      "test")
    elif args.test and os.path.exists(test_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                            args.test_history_len,
                                                            train_list+valid_list,
                                                            test_list,
                                                            num_rels,
                                                            num_nodes,
                                                            use_cuda,
                                                            all_ans_list_test_time_filter,
                                                            all_ans_list_r_test_time_filter,
                                                            test_state_file,
                                                            "test")
    elif args.cur_train:
        # load best model with start history length
        init_state_file = '../models/{}/'.format(args.dataset) + "{}-{}-{}-ly{}-his{}-dp{}|{}|{}|{}-gpu{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.train_history_len,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu)
        init_checkpoint = torch.load(init_state_file, map_location=torch.device(args.gpu))
        print("Load Previous Model name: {}. Using best epoch : {}".format(init_state_file, init_checkpoint[
            'epoch']))  # use best stat checkpoint
        print("\n" + "-" * 10 + "Load model with history length {}".format(args.train_history_len) + "-" * 10 + "\n")
        model.load_state_dict(init_checkpoint['state_dict'])
        test_history_len = args.train_history_len
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                            args.train_history_len,
                                                            train_list + valid_list,
                                                            test_list,
                                                            num_rels,
                                                            num_nodes,
                                                            use_cuda,
                                                            all_ans_list_test_time_filter,
                                                            all_ans_list_r_test_time_filter,
                                                            init_state_file,
                                                            "test")
        mrr_raw_ent_rel = mrr_raw * args.task_weight + mrr_raw_r
        best_mrr_list = [mrr_raw_ent_rel.item()]
        # start knowledge distillation
        ks_idx = 0
        for history_len in range(args.train_history_len + 1, 10 + 1, 1):
            # current model
            # print("best mrr list :", best_mrr_list)
            # lr = 0.1*args.lr - 0.002*args.lr*ks_idx
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1 * args.lr, weight_decay=0.00001)
            model_name = "{}-{}-{}-ly{}-his{}-dp{}|{}|{}|{}-gpu{}"\
                .format(args.dataset, args.encoder, args.decoder, args.n_layers, history_len,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu)
            model_state_file = '../models/{}/'.format(args.dataset) + model_name
            print("Sanity Check: stat name : {}".format(model_state_file))

            # load model with the least history length
            prev_model_name = "{}-{}-{}-ly{}-his{}-dp{}|{}|{}|{}-gpu{}"\
                .format(args.dataset, args.encoder, args.decoder, args.n_layers, history_len-1,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu)
            prev_state_file = '../models/{}/'.format(args.dataset) + prev_model_name
            checkpoint = torch.load(prev_state_file, map_location=torch.device(args.gpu))
            # print("Load Previous Model name: {}. Using best epoch : {}".format(prev_model_name, checkpoint['epoch']))  # use best stat checkpoint
            model.load_state_dict(checkpoint['state_dict'])
            # prev_model = copy.deepcopy(model)
            # prev_model.eval()
            print("\n" + "-" * 10 + "start knowledge distillation for history length at " + str(history_len) + "-" * 10 + "\n")

            best_mrr = 0
            best_epoch = 0
            for epoch in range(args.n_epochs):
                model.train()
                losses = []
                # losses_e = []
                # losses_r = []

                idx = [_ for _ in range(len(train_list))]
                random.shuffle(idx)
                for train_sample_num in idx:
                    if train_sample_num == 0 or train_sample_num == 1: continue
                    if train_sample_num - history_len < 0:
                        input_list = train_list[0: train_sample_num]
                        output = train_list[1:train_sample_num + 1]
                    else:
                        input_list = train_list[train_sample_num - history_len: train_sample_num]
                        output = train_list[train_sample_num - history_len + 1:train_sample_num + 1]

                    # generate super history graph
                    history_super_glist = []
                    for sub_g in input_list:
                        rel_head, rel_tail = get_relhead_reltal(sub_g, num_nodes, num_rels)
                        super_sub_g = build_super_g(num_rels, rel_head, rel_tail, use_cuda, args.gpu)
                        history_super_glist.append(super_sub_g)

                    # generate history graph
                    history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                    output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]

                    loss_e, loss_r = model.get_loss(history_glist, history_super_glist, output[0], use_cuda)
                    # print(loss)
                    loss = args.task_weight*loss_e + (1-args.task_weight)*loss_r
                    losses.append(loss.item())

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()

                print("His {:04d}, Epoch {:04d} | Ave Loss: {:.4f} |Best MRR {:.4f} | Model {} ".format(history_len, epoch, np.mean(losses), best_mrr, model_name))

                # validation
                if epoch % args.evaluate_every == 0:
                    mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                                        history_len,
                                                                        train_list,
                                                                        valid_list,
                                                                        num_rels,
                                                                        num_nodes,
                                                                        use_cuda,
                                                                        all_ans_list_valid_time_filter,
                                                                        all_ans_list_r_valid_time_filter,
                                                                        model_state_file,
                                                                        mode="train")
                    mrr_raw_ent_rel = mrr_raw * args.task_weight + mrr_raw_r
                    if mrr_raw_ent_rel < best_mrr:
                        if epoch >= args.n_epochs or epoch - best_epoch > 2:
                            break
                    else:
                        best_mrr = mrr_raw_ent_rel
                        best_epoch = epoch
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                                history_len,
                                                                train_list,
                                                                valid_list,
                                                                num_rels,
                                                                num_nodes,
                                                                use_cuda,
                                                                all_ans_list_valid_time_filter,
                                                                all_ans_list_r_valid_time_filter,
                                                                model_state_file,
                                                                mode="test")
            ks_idx += 1
            mrr_raw_ent_rel = mrr_raw * args.task_weight + mrr_raw_r
            if mrr_raw_ent_rel < max(best_mrr_list):
                test_history_len = history_len - 1
                break
            else:
                best_mrr_list.append(mrr_raw_ent_rel)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                            test_history_len,
                                                            train_list + valid_list,
                                                            test_list,
                                                            num_rels,
                                                            num_nodes,
                                                            use_cuda,
                                                            all_ans_list_test_time_filter,
                                                            all_ans_list_r_test_time_filter,
                                                            prev_state_file,
                                                            mode="test")
    else:
        print("----------------------------------------start training----------------------------------------\n")
        model_name = "{}-{}-{}-ly{}-his{}-dp{}|{}|{}|{}-gpu{}" \
            .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.train_history_len,
                    args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu)
        if not os.path.exists('../models/{}/'.format(args.dataset)):
            os.makedirs('../models/{}/'.format(args.dataset))
        model_state_file = '../models/{}/{}'.format(args.dataset, model_name)
        print("Sanity Check: stat name : {}".format(model_state_file))
        print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

        best_mrr = 0
        best_epoch = 0
        for epoch in range(args.n_epochs): # 对于每一个epoch
            model.train()
            # 两个不同的loss
            losses = []
            losses_e = []
            losses_r = []

            idx = [_ for _ in range(len(train_list))] # train_list: 列表，每一个时间戳内包含的所有事实; idx: 顺序进行的时间戳
            random.shuffle(idx) # 将包含训练集所有时间戳的idx列表打乱

            # 按照时间戳进行训练
            for train_sample_num in tqdm(idx): # 对于每一个时间戳（idx编号是打乱的，但是还对应相应的第几个时间戳）
                if train_sample_num == 0: continue # 第0个时间戳没有历史信息

                # 往下走的必定不是第0个时间戳
                output = train_list[train_sample_num:train_sample_num+1] # 取出当前时间戳下的所有事实array [[s, r, o], [], ...]
                if train_sample_num - args.train_history_len<0: # 当前时间戳的历史深度不够train_history_len
                    input_list = train_list[0: train_sample_num] # 历史信息从第0个时间戳取出来
                else:
                    input_list = train_list[train_sample_num - args.train_history_len: train_sample_num] # 历史信息取前train_history_len个时间戳

                # 聚合关系的RGCN，构造关系超图的DGL对象，同样组织为DGL对象的列表
                history_super_glist = []
                for sub_g in input_list:
                    rel_head, rel_tail = get_relhead_reltal(sub_g, num_nodes, num_rels)
                    super_sub_g = build_super_g(num_rels, rel_head, rel_tail, use_cuda, args.gpu)
                    history_super_glist.append(super_sub_g)

                # 聚合实体的RGCN，返回一个DGL对象列表
                # generate history graph; input_list: [[[s, r, o], [], ...], [], ...] 针对当前时间戳, 前train_history_len个时间戳的历史信息
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                # 对于当前时间戳下的每一个事实三元组
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
                # history_glist：历史子图列表；output[0]：当前时间戳下的所有事实
                loss_e, loss_r = model.get_loss(history_glist, history_super_glist, output[0], use_cuda)
                loss = args.task_weight*loss_e + (1-args.task_weight)*loss_r # 0.7

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss:{:.4f} | entity-relation:{:.4f}-{:.4f} Best MRR {:.4f} | Model {} ".format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                                    args.train_history_len,
                                                                    train_list, # 列表，snapshot_list，按时间戳划分的事实array：[[[s, r, o], [], ...], [], ...]
                                                                    valid_list, # [[[s, r, o], [], ...], [], ...]
                                                                    num_rels,
                                                                    num_nodes,
                                                                    use_cuda,
                                                                    all_ans_list_valid_time_filter,
                                                                    all_ans_list_r_valid_time_filter,
                                                                    model_state_file,
                                                                    mode="train")
                # entity&relation prediction evalution
                if args.dataset == 'YAGO' or args.dataset == 'WIKI' or args.dataset == 'ICEWS14':
                    mrr_raw_ent_rel = mrr_raw * args.task_weight + mrr_raw_r
                else:
                    mrr_raw_ent_rel = mrr_raw * args.task_weight + mrr_raw_r * (1 - args.task_weight)
                if mrr_raw_ent_rel < best_mrr:
                    if epoch >= args.n_epochs or epoch - best_epoch > 5:
                        break
                else:
                    best_mrr = mrr_raw_ent_rel
                    best_epoch = epoch
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                            args.train_history_len,
                                                            train_list+valid_list,
                                                            test_list,
                                                            num_rels,
                                                            num_nodes,
                                                            use_cuda,
                                                            all_ans_list_test_time_filter,
                                                            all_ans_list_r_test_time_filter,
                                                            model_state_file,
                                                            mode="test")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--cur-train", action='store_true', default=False,
                        help="curriculum training the train set")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--test-valid", action='store_true', default=False,
                        help="online train and test the valid set")
    parser.add_argument("--test-test", action='store_true', default=False,
                        help="online train and test the test set")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")

    # configuration for encoder RGCN stat
    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")
    parser.add_argument("--task-weight", type=float, default=1,
                        help="weight of entity prediction task")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=250,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--ft_epochs", type=int, default=30,
                        help="number of minimum fine-tuning epoch")
    parser.add_argument("--ft_lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--norm_weight", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")

    # configuration for optimal parameters
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")


    args = parser.parse_args()
    print(args)
    run_experiment(args)
    sys.exit()



