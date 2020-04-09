#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import csv
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional
import numpy as np
from progress.bar import Bar
import pandas as pd
from matplotlib import pyplot as plt

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.h36motion import H36motion
import utils.model as nnmodel
import utils.data_utils as data_utils
import utils.viz as viz


def save_loss_file(act, pred_expmap, targ_expmap, output_n):
    start = 0
    errors = []
    head = ['act', 'frame', 'error']
    filename = 'checkpoint/logs/main_ar_errors_{:d}.csv'.format(output_n)
    for id in range(output_n):
        err = np.linalg.norm(targ_expmap[:, start:id, :] - pred_expmap[:, start:id, :])
        #err = torch.sum((pred_expmap - targ_expmap)**2)
        start = id
        errors.append([act, id, err])

    with open(filename, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(head)
        writer.writerows(errors)
    return filename

def main(opt, stepsize=10):
    is_cuda = torch.cuda.is_available()
    desired_acts = ['eating', 'posing', 'sitting', 'posing', 'walkingdog']
    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    sample_rate = opt.sample_rate
    model = nnmodel.GCN(input_feature=(input_n + stepsize), hidden_feature=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=48)
    if is_cuda:
        model.cuda()
    model_path_len = "checkpoint/pretrained/h36m_in10_out10_dctn20.pth.tar"
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    if is_cuda:
        ckpt = torch.load(model_path_len)
    else:
        ckpt = torch.load(model_path_len, map_location='cpu')
    err_best = ckpt['err']
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # data loading
    print(">>> loading data")
    acts = data_utils.define_actions('all')
    test_data = dict()
    for act in acts:
        test_dataset = H36motion(path_to_data=opt.data_dir, actions=act, input_n=input_n, output_n=output_n,
                                 dct_n=(stepsize + input_n), split=1, sample_rate=sample_rate)
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
    dim_used = test_dataset.dim_used
    print(">>> data loaded !")
    model.eval()
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    #filename = None
    iterations = int(output_n / stepsize)
    print('iterations: {}'.format(iterations))
    for act in acts:
        for i, (inputs, targets, all_seq) in enumerate(test_data[act]):
            inputs = Variable(inputs).float()
            print(inputs.shape)
            all_seq = Variable(all_seq).float()
            dim_used_len = len(dim_used)
            if is_cuda:
                inputs = inputs.cuda()
                all_seq = all_seq.cuda()
            outputs = None
            sLen = input_n + stepsize
            dct_m_in, _ = data_utils.get_dct_matrix(sLen)
            dct_m_in = Variable(torch.from_numpy(dct_m_in)).float().cuda()
            _, idct_m = data_utils.get_dct_matrix(sLen)
            idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
            for idx in range(iterations):
                print('index: {}'.format(idx))
                if idx > 0:
                    input_dct_seq = inputs
                else:
                    start = input_n + idx * stepsize
                    stop = start + stepsize
                    input_seq = torch.cat((y_hat, all_seq[:, start:stop, dim_used]), 1)
                    input_dct_seq = torch.matmul(dct_m_in, input_seq).transpose(1, 2)
                    if is_cuda:
                        input_dct_seq = input_dct_seq.cuda()
                y = model(input_dct_seq)
                y_t = y.view(-1, sLen).transpose(0, 1)
                y_exp = torch.matmul(idct_m, y_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                        sLen).transpose(1, 2)
                y_hat = y_exp[:, input_n:, :]
                if idx == 0:
                    outputs = y_exp
                else:
                    outputs = torch.cat((outputs, y_hat), 1)
            pred_expmap = all_seq.clone()
            dim_used = np.array(dim_used)
            pred_expmap[:, :, dim_used] = outputs
            pred_expmap = pred_expmap.cpu().data.numpy()
            targ_expmap = all_seq.cpu().data.numpy()
            #save_loss_file(act, pred_expmap, targ_expmap, output_n)
            if act in desired_acts:
                for k in range(8):
                    plt.cla()
                    figure_title = "action:{}, seq:{},".format(act, (k + 1))
                    viz.plot_predictions(targ_expmap[k, :, :], pred_expmap[k, :, :], fig, ax, figure_title)
                    plt.pause(1)


if __name__ == "__main__":
    option = Options().parse()
    main(option)