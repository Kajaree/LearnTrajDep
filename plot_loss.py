#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

def plot_loss(losses, output_n, filename, title):
    dir = "checkpoint/plots"
    plt.figure()
    #plt.plot(losses[output_n[0]], 'o-', label='0.5 sec')
    plt.plot(losses[output_n[1]], 'co-', label='1 sec')
    plt.plot(losses[output_n[2]], 'r-', label='2 sec')
    plt.plot(losses[output_n[3]], 'y--', label='4 sec')
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('frames')
    plt.ylabel('loss')
    plt.tight_layout(.5)
    plt.legend(loc='lower right')
    plt.savefig("{}/{}.png".format(dir, filename), bbox_inches='tight')
    plt.show()


def plot_loss_per_action(losses, input_n, output_n):
    for act, act_loss in losses.items():
        title = 'Loss per frame for {} in Auto Regression \n Input - {} frames'.format(act, input_n)
        filename = "main_ar_errors_{}_{}_{}".format(act, input_n, output_n)
        plot_loss(act_loss, output_n, filename, title)



def calculate_loss(filename, frames):
    losses = []
    acts = []
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        loss = []
        for row in readCSV:
            if row[1] == 'frame':
                continue
            act = row[0]
            if len(loss) == frames:
                losses.append(loss)
                loss = []
            loss.append(float(row[2]))
            acts.append(act)
    losses = np.array(losses).astype(np.float)
    acts = np.unique(np.array(acts))
    return acts.tolist(), losses

def get_table(input_n, output_n, dct_n, filename='autoRegression'):
    acts = ['walking', 'eating', 'smoking', 'discussion', 'directions', 'greeting', 'phoning', 'posing', 'purchases',
     'sitting', 'sittingdown', 'takingphoto', 'waiting', 'walkingdog', 'walkingtogether']
    if output_n == 50:
        frames = [560, 1000, 2000]
    else:
        frames = [560, 1000, 2000, 4000]
    avg_losses = {frame:0 for frame in frames}
    dir = "checkpoint/test/"
    filename = dir + filename + "_in{:d}_out{:d}_dctn{:d}.csv".format(input_n, output_n, dct_n)
    readCSV = pd.read_csv(filename)

    for frame in frames:
        loss = 0
        headings = [act + '3d' + str(frame) for act in acts]
        for idx in range(len(headings)):
            loss += readCSV[headings[idx]][0]
        avg_losses[frame] = loss / len(acts)
    return avg_losses




def compare_loss():
    output_n = [50, 100]
    input_n = 25
    dir = "checkpoint/plots"
    plt.figure()
    for n in output_n:
        filename = 'checkpoint/errors/main_errors_{:d}_{:d}.csv'.format(input_n, n)
        filename_ar = 'checkpoint/errors/main_ar_errors_{:d}_{:d}.csv'.format(input_n, n)
        _, losses = calculate_loss(filename, n)
        _, losses_ar = calculate_loss(filename_ar, n)
        mean_losses = np.mean(losses, axis=0)
        mean_losses_ar = np.mean(losses_ar, axis=0)
        plt.plot(mean_losses, 'r-', label='direct')
        plt.plot(mean_losses_ar, 'k-', label='auto-regression')
        plt.yscale('log')
        plt.title('Direct vs Autoregression: \n Input - {} Output - {} frames'.format(input_n, n))
        plt.xlabel('frames')
        plt.ylabel('loss')
        plt.tight_layout(.5)
        plt.legend(loc='lower right')
        plt.savefig("{}/{}.png".format(dir, 'dva_{:d}_{}'.format(input_n, n)), bbox_inches='tight')
        plt.show()



if __name__ == "__main__":
    output_n = [50, 100]
    input_n = [10, 25]
    filenames = ['main_eval', 'autoRegression']
    '''avg_losses = dict()
    acts = None
    for n in output_n:
        if input_n <= n:
            filename = 'checkpoint/errors/main_errors_{:d}_{:d}.csv'.format(input_n, n)
            acts, losses = calculate_loss(filename, n)
            mean_losses = np.mean(losses, axis=0)
            avg_losses[n] = mean_losses.tolist()
    #plot_loss_per_action(acts_losses, input_n, output_n)
    plot_loss(avg_losses, output_n, 'main_avg_errors_input_{:d}'.format(input_n),
              'Average Loss per frame: Direct Approach \n Input - {} frames'.format(input_n))
    compare_loss()'''
    for m in input_n:
        for n in output_n:
            for filename in filenames:
                if filename == 'main_eval':
                    dct_n = m + n
                else:
                    dct_n = m * 2
                losses = get_table(m, n, dct_n, filename)
                print('3d losses for {}: \n input: {},  output: {} \n {}'.format(filename, m, n, losses))


