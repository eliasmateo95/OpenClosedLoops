import os
import sys 
import time
import random
import datetime
import itertools
import brian2 as b2
import pandas as pd
from brian2 import *
import seaborn as sns
import scipy.io as sio
from scipy import signal
from brian2tools import *
from random import randrange
import ipywidgets as widgets
import matplotlib.pyplot as plt
from statistics import variance
from collections import Counter
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors
from IPython.display import display, Image
from slack_sdk.webhook import WebhookClient
from ipywidgets import interact, interactive
from matplotlib.collections import PolyCollection
url = "https://hooks.slack.com/services/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"
webhook = WebhookClient(url)

class Struct:
    pass

def visualise(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

def rand_params(parameter, unit, n_cells, step):
    if n_cells == 1:
        return [parameter * unit]
    n1 = n_cells // 2
    n2 = n_cells - n1
    n_list = [n1, n2]
    np.random.shuffle(n_list)
    base = int(1 / step)
    start = int(base * parameter)
    begin = start - n_list[0]
    end = start + n_list[1]
    param_list = [x / float(base) for x in range(begin, end)]
    np.random.shuffle(param_list)
    if isinstance(unit, Unit):
        param_list = [x * unit for x in param_list]
    else:
        param_list = [x for x in param_list]

    return param_list

def cells_connected_to_noise(PC_DCN_Synapse_Targets,PC_DCN_Synapse_Sources,DCN_IO_Synapse_Targets,IO_PC_Synapse_Sources):
    IO_Cells_Connected = []
    for ii in range(0,size(PC_DCN_Synapse_Targets)):
        IO_Cells_Connected.append(DCN_IO_Synapse_Targets[PC_DCN_Synapse_Targets[ii]])
    DCN_Cells_Connected = PC_DCN_Synapse_Targets

    IO_Cell_to_show = []
    PC_Cell_to_show = []
    for ii in range(0,size(IO_PC_Synapse_Sources)):
        if IO_Cells_Connected[ii] == IO_PC_Synapse_Sources[ii]:
            IO_Cell_to_show.append(IO_Cells_Connected[ii])
            PC_Cell_to_show.append(PC_DCN_Synapse_Sources[ii])
            
    return IO_Cell_to_show,PC_Cell_to_show,DCN_Cells_Connected,IO_Cells_Connected

def most_frequent(List):
    count = Counter(List)
    return count.most_common(1)[0][0]