#plots a random file in AFs_fixed and TVs_fixed_new/ 

import numpy as np
import matplotlib.pyplot as plt
import os
import random


directories = ['../data/XRMB_original_delimited/AFs_wav_files_delimited/', "../data/XRMB_original_delimited/TVs_Ahmed\'s_x-constriction/"]
fixed_directories = ['../data/XRMB_original_delimited/AFs_wav_files_delimited_test_fixed//', "../data/XRMB_original_delimited/TVs_new_test_fixed//"]
TV_names = ["LA", "LP", "TBCL", "TBCD", "TTCL", "TTCD"]
AFs_names  = ["UL", "LL", "T1", "T2", "T3", "T4", " MNI", " MNM"]
file_list = os.listdir(directories[1])
plt_dir = '../data/UWXRMB/trimmed/AFs_fixed_plots/'
if not os.path.exists(plt_dir):
    os.mkdir(plt_dir)
for file in file_list:
    afs = np.load(directories[0] + file)
    tvs = np.load(directories[1] + file)
    fixed_afs = np.load(fixed_directories[0] + file)
    fixed_tvs = np.load(fixed_directories[1] + file)
    #plot the AFs
    for i in range(8):
        plt.subplot(8,1,i+1)
        plt.plot(afs[:,i, 0], label = "x")
        plt.plot(afs[:,i, 1], label = "y")
        plt.plot(fixed_afs[:,i, 0], label = "fixed x")
        plt.plot(fixed_afs[:,i, 1], label = "fixed y")
        plt.title(f"{AFs_names[i]}")
    plt.legend()
    
    plt.savefig(plt_dir + file + f"_AF.png")
    plt.close()
    #plot the TVs
    for i in range(6):
        plt.subplot(6,1,i+1)
        plt.plot(tvs[:,i], label = "original")
        plt.plot(fixed_tvs[:,i], label = "fixed")
        plt.title(f"{TV_names[i]}")
    plt.legend()
    plt.savefig(plt_dir + file + f"_TV.png")