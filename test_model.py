from enum import auto
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations
from metrics import *
from matplotlib.ticker import FormatStrFormatter
import os
import random
def prop_n_masked_afs(num_masked_afs_max, num_masked_afs, num_afs):
    k = num_afs
    n = num_masked_afs_max
    m = num_masked_afs
    
    p = []
    for i in range(m): 
        p.append((-1)**i * math.comb(m,i) * ((m-i)/k)**n)
    
    p = sum(p)
    p *= math.comb(k,m)
    return p

# def autolabel(bar_plot, bar_label):
#     for idx,rect in enumerate(bar_plot):
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#                 bar_label[idx],
#                 ha='center', va='bottom', rotation=0)

def test_model(model, afs_test, model_dir):
    afs_names = ["UL", "LL", "T1", "T2", "T3","T4", "MNI", "MNM"]
    num_masked_afs_max = model_dir.split("_")[-2]
    corr_per_num_masked_afs_max_x = []
    corr_per_num_masked_afs_max_y = []
    bar_prop = []

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    for num_afs_to_test in range(8):
        avg_corrs_x = []
        avg_corrs_y = []
        hundreds = np.ones((afs_test.shape[0], afs_test.shape[1],num_afs_to_test+1, 2)) * 100
        for afs in combinations(range(8), num_afs_to_test+1):
            model_input = afs_test.copy()
            model_input[:, : ,afs, :] = hundreds
            out = model.predict(model_input, verbose = 0)
            corr_afs_avg, avg_corr_afs_x = compute_corr_score(out[:, : ,afs, 0], afs_test[:, : ,afs, 0])
            corr_afs_avg, avg_corr_afs_y = compute_corr_score(out[:, : ,afs, 1], afs_test[:, : ,afs, 1])
            avg_corrs_x.append(avg_corr_afs_x)
            avg_corrs_y.append(avg_corr_afs_y)
        # bar_prop.append(prop_n_masked_afs(num_masked_afs_max, num_afs_to_test, 6))
        corr_per_num_masked_afs_max_x.append(np.mean(avg_corrs_x))
        corr_per_num_masked_afs_max_y.append(np.mean(avg_corrs_y))

    training_log = f"\nCorrelation on X axis per number of masked AFs: {corr_per_num_masked_afs_max_x}"
    training_log += f"\nCorrelation on Y axis per number of masked AFs: {corr_per_num_masked_afs_max_y}"
    print(training_log)

    with open(f'{model_dir}/log.txt', 'a') as f:
        f.write(training_log)
    
    
    y_pos = np.arange(1,9)
    width = 0.35
    fig, ax = plt.subplots(figsize=(11.69,8.27))
    bars = ax.bar(y_pos, corr_per_num_masked_afs_max_x, width, color = 'royalblue')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    bars.set_label('x-axis')
    ax.bar_label(bars)
    bars = ax.bar(y_pos+width, corr_per_num_masked_afs_max_y, width, color = 'seagreen')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    bars.set_label('y-axis')
    ax.bar_label(bars)
    ax.legend()

    # autolabel(bars, bar_prop)
    plt.xlabel("Num of masked afs during training")
    plt.ylabel("Average Correlation")
    plt.title(f"Average Correlation For Model Trained With {num_masked_afs_max}\nMasked afs On Different Numbers Of Masked afs")

    plt.savefig(f"{model_dir}/bar_chart.png", bbox_inches='tight')
    plt.close()
        # plt.show()

# if num_masked_afs_max !=0:
#     plt.figure(figsize=(20,5* num_masked_afs_max))
#     for i in range(num_masked_afs_max):
#         ax = plt.subplot(num_masked_afs_max, 1, i+1)
#         ax.plot(out[...,i].numpy().reshape(-1,1), label='predicted')
#         ax.plot(afs_test[...,i].reshape(-1,1), label='True')
#         plt.legend()
#         ax.set_title(f'af Predictions - {afs_names[i]}')
#     plt.savefig(f"model_outs/MAE_PRETRAINED_BEST_afS_{num_masked_afs_max}/plot_{[afs_names[j] for j in afs]}.png", bbox_inches='tight',)
#     # pyplot.show()
#     plt.close()

def test_model_by_af(model, afs_test, model_dir, is_limiting = False):
    afs_names = ["UL", "LL", "T1", "T2", "T3","T4", "MNI", "MNM"]
    num_masked_afs_max = model_dir.split("_")[-2]
    spkr = model_dir.split("/")[-2]
    bar_prop = []
    if is_limiting:
        log_dir = "/".join(model_dir.split("/")[:-2]) + f"/barcharts/per_af/limiting"
    else:
        log_dir = "/".join(model_dir.split("/")[:-2]) + f"/barcharts/per_af/"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log = []
    log.append(f"SPEAKER: {spkr} \n")
    log.append(f"MODEL: Masking up to {num_masked_afs_max} \n\n")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    
    fig = plt.figure()
    fig.set_figheight(30)
    fig.set_figwidth(60)
    total_cases = 0
    bad_cases = 0
    with open(f'{log_dir}/avg_min_max.txt', 'a') as f:
        f.write(f"{spkr}: Masking up to {num_masked_afs_max}.\n")

    for num_afs_to_test in range(3):
        is_log = False
        print(f"Masking {num_afs_to_test + 1} AFs")
        corr_per_af_x = {"UL":[], "LL":[], "T1":[], "T2":[], "T3":[],"T4":[], "MNI":[], "MNM":[]}
        corr_per_af_y = {"UL":[], "LL":[], "T1":[], "T2":[], "T3":[],"T4":[], "MNI":[], "MNM":[]}
        avg_corrs_x = []
        avg_corrs_y = []
        hundreds = np.ones((afs_test.shape[0], afs_test.shape[1],num_afs_to_test+1, 2)) * 100
        i = 0
        avg_x = 0
        avg_y = 0
        
        if num_afs_to_test + 1 > int(num_masked_afs_max):
            continue
        log.append(f"Testing With Masking {num_afs_to_test + 1} \n")
        # else:
        #     continue

        for afs in combinations(range(8), num_afs_to_test+1):
            if is_limiting:
                if set([0,1]) <= set(afs) or set([6,7]) <= set(afs) or set([2,3,4]) <= set(afs) or set([2,3,5]) <= set(afs) or set([2,4,5]) <= set(afs) or set([3,4,5]) <= set(afs):
                    continue
            i+=1
            total_cases += 1
            model_input = afs_test.copy()
            model_input[:, : ,afs, :] = hundreds
            out = model.predict(model_input, verbose = 0)
            corr_afs_avg_x, avg_corr_afs_x = compute_corr_score(out[:, : ,afs, 0], afs_test[:, : ,afs, 0])
            corr_afs_avg_y, avg_corr_afs_y = compute_corr_score(out[:, : ,afs, 1], afs_test[:, : ,afs, 1])
            
            avg_x += avg_corr_afs_x    
            avg_y += avg_corr_afs_y

            for idx, af in enumerate(afs):
                corr_per_af_x[afs_names[af]].append(corr_afs_avg_x[idx])
                corr_per_af_y[afs_names[af]].append(corr_afs_avg_y[idx])

                if num_afs_to_test + 1 <= int(num_masked_afs_max):
                    if corr_afs_avg_x[idx] < 0.7 or corr_afs_avg_y[idx] < 0.7:
                        if not is_log:
                            is_log = True
                            with open(f'{log_dir}/log.txt', 'a') as f:
                                for l in log:
                                    f.write(l)
                        with open(f'{log_dir}/log.txt', 'a') as f:
                            f.write(f"Masking {afs} results in correlation ({corr_afs_avg_x[idx]:.3f}, {corr_afs_avg_y[idx]:.3f}) for AF {afs[idx]} \n")
        if is_log:
            log = []


        print(f"Testing Done")
   
        avg_corr_x = []
        avg_corr_y = []
        
        max_x = []
        max_y = []
        
        min_x = []
        min_y = []
        
        upper_bound_x =[]
        upper_bound_y =[]

        lower_bound_x =[]
        lower_bound_y =[]

        for key in afs_names:

            avg_corr_x.append(np.mean(corr_per_af_x[key]))
            avg_corr_y.append(np.mean(corr_per_af_y[key]))

            upper_bound_x.append(- np.max(corr_per_af_x[key]) + np.mean(corr_per_af_x[key]))
            upper_bound_y.append(- np.max(corr_per_af_y[key]) + np.mean(corr_per_af_y[key]))

            max_x.append(np.max(corr_per_af_x[key]))
            max_y.append(np.max(corr_per_af_y[key]))

            min_x.append(np.min(corr_per_af_x[key]))
            min_y.append(np.min(corr_per_af_y[key]))


            lower_bound_x.append(np.min(corr_per_af_x[key]) - np.mean(corr_per_af_x[key]))
            lower_bound_y.append(np.min(corr_per_af_y[key]) - np.mean(corr_per_af_y[key]))
        with open(f'{log_dir}/avg_min_max.txt', 'a') as f:
            f.write(f"Testing {num_afs_to_test+1} masked.\n")
            for pt, a, maxi, mini in zip(list(corr_per_af_x.keys()), avg_corr_x, upper_bound_x, lower_bound_x):
                f.write(f"{pt}    {a:4f}    {-maxi:4f}    {-mini:4f}\n")
            f.write(f"Y- Coordinates\n")
            for pt, a, maxi, mini in zip(list(corr_per_af_y.keys()), avg_corr_y, upper_bound_y, lower_bound_y):
                f.write(f"{pt}    {a:4f}    {-maxi:4f}    {-mini:4f}\n")
            f.write(f"######################\n")

        ax = fig.add_subplot(2,3,num_afs_to_test+1, frameon=True)
        y_pos = np.arange(1,9)
        width = 0.35
        # fig, ax = plt.subplots(figsize=(11.69,8.27))
        ax.set_xticks(y_pos+width/2, afs_names)
        bars = ax.bar(y_pos, avg_corr_x, width, yerr =[upper_bound_x, lower_bound_x],  color = 'royalblue')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        bars.set_label('x-axis')
        ax.bar_label(bars)
        bars = ax.bar(y_pos+width, avg_corr_y, width, yerr =[upper_bound_y, lower_bound_y], color = 'seagreen')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        bars.set_label('y-axis')
        ax.bar_label(bars)
        ax.legend()

        # autolabel(bars, bar_prop)
        plt.xlabel("Masked AFs")
        plt.ylabel("Average Correlation")
        plt.title(f"Average Correlation And Range Of Correlation Scores Per Each Masked AF\n For Model Trained With Upto {num_masked_afs_max} AFs When Masking {num_afs_to_test+1} AFs During Testing")

    fig.subplots_adjust(wspace=0, hspace=0)
    if is_limiting:
        bar_charts_dir ="/".join(model_dir.split("/")[:-2]) + f"/barcharts/per_af/limiting/masking_{num_masked_afs_max}_during_training"
    else:
        bar_charts_dir ="/".join(model_dir.split("/")[:-2]) + f"/barcharts/per_af/masking_{num_masked_afs_max}_during_training"
    if not os.path.isdir(bar_charts_dir):
        os.makedirs(bar_charts_dir)
    fig.savefig(f"{bar_charts_dir}/{spkr}.png", bbox_inches='tight')
    plt.close()
    print(f"Plot Done")

    with open(f'{log_dir}/log.txt', 'a') as f:
        f.write(f"Testing Model Done.\n")
        f.write(f"######################\n")
        
        


def test_on_corrupted_files(model, model_dir):
    afs_names = ["UL", "LL", "T1", "T2", "T3","T4", "MNI", "MNM"]
    corr_per_num_masked_afs_max_x = [[] for _ in range(8)]
    corr_per_num_masked_afs_max_y = [[] for _ in range(8)]
    spkr = model_dir.split("/")[-2]
   
    with open(f'{log_dir}/avg_min_max.txt', 'a') as f:
        f.write(f"{spkr}: Masking up to {num_masked_afs_max}.\n")

    num_masked_afs_max = model_dir.split("_")[-2]

    for file_path in os.scandir(f'../data/UWXRMB/trimmed/afs/'):
        file_name = file_path.path.split("/")[-1]
        
        if file_name.split("_")[0] != spkr:
            continue
        afs_test = np.load(f'../data/UWXRMB/trimmed/afs/{file_name}')
        
        not_all_nans = np.where(np.isnan(afs_test).mean((-1,-2)) != 1)[0]
        afs_test = afs_test[not_all_nans,...]
        corrupted_af = np.where(np.isnan(afs_test).mean((-1,0)).astype(bool))[0]

        afs_test = afs_test[:afs_test.shape[0]//200*200]
        afs_test = afs_test.reshape(-1,200,8,2)

        if corrupted_af.shape[0] == 0:
            corrupted_af = None
        if corrupted_af is not None:
            manual_masking_range = range(1, 9-corrupted_af.shape[0])
        else:
            manual_masking_range = range(1, 9)
            
        for i in manual_masking_range:
            if corrupted_af is not None:
                masked_afs = combinations(set(range(8)) - set(corrupted_af), i)
            else:
                masked_afs = combinations(range(8), i)

            # print(corrupted_af)
            for masked_af in masked_afs:
                afs = []
                afs +=masked_af
                if corrupted_af is not None:
                    for af in corrupted_af:
                        afs.append(af)    
                hundreds = np.ones((afs_test.shape[0], afs_test.shape[1],len(afs), 2)) * 100
                model_input = afs_test.copy()

                try:
                    model_input[:, : ,afs, :] = hundreds
                except:
                    print(afs)
                    exit()
                out = model.predict(model_input, verbose = 0)
                corr_afs_avg_x, avg_corr_afs_x = compute_corr_score(out[:, : ,masked_af, 0], afs_test[:, : ,masked_af, 0])
                corr_afs_avg_y, avg_corr_afs_y = compute_corr_score(out[:, : ,masked_af, 1], afs_test[:, : ,masked_af, 1])
                corr_per_num_masked_afs_max_x[i-1].append(avg_corr_afs_x)        
                corr_per_num_masked_afs_max_y[i-1].append(avg_corr_afs_y)        
    corr_per_num_masked_afs_max_x = [np.mean(l) for l in corr_per_num_masked_afs_max_x]
    corr_per_num_masked_afs_max_y = [np.mean(l) for l in corr_per_num_masked_afs_max_y]
    with open("sparse_new.txt", "a") as f:
        f.write(f"{corr_per_num_masked_afs_max_x}, {corr_per_num_masked_afs_max_y}\n")
    

    y_pos = np.arange(1,9)
    width = 0.35
    fig, ax = plt.subplots(figsize=(11.69,8.27))
    bars = ax.bar(y_pos, corr_per_num_masked_afs_max_x, width, color = 'royalblue')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    bars.set_label('x-axis')
    ax.bar_label(bars)
    bars = ax.bar(y_pos+width, corr_per_num_masked_afs_max_y, width, color = 'seagreen')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    bars.set_label('y-axis')
    ax.bar_label(bars)
    ax.legend()
    bar_charts_dir ="/".join(model_dir.split("/")[:-2]) + f"/barcharts/"
    # autolabel(bars, bar_prop)
    plt.xlabel("Num of masked afs during training")
    plt.ylabel("Average Correlation")
    plt.title(f"Average Correlation For Model Trained With {num_masked_afs_max}\nMasked afs On Different Numbers Of Masked afs")

    plt.savefig(f"{bar_charts_dir}/{spkr}.png", bbox_inches='tight')
    plt.close()

def test_on_corrupted_files_per_af(model, model_dir, is_limiting= False):
    afs_names = ["UL", "LL", "T1", "T2", "T3","T4", "MNI", "MNM"]
    num_masked_afs_max = model_dir.split("_")[-2]
    spkr = model_dir.split("/")[-2]
    bar_prop = []
    if is_limiting:
        log_dir = "/".join(model_dir.split("/")[:-2]) + f"/barcharts/per_af/limiting"
    else:
        log_dir = "/".join(model_dir.split("/")[:-2]) + f"/barcharts/per_af/"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(f'{log_dir}/avg_min_max.txt', 'a') as f:
        f.write(f"{spkr}: Masking up to {num_masked_afs_max}.\n")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log = []
    log.append(f"SPEAKER: {spkr} \n")
    log.append(f"MODEL: Masking up to {num_masked_afs_max} \n\n")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    total_cases = 0
    bad_cases = 0
    corr_per_af_x = [{"UL":[], "LL":[], "T1":[], "T2":[], "T3":[],"T4":[], "MNI":[], "MNM":[]} for _ in range(3)]
    corr_per_af_y = [{"UL":[], "LL":[], "T1":[], "T2":[], "T3":[],"T4":[], "MNI":[], "MNM":[]} for _ in range(3)]
    avg_corr_x =  [{"UL":[], "LL":[], "T1":[], "T2":[], "T3":[],"T4":[], "MNI":[], "MNM":[]} for _ in range(3)]
    avg_corr_y =  [{"UL":[], "LL":[], "T1":[], "T2":[], "T3":[],"T4":[], "MNI":[], "MNM":[]} for _ in range(3)]

    upper_bound_x = [{"UL":[], "LL":[], "T1":[], "T2":[], "T3":[],"T4":[], "MNI":[], "MNM":[]} for _ in range(3)]
    upper_bound_y = [{"UL":[], "LL":[], "T1":[], "T2":[], "T3":[],"T4":[], "MNI":[], "MNM":[]} for _ in range(3)]

    lower_bound_x = [{"UL":[], "LL":[], "T1":[], "T2":[], "T3":[],"T4":[], "MNI":[], "MNM":[]} for _ in range(3)]
    lower_bound_y = [{"UL":[], "LL":[], "T1":[], "T2":[], "T3":[],"T4":[], "MNI":[], "MNM":[]} for _ in range(3)]
    # files_ds = tf.data.Dataset.list_files(f'../data/UWXRMB/trimmed/afs/{spkr}*.npy')
    # files_ds = files_ds.map(lambda x: tf.pyfunc(func = np.load, inp = [x], Tout = tf.float32))
    
    for file_path in os.scandir(f'../data/UWXRMB/trimmed/afs/'):
        file_name = file_path.path.split("/")[-1]
        
        if file_name.split("_")[0] != spkr:
            continue
        afs_test = np.load(f'../data/UWXRMB/trimmed/afs/{file_name}')
        not_all_nans = np.where(np.isnan(afs_test).mean((-1,-2)) != 1)[0]
        afs_test = afs_test[not_all_nans,...]
        corrupted_af = np.where(np.isnan(afs_test).mean((-1,0)).astype(bool))[0]

        afs_test = afs_test[:afs_test.shape[0]//200*200]
        afs_test = afs_test.reshape(-1,200,8,2)
        if corrupted_af.shape[0] == 0:
            corrupted_af = None
            manual_masking_range = range(1, min(4, int(num_masked_afs_max)))
        else:
            manual_masking_range = range(1, min(4-corrupted_af.shape[0],  int(num_masked_afs_max)))
        
        for i in manual_masking_range:
            if corrupted_af is not None:
                masked_afs = combinations(set(range(8)) - set(corrupted_af), i)
            else:
                masked_afs = combinations(range(8), i)
            # print(corrupted_af)
            for masked_af in masked_afs:
                afs = []
                afs +=masked_af
                if corrupted_af is not None:
                    for af in corrupted_af:
                        afs.append(af)    
                if is_limiting:
                    if set([0,1]) <= set(afs) or set([6,7]) <= set(afs) or set([2,3,4]) <= set(afs) or set([2,3,5]) <= set(afs) or set([2,4,5]) <= set(afs) or set([3,4,5]) <= set(afs):
                        continue
                hundreds = np.ones((afs_test.shape[0], afs_test.shape[1],len(afs), 2)) * 100
                model_input = afs_test.copy()

                model_input[:, : ,afs, :] = hundreds
          
                out = model.predict(model_input, verbose = 0)
                corr_afs_avg_x, avg_corr_afs_x = compute_corr_score(out[:, : ,masked_af, 0], afs_test[:, : ,masked_af, 0])
                corr_afs_avg_y, avg_corr_afs_y = compute_corr_score(out[:, : ,masked_af, 1], afs_test[:, : ,masked_af, 1])
                for idx, af in enumerate(masked_af):
                    corr_per_af_x[len(afs) -1][afs_names[af]].append(corr_afs_avg_x[idx])
                    corr_per_af_y[len(afs) -1][afs_names[af]].append(corr_afs_avg_y[idx])
                    if corr_afs_avg_x[idx] < 0.7 or corr_afs_avg_y[idx] < 0.7:
                        with open(f'{log_dir}/log.txt', 'a') as f:
                            f.write(f"{file_name} Masking {afs} results in correlation ({corr_afs_avg_x[idx]:.3f}, {corr_afs_avg_y[idx]:.3f}) for AF {afs[idx]} \n")
    for i in range(3):
        for key in afs_names:
            avg_corr_x[i][key] = np.mean(corr_per_af_x[i][key])
            avg_corr_y[i][key] = np.mean(corr_per_af_y[i][key])
            upper_bound_x[i][key] = - np.max(corr_per_af_x[i][key]) + np.mean(corr_per_af_x[i][key])
            upper_bound_y[i][key] = - np.max(corr_per_af_y[i][key]) + np.mean(corr_per_af_y[i][key])
            lower_bound_x[i][key] = np.min(corr_per_af_x[i][key]) - np.mean(corr_per_af_x[i][key])
            lower_bound_y[i][key]= np.min(corr_per_af_y[i][key]) - np.mean(corr_per_af_y[i][key])
    for i in range(3):
        with open(f'{log_dir}/avg_min_max.txt', 'a') as f:
            f.write(f"Testing {i+1} masked.\n")
            for pt, a, maxi, mini in zip(list(corr_per_af_x[i].keys()), avg_corr_x[i].values(), upper_bound_x[i].values(), lower_bound_x[i].values()):
                f.write(f"{pt}    {a:4f}    {-maxi:4f}    {-mini:4f}\n")
            f.write(f"Y- Coordinates\n")
            for pt, a, maxi, mini in zip(list(corr_per_af_y[i].keys()), avg_corr_y[i].values(), upper_bound_y[i].values(), lower_bound_y[i].values()):
                f.write(f"{pt}    {a:4f}    {-maxi:4f}    {-mini:4f}\n")
            f.write(f"######################\n")
    return 
    fig = plt.figure()
    fig.set_figheight(30)
    fig.set_figwidth(60)

    for i in range(3):
        ax = fig.add_subplot(1,3,i+1, frameon=True)
        y_pos = np.arange(1,9)
        width = 0.35
        ax.set_xticks(y_pos+width/2, afs_names)
        bars = ax.bar(x= y_pos, height= list(avg_corr_x[i].values()), width = width, yerr =[list(upper_bound_x[i].values()), list(lower_bound_x[i].values())],  color = 'royalblue')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        bars.set_label('x-axis')
        ax.bar_label(bars)
        bars = ax.bar(x = y_pos+width, height = list(avg_corr_y[i].values()), width = width, yerr =[list(upper_bound_y[i].values()), list(lower_bound_y[i].values())], color = 'seagreen')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        bars.set_label('y-axis')
        ax.bar_label(bars)
        ax.legend()

        # autolabel(bars, bar_prop)
        plt.xlabel("Masked AFs")
        plt.ylabel("Average Correlation")
        plt.title(f"Average Correlation And Range Of Correlation Scores Per Each Masked AF\n For Model Trained With Upto {num_masked_afs_max} AFs When Masking {i +1} AFs During Testing: {spkr}")

    fig.subplots_adjust(wspace=0, hspace=0)
    if is_limiting:
        bar_charts_dir ="/".join(model_dir.split("/")[:-2]) + f"/barcharts/per_af/limiting/masking_{num_masked_afs_max}_during_training"
    else:
        bar_charts_dir ="/".join(model_dir.split("/")[:-2]) + f"/barcharts/per_af/masking_{num_masked_afs_max}_during_training"
    if not os.path.isdir(bar_charts_dir):
        os.makedirs(bar_charts_dir)
    fig.savefig(f"{bar_charts_dir}/{spkr}.png", bbox_inches='tight')
    plt.close()
    print(f"Plot Done")

if __name__ == '__main__':
    from masked_autoencoder import MaskedAutoEncoder
    from tqdm import tqdm
    directory = "./model_outs/AFS/trimmed/new_testset-overlap/pretrain/sparse_speakers/resampled_data"
    if not os.path.isdir(f"{directory}/barcharts"):
        os.mkdir(f"{directory}/barcharts")
    spkrs_dirs = [f.path for f in os.scandir(directory) if f.is_dir() and f.path.split("/")[-1][:2] == "JW" or f.path.split("/")[-1] == 'JW29' ]
    for spkr_dir in tqdm(spkrs_dirs):
        spkr = spkr_dir.split("/")[-1]
        if spkr not in ["JW60"]:
            continue
        input_file=np.load(f'../data/UWXRMB/trimmed/afs/JW32_TP014.npy')
        models_dirs = [f.path for f in os.scandir(spkr_dir) if f.is_dir()]
        models_dirs.sort()
        for model_dir in models_dirs[2:3]:
            print(model_dir)
            with open("sparse_new.txt", "a") as f:
                f.write(f"{model_dir}\n")
            model = MaskedAutoEncoder(num_TVs=8)
            model(input_file[0:1,...])
            model.load_weights("/".join(model_dir.split("/") + [model_dir.split("/")[-1]])+".tf").expect_partial()
            
            test_on_corrupted_files_per_af(model, model_dir)
            # test_on_corrupted_files_per_af(model, model_dir, is_limiting = True)

    
    # directory = "./model_outs/AFS/trimmed/new_testset-overlap/pretrain/Original_Speakers"
    # if not os.path.isdir(f"{directory}/barcharts"):
    #     os.mkdir(f"{directory}/barcharts")
    # spkrs_dirs = [f.path for f in os.scandir(directory) if f.is_dir() and f.path.split("/")[-1][:2] == "JW"]
    # spkrs_dirs.sort()
    # for spkr_dir in tqdm(spkrs_dirs):
    #     spkr = spkr_dir.split("/")[-1]

    #     af_test=np.load(f'../data/UWXRMB/trimmed/Train_files/speaker_files/{spkr}/TVs_{spkr}_test_200.npy')
    #     models_dirs = [f.path for f in os.scandir(spkr_dir) if f.is_dir() and  "MAE_pretrained" not in  f.path]
    #     models_dirs.sort()
    #     for model_dir in models_dirs:
    #         print(model_dir)

    #         model = MaskedAutoEncoder(num_TVs=8)
    #         model(af_test[0:1,...])
    #         model.load_weights("/".join(model_dir.split("/") + [model_dir.split("/")[-1]])+".tf").expect_partial()
            
    #         test_model_by_af(model, af_test, model_dir, is_limiting = True)