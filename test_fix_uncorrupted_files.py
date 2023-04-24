import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from statistics import mode
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
from tqdm import tqdm
from masked_autoencoder import MaskedAutoEncoder
from scipy.io import loadmat, savemat
from copy import copy
import math
from metrics import ppmc
import shutil
from itertools import combinations
directory = "../data/XRMB_original_delimited/AFs_wav_files_delimited/"
# directory = "../data/UWXRMB/trimmed/afs/"
mat_directory = "../data/UWXRMB/mat"
fixed_mat_directory = "../data/UWXRMB/trimmed/mat_fixed"
test_mat_directory = "../data/UWXRMB/trimmed/mat_test"
fixed_np_directory = "../data/XRMB_original_delimited/AFs_wav_files_delimited_test_fixed/"
test_np_directory = "../data/UWXRMB/trimmed/afs_test/"
window_size =200
if not os.path.isdir(fixed_np_directory):
    os.makedirs(fixed_np_directory)
files = os.listdir(directory)
files.sort()
print("Total files:", len(files))

fixed_files = os.listdir(fixed_np_directory)
# files = [file for file in files if file not in fixed_files]
files_dict = {}
trained_spkrs = open("../data/UWXRMB/speakers_list.txt", "r").readlines() #list of speakers
sparse_speakers = open("../data/UWXRMB/sparse_speakers.txt", "r").readlines() #list of speakers with less than few utterances
trained_spkrs = [spkr for spkr in trained_spkrs  if spkr not in sparse_speakers] #only train on speakers with enough utterances
trained_spkrs = [spkr.strip("\n") for spkr in trained_spkrs]
# spkr_model_file = open("model_to_use.txt")

# for l in spkr_model_file.readlines():
#     spkr, model_id, _ = l.split("	")
#     if model_id != "N/A":
#         spkr_best_models_dict[spkr] = 3
trained_spkrs = [spkr.strip("\n") for spkr in trained_spkrs]
spkr = None
corrupted_count = 0
non_corrupted_count = 0
for file in files:
    if spkr != file.split("_")[1]:
        spkr = file.split("_")[1]
        files_dict[spkr] = []

    files_dict[spkr].append(file)

for i,(spkr, files) in enumerate(files_dict.items()):

    if spkr in trained_spkrs:
        model = MaskedAutoEncoder(num_PTs=8)
        model.load_weights(f"./model_outs/AFS/trimmed/new_testset-overlap/pretrain/resampled/{spkr}/MAE_Trained_on_upto_4_AFs/MAE_Trained_on_upto_4_AFs.tf").expect_partial()
    for file in tqdm(files, desc= f"{i+1}th spkr out of {len(files_dict)} files: {spkr}"):
        file=file.strip("\n")
        
        TV_data = np.load(directory + file)
        orig_len = TV_data.shape[0]
        # is_not_entire_AF_nan = np.where(~np.isnan(TV_data.mean((-1))).all(0))[0]
        # avg_over_AFs = TV_data[:,is_not_entire_AF_nan, :].mean((-1))
        # try:
        #     first_non_nan_element = avg_over_AFs[np.isfinite(avg_over_AFs)][0]
        #     last_non_nan_element = avg_over_AFs[np.isfinite(avg_over_AFs)][-1]
        # except:
        #     print("error at", file)
        #     continue
        # idx_of_first_non_nan = np.where(avg_over_AFs == first_non_nan_element)[0][0]
        # idx_of_last_non_nan = np.where(avg_over_AFs == last_non_nan_element)[0][0]
    
        # TV_data = TV_data[idx_of_first_non_nan+1:idx_of_last_non_nan]
        afs_len = TV_data.shape[0]
        
        try:
            TV_data = np.pad(TV_data, [[0,window_size - afs_len % window_size], [0,0], [0,0]])
        except:
            print(file)
        nan_idx =  (np.isnan(TV_data.mean((-1, -2)))) 
        #bug here, corrupted files from non trained speakers are being saved as is
        if nan_idx.any() and spkr in trained_spkrs:
            corrupted_count += 1
            TV_data = np.nan_to_num(TV_data, nan = 100)
            model_input = TV_data.reshape(TV_data.shape[0] // window_size, window_size, TV_data.shape[1], TV_data.shape[2]) #reshaping the arrays into windows of length window_size
            fixed_file = model(model_input).numpy().reshape(-1,8,2)

            TV_data[nan_idx] = fixed_file[nan_idx]
            TV_data = TV_data[0: TV_data.shape[0]-(window_size - afs_len % window_size)]
            assert TV_data.shape[0] == orig_len
            assert not np.isnan(TV_data).any()
            np.save(fixed_np_directory + file, TV_data)
            #save the fixed file
            

        else:
            non_corrupted_count += 1
            if spkr in trained_spkrs:
                masked_afs = []
                corrs = []
                test_file = TV_data.copy()
                n_masked_afs = np.random.randint(1,4)
                combs = tuple(combinations(range(8), n_masked_afs+1))
                while True:
                    #pick a random combination of afs to mask
                    n = len(combs)
                    idx = sorted(random.sample(range(n), 1))[0]
                    masked_afs = tuple(combs[idx])
                    if set([0,1]) <= set(masked_afs) or set([6,7]) <= set(masked_afs) or set([2,3,4]) <= set(masked_afs) or set([2,3,5]) <= set(masked_afs) or set([2,4,5]) <= set(masked_afs) or set([3,4,5]) <= set(masked_afs):
                        continue
                    else:
                        break
                
                for rand_af in masked_afs:
                    test_file[:,rand_af, :] = 100
                model_input = test_file.reshape(test_file.shape[0] // window_size, window_size, test_file.shape[1], test_file.shape[2]) #reshaping the arrays into windows of length window_size
                fixed_file = model(model_input).numpy().reshape(-1,8,2)
                for rand_af in masked_afs:
                    test_file[:,rand_af, :] = fixed_file[:,rand_af, :]
                    corrs.append(ppmc(TV_data[:,rand_af,:], fixed_file[:,rand_af, :]))

                # test_file = test_file[0: test_file.shape[0]-(window_size - afs_len % window_size)]
                # test_file = np.pad(test_file, [[idx_of_first_non_nan+1, orig_len - idx_of_last_non_nan], [0,0],[0,0]], 'constant', constant_values = np.math.nan)
                #TODO: comment this out
                TV_data = test_file
                with open('../data/UWXRMB/trimmed/test_files.txt', 'a') as f:
                    f.write(f"{file}: {masked_afs}, {corrs}\n")
            if not np.isnan(TV_data).any():
                TV_data = TV_data[0: TV_data.shape[0]-(window_size - afs_len % window_size)]
                try:
                    assert TV_data.shape[0] == orig_len
                except:
                    print(file, TV_data.shape, orig_len)
                    exit()
                assert not np.isnan(TV_data).any()
                np.save(fixed_np_directory + file, TV_data)
            
        # TV_data = TV_data[0: TV_data.shape[0]-(window_size - afs_len % window_size)]
        # TV_data = np.pad(TV_data, [[idx_of_first_non_nan+1, orig_len - idx_of_last_non_nan], [0,0],[0,0]], 'constant', constant_values = np.math.nan)
with open('stats.txt', 'w') as f:
    f.write(f"corrupted_count: {corrupted_count}\n")
    f.write(f"non_corrupted_count: {non_corrupted_count}\n")
    f.write(f"total: {corrupted_count + non_corrupted_count}\n")
    f.write(f"corrupted percentage: {corrupted_count / (corrupted_count + non_corrupted_count)}\n")
  
        # try: assert TV_data.shape[0] == orig_len
        # except AssertionError:
        #     print(file, TV_data.shape[0], orig_len)

        # tag = file.split(".")[0] #tag is the name of the file without the extension

        # # loading the .mat file
        # mat_file = loadmat(f"{mat_directory}/{tag}.mat")

        # # extracting the data from the .mat file
        # file_data = mat_file[tag][0] #Extracting the data from the .mat file. data[tag] is a list of len 1 so the index [0] is used to extract the data from the list.
        
        # for i in range(2, len(file_data)): #element 0 and 1 are the audio data and periodicity information.
        #     orig_len = file_data[i][-1].shape[0]
        #     fixed_af = TV_data[:,i-2,:]
        #     fixed_len = fixed_af.shape[0]
        #     try: assert fixed_len == orig_len
        #     except AssertionError:
        #         print(tag,fixed_file.shape, fixed_len, orig_len)
        #         break
        #     file_data[i][-1] = fixed_af

        # for i in range(2):
        #     file_data[i][1] = file_data[i][1].astype(np.float)
        # mat_file[tag][0] = file_data

        # savemat(f"{fixed_mat_directory}/{tag}.mat", mat_file)
        # np.save(fixed_np_directory+file, TV_data)
        # if file_type == "test":
        #     # loading the .mat file
        #     mat_file = loadmat(f"{mat_directory}/{tag}.mat")

        #     # extracting the data from the .mat file
        #     file_data = mat_file[tag][0] #Extracting the data from the .mat file. data[tag] is a list of len 1 so the index [0] is used to extract the data from the list.
            
        #     for i in range(2, len(file_data)): #element 0 and 1 are the audio data and periodicity information.
        #         orig_len = file_data[i][-1].shape[0]
        #         fixed_af = test_file[:,i-2,:]
        #         fixed_len = fixed_af.shape[0]
        #         try: assert fixed_len == orig_len
        #         except AssertionError:
        #             print(tag,fixed_file.shape, fixed_len, orig_len)
        #             break
        #         file_data[i][-1] = fixed_af

        #     for i in range(2):
        #         file_data[i][1] = file_data[i][1].astype(np.float)
        #     mat_file[tag][0] = file_data
        #     print(f"./{tag}.mat")
        #     savemat(f"./{tag}.mat", mat_file)
        #     np.save(test_np_directory+file, test_file)
        #     exit()