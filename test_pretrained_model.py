from enum import auto
import tensorflow as tf
from masked_autoencoder import *
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations

spkr = "JW12"
TV_NAMES = ["UL", "LL", "T1", "T2", "T3","T4", "MNI", "MNM"]

TVs_test=np.load(f'../data/UWXRMB/Train_files/speaker_files/{spkr}/TVs_{spkr}_test_200.npy')

num_masked_tvs = 6
model = MaskedAutoEncoder(num_masked_tvs = num_masked_tvs, num_TVs = 8)
# model.summary()
model.enable_masking = False
model(TVs_test)
model.load_weights(f"model_outs/AFS/MAE_pretrained/MAE_pretrained.h5")

TVs_predict = model(TVs_test).numpy()

corr_TVs_avg, avg_corr_tvs = compute_corr_score(TVs_predict[...,0], TVs_test[...,0])
corr_TVs_avg, avg_corr_tvs = compute_corr_score(TVs_predict[...,1], TVs_test[...,1])

for i in range(8):
    ax = plt.subplot(2, 1, 1)
    ax.plot(TVs_predict[:,:,i,0].reshape(-1,1)[:1000], label='x-predicted')
    ax.plot(TVs_test[:,:,i,0].reshape(-1,1)[:1000], label='x-true')
    plt.legend()
    ax.set_title(f'TV Predictions - {TV_NAMES[i]} - X-AXIS')
    ax = plt.subplot(2, 1, 2)
    ax.plot(TVs_predict[:,:,i,1].reshape(-1,1)[:1000], label='y-predicted')
    ax.plot(TVs_test[:,:,i,1].reshape(-1,1)[:1000], label='y-true')
    plt.legend()
    ax.set_title(f'TV Predictions - {TV_NAMES[i]} - Y-AXIS')
    plt.savefig(f"model_outs/AFS/MAE_pretrained/plot_{TV_NAMES[i]}.png")
    plt.close()
    plt.show()        

