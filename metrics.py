import numpy as np
import tensorflow as tf

def ppmc(x, y):
    xbar = np.nanmean(x)
    ybar = np.nanmean(y)
    num = np.nansum((x - xbar)*(y - ybar))
    den = np.sqrt(np.nansum((x - xbar)**2))*np.sqrt(np.nansum((y - ybar)**2))
    corr = num/den
    return corr

# computing average correlations on both test and validation sets
def compute_corr_score(y_predict, y_true):
    num_tvs = y_predict.shape[2]
    corr_TVs = np.zeros(num_tvs)
    print(y_predict.shape)
    if len(y_predict.shape) >3:
        y_predict = y_predict.reshape(-1, num_tvs, 2)
        y_true = y_true.reshape(-1, num_tvs, 2)
        af_dimention = 2
    else:
        y_predict = y_predict.reshape(-1, num_tvs, 1)
        y_true = y_true.reshape(-1, num_tvs, 1)
        af_dimention = 1

    for i in range(0, num_tvs):
        corr = 0
        for j in range(0, af_dimention):
            corr += ppmc(y_predict[:,i,j], y_true[:, i,j])/af_dimention
        if tf.math.is_nan(corr):
            corr_TVs[i] += np.mean(corr_TVs[:i-1])
        else:
            corr_TVs[i] +=corr
            
    corr_TVs_avg = corr_TVs
    avg_corr_tvs = np.mean(corr_TVs_avg)

    return corr_TVs_avg, avg_corr_tvs
