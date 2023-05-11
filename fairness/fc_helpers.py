import numpy as np

def msd_one(x1, t1):
    res = np.mean(x1[t1], axis=0)
    return res

def msd_two(x1, t1, x2, t2):
    res = np.mean(x1[t1], axis=0) + np.mean(x2[t2], axis=0)
    return res

def msd_three(x1, t1, x2, t2, x3, t3):
    res = np.mean(x1[t1], axis=0) + np.mean(x2[t2], axis=0) + np.mean(x3[t3], axis=0)
    return res