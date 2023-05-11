import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from flaml import AutoML
from sklearn.model_selection import train_test_split
import sys



import utils.estimators as models
from fairness.fc_helpers import msd_one, msd_two, msd_three
from fairness.evaluate_fairness import evaluate

def automl_estimator_wrapper(X, Y, W, num_train, estimator_name=""):
    automl = AutoML()
    automl_settings = {
            "time_budget": 5,  # in seconds
            "task": 'regression',
            "eval_method": 'cv',
            "n_splits": 3,
            "verbose": 3
        }
    X_train = np.concatenate((X[0:num_train], W[0:num_train]), axis=1)
    y_train = Y[0:num_train]
    automl.fit(X_train[0:num_train], y_train[0:num_train], **automl_settings)
    y0_in = automl.predict(np.concatenate((X[0:num_train], np.zeros(W[0:num_train].shape)), axis=1))
    y1_in = automl.predict(np.concatenate((X[0:num_train], np.ones(W[0:num_train].shape)), axis=1))
    y0_out = automl.predict(np.concatenate((X[num_train:], np.zeros(W[num_train:].shape)), axis=1))
    y1_out = automl.predict(np.concatenate((X[num_train:], np.ones(W[num_train:].shape)), axis=1))

    return y0_in, y1_in, y0_out, y1_out

def wen_estimator_wrapper(X, Y, W, num_train, estimator_name="OLS1"):

    t = W
    if t.shape[-1] == 1:
        t = np.squeeze(t, axis=-1)
    yf = Y
    if yf.shape[-1] == 1:
        yf = np.squeeze(yf, axis=-1)

    estimator_results = models.train_and_evaluate(x=X[0:num_train], t=t[0:num_train], yf=yf[0:num_train], x_test=X[num_train:], estimator_name=estimator_name)
    y0_in = estimator_results[0]
    y1_in = estimator_results[1]
    y0_out = estimator_results[3]
    y1_out = estimator_results[4]

    return y0_in, y1_in, y0_out, y1_out


def estimator_wrapper(X, Y, W, estimator_name, id0, id1):
    # Estimates tau(X). tau(X) = E[Y(1) - Y(0) | X = x].

    test_ratio = 0.1
    num_samples = X.shape[0]
    num_train = int(num_samples * (1 - test_ratio))

    #TODO
    # indices_all = np.arange(x_all.shape[0])
    # x_train, x_eval, t_train, t_eval, yf_train, yf_eval, inidices_train, indices_eval = train_test_split(
    #     x_all, t_all, yf_all, indices_all, test_size=0.2, random_state=seed
    #     )

    if estimator_name == "AutoML":
        y0_in, y1_in, y0_out, y1_out = automl_estimator_wrapper(X, Y, W, num_train)
    else:
        y0_in, y1_in, y0_out, y1_out = wen_estimator_wrapper(X, Y, W, num_train, estimator_name)

    ##TODO train test eval 2- why does it have to be the same seed
    dataset_name = "fairness_dataset"
    evaluate(X[0:num_train], W[0:num_train], Y[0:num_train], np.arange(num_train), estimator_name, y0_in, y1_in, dataset_name)


    if len(y0_in.shape) == 1:
        y0_in = np.expand_dims(y0_in, axis=1)
        y1_in = np.expand_dims(y1_in, axis=1)
        y0_out = np.expand_dims(y0_out, axis=1)
        y1_out = np.expand_dims(y1_out, axis=1)
    
    y0 = np.concatenate((y0_in, y0_out), axis=0)
    y1 = np.concatenate((y1_in, y1_out), axis=0)

    res = (y1 - y0)

    y_pred = np.zeros(Y.shape)
    y_pred[id0] = y0[id0]
    y_pred[id1] = y1[id1]
    err = np.mean((Y - y_pred)[num_train:] ** 2)
    return res, err

def normalize_data(data):
    data_scaler = StandardScaler().fit(data)
    data_scaled = data_scaler.transform(data)
    return data_scaled

def fairness_cookbook(data, X, Z, Y, W , x0, x1, estimator_name="RF2"):
    metrics = {}

    # TODO: Change sampling method.
    num_obs = len(data)
    idx = np.array([i for i in range(num_obs)])
    id0 = idx[(data[:, X][idx] == [x0])[:, 0]]
    id1 = idx[(data[:, X][idx] == [x1])[:, 0]]

    norm_cols = Y + Z + W
    data[:, norm_cols] = normalize_data(data[:, norm_cols])
    y = data[:, Y]
    x = data[:, X]
    z = data[:, Z]
    
    tv = msd_two(y, id1, -y, id0)
    metrics["tv"] = tv

    if len(Z) == 0:
        te = tv
        ett = tv
        expse_x1 = 0
        expse_x0 = 0
        ctfse = 0
        crf_te = np.array([tv] * len(idx))
    else:
        crf_te, err = estimator_wrapper(X = z, 
                       Y = y, 
                       W = x,
                       estimator_name=estimator_name, id0=id0, id1=id1)
        print("Estimator error for E[Y(x1) - Y(x0) | C = z] = ", err)
        te = msd_one(crf_te, idx)
        ett = msd_one(crf_te, id0)
        ctfse = msd_three(crf_te, id0, -y, id1, y, id0)
    expse_x0 = tv 
    expse_x1 = tv

    metrics["te"] = te
    metrics["ett"] = ett
    metrics["ctfse"] = ctfse
    metrics["expse_x0"] = expse_x0
    metrics["expse_x1"] = expse_x1

    if len(W) == 0:
        nde = te
        ctfde = ett
        ctfie = 0
        nie = 0
    else:
        ZW = Z + W
        zw = data[:, ZW]
        crf_med, err = estimator_wrapper(X = zw, 
                        Y = y, 
                        W = x,
                        estimator_name=estimator_name, id0=id0, id1=id1)
        print("Estimator error for E[Y(x1) - Y(x0) | C = zw] = ", err)
        nde = msd_one(crf_med, idx)
        ctfde = msd_one(crf_med, id0)
        nie = msd_two(crf_med, idx, -crf_te, idx)
        ctfie = msd_two(crf_med, id0, -crf_te, id0)
    metrics["nde"] = nde
    metrics["ctfde"] = ctfde
    metrics["nie"] = nie
    metrics["ctfie"] = ctfie

    return metrics