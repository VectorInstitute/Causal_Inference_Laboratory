import os
from sklearn.model_selection import train_test_split
from typing import Dict
import numpy as np
import sys
from pathlib import Path
import pickle
import random
from tqdm import tqdm
from flaml import AutoML


import utils.metrics as metrics
from utils.preprocessing import sys_config
from utils.evaluation import *
from utils.helpers import get_nuisance_models_list
from fairness.nuisance_model_selection import *


results_folder = sys_config["results_folder"]
datasets_folder = sys_config["datasets_folder"]

seed = 0 # same seed as in nuisance_model_selection.py
np.random.seed(seed)

metrics_set = [
    "value_score",
    "value_dr_score",
    "value_dr_clip_prop_score",
    "tau_t_score",
    "tau_s_score",
    "tau_match_score",
    "tau_iptw_score",
    "tau_iptw_clip_prop_score",
    "tau_dr_score",
    "tau_dr_clip_prop_score",
    "influence_score",
    "influence_clip_prop_score",
    "r_score",
]

def evaluate(x_all, t_all, yf_all, indices_all, estimator_name, y0_in, y1_in, dataset_name):
    print(f'{" Evaluation ":-^79}')
    results_in: Dict[str, Dict] = {}
    results_out: Dict[str, Dict] = {}

    if t_all.shape[-1] == 1:
        t_all = np.squeeze(t_all, axis=-1)
    if yf_all.shape[-1] == 1:
        yf_all = np.squeeze(yf_all, axis=-1)
    
    x_all = np.expand_dims(x_all, axis=-1)
    t_all = np.expand_dims(t_all, axis=-1)
    yf_all = np.expand_dims(yf_all, axis=-1)

    results_folder = sys_config["results_folder"]

    indices_all = np.arange(x_all.shape[0])

    x_train, x_eval, t_train, t_eval, yf_train, yf_eval, inidices_train, indices_eval = train_test_split(
        x_all, t_all, yf_all, indices_all, test_size=0.2, random_state=seed
        )

    save_dir= results_folder + '/' + dataset_name + '/models/'
    print(save_dir, flush=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    outcome_models, prop_models= get_nuisance_models_list()

    random.seed(seed)
    np.random.seed(seed)

    w= { 'tr': x_train, 'te': x_eval, 'all': np.concatenate((x_train, x_eval), axis=0)}
    t= { 'tr': t_train, 'te': t_eval, 'all': np.concatenate((t_train, t_eval), axis=0)}
    y= { 'tr': yf_train, 'te': yf_eval, 'all': np.concatenate((yf_train, yf_eval), axis=0)}

    for key in t.keys():
        data_size= w[key].shape[0]
        num_realizations = w[key].shape[-1]

        # print(w[key].shape, "00000")
        # stack num_realizations vertically under data_size
        new_w = np.zeros((data_size * num_realizations, w[key].shape[1]))
        for i in range(num_realizations):
            new_w[i * data_size : (i + 1) * data_size, :] = w[key][:, :, i]
        w[key] = new_w

        # stack num_realizations vertically under data_size
        new_t = np.zeros((data_size * num_realizations))
        for i in range(num_realizations):
            new_t[i * data_size : (i + 1) * data_size] = t[key][:, i]
        t[key] = new_t

        # stack num_realizations vertically under data_size
        new_y = np.zeros((data_size * num_realizations))
        for i in range(num_realizations):
            new_y[i * data_size : (i + 1) * data_size] = y[key][:, i]
        y[key] = new_y


        w[key]= np.reshape(w[key], (data_size * num_realizations, -1)) # TODO: fix for mutliple realizations
        t[key]= np.reshape(t[key], (data_size * num_realizations))
        y[key]= np.reshape(y[key], (data_size * num_realizations))


    model_sel_res={}
    for key in ['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop']:
        model_sel_res[key]= {}
        model_sel_res[key]['score']= -sys.maxsize - 1
        model_sel_res[key]['model']= -sys.maxsize - 1


    sys.stderr.flush()
    for model_case in tqdm(['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop'], file=sys.stdout):
    # for model_case in  ['s_learner']:
        if model_case == 'prop':
            automl_settings = {
                "time_budget": 2,  # in seconds
                "task": 'classification',
                "eval_method": 'cv',
                "n_splits": 3,
                "verbose": 0
            }
            nuisance_list= prop_models
        else:
            automl_settings = {
                "time_budget": 2,  # in seconds
                "task": 'regression',
                "eval_method": 'cv',
                "n_splits": 3,
                "verbose": 0
            }
            nuisance_list= outcome_models

        #AutoML
        selection_case = "metric"
        automl = AutoML()
        if model_case == 'prop':
            print('Propensity Model')
            score, best_model = get_propensity_model(automl, w, t, automl= 1, automl_settings= automl_settings, selection_case= selection_case)
            pickle.dump(best_model, open(save_dir + 'prop' + '.p', "wb"))
        elif model_case == 't_learner_0':
            print('T-Learner 0')
            score, best_model = get_outcome_model(automl, w, t, y, case='t_0', automl= 1, automl_settings= automl_settings, selection_case= selection_case)
            pickle.dump(best_model, open(save_dir + 'mu_0' + '.p', "wb"))
        elif model_case == 't_learner_1':
            print('T-Learner 1')
            score, best_model = get_outcome_model(automl, w, t, y, case='t_1', automl= 1, automl_settings= automl_settings, selection_case= selection_case)
            pickle.dump(best_model, open(save_dir + 'mu_1' + '.p', "wb"))
        elif model_case == 's_learner':
            print('S-Learner')
            score, best_model = get_s_learner_model(automl, w, t, y, automl= 1, automl_settings= automl_settings, selection_case= selection_case)
            pickle.dump(best_model, open(save_dir + 'mu_s' + '.p', "wb"))
        elif model_case == 'dml':
            print('DML')
            score, best_model = get_r_score_model(automl, w, y, automl= 1, automl_settings= automl_settings, selection_case= selection_case)
            pickle.dump(best_model, open(save_dir + 'mu_r_score' + '.p', "wb"))

        model_sel_res[model_case]['score']= score
        model_sel_res[model_case]['model']= best_model
        print(model_case, score, best_model)

    data_size = x_eval.shape[0]
    if len(x_eval.shape) == 3:
        data_size = x_eval.shape[0] * x_eval.shape[2]
    
    # squeeze all eval data
    x_eval = np.reshape(x_eval, (data_size, x_eval.shape[1]))
    t_eval = np.reshape(t_eval, (data_size))
    yf_eval = np.reshape(yf_eval, (data_size))

    #Computing relevant evaluation metric for ensemble
    nuisance_stats_dir= results_folder + '/' + dataset_name + '/models/'
    # Nuisance Models
    # prop_prob, prop_score = get_nuisance_propensity_pred(x_all, t_all, save_dir=nuisance_stats_dir)
    prop_prob, prop_score = get_nuisance_propensity_pred(x_eval, t_eval, save_dir=nuisance_stats_dir)
    outcome_s_pred = get_nuisance_outome_s_pred(x_eval, t_eval, save_dir=nuisance_stats_dir)
    outcome_t_pred = get_nuisance_outcome_t_pred(x_eval, t_eval, save_dir=nuisance_stats_dir)
    outcome_r_pred = get_nuisance_outcome_r_pred(x_eval, save_dir=nuisance_stats_dir)

    results_in[estimator_name] = {}
    results_out[estimator_name] = {}
    for metric in metrics_set:
        print(metric)
        metric_in = None
        ite_estimate_eval = (y1_in[indices_eval] - y0_in[indices_eval])
        if metric == "value_score":
            # ite_estimate = (y1_in - y0_in)
            # metric_in = metrics.calculate_value_risk(
            #     ite_estimate, x_all, t_all, yf_all, dataset_name=dataset_name, prop_score=prop_score
            # )

            metric_in = metrics.calculate_value_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, dataset_name=dataset_name, prop_score=prop_score
            )
        elif metric == "value_dr_score":
            metric_in = metrics.calculate_value_dr_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, dataset_name=dataset_name, prop_score=prop_score
            )
        elif metric == "value_dr_clip_prop_score":
            metric_in = metrics.calculate_value_dr_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, dataset_name=dataset_name, prop_score=prop_score, min_propensity=0.1
            )
        elif metric == "tau_match_score":
            metric_in = metrics.calculate_tau_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval
            )
        elif metric == "tau_iptw_score":
            metric_in = metrics.calculate_tau_iptw_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, prop_score=prop_score
            )
        elif metric == "tau_iptw_clip_prop_score":
            metric_in = metrics.calculate_tau_iptw_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, prop_score=prop_score, min_propensity=0.1
            )
        elif metric == "tau_dr_score":
            metric_in = metrics.calculate_tau_dr_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, prop_score=prop_score
            )
        elif metric == "tau_dr_clip_prop_score":
            metric_in = metrics.calculate_tau_dr_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, prop_score=prop_score, min_propensity=0.1
            )
        elif metric == "tau_s_score":
            metric_in = metrics.calculate_tau_s_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_s_pred
            )
        elif metric == "tau_t_score":
            metric_in = metrics.calculate_tau_t_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred
            )
        elif metric == "influence_score":
            metric_in = metrics.calculate_influence_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, prop_prob=prop_prob
            )
        elif metric == "influence_clip_prop_score":
            metric_in = metrics.calculate_influence_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, prop_prob=prop_prob, min_propensity=0.1
            )
        elif metric == "r_score":
            metric_in = metrics.calculate_r_risk(
                ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_r_pred, treatment_prob=prop_prob[:, 1]
            )

        if metric_in is None:
            results_in[estimator_name][metric] = {"mean": None}
        else:
            results_in[estimator_name][metric] = {
                "mean": np.mean(metric_in, where=(metric_in != 0)),
            }
        
        metric_out = None
  
        if metric_out is None:
            results_out[estimator_name][metric] = {"mean": None}
        else:
            results_out[estimator_name][metric] = {
                "mean": np.mean(metric_out, where=(metric_out != 0)),
            }
    print(f'{" In-sample results ":-^79}')
    for metric in metrics_set:
        print(metric, estimator_name, results_in[estimator_name][metric])