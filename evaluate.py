import os
from sklearn.model_selection import train_test_split
from typing import Dict

import numpy as np

import utils.metrics as metrics
import utils.preprocessing as helper
from utils.preprocessing import sys_config
from utils.evaluation import *

results_folder = sys_config["results_folder"]
datasets_folder = sys_config["datasets_folder"]

seed = 0 # same seed as in nuisance_model_selection.py
np.random.seed(seed)

dataset_name = "TWINS" # optionas are "Jobs", "IHDP-100", "TWINS", "census_e1", "census_e2

estimator_set = [
    "OLS1",
    "RF1",
    "NN1",
    "OLS2",
    "RF2",
    "NN2",
    "TARNet",
    "Dragonnet",
    "DML",
]

metrics_set = [
    "MAE", 
    "PEHE", 
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
    "abs_diff_ate_t",
    "abs_diff_ate_s"
]

if __name__ == "__main__":
    print(f'{" Evaluation ":-^79}')
    results_in: Dict[str, Dict] = {}
    results_out: Dict[str, Dict] = {}
    if dataset_name == "Jobs":
        ate_in_gt, ate_out_gt = helper.load_Jobs_ground_truth(
            datasets_folder, dataset_name, details=False
        )
        mu0_in, mu1_in, mu0_out, mu1_out = None, None, None, None
        x_all, t_all, yf_all = helper.load_Jobs_observational(
            datasets_folder, dataset_name, details=False
        )
        x_test_all, t_test_all, yf_test_all = helper.load_Jobs_out_of_sample(
            datasets_folder, dataset_name, details=False
        )
    elif "IHDP" in dataset_name:
        mu0_in, mu1_in, mu0_out, mu1_out = helper.load_IHDP_ground_truth(
            datasets_folder, dataset_name, details=False
        )
        ate_in_gt = np.mean(mu1_in - mu0_in)
        ate_out_gt = np.mean(mu1_out - mu0_out)
        x_all, t_all, yf_all = helper.load_IHDP_observational(
            datasets_folder, dataset_name, details=False
        )
        yf_all = helper.scale_y(yf_all)
        x_test_all, t_test_all, yf_test_all = helper.load_IHDP_out_of_sample(
            datasets_folder, dataset_name, details=False
        )
        yf_test_all = helper.scale_y(yf_test_all)
    elif dataset_name == "TWINS":
        mu0_in, mu1_in, mu0_out, mu1_out = helper.load_TWINS_ground_truth(
            datasets_folder, dataset_name, details=False
        )
        ate_in_gt = np.mean(mu1_in - mu0_in)
        ate_out_gt = np.mean(mu1_out - mu0_out)
        x_all, t_all, yf_all = helper.load_TWINS_observational(
            datasets_folder, dataset_name, details=False
        )
        x_test_all, t_test_all, yf_test_all = helper.load_TWINS_out_of_sample(
            datasets_folder, dataset_name, details=False
        )
        mu0_in, mu1_in, mu0_out, mu1_out = helper.load_TWINS_ground_truth(
            datasets_folder, dataset_name, details=False
        )
    elif "census" in dataset_name:
        datasets_folder = os.path.join(datasets_folder, "CFA")
        mu0_in, mu1_in, mu0_out, mu1_out = None, None, None, None
        ate_in_gt = None
        ate_out_gt = None
        x_all, t_all, yf_all = helper.load_census_observational(
            datasets_folder, dataset_name, details=False
        )
        x_test_all, t_test_all, yf_test_all = helper.load_census_out_of_sample(
            datasets_folder, dataset_name, details=False
        )
    

    # process in sample data
    data_size_in = x_all.shape[0]
    num_realizations_in = 1
    if len(x_all.shape) == 3:
        num_realizations_in = x_all.shape[2]
        new_x_all = np.zeros((data_size_in * num_realizations_in, x_all.shape[1]))
        for i in range(num_realizations_in):
            new_x_all[i * data_size_in : (i + 1) * data_size_in, :] = x_all[:, :, i]
        x_all = new_x_all

    
    # squeeze all eval data
    x_all = np.reshape(x_all, (data_size_in*num_realizations_in, x_all.shape[1]))
    t_all = np.reshape(t_all, (data_size_in * num_realizations_in), order='F')
    yf_all = np.reshape(yf_all, (data_size_in * num_realizations_in), order='F')

    indices_all = np.arange(x_all.shape[0])

    x_train, x_eval_orig, t_train, t_eval_orig, yf_train, yf_eval_orig, indices_train, indices_eval = train_test_split(
        x_all, t_all, yf_all, indices_all, test_size=0.2, random_state=seed
        )
    
    # process out of sample data
    data_size_out = x_test_all.shape[0]
    num_realizations_out = 1
    if len(x_test_all.shape) == 3:
        num_realizations_out = x_test_all.shape[2]
        new_x_test_all = np.zeros((data_size_out * num_realizations_out, x_test_all.shape[1]))
        for i in range(num_realizations_out):
            new_x_test_all[i * data_size_out : (i + 1) * data_size_out, :] = x_test_all[:, :, i]
        x_test_all = new_x_test_all
    
    # squeeze all test data
    x_test_all = np.reshape(x_test_all, (data_size_out*num_realizations_out, x_test_all.shape[1]))
    t_test_all = np.reshape(t_test_all, (data_size_out * num_realizations_out), order='F')
    yf_test_all = np.reshape(yf_test_all, (data_size_out * num_realizations_out), order='F')

    #Computing relevant evaluation metric for ensemble for in sample data
    nuisance_stats_dir= results_folder + '//..//models//' + dataset_name + '//'
    # Nuisance Models
    prop_prob_orig, prop_score_orig = get_nuisance_propensity_pred(x_eval_orig, t_eval_orig, save_dir=nuisance_stats_dir)
    outcome_s_pred = get_nuisance_outome_s_pred(x_eval_orig, t_eval_orig, save_dir=nuisance_stats_dir)
    outcome_t_pred = get_nuisance_outcome_t_pred(x_eval_orig, t_eval_orig, save_dir=nuisance_stats_dir)
    outcome_r_pred = get_nuisance_outcome_r_pred(x_eval_orig, save_dir=nuisance_stats_dir)

    outcome_s_pred_orig = np.array(outcome_s_pred)
    outcome_t_pred_orig = np.array(outcome_t_pred)
    outcome_r_pred_orig = np.array(outcome_r_pred)

    #Computing relevant evaluation metric for ensemble for out of sample data
    # Nuisance Models
    prop_prob_test, prop_score_test = get_nuisance_propensity_pred(x_test_all, t_test_all, save_dir=nuisance_stats_dir)
    outcome_s_pred = get_nuisance_outome_s_pred(x_test_all, t_test_all, save_dir=nuisance_stats_dir)
    outcome_t_pred = get_nuisance_outcome_t_pred(x_test_all, t_test_all, save_dir=nuisance_stats_dir)
    outcome_r_pred = get_nuisance_outcome_r_pred(x_test_all, save_dir=nuisance_stats_dir)

    outcome_s_pred_test = np.array(outcome_s_pred)
    outcome_t_pred_test = np.array(outcome_t_pred)
    outcome_r_pred_test = np.array(outcome_r_pred)

    for estimator_name in estimator_set:
        results_in[estimator_name] = {}
        results_out[estimator_name] = {}
        for z in range(10):
            estimation_result_folder = os.path.join(
                results_folder, dataset_name+"_"+str(z+1), estimator_name
            )
            (
                y0_in,
                y1_in,
                ate_in,
                y0_out,
                y1_out,
                ate_out,
            ) = helper.load_in_and_out_results(estimation_result_folder)

            if dataset_name == "TWINS":
                y0_in = y0_in.reshape((-1, 1))
                y1_in = y1_in.reshape((-1, 1))
                y0_out = y0_out.reshape((-1, 1))
                y1_out = y1_out.reshape((-1, 1))
                ate_in = ate_in.reshape((-1, 1))
                ate_out = ate_out.reshape((-1, 1))

            if estimator_name == "DML":
                # create dummy y0_in and y1_in and y0_out and y1_out
                y0_in = np.zeros((t_all.shape[0], 1))
                y1_in = np.zeros((t_all.shape[0], 1))
                y0_out = np.zeros((t_test_all.shape[0], 1))
                y1_out = np.zeros((t_test_all.shape[0], 1))

            
            # process in sample data
            ite_estimate_in = y1_in.reshape((-1, 1), order='F') - y0_in.reshape((-1, 1), order='F')
            ite_estimate_eval = ite_estimate_in[indices_eval]

            # get number of nan values
            num_nan = np.sum(np.isnan(ite_estimate_eval))

            # get indices of non nan values
            non_nan = ~np.isnan(ite_estimate_eval)
            non_nan_inds = np.where(non_nan)[0]
            ite_estimate_eval = ite_estimate_eval[non_nan_inds]
            x_eval = np.take(x_eval_orig, non_nan_inds, axis=0)
            t_eval = t_eval_orig[non_nan.squeeze()]
            yf_eval = yf_eval_orig[non_nan.squeeze()]
            
            
            prop_score = prop_score_orig[non_nan.squeeze()]
            outcome_s_pred = np.transpose(np.transpose(outcome_s_pred_orig)[non_nan.squeeze()])
            outcome_t_pred = np.transpose(np.transpose(outcome_t_pred_orig)[non_nan.squeeze()])
            outcome_r_pred = outcome_r_pred_orig[non_nan.squeeze()]
            prop_prob = prop_prob_orig[non_nan.squeeze()]

            # process out of sample data
            ite_estimate_out = y1_out.reshape((-1, 1), order='F') - y0_out.reshape((-1, 1), order='F')
            ite_estimate_eval_out = ite_estimate_out

            # get number of nan values
            num_nan = np.sum(np.isnan(ite_estimate_eval_out))

            # get indices of non nan values
            non_nan = ~np.isnan(ite_estimate_eval_out)
            non_nan_inds = np.where(non_nan)[0]
            ite_estimate_eval_out = ite_estimate_eval_out[non_nan_inds]
            x_eval_out = np.take(x_test_all, non_nan_inds, axis=0)
            t_eval_out = t_test_all[non_nan.squeeze()]
            yf_eval_out = yf_test_all[non_nan.squeeze()]

            prop_score_out = prop_score_test[non_nan.squeeze()]
            outcome_s_pred_out = np.transpose(np.transpose(outcome_s_pred_test)[non_nan.squeeze()])
            outcome_t_pred_out = np.transpose(np.transpose(outcome_t_pred_test)[non_nan.squeeze()])
            outcome_r_pred_out = outcome_r_pred_test[non_nan.squeeze()]
            prop_prob_out = prop_prob_test[non_nan.squeeze()]

            # print("Number of nan values in ITE estimate: ", num_nan)
            # print("Number of non nan values in ITE estimate: ", len(non_nan_inds))

            for metric in metrics_set:
                metric_in = None
                metric_out = None

                if metric == "MAE" and estimator_name == "DML":
                    metric_in = np.abs(ate_in_gt - ate_in)
                    metric_out = np.abs(ate_out_gt - ate_out)                
                elif metric in ["MAE", "PEHE"]:
                    metric_in = metrics.calculate_metrics(
                        y0_in, y1_in, ate_in, mu0_in, mu1_in, ate_in_gt, metric=metric
                    )
                    metric_out = metrics.calculate_metrics(
                        y0_out, y1_out, ate_out, mu0_out, mu1_out, ate_out_gt, metric=metric
                    )
                elif metric == "value_score":
                    metric_in = metrics.calculate_value_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, dataset_name=dataset_name, prop_score=prop_score
                    )
                    metric_out = metrics.calculate_value_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, dataset_name=dataset_name, prop_score=prop_score_out
                    )
                elif metric == "value_dr_score":
                    metric_in = metrics.calculate_value_dr_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, dataset_name=dataset_name, prop_score=prop_score
                    )
                    metric_out = metrics.calculate_value_dr_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, outcome_pred=outcome_t_pred_out, dataset_name=dataset_name, prop_score=prop_score_out
                    )
                elif metric == "value_dr_clip_prop_score":
                    metric_in = metrics.calculate_value_dr_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, dataset_name=dataset_name, prop_score=prop_score, min_propensity=0.1
                    )
                    metric_out = metrics.calculate_value_dr_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, outcome_pred=outcome_t_pred_out, dataset_name=dataset_name, prop_score=prop_score_out, min_propensity=0.1
                    )
                elif metric == "tau_match_score":
                    metric_in = metrics.calculate_tau_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval
                    )
                    metric_out = metrics.calculate_tau_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out
                    )
                elif metric == "tau_iptw_score":
                    metric_in = metrics.calculate_tau_iptw_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, prop_score=prop_score
                    )
                    metric_out = metrics.calculate_tau_iptw_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, prop_score=prop_score_out
                    )
                elif metric == "tau_iptw_clip_prop_score":
                    metric_in = metrics.calculate_tau_iptw_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, prop_score=prop_score, min_propensity=0.1
                    )
                    metric_out = metrics.calculate_tau_iptw_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, prop_score=prop_score_out, min_propensity=0.1
                    )
                elif metric == "tau_dr_score":
                    metric_in = metrics.calculate_tau_dr_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, prop_score=prop_score
                    )
                    metric_out = metrics.calculate_tau_dr_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, outcome_pred=outcome_t_pred_out, prop_score=prop_score_out
                    )
                elif metric == "tau_dr_clip_prop_score":
                    metric_in = metrics.calculate_tau_dr_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, prop_score=prop_score, min_propensity=0.1
                    )
                    metric_out = metrics.calculate_tau_dr_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, outcome_pred=outcome_t_pred_out, prop_score=prop_score_out, min_propensity=0.1
                    )
                elif metric == "tau_s_score":
                    metric_in = metrics.calculate_tau_s_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_s_pred
                    )
                    metric_out = metrics.calculate_tau_s_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, outcome_pred=outcome_s_pred_out
                    )
                elif metric == "tau_t_score":
                    metric_in = metrics.calculate_tau_t_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred
                    )
                    metric_out = metrics.calculate_tau_t_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, outcome_pred=outcome_t_pred_out
                    )
                elif metric == "influence_score":
                    metric_in = metrics.calculate_influence_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, prop_prob=prop_prob
                    )
                    metric_out = metrics.calculate_influence_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, outcome_pred=outcome_t_pred_out, prop_prob=prop_prob_out
                    )
                elif metric == "influence_clip_prop_score":
                    metric_in = metrics.calculate_influence_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_t_pred, prop_prob=prop_prob, min_propensity=0.1
                    )
                    metric_out = metrics.calculate_influence_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, outcome_pred=outcome_t_pred_out, prop_prob=prop_prob_out, min_propensity=0.1
                    )
                elif metric == "r_score":
                    metric_in = metrics.calculate_r_risk(
                        ite_estimate_eval, x_eval, t_eval, yf_eval, outcome_pred=outcome_r_pred, treatment_prob=prop_prob[:, 1]
                    )
                    metric_out = metrics.calculate_r_risk(
                        ite_estimate_eval_out, x_eval_out, t_eval_out, yf_eval_out, outcome_pred=outcome_r_pred_out, treatment_prob=prop_prob_out[:, 1]
                    )
                elif metric == "abs_diff_ate_t":
                    metric_in = metrics.calculate_abs_diff_ate(
                        ate_in, outcome_pred=outcome_t_pred
                    )
                    metric_out = metrics.calculate_abs_diff_ate(
                        ate_out, outcome_pred=outcome_t_pred_out
                    )
                elif metric == "abs_diff_ate_s":
                    metric_in = metrics.calculate_abs_diff_ate(
                        ate_in, outcome_pred=outcome_s_pred
                    )
                    metric_out = metrics.calculate_abs_diff_ate(
                        ate_out, outcome_pred=outcome_s_pred_out
                    )

                if metric not in results_in[estimator_name]:
                    results_in[estimator_name][metric] = []
                if metric_in is not None:
                    results_in[estimator_name][metric] += [metric_in]

                if metric not in results_out[estimator_name]:
                    results_out[estimator_name][metric] = []
                if metric_out is not None:
                    results_out[estimator_name][metric] += [metric_out]
    print(f'{" In-sample results ":-^79}')
    for metric in metrics_set:
        for estimator_name in estimator_set:
            print(metric, estimator_name, "mean", np.mean(results_in[estimator_name][metric]), "std", np.std(results_in[estimator_name][metric]))
    print(f'{" Out-of-sample results ":-^79}')
    for metric in metrics_set:
        for estimator_name in estimator_set:
            print(metric, estimator_name, "mean", np.mean(results_out[estimator_name][metric]), "std", np.std(results_out[estimator_name][metric]))
