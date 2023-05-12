from typing import List
from statistics import mean
# from math import sqrt
# import sys
import os
import pickle
# import copy
import numpy as np
# import pandas as pd
# import sklearn
# from sklearn.model_selection import cross_validate, GridSearchCV
# from sklearn.model_selection._search import BaseSearchCV
# from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

# from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet, RidgeClassifier

# from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,\
#     RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# from models.base import BaseGenModel
# from causal_estimators.base import BaseEstimator, BaseIteEstimator

# from econml.dml import CausalForestDML
# from econml.score import RScorer

# from helpers import sample_dataset

STACK_AXIS = 0
CONF = 0.95
REGRESSION_SCORES = ['max_error', 'neg_mean_absolute_error', 'neg_median_absolute_error',
                     'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
REGRESSION_SCORE_DEF = 'r2'
CLASSIFICATION_SCORES = ['accuracy', 'balanced_accuracy', 'average_precision',
                         'f1',
                         'precision',
                         'recall', 'roc_auc']
CLASSIFICATION_SCORE_DEF = 'accuracy'

def get_nuisance_propensity_pred(w, t, save_dir=''):

    if os.path.isfile(save_dir + 'prop' + '.p'):

        #Propensity Model
        data_size= w.shape[0]
        num_realizations = 1
        if len(t.shape) > 1:
            num_realizations = t.shape[1]
        data_size= data_size*num_realizations
        w = np.reshape(w, (data_size, -1))
        t= np.reshape(t, (data_size))
        
        prop_model= pickle.load( open(save_dir + 'prop' + '.p', "rb") )                  
        pred_prob= prop_model.predict_proba(w)
        prop_score= pred_prob[:, 0] * (1-t) + pred_prob[:, 1] * (t)
    
    else:        
        print('Error: Nuisance model not trained for evaluation purposes on this dataset')
    
    return pred_prob, prop_score
    
def get_nuisance_outcome_t_pred(w, t, save_dir=''):

    if os.path.isfile(save_dir + 'mu_0' + '.p') and os.path.isfile(save_dir + 'mu_1' + '.p'):
        
        data_size= w.shape[0]
        num_realizations = 1
        if len(t.shape) > 1:
            num_realizations = t.shape[1]
        data_size= data_size*num_realizations
        w = np.reshape(w, (data_size, -1))
        t= np.reshape(t, (data_size))

        #Outcome Models
        out_model_0= pickle.load( open(save_dir + 'mu_0' + '.p', "rb") )    
        out_model_1= pickle.load( open(save_dir + 'mu_1' + '.p', "rb") )        

        mu_0= out_model_0.predict(w)
        mu_1= out_model_1.predict(w)
    
    else:
        print('Error: Nuisance model not trained for evaluation purposes on this dataset')        
        
    return (mu_0, mu_1)

def get_nuisance_outome_s_pred(w, t, save_dir=''):

    if os.path.isfile(save_dir + 'mu_s' + '.p'):
        
        data_size= w.shape[0]
        num_realizations = 1
        if len(t.shape) > 1:
            num_realizations = t.shape[1]
        data_size= data_size*num_realizations
        w = np.reshape(w, (data_size, -1))
        t= np.reshape(t, (data_size, 1))
        
        t0= t*0
        t1= t*0 + 1        
        w0= np.hstack([w, t0])
        w1= np.hstack([w, t1])

        out_model= pickle.load( open(save_dir + 'mu_s' + '.p', "rb") )        
        mu_0= out_model.predict(w0)
        mu_1= out_model.predict(w1)

    else:        
        print('Error: Nuisance model not trained for evaluation purposes on this dataset')
        
    return (mu_0, mu_1)


def get_nuisance_outcome_r_pred(w, save_dir=''):

    if os.path.isfile(save_dir + 'mu_r_score' + '.p'):

        data_size= w.shape[0]
        num_realizations = 1
        if len(w.shape) > 2:
            num_realizations = w.shape[-1]
        data_size= data_size*num_realizations
        w = np.reshape(w, (data_size, -1))

        out_model = pickle.load(open(save_dir + 'mu_r_score' + '.p', "rb"))
        mu_r= out_model.predict(w)

    else:
        print('Error: Nuisance model not trained for evaluation purposes on this dataset')

    return mu_r


# def calculate_metrics(dataset_name, dataset_obj,
#                       estimator_name, estimator: BaseEstimator, nuisance_model_config,
#                       seed: int, conf_ints=True, return_ite_vectors=False,
#                       nuisance_stats_dir= ''):
    

#     fitted_estimators=[]
#     fitted_estimators.append(copy.deepcopy(estimator))
    
#     #Evaluation Data
#     dataset_samples= sample_dataset(dataset_name, dataset_obj, seed=seed, case='eval')
#     eval_w, eval_t, eval_y, ate, ite= dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], dataset_samples['ate'], dataset_samples['ite']

#     #Nuisance Models
#     prop_prob, prop_score= get_nuisance_propensity_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
#     outcome_s_pred= get_nuisance_outome_s_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
#     outcome_t_pred= get_nuisance_outcome_t_pred(eval_w, eval_t, save_dir=nuisance_stats_dir)
#     outcome_r_pred= get_nuisance_outcome_r_pred(eval_w, save_dir=nuisance_stats_dir)

#     #ITE Metrics
#     ite_metrics = calculate_ite_metrics(ite, fitted_estimators, eval_w, eval_t)
#     ite_mean_metrics = {k: np.mean(v) for k, v in ite_metrics.items()}
# #         ite_std_metrics = {'std_of_' + k: np.std(v) for k, v in ite_metrics.items()}    

#     metrics= {}
#     metrics.update(ite_mean_metrics)
#     if return_ite_vectors:
#         metrics.update(ite_metrics)

#     # Estimator's ITE
#     eval_t0 = eval_t * 0
#     eval_t1 = eval_t * 0 + 1

#     ite_estimates = np.stack([fitted_estimator.effect(eval_w, eval_t0, eval_t1) for
#                               fitted_estimator in fitted_estimators],
#                              axis=STACK_AXIS)

#     #Saving estimators ITE estimate
#     metrics.update({'ite-estimates': ite_estimates})

#     #Compute Value Risk
#     value_score= calculate_value_risk(ite_estimates, eval_w, eval_t, eval_y, dataset_name, prop_score= prop_score)
#     metrics.update(value_score)

#     #Compute Value DR Risk
#     value_score= calculate_value_dr_risk(ite_estimates, eval_w, eval_t, eval_y, dataset_name, outcome_pred= outcome_t_pred, prop_score= prop_score, min_propensity= 0.1)
#     metrics.update(value_score)

#     #Compute Tau Matching Risk
#     tau_score= calculate_tau_risk(ite_estimates, eval_w, eval_t, eval_y)
#     metrics.update(tau_score)

#     #Compute Tau Risk with IPTW
#     tau_score= calculate_tau_iptw_risk(ite_estimates,eval_w, eval_t, eval_y, prop_score= prop_score, min_propensity= 0.1)
#     metrics.update(tau_score)

#     #Compute Tau DR Risk
#     tau_score= calculate_tau_dr_risk(ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_score= prop_score, min_propensity= 0.1)
#     metrics.update(tau_score)

#     #Compute Plug In Tau Score from Van der Schaar paper using S-Learner
#     tau_s_score= calculate_tau_s_risk(ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_s_pred)
#     metrics.update(tau_s_score)

#     #Compute Plug In Tau Score from Van der Schaar paper using T-Learner
#     tau_plugin_score= calculate_tau_t_risk(ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred)
#     metrics.update(tau_plugin_score)

#     #Compute Van de Schaar Influence function
#     influence_score= calculate_influence_risk(ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_t_pred, prop_prob= prop_prob, min_propensity= 0.1)
#     metrics.update(influence_score)

#     # Compute RScore Metric
#     rscore_metrics= calculate_r_risk(ite_estimates, eval_w, eval_t, eval_y, outcome_pred= outcome_r_pred, treatment_prob=prop_prob[:, 1])
#     metrics.update(rscore_metrics)

#     return metrics


def calculate_r_risk(ite_estimates, w, t, y, outcome_pred= [], treatment_prob=[]):

    data_size = w.shape[0]
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))
    ite_estimates = np.reshape(ite_estimates, (data_size))

    mu= outcome_pred
    # print(y.shape, mu.shape, t.shape, ite_estimates.shape, prop_score.shape)

    # R Score
    r_score= np.mean(( (y-mu) - ite_estimates*(t-treatment_prob)) ** 2)

    out = {}
    out['rscore'] = r_score

    return out


def calculate_value_risk(ite_estimates, w, t, y, dataset_name, prop_score=[]):
    # TODO: Defining (t0, t1) assumes the treatment is binary. Are we going to work with non binary treatments?

    t0 = t * 0
    t1 = t * 0 + 1

    data_size = w.shape[0]
    ite_estimates = np.reshape(ite_estimates, (data_size))
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))

    #     print('W', w.shape)
    #     print('T', t.shape)
    #     print('Y', y.shape)
    #     print('Prop Score', prop_score.shape)
    #     print('ITE', ite_estimates.shape)

    # Decision policy: Recommend treatment based ITE. Check whether positive ITE is desirable for the datsaet or not
    if dataset_name not in ['twins']:
        decision_policy = 1 * (ite_estimates > 0)
    else:
        decision_policy = 1 * (ite_estimates < 0)

    decision_policy = np.reshape(decision_policy, (data_size))
    #     print('Decision Policy', decision_policy.shape)

    #     print(np.unique(t, return_counts=True))
    #     print(np.unique(decision_policy, return_counts=True))
    #     print(np.sum(t == decision_policy))

    indices = t == decision_policy
    weighted_outcome = y / prop_score

    #     print(np.sum(indices), data_size)
    #     print(np.mean(weighted_outcome[indices]), np.sum(weighted_outcome[indices])/np.sum(indices))
    #     print(np.mean(weighted_outcome), np.sum(weighted_outcome)/data_size)
    value_score = np.sum(weighted_outcome[indices]) / data_size
    if dataset_name not in ['twins']:
        value_score = -1*value_score

    out = {}
    out['value_score'] = value_score

    return out


def calculate_value_dr_risk(ite_estimates, w, t, y, dataset_name, outcome_pred=[], prop_score=[], min_propensity=0):
    # TODO: Defining (t0, t1) assumes the treatment is binary
    t0 = t * 0
    t1 = t * 0 + 1

    data_size = w.shape[0]
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))
    ite_estimates = np.reshape(ite_estimates, (data_size))

    mu_0, mu_1 = outcome_pred
    mu = mu_0 * (1 - t) + mu_1 * (t)

    # Decision Policy: Recommend treatment based ITE. Check whether positive ITE is desirable for the datsaet or not
    if dataset_name not in ['twins']:
        decision_policy = 1 * (ite_estimates > 0)
    else:
        decision_policy = 1 * (ite_estimates < 0)
    decision_policy = np.reshape(decision_policy, (data_size))
    #     print('Decision Policy', decision_policy.shape)

    # Value DR Score
    value_dr_score = decision_policy * (mu_1 - mu_0 + (2 * t - 1) * (y - mu) / prop_score)
    if dataset_name not in ['twins']:
        value_dr_score = -1*value_dr_score

    out={}
    out['value_dr_score'] = np.mean(value_dr_score)

    if min_propensity:
        indices= np.where(np.logical_and(prop_score >= min_propensity, prop_score <= 1-min_propensity))[0]
        out['value_dr_clip_prop_score']= np.mean(value_dr_score[indices])

    return out


def nearest_observed_counterfactual(x, X, Y):
    
    dist= np.sum((X-x)**2, axis=1)
    idx= np.argmin(dist)
    return Y[idx]

def calculate_tau_risk(ite_estimates, w, t, y, iptw=False, prop_score=[]):
    
    #TODO: Defining (t0, t1) assumes the treatment is binary  
    t0= t*0
    t1= t*0 + 1

    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))
    
    X0= w[t==0, :]
    X1= w[t==1, :]
    Y0= y[t==0]
    Y1= y[t==1]    
    
    match_estimates_ite= np.zeros(data_size)
    
    cf_y=np.zeros(data_size)
    for idx in range(data_size):
        curr_x= w[idx]
        curr_t= t[idx]
        curr_y= y[idx]
        #Approximating counterfactual by mathching
        if curr_t == 1:
            cf_y[idx] = nearest_observed_counterfactual(curr_x, X0, Y0) 
        elif curr_t == 0:
            cf_y[idx] = nearest_observed_counterfactual(curr_x, X1, Y1) 
                
    match_estimates_ite= (2*t -1)*(y - cf_y)        

    tau_score= np.mean((ite_estimates - match_estimates_ite)**2)
    
    out={}    
    out['tau_match_score']= tau_score
    
    return out

def calculate_tau_iptw_risk(ite_estimates, w, t, y, prop_score= [], min_propensity= 0):
    
    #TODO: Defining (t0, t1) assumes the treatment is binary
    t0= t*0
    t1= t*0 + 1
    
    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))

    #Compute the Tau IPTW Score
    tau_iptw_score = (ite_estimates - (2 * t - 1) * (y) / prop_score) ** 2

    out={}
    out['tau_iptw_score']= np.mean(tau_iptw_score)

    #Propesnity clipping version
    if min_propensity:
        indices= np.where(np.logical_and(prop_score >= min_propensity, prop_score <= 1-min_propensity))[0]
        out['tau_iptw_clip_prop_score']= np.mean( tau_iptw_score[indices] )

    return out

def calculate_tau_dr_risk(ite_estimates, w, t, y, outcome_pred= [], prop_score=[], min_propensity=0):
    
    #TODO: Defining (t0, t1) assumes the treatment is binary  
    t0= t*0
    t1= t*0 + 1
    
    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))    
    
    mu_0, mu_1= outcome_pred
    mu= mu_0 * (1-t) + mu_1 * (t)
    
    #Tau DR Score
    tau_dr_score = (ite_estimates - (mu_1 - mu_0 + (2 * t - 1) * (y - mu) / prop_score)) ** 2

    out={}
    out['tau_dr_score']= np.mean(tau_dr_score)

    #Propesnity clipping version
    if min_propensity:
        indices= np.where(np.logical_and(prop_score >= min_propensity, prop_score <= 1-min_propensity))[0]
        out['tau_dr_clip_prop_score']= np.mean( tau_dr_score[indices] )

    return out
    

def calculate_tau_s_risk(ite_estimates, w, t, y, outcome_pred=[]):
    
    #TODO: Defining (t0, t1) assumes the treatment is binary  
    t0= t*0
    t1= t*0 + 1
    
    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))
    
    mu_0, mu_1= outcome_pred
    s_learner_ite= mu_1 - mu_0

    #Plug In Score
    tau_plugin_score= np.mean( (ite_estimates - s_learner_ite)**2 )

    out={}
    out['tau_s_score']= tau_plugin_score
    
    return out


def calculate_tau_t_risk(ite_estimates, w, t, y, outcome_pred=[]):
    
    #TODO: Defining (t0, t1) assumes the treatment is binary  
    t0= t*0
    t1= t*0 + 1
    
    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))
    
    mu_0, mu_1= outcome_pred
    t_learner_ite= mu_1 - mu_0
    
    #Influence Score
    tau_plugin_score= np.mean( (ite_estimates - t_learner_ite)**2 )
        
    out={}    
    out['tau_t_score']= tau_plugin_score
    
    return out

    
def calculate_influence_risk(ite_estimates, w, t, y, outcome_pred=[], prop_prob=[], min_propensity=0):
    
    #TODO: Defining (t0, t1) assumes the treatment is binary  
    t0= t*0
    t1= t*0 + 1

    # print inputs
    # print outcome_pred
    # print prop_prob
    # print min_propensity

    print("ite_estimates", ite_estimates.shape)
    print("w", w.shape)
    print("t", t.shape)
    print("y", y.shape)
    print("outcome_pred", outcome_pred)
    print("prop_prob", prop_prob)
    print("min_propensity", min_propensity)
    
    data_size= w.shape[0]
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))
    
    mu_0, mu_1= outcome_pred
    mu= mu_0 * (1-t) + mu_1 * (t)
    prop_score = prop_prob[:, 0] * (1 - t) + prop_prob[:, 1] * (t)

    t_learner_ite= mu_1 - mu_0
    plug_in_estimate= t_learner_ite - ite_estimates
    A= t - prop_prob[:, 1]
    C= prop_prob[:, 0] * prop_prob[:, 1]
    B= 2*t*(t- prop_prob[:, 1])*(1/C)

    #Influence Score
    influence_score = (1 - B) * (t_learner_ite ** 2) + B * y * plug_in_estimate - A * (plug_in_estimate ** 2) + ite_estimates ** 2

    out= {}
    out['influence_score'] = np.mean(influence_score)

    #Propesnity clipping version
    if min_propensity:
        indices= np.where(np.logical_and(prop_score >= min_propensity, prop_score <= 1-min_propensity))[0]
        out['influence_clip_prop_score']= np.mean(influence_score[indices])

    return out


# def calculate_ite_metrics(ite: np.ndarray, fitted_estimators: List[BaseIteEstimator], w, t):
    
#     #TODO: Defining (t0, t1) assumes the treatment is binary  
#     t0= t*0
#     t1= t*0 + 1
    
#     ite_estimates = np.stack([fitted_estimator.effect(w, t0, t1) for
#                               fitted_estimator in fitted_estimators],
#                              axis=STACK_AXIS)

#     # Calculated for each unit/individual, this is the a vector of num units
#     mean_ite_estimate = ite_estimates.mean(axis=STACK_AXIS)
#     ite_bias = mean_ite_estimate - ite
#     ite_abs_bias = np.abs(ite_bias)
#     ite_squared_bias = ite_bias**2
#     ite_variance = calc_vector_variance(ite_estimates, mean_ite_estimate)
#     ite_std_error = np.sqrt(ite_variance)
#     ite_mse = calc_vector_mse(ite_estimates, ite)
#     ite_rmse = np.sqrt(ite_mse)

#     # Calculated for a single dataset, so this is a vector of num datasets
#     pehe_squared = calc_vector_mse(ite_estimates, ite, reduce_axis=(1 - STACK_AXIS))
#     pehe = np.sqrt(pehe_squared)

#     #True Variance of ITE estimates
#     true_ite_var= [np.var(ite)]

# #     #Standard Scale
# #     pehe_vec= np.sqrt((ite_estimates - ite)**2)
# #     standard_scale= [(pehe_vec - np.mean(pehe_vec))/np.sqrt(np.var(pehe_vec))]

#     # TODO: ITE coverage
#     # ate_coverage = calc_coverage(ate_conf_ints, ate)
#     # ate_mean_int_length = calc_mean_interval_length(ate_conf_ints)

#     return {
#         'ite_bias': ite_bias,
#         # 'ite_abs_bias': ite_abs_bias,
#         # 'ite_squared_bias': ite_squared_bias,
#         'ite_variance': ite_variance,
#         'ite_std_error': ite_std_error,
# #         'ite_mse': ite_mse,
# #         'ite_rmse': ite_rmse,
#         # 'ite_coverage': ite_coverage,
#         # 'ite_mean_int_length': ite_mean_int_length,
#         'true_ite_var': true_ite_var,
#         'pehe_squared': pehe_squared,
#         'pehe': pehe,
#     }


def calc_variance(estimates, mean_estimate):
    return calc_mse(estimates, mean_estimate)


def calc_mse(estimates, target):
    if isinstance(estimates, (list, tuple)):
        estimates = np.array(estimates)
    return ((estimates - target) ** 2).mean()


def calc_coverage(intervals: List[tuple], estimand):
    n_covers = sum(1 for interval in intervals if interval[0] <= estimand <= interval[1])
    return n_covers / len(intervals)


def calc_mean_interval_length(intervals: List[tuple]):
    return mean(interval[1] - interval[0] for interval in intervals)


def calc_vector_variance(estimates: np.ndarray, mean_estimate: np.ndarray):
    return calc_vector_mse(estimates, mean_estimate)


def calc_vector_mse(estimates: np.ndarray, target: np.ndarray, reduce_axis=STACK_AXIS):
    assert isinstance(estimates, np.ndarray) and estimates.ndim == 2
    assert isinstance(target, np.ndarray) and target.ndim == 1
    assert target.shape[0] == estimates.shape[1 - STACK_AXIS]

    n_seeds = estimates.shape[STACK_AXIS]
    target = np.expand_dims(target, axis=STACK_AXIS).repeat(n_seeds, axis=STACK_AXIS)
    return ((estimates - target) ** 2).mean(axis=reduce_axis)
