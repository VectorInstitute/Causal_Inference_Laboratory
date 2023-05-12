import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error


def calculate_mae_ATE(y0, y1, mu0, mu1):
    """
    Calculate the mean absolute error of ATE estimation
    :param y0: estimation of y0
    :param y1: estimation of y1
    :param mu0: ground-truth of potential outcome of T = 0
    :param mu1: ground-truth of potential outcome of T = 1
    :return:
    """
    ATE_gt = np.mean(mu1 - mu0)
    ATE_pred = np.mean(y1 - y0)
    return np.abs(ATE_gt - ATE_pred)


def calculate_mae_ATEs(y0, y1, mu0, mu1):
    """
    Calculate the mean absolute error of ATE estimation of
    multiple realizations/datasets
    :param y0: estimation of y0 of multiple realizations
    :param y1: estimation of y1 of multiple realizations
    :param mu0: ground-truth of y0 of multiple realizations
    :param mu1: ground-truth of y1 of multiple realizations
    :return: ndarray of PEHEs of all realizations
    """
    assert y0.shape == mu0.shape, f"shape of y0 and mu0: {y0.shape}{mu0.shape}"
    assert y1.shape == mu1.shape, f"shape of y1 and mu1: {y1.shape}{mu1.shape}"
    assert y0.shape == y1.shape, f"shape of y0 and y1: {y0.shape}{y1.shape}"
    num_realizations = y0.shape[-1]
    mae_ATE = np.zeros(num_realizations)
    for i in range(num_realizations):
        if np.isnan(np.min(y0[:, i])) or np.isnan(np.min(y1[:, i])):
            mae_ATE[i] = 0.0  # will compensate for this later
            continue
        mae_ATE[i] = calculate_mae_ATE(
            y0[:, i], y1[:, i], mu0[:, i], mu1[:, i]
        )
    return mae_ATE


def calculate_mae_ATEs_scalar(y0, y1, ate):
    """
    Calculate the mean absolute error of ATE estimation of
    multiple realizations/datasets
    :param y0: estimation of y0 of multiple realizations
    :param y1: estimation of y1 of multiple realizations
    :param ate: ground-truth ate of multiple realizations
    :return: ndarray of PEHEs of all realizations
    """
    assert y0.shape == y1.shape, f"shape of y0 and y1: {y0.shape}{y1.shape}"
    num_realizations = y0.shape[-1]
    # each realization has an ate
    if np.isscalar(ate) or ate.shape == (1,):
        ate = ate * np.ones(num_realizations)
    elif ate.shape == (1, 1):
        ate = ate.item() * np.ones(num_realizations)
    mae_ATE = np.zeros(num_realizations)
    for i in range(num_realizations):
        mae_ATE[i] = np.abs(np.mean(y1[:, i] - y0[:, i]) - ate[i])
    return mae_ATE


def calculate_mae_ATEs_scalar_scalar(ate, ate_gt):
    """
    Calculate the mean absolute error of ATE estimation of
    multiple realizations/datasets
    :param ate: prediction of ate of multiple realizations
    :param ate_gt: ground-truth ate of multiple realizations
    :return: ndarray of PEHEs of all realizations
    """
    num_realizations = ate.size
    if np.isscalar(ate_gt) or ate_gt.shape == (1,):
        ate_gt = ate_gt * np.ones(num_realizations)
    elif ate_gt.shape == (1, 1):
        ate_gt = ate_gt.item() * np.ones(num_realizations)
    # each realization has an ate
    mae_ATE = np.zeros(num_realizations)
    for i in range(num_realizations):
        mae_ATE[i] = np.abs(ate[i] - ate_gt[i])
    return mae_ATE


def calculate_PEHE(y0, y1, mu0, mu1):
    """
    Calculate the Precision Estimation of Heterogeneous Effect (PEHE) of
    one realization/dataset
    :param y0: estimation of y0
    :param y1: estimation of y1
    :param mu0: ground-truth of potential outcome of T = 0
    :param mu1: ground-truth of potential outcome of T = 1
    :return:
    """
    return np.sqrt(mean_squared_error(mu1 - mu0, y1 - y0))


def calculate_PEHEs(y0, y1, mu0, mu1):
    """
    Calculate the Precision Estimation of Heterogeneous Effect (PEHE) of
    multiple realizations/datasets
    :param y0: estimation of y0 of multiple realizations
    :param y1: estimation of y1 of multiple realizations
    :param mu0: ground-truth of y0 of multiple realizations
    :param mu1: ground-truth of y1 of multiple realizations
    """
    assert y0.shape == mu0.shape, f"shape of y0 and mu0: {y0.shape}{mu0.shape}"
    assert y1.shape == mu1.shape, f"shape of y1 and mu1: {y1.shape}{mu1.shape}"
    assert y0.shape == y1.shape, f"shape of y0 and y1: {y0.shape}{y1.shape}"
    num_realizations = y0.shape[-1]
    pehe = np.zeros(num_realizations)
    for i in range(num_realizations):
        if np.isnan(np.min(y0[:, i])) or np.isnan(np.min(y1[:, i])):
            pehe[i] = 0.0  # will compensate for this later
            continue
        pehe[i] = calculate_PEHE(y0[:, i], y1[:, i], mu0[:, i], mu1[:, i])
    return pehe


def calculate_metrics(y0, y1, ate, mu0, mu1, ate_gt, metric="PEHE"):
    if ate is None or ate[0] is None:
        return None
    if y0[0] is None:  # no individual prediction
        return calculate_mae_ATEs_scalar_scalar(ate, ate_gt)
    if mu0 is None:
        metric_over_realizations = calculate_mae_ATEs_scalar(y0, y1, ate_gt)
        return metric_over_realizations
    if metric == "PEHE":
        metric_over_realizations = calculate_PEHEs(y0, y1, mu0, mu1)
    elif metric == "MAE":
        metric_over_realizations = calculate_mae_ATEs(y0, y1, mu0, mu1)
    return metric_over_realizations

def calculate_value_risk(ite_estimates, w, t, y, dataset_name, prop_score=[]):
    # TODO: Defining (t0, t1) assumes the treatment is binary. Are we going to work with non binary treatments?
    data_size = t.shape[0]
    num_realizations = 1
    if len(t.shape) > 1:
        num_realizations = t.shape[1]
    ite_estimates = np.reshape(ite_estimates, (data_size * num_realizations))
    t = np.reshape(t, (data_size * num_realizations))

    # Decision policy: Recommend treatment based ITE. Check whether positive ITE is desirable for the datsaet or not
    if dataset_name != "TWINS":
        decision_policy = 1 * (ite_estimates > 0)
    else:
        decision_policy = 1 * (ite_estimates < 0)

    decision_policy = np.reshape(decision_policy, (data_size * num_realizations))

    indices = t == decision_policy
    weighted_outcome = y / (prop_score + 1e-8)

    value_score = np.sum(weighted_outcome[indices]) / data_size
    if dataset_name != "TWINS":
        value_score = -1*value_score

    return value_score

def calculate_value_dr_risk(ite_estimates, w, t, y, outcome_pred=[], prop_score=[], min_propensity=0, dataset_name=None):
    # TODO: Defining (t0, t1) assumes the treatment is binary
    t0 = t * 0
    t1 = t * 0 + 1

    data_size = t.shape[0]
    num_realizations = 1
    if len(t.shape) > 1:
        num_realizations = t.shape[1]
    ite_estimates = np.reshape(ite_estimates, (data_size * num_realizations))
    t = np.reshape(t, (data_size * num_realizations))

    mu_0, mu_1 = outcome_pred
    mu = mu_0 * (1 - t) + mu_1 * (t)

    # Decision Policy: Recommend treatment based ITE. Check whether positive ITE is desirable for the datsaet or not
    if dataset_name != "TWINS":
        decision_policy = 1 * (ite_estimates > 0)
    else:
        decision_policy = 1 * (ite_estimates < 0)
    decision_policy = np.reshape(decision_policy, (data_size * num_realizations))
    #     print('Decision Policy', decision_policy.shape)

    # Value DR Score
    value_dr_score = decision_policy * (mu_1 - mu_0 + (2 * t - 1) * (y - mu) / (prop_score + 1e-8))
    if dataset_name != "TWINS":
        value_dr_score = -1*value_dr_score
    
    if min_propensity:
        indices= np.where(np.logical_and(prop_score >= min_propensity, prop_score <= 1-min_propensity))[0]
        return np.mean(value_dr_score[indices])

    return np.mean(value_dr_score)

def nearest_observed_counterfactual(x, X, Y):
    
    dist= np.sum((X-x)**2, axis=1)
    idx= np.argmin(dist)
    return Y[idx]

def calculate_tau_risk(ite_estimates, w, t, y):
    t0= t*0
    t1= t*0 + 1

    data_size= w.shape[0]
    num_realizations = 1
    if len(t.shape) > 1:
        num_realizations = t.shape[1]
    data_size= data_size*num_realizations
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
    
    return tau_score

def calculate_tau_iptw_risk(ite_estimates, w, t, y, prop_score= [], min_propensity= 0):
    
    #TODO: Defining (t0, t1) assumes the treatment is binary
    t0= t*0
    t1= t*0 + 1
    
    data_size= w.shape[0]
    num_realizations = 1
    if len(t.shape) > 1:
        num_realizations = t.shape[1]
    data_size= data_size*num_realizations
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))

    #Compute the Tau IPTW Score
    tau_iptw_score = (ite_estimates - (2 * t - 1) * (y) / (prop_score + 1e-8)) ** 2

    #Propesnity clipping version
    if min_propensity:
        indices= np.where(np.logical_and(prop_score >= min_propensity, prop_score <= 1-min_propensity))[0]
        return np.mean(tau_iptw_score[indices])

    return np.mean(tau_iptw_score)

def calculate_tau_dr_risk(ite_estimates, w, t, y, outcome_pred= [], prop_score=[], min_propensity=0):
    
    #TODO: Defining (t0, t1) assumes the treatment is binary  
    t0= t*0
    t1= t*0 + 1
    
    data_size= w.shape[0]
    num_realizations = 1
    if len(t.shape) > 1:
        num_realizations = t.shape[1]
    data_size= data_size*num_realizations
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))    
    
    mu_0, mu_1= outcome_pred
    mu= mu_0 * (1-t) + mu_1 * (t)
    
    #Tau DR Score
    tau_dr_score = (ite_estimates - (mu_1 - mu_0 + (2 * t - 1) * (y - mu) / (prop_score + 1e-8))) ** 2

    #Propesnity clipping version
    if min_propensity:
        indices= np.where(np.logical_and(prop_score >= min_propensity, prop_score <= 1-min_propensity))[0]
        return np.mean(tau_dr_score[indices])

    return np.mean(tau_dr_score)

def calculate_tau_s_risk(ite_estimates, w, t, y, outcome_pred=[]):
    
    #TODO: Defining (t0, t1) assumes the treatment is binary  
    t0= t*0
    t1= t*0 + 1
    
    data_size= w.shape[0]
    num_realizations = 1
    if len(t.shape) > 1:
        num_realizations = t.shape[1]
    data_size= data_size*num_realizations
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))
    
    mu_0, mu_1= outcome_pred
    s_learner_ite= mu_1 - mu_0

    #Plug In Score
    tau_plugin_score= np.mean( (ite_estimates - s_learner_ite)**2 )

    return tau_plugin_score

def calculate_tau_t_risk(ite_estimates, w, t, y, outcome_pred=[]):
    
    #TODO: Defining (t0, t1) assumes the treatment is binary  
    t0= t*0
    t1= t*0 + 1
    
    data_size= w.shape[0]
    num_realizations = 1
    if len(t.shape) > 1:
        num_realizations = t.shape[1]
    data_size= data_size*num_realizations
    t= np.reshape(t, (data_size))
    y= np.reshape(y, (data_size))
    ite_estimates= np.reshape(ite_estimates, (data_size))
    
    mu_0, mu_1= outcome_pred
    t_learner_ite= mu_1 - mu_0
    
    #Influence Score
    tau_plugin_score= np.mean( (ite_estimates - t_learner_ite)**2 )

    return tau_plugin_score

def calculate_influence_risk(ite_estimates, w, t, y, outcome_pred=[], prop_prob=[], min_propensity=0):

    # print inputs
    # print("ite_estimates", ite_estimates.shape)
    # print("w", w.shape)
    # print("t", t.shape)
    # print("y", y.shape)
    # print("outcome_pred", outcome_pred)
    # print("prop_prob", prop_prob)
    # print("min_propensity", min_propensity)
    
    #TODO: Defining (t0, t1) assumes the treatment is binary  
    t0= t*0
    t1= t*0 + 1
    
    data_size= w.shape[0]
    num_realizations = 1
    if len(t.shape) > 1:
        num_realizations = t.shape[1]
    data_size= data_size*num_realizations
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
    B= 2*t*(t- prop_prob[:, 1])*(1/(C+1e-8))

    #Influence Score
    influence_score = (1 - B) * (t_learner_ite ** 2) + B * y * plug_in_estimate - A * (plug_in_estimate ** 2) + ite_estimates ** 2

    #Propesnity clipping version
    if min_propensity:
        indices= np.where(np.logical_and(prop_score >= min_propensity, prop_score <= 1-min_propensity))[0]
        return np.mean(influence_score[indices])

    return np.mean(influence_score)

def calculate_r_risk(ite_estimates, w, t, y, outcome_pred= [], treatment_prob=[]):

    data_size = w.shape[0]
    num_realizations = 1
    if len(t.shape) > 1:
        num_realizations = t.shape[1]
    data_size = data_size * num_realizations
    t = np.reshape(t, (data_size))
    y = np.reshape(y, (data_size))
    ite_estimates = np.reshape(ite_estimates, (data_size))

    mu= outcome_pred
    # print(y.shape, mu.shape, t.shape, ite_estimates.shape, prop_score.shape)

    # R Score
    r_score= np.mean(( (y-mu) - ite_estimates*(t-treatment_prob)) ** 2)

    return r_score

def binary_classification_loss(concat_true, concat_pred):
    # concat_true: y, t,
    # concat_pred: [y0_predictions, y1_predictions, t_predictions, epsilons]
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    # losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))
    bce = tf.keras.losses.BinaryCrossentropy()
    losst = tf.reduce_sum(bce(t_true, t_pred))
    return losst


def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1.0 - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))
    return loss0 + loss1


def dragonnet_loss_binarycross(concat_true, concat_pred):
    return regression_loss(
        concat_true, concat_pred
    ) + binary_classification_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return tf.keras.metrics.binary_accuracy(t_true, t_pred)


def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


def make_tarreg_loss(ratio=1.0, dragonnet_loss=dragonnet_loss_binarycross):
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss
