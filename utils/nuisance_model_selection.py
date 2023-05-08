import numpy as np
import sys
import os
from pathlib import Path
import argparse
import pickle

from helpers import *
from preprocessing import sys_config
import preprocessing as helper

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from flaml import AutoML
from tqdm import tqdm

def get_propensity_model(prop_model, w, t, automl= 0, automl_settings={}, selection_case= 'estimator'):
    
    #Propensity Model    
    if automl:
        prop_model.fit(X_train=w['tr'], y_train=t['tr'], **automl_settings)
        prop_model = clone(prop_model.model.estimator)

    prop_model.fit(w['tr'], t['tr'])
    score= prop_model.score(w['te'], t['te'])

    if automl:
        return score, prop_model
    else:
        return score
    
def get_outcome_model(out_model, w, t, y, case='t_0', automl= 0, automl_settings={}, selection_case= 'estimator'):
        
    #Outcome Models
    if case == 't_0':
        indices= t['tr'] == 0
        indices_eval= t['te'] == 0
    elif case == 't_1':
        indices= t['tr'] == 1
        indices_eval= t['te'] == 1


    # indices_full = np.repeat(indices[:, np.newaxis, :], w['tr'].shape[1], axis=1)
    # indices_eval_full = np.repeat(indices_eval[:, np.newaxis, :], w['te'].shape[1], axis=1)


    if automl:
        print(indices.shape)
        print(w['tr'][indices, :].shape, y['tr'][indices].shape)
        out_model.fit(X_train=w['tr'][indices, :], y_train=y['tr'][indices], **automl_settings)
        out_model = clone(out_model.model.estimator)

    out_model.fit(w['tr'][indices, :], y['tr'][indices])
    score = out_model.score(w['te'][indices_eval, :], y['te'][indices_eval])

    if automl:
        return score, out_model
    else:
        return score


def get_s_learner_model(out_model, w, t, y, automl= 0, automl_settings={}, selection_case= 'estimator'):
        
    #Outcome Models
    #Since these nuisance models would be used as part of the metric computation, we train them on the actual evaluation/validation set and test on the actual training test
    
    for key in w.keys():
        t[key] = t[key].reshape((-1,1))
        y[key] = y[key].reshape((-1,1))

    print(w['tr'].shape, t['tr'].shape, y['tr'].shape)

    w_upd={'te':'', 'tr':''}
    w_upd['te']= np.hstack([w['te'],t['te']])
    w_upd['tr']= np.hstack([w['tr'],t['tr']])

    if automl:
        out_model.fit(X_train=w_upd['tr'], y_train=y['tr'], **automl_settings)
        out_model = clone(out_model.model.estimator)

    out_model.fit(w_upd['tr'], y['tr'])
    score = out_model.score(w_upd['te'], y['te'])

    if automl:
        return score, out_model
    else:
        return score


def get_r_score_model(out_model, w, y, automl= 0, automl_settings={}, selection_case= 'estimator'):
        
    #Outcome Models
    #Since these nuisance models would be used as part of the metric computation, we train them on the actual evaluation/validation set and test on the actual training test    

    if automl:
        out_model.fit(X_train=w['tr'], y_train=y['tr'], **automl_settings)
        out_model = clone(out_model.model.estimator)

    out_model.fit(w['tr'], y['tr'])
    score = out_model.score(w['te'], y['te'])

    if automl:
        return score, out_model
    else:
        return score

datasets_folder = sys_config["datasets_folder"]
results_folder = sys_config["results_folder"]

# Input Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='TWINS',
                    help='Datasets: lalonde_psid1; lalonde_cps1; twins; lbidd')
parser.add_argument('--seed', type=int, default=0,
                    help='Total seeds for causal effect estimation experiments')
parser.add_argument('--selection_case', type=str, default='metric', help='model selection for estimator or metric')
parser.add_argument('--slurm_exp', type=int, default=0,
                   help='')

args = parser.parse_args()
print(vars(args))

dataset_name = args.dataset
seed = args.seed

save_dir= results_folder + '//..//models//' + dataset_name + '//'
print(save_dir, flush=True)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#Experiments on Slurm
if args.slurm_exp:
    dataset_list = ['twins', 'lalonde_psid1', 'lalonde_cps1', 'orthogonal_ml_dgp']
    # for idx in range(100):
    #     dataset_list.append('lbidd_' + str(idx))
    # for idx in range(77):
    #     dataset_list.append('acic_2016_' + str(idx))

    # dataset_list = pickle.load(open('datasets/acic_2018_heterogenous_list.p', "rb"))
    # dataset_list = pickle.load(open('datasets/acic_2016_heterogenous_list.p', "rb"))

    slurm_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    args.dataset = dataset_list[slurm_idx]

#Obtain list of nuisance (outcome, propensity) models
outcome_models, prop_models= get_nuisance_models_list()

# for key in outcome_models.keys():
#     print(key)
#
# for key in prop_models.keys():
#     print(key)

grid_size= 80

# Loop over datasets, seeds, estimators with their hyperparams and nusiance models
print('SEED: ', seed)
random.seed(seed)
np.random.seed(seed)

print('DATASET:', dataset_name)
if "IHDP" in dataset_name:
    x_all, t_all, yf_all = helper.load_IHDP_observational(
        datasets_folder, dataset_name, details=False
    )
    x_test_all = helper.load_IHDP_out_of_sample(
        datasets_folder, dataset_name, details=False
    )
elif dataset_name == "Jobs":
    x_all, t_all, yf_all = helper.load_Jobs_observational(
        datasets_folder, dataset_name, details=False
    )
    x_test_all = helper.load_Jobs_out_of_sample(
        datasets_folder, dataset_name, details=False
    )
elif dataset_name == "TWINS":
    x_all, t_all, yf_all = helper.load_TWINS_observational(
        datasets_folder, dataset_name, details=False
    )
    x_test_all, t_test_all = helper.load_TWINS_out_of_sample(
        datasets_folder, dataset_name, details=False
    )


# dataset_samples= sample_dataset(dataset_name, dataset_obj, seed=seed, case='eval')
# eval_w, eval_t, eval_y, ate, ite= dataset_samples['w'], dataset_samples['t'], dataset_samples['y'], dataset_samples['ate'], dataset_samples['ite']

# split data into train and validation
x_train, x_eval, t_train, t_eval, yf_train, yf_eval = train_test_split(x_all, t_all, yf_all, test_size=0.2, random_state=seed)

# eval_w, eval_t, eval_y= x_all, t_all, yf_all # TODO: change to actual test data, just temporarily using train data

w= { 'tr': x_train, 'te': x_eval, 'all': np.concatenate((x_train, x_eval), axis=0)}
t= { 'tr': t_train, 'te': t_eval, 'all': np.concatenate((t_train, t_eval), axis=0)}
y= { 'tr': yf_train, 'te': yf_eval, 'all': np.concatenate((yf_train, yf_eval), axis=0)}

for key in t.keys():
    data_size= w[key].shape[0]
    num_realizations = w[key].shape[-1]

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

# print('Shape Check')
# print(w['tr'].shape, w['te'].shape, w['all'].shape)
# print(t['tr'].shape, t['te'].shape, t['all'].shape)
# print(y['tr'].shape, y['te'].shape, y['all'].shape)
# sys.exit()

model_sel_res={}
for key in ['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop']:
    model_sel_res[key]= {}
    model_sel_res[key]['score']= -sys.maxsize - 1
    model_sel_res[key]['model']= -sys.maxsize - 1

sys.stderr.flush()
for model_case in tqdm(['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop'], file=sys.stdout):
# for model_case in  tqdm(['s_learner']):
    if model_case == 'prop':
        automl_settings = {
            "time_budget": 1800,  # in seconds
            "task": 'classification',
            "eval_method": 'cv',
            "n_splits": 3,
            "verbose": 0
        }
        nuisance_list= prop_models
    else:
        automl_settings = {
            "time_budget": 1800,  # in seconds
            "task": 'regression',
            "eval_method": 'cv',
            "n_splits": 3,
            "verbose": 0
        }
        nuisance_list= outcome_models

    #AutoML
    automl = AutoML()
    if model_case == 'prop':
        print('Propensity Model')
        score, best_model = get_propensity_model(automl, w, t, automl= 1, automl_settings= automl_settings, selection_case= args.selection_case)
        pickle.dump(best_model, open(save_dir + 'prop' + '.p', "wb"))
    elif model_case == 't_learner_0':
        print('T-Learner 0')
        score, best_model = get_outcome_model(automl, w, t, y, case='t_0', automl= 1, automl_settings= automl_settings, selection_case= args.selection_case)
        pickle.dump(best_model, open(save_dir + 'mu_0' + '.p', "wb"))
    elif model_case == 't_learner_1':
        print('T-Learner 1')
        score, best_model = get_outcome_model(automl, w, t, y, case='t_1', automl= 1, automl_settings= automl_settings, selection_case= args.selection_case)
        pickle.dump(best_model, open(save_dir + 'mu_1' + '.p', "wb"))
    elif model_case == 's_learner':
        print('S-Learner')
        score, best_model = get_s_learner_model(automl, w, t, y, automl= 1, automl_settings= automl_settings, selection_case= args.selection_case)
        pickle.dump(best_model, open(save_dir + 'mu_s' + '.p', "wb"))
    elif model_case == 'dml':
        print('DML')
        score, best_model = get_r_score_model(automl, w, y, automl= 1, automl_settings= automl_settings, selection_case= args.selection_case)
        pickle.dump(best_model, open(save_dir + 'mu_r_score' + '.p', "wb"))

    model_sel_res[model_case]['score']= score
    model_sel_res[model_case]['model']= best_model
    print(model_case, score, best_model)

    # print(automl.model.estimator)
    # best_est= automl.best_estimator
    # print(best_est)
    # print(automl.best_model_for_estimator(best_est))
    # sys.exit(-1)

    # #Searching over the existing grid
    # count=0
    # for key in nuisance_list.keys():
    #     for model in nuisance_list[key]:
    #         count+=1
    #
    #         curr_model = model['model_func']
    #
    #         start_time = time.time()
    #
    #         if model_case == 'prop':
    #             score = get_propensity_model(curr_model, w, t)
    #         elif model_case == 't_learner_0':
    #             score = get_outcome_model(curr_model, w, t, y, case='t_0')
    #         elif model_case == 't_learner_1':
    #             score = get_outcome_model(curr_model, w, t, y, case='t_1')
    #         elif model_case == 's_learner':
    #             score = get_s_learner_model(curr_model, w, t, y)
    #         elif model_case == 'dml':
    #             score = get_r_score_model(curr_model, w, y)
    #
    #         #             print(model['model_y']['name'], model['model_y']['hparam'])
    #         #             print(score, 'time: ', time.time() - start_time)
    #         if score > model_sel_res[model_case]['score']:
    #             model_sel_res[model_case]['score'] = score
    #             model_sel_res[model_case]['model'] = curr_model
    #         #             print('Curr Best Model: ', best_outcome_score_0)
    #
    # print(count)



# for model_case in tqdm(['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop']):
#     print('Best ', model_case)
#     print(model_sel_res[model_case]['score'])
#     print(model_sel_res[model_case]['model'])

#     if model_case == 'prop':
#         pickle.dump(model_sel_res[model_case]['model'], open(save_dir + 'prop' + '.p', "wb"))
#     elif model_case == 't_learner_0':
#         pickle.dump(model_sel_res[model_case]['model'], open(save_dir + 'mu_0' + '.p', "wb"))
#     elif model_case == 't_learner_1':
#         pickle.dump(model_sel_res[model_case]['model'], open(save_dir + 'mu_1' + '.p', "wb"))
#     elif model_case == 's_learner':
#         pickle.dump(model_sel_res[model_case]['model'], open(save_dir + 'mu_s' + '.p', "wb"))
#     elif model_case == 'dml':
#         pickle.dump(model_sel_res[model_case]['model'], open(save_dir + 'mu_r_score' + '.p', "wb"))
