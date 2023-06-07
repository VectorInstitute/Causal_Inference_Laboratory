from preprocessing import sys_config

import preprocessing as helper
from evaluation import *
from helpers import *
from sklearn.model_selection import train_test_split


seed = 0

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

dataset_name = "berkeley_e1"

save_dir= results_folder + '//..//models//' + dataset_name + '//'

print(save_dir, flush=True)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#Obtain list of nuisance (outcome, propensity) models
outcome_models, prop_models= get_nuisance_models_list()
grid_size= 80

# Loop over datasets, seeds, estimators with their hyperparams and nusiance models
print('SEED: ', seed)


print('DATASET:', dataset_name)
if "IHDP" in dataset_name:
    x_all, t_all, yf_all = helper.load_IHDP_observational(
        datasets_folder, dataset_name, details=False
    )
    x_test_all, t_test_all, yf_test_all = helper.load_IHDP_out_of_sample(
        datasets_folder, dataset_name, details=False
    )
elif dataset_name == "Jobs":
    x_all, t_all, yf_all = helper.load_Jobs_observational(
        datasets_folder, dataset_name, details=False
    )
    x_test_all, t_test_all, yf_test_all = helper.load_Jobs_out_of_sample(
        datasets_folder, dataset_name, details=False
    )
elif dataset_name == "TWINS":
    x_all, t_all, yf_all = helper.load_TWINS_observational(
        datasets_folder, dataset_name, details=False
    )
    x_test_all, t_test_all, yf_test_all = helper.load_TWINS_out_of_sample(
        datasets_folder, dataset_name, details=False
    )
elif dataset_name == "berkeley_e1" or dataset_name == "berkeley_e2":
    # datasets_folder_CFA = os.path.join(datasets_folder, "CFA")
    datasets_folder_CFA = "data/CFA"
    print(datasets_folder_CFA)
    print(dataset_name)
    x_all, t_all, yf_all = helper.load_berkeley_observational(
        datasets_folder_CFA, dataset_name, details=False
    )
    x_test_all, t_test_all, yf_test_all = helper.load_berkeley_out_of_sample(
        datasets_folder_CFA, dataset_name, details=False
    )

x_train, x_eval, t_train, t_eval, yf_train, yf_eval = train_test_split(x_all, t_all, yf_all, test_size=0.2, random_state=seed)
print("X TRAIN")
print(x_train)
print(x_all)


w= { 'tr': x_train, 'te': x_eval, 'all': np.concatenate((x_train, x_eval), axis=0)}
t= { 'tr': t_train, 'te': t_eval, 'all': np.concatenate((t_train, t_eval), axis=0)}
y= { 'tr': yf_train, 'te': yf_eval, 'all': np.concatenate((yf_train, yf_eval), axis=0)}

# for key in t.keys():
#     data_size= w[key].shape[0]
#     num_realizations = w[key].shape[-1]

#     # stack num_realizations vertically under data_size
#     new_w = np.zeros((data_size * num_realizations, w[key].shape[1]))
#     for i in range(num_realizations):
#         new_w[i * data_size : (i + 1) * data_size, :] = w[key][:, :, i]
#     w[key] = new_w

#     # stack num_realizations vertically under data_size
#     new_t = np.zeros((data_size * num_realizations))
#     for i in range(num_realizations):
#         new_t[i * data_size : (i + 1) * data_size] = t[key][:, i]
#     t[key] = new_t

#     # stack num_realizations vertically under data_size
#     new_y = np.zeros((data_size * num_realizations))
#     for i in range(num_realizations):
#         new_y[i * data_size : (i + 1) * data_size] = y[key][:, i]
#     y[key] = new_y


#     w[key]= np.reshape(w[key], (data_size * num_realizations, -1)) # TODO: fix for mutliple realizations
#     t[key]= np.reshape(t[key], (data_size * num_realizations))
#     y[key]= np.reshape(y[key], (data_size * num_realizations))


# model_sel_res={}
# for key in ['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop']:
#     model_sel_res[key]= {}
#     model_sel_res[key]['score']= -sys.maxsize - 1
#     model_sel_res[key]['model']= -sys.maxsize - 1

# for model_case in tqdm(['t_learner_0', 't_learner_1', 's_learner', 'dml', 'prop'], file=sys.stdout):
# # for model_case in  tqdm(['s_learner']):
#     if model_case == 'prop':
#         automl_settings = {
#             "time_budget": 180,  # in seconds
#             "task": 'classification',
#             "eval_method": 'cv',
#             "n_splits": 3,
#             "verbose": 1
#         }
#         nuisance_list= prop_models
#     else:
#         automl_settings = {
#             "time_budget": 180,  # in seconds
#             "task": 'regression',
#             "eval_method": 'cv',
#             "n_splits": 3,
#             "verbose": 1
#         }
#         nuisance_list= outcome_models

#     #AutoML
#     automl = AutoML()
#     if model_case == 'prop':
#         print('Propensity Model')
#         score, best_model = get_propensity_model(automl, w, t, automl= 1, automl_settings= automl_settings, selection_case='metric')
#         pickle.dump(best_model, open(save_dir + 'prop' + '.p', "wb"))
#     elif model_case == 't_learner_0':
#         print('T-Learner 0')
#         score, best_model = get_outcome_model(automl, w, t, y, case='t_0', automl= 1, automl_settings= automl_settings, selection_case='metric')
#         pickle.dump(best_model, open(save_dir + 'mu_0' + '.p', "wb"))
#     elif model_case == 't_learner_1':
#         print('T-Learner 1')
#         score, best_model = get_outcome_model(automl, w, t, y, case='t_1', automl= 1, automl_settings= automl_settings, selection_case='metric')
#         pickle.dump(best_model, open(save_dir + 'mu_1' + '.p', "wb"))
#     elif model_case == 's_learner':
#         print('S-Learner')
#         score, best_model = get_s_learner_model(automl, w, t, y, automl= 1, automl_settings= automl_settings, selection_case='metric')
#         pickle.dump(best_model, open(save_dir + 'mu_s' + '.p', "wb"))
#     elif model_case == 'dml':
#         print('DML')
#         score, best_model = get_r_score_model(automl, w, y, automl= 1, automl_settings= automl_settings, selection_case='metric')
#         pickle.dump(best_model, open(save_dir + 'mu_r_score' + '.p', "wb"))

#     model_sel_res[model_case]['score']= score
#     model_sel_res[model_case]['model']= best_model
#     print(model_case, score, best_model)