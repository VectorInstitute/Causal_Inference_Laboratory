import os
import sys
from datetime import datetime
import random
import copy

import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet, RidgeClassifier
from sklearn.svm import SVR, LinearSVR, SVC, LinearSVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,\
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.exceptions import UndefinedMetricWarning


#EconML Estimators
from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML, NonParamDML
from econml.dr import DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner
from econml.metalearners import XLearner, TLearner, SLearner, DomainAdaptationLearner
from econml.orf import DMLOrthoForest, DROrthoForest
from econml._cate_estimator import LinearCateEstimator

# from data.loading import load_from_folder
# from data.lbidd import lbidd_main_loader
# from data.acic_2016 import acic_2016_main_loader
# from data.orthogonal_ml_dgp import get_data_generator

class SLearnerUpdated(LinearCateEstimator):

    def __init__(self, *, overall_model, final_model):
        self.overall_model = overall_model
        self.final_model = final_model
        return

    def fit(self, y, T, X, W=None):
        XW = X
        if W is not None:
            XW = np.hstack([X, W])
        self.model_ = clone(self.overall_model)
        self.model_.fit(np.hstack([T.reshape(-1, 1), XW]), y)
        ones = np.hstack([np.ones((X.shape[0], 1)), XW])
        zeros = np.hstack([np.zeros((X.shape[0], 1)), XW])
        diffs = self.model_.predict(ones) - self.model_.predict(zeros)

        self.model_final_ = clone(self.final_model)
        self.model_final_.fit(X, diffs)
        return self

    def effect(self, X, T0=0, T1=1):
        return self.const_marginal_effect(X)

    def const_marginal_effect(self, X):
        return self.model_final_.predict(X)


#Hyperparam search grids                    
alphas = {'alpha': np.logspace(-4, 5, 10)}
# gammas = [] + ['scale']
Cs = np.logspace(-4, 5, 10)
d_Cs = {'C': Cs}
SVM = 'svm'
d_Cs_pipeline = {SVM + '__C': Cs}
max_depths = list(range(2, 10 + 1)) + [None]
d_max_depths = {'max_depth': max_depths}
d_max_depths_base = {'base_estimator__max_depth': max_depths}
# Ks = {'n_neighbors': [1, 2, 3, 5, 10, 15, 25, 50, 100, 200]}
Ks = {'n_neighbors': [1, 2, 3, 5, 10, 15, 20, 25, 40, 50]}


OUTCOME_MODEL_GRID = { 'no_hparam' : [], 'regularized_lr': [],  'svr': [], 'forest': [], 'misc': [] }
PROP_SCORE_MODEL_GRID = { 'no_hparam' : [], 'logistic': [],  'svm': [], 'misc': []  }

OUTCOME_MODEL_GRID['no_hparam']= [
    
    ('LinearRegression', LinearRegression(), {}),
    ('LinearRegression_interact',
     make_pipeline(PolynomialFeatures(degree=2, interaction_only=True),
                   LinearRegression()),
     {}),
    ('LinearRegression_degree2',
     make_pipeline(PolynomialFeatures(degree=2), LinearRegression()), {}),    
    
]

OUTCOME_MODEL_GRID['regularized_lr']= [
    
    ('Ridge', lambda x: Ridge(alpha=x),  alphas),
    ('Lasso', lambda x: Lasso(alpha=x), alphas),
    ('ElasticNet', lambda x: ElasticNet(alpha=x), alphas),
    ('KernelRidge', lambda x: KernelRidge(alpha=x), alphas),    
    
]

OUTCOME_MODEL_GRID['svr']= [
    
    ('SVR_rbf', lambda x: SVR(kernel='rbf', C=x), d_Cs),
    ('SVR_sigmoid', lambda x: SVR(kernel='sigmoid', C=x), d_Cs),    
    ('LinearSVR', lambda x: LinearSVR(), d_Cs),
  
]

OUTCOME_MODEL_GRID['forest']= [
    
    # TODO: also cross-validate over min_samples_split and min_samples_leaf
    ('DecisionTree', lambda x: DecisionTreeRegressor(max_depth= x), d_max_depths),
    ('RandomForest', lambda x: RandomForestRegressor(max_depth= x), d_max_depths),
    
]
    
OUTCOME_MODEL_GRID['misc']= [
    
    #('kNN', lambda x: KNeighborsRegressor(n_neighbors=x), Ks),    
    
    # TODO: also cross-validate over learning_rate
    ('GradientBoosting', lambda x: GradientBoostingRegressor(max_depth=x), d_max_depths),
    
]



PROP_SCORE_MODEL_GRID['no_hparam']= [
    
    ('LogisticRegression',  LogisticRegression(penalty='none'), {}),    
    ('LDA', LinearDiscriminantAnalysis(), {}),
    ('LDA_shrinkage', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), {}),
    ('QDA', QuadraticDiscriminantAnalysis(), {}),    
    ('GaussianNB', GaussianNB(), {}),    
    
]

PROP_SCORE_MODEL_GRID['logistic']= [
    
    ('LogisticRegression_l2', lambda x: LogisticRegression(penalty='l2', C=x),  d_Cs),
    ('LogisticRegression_l1_liblinear', lambda x: LogisticRegression(penalty='l1', solver='liblinear', C=x), d_Cs),
    ('LogisticRegression_l2_liblinear', lambda x: LogisticRegression(penalty='l2', solver='liblinear', C=x), d_Cs),
    ('LogisticRegression_l1_saga', lambda x: LogisticRegression(penalty='l1', solver='saga', C=x), d_Cs),
    
]

PROP_SCORE_MODEL_GRID['svm']= [
    
    ('SVM_rbf', lambda x: SVC(kernel='rbf', probability=True, C=x), d_Cs),    
    ('SVM_sigmoid', lambda x: SVC(kernel='sigmoid', probability=True, C=x), d_Cs),   
    
]

PROP_SCORE_MODEL_GRID['misc']= [
    
    ('kNN', lambda x: KNeighborsClassifier(n_neighbors=x), Ks),    
    
    # TODO: also cross-validate over learning_rate
    ('GradientBoosting', lambda x: GradientBoostingClassifier(max_depth=x), d_max_depths),    

]

# OUTCOME_MODEL_GRID = [
    
# #     ('LinearRegression_degree3',
# #      make_pipeline(PolynomialFeatures(degree=3), LinearRegression()), {}),        
    
#     # SVMs are sensitive to input scale
# #     ('Standardized_SVM_rbf', Pipeline([('standard', StandardScaler()), (SVM, SVR(kernel='rbf'))]),
# #      d_Cs_pipeline),
# #     ('Standardized_SVM_sigmoid', Pipeline([('standard', StandardScaler()), (SVM, SVR(kernel='sigmoid'))]),
# #      d_Cs_pipeline),
# #     ('Standardized_LinearSVM', Pipeline([('standard', StandardScaler()), (SVM, LinearSVR())]),
# #      d_Cs_pipeline),    
    
# ]
    
# PROP_SCORE_MODEL_GRID = [
    
#     # SVMs are sensitive to input scale
# #     ('Standardized_SVM_rbf', Pipeline([('standard', StandardScaler()), (SVM, SVC(kernel='rbf', probability=True))]),
# #      d_Cs_pipeline),
# #     ('Standardized_SVM_sigmoid', Pipeline([('standard', StandardScaler()),
# #                                            (SVM, SVC(kernel='sigmoid', probability=True))]),
# #      d_Cs_pipeline),    
    
# #     ('SVM_linear', lambda x: SVC(kernel='linear', probability=True, C=x), d_Cs),   # doesn't seem to work (runs forever)
    
#     #TODO: also cross-validate over min_samples_split and min_samples_leaf
#     # Doesn't fit with econml estimators, AttributeError: 'RandomForestRegressor' object has no attribute 'predict_proba'
# #     ('DecisionTree', lambda x: DecisionTreeRegressor(max_depth= x), d_max_depths),
# #     ('RandomForest', lambda x: RandomForestRegressor(max_depth= x), d_max_depths),
    
# ]

# Pass a list of nuisance models required for the Meta Estimator under the following scheme: ( (model_name, model_y), (model_name, model_t) )
META_EST_GRID={
    
    'dml_learner': ( ['model_final'], lambda model_dict, model_final: DML(model_t= model_dict['model_t']['model_func'], model_y= model_dict['model_y']['model_func'], model_final= model_final['model_func'],  discrete_treatment=True, linear_first_stages=False, cv=3)),

#     ('linear_dml',  [], lambda model_dict: LinearDML(model_t= model_dict['model_t']['model_func'], model_y= model_dict['model_y']['model_func'], discrete_treatment=True)),
    
#     ('sparse_linear_dml',  [], lambda model_dict: SparseLinearDML(model_t= model_dict['model_t']['model_func'], model_y= model_dict['model_y']['model_func'], discrete_treatment=True)),

#     ('causal_forest_dml', [], lambda model_dict: CausalForestDML(model_t= model_dict['model_t']['model_func'], model_y= model_dict['model_y']['model_func'], discrete_treatment=True)),

    'dr_learner': ( ['model_final'], lambda model_dict, model_final: DRLearner(model_propensity= model_dict['model_t']['model_func'], model_regression= model_dict['model_y']['model_func'],
                   model_final= model_final['model_func'], cv=3)),    

#     ('linear_dr', [], lambda model_dict: LinearDRLearner(model_propensity= model_dict['model_t']['model_func'], model_regression= model_dict['model_y']['model_func'], cv=3)),    

#     ('sparse_linear_dr', [],  lambda model_dict: SparseLinearDRLearner(model_propensity= model_dict['model_t']['model_func'], model_regression= model_dict['model_y']['model_func'], cv=3)),    

#     ('forest_dr', [], lambda model_dict: ForestDRLearner(model_propensity= model_dict['model_t']['model_func'], model_regression= model_dict['model_y']['model_func'], cv=3)),
    

    'dr_learner_tune_0.1': ( ['model_final'], lambda model_dict, model_final: DRLearner(model_propensity= model_dict['model_t']['model_func'], model_regression= model_dict['model_y']['model_func'],
                   model_final= model_final['model_func'], cv=3, min_propensity=0.1)),

    'dr_learner_tune_0.01': ( ['model_final'], lambda model_dict, model_final: DRLearner(model_propensity= model_dict['model_t']['model_func'], model_regression= model_dict['model_y']['model_func'],
                   model_final= model_final['model_func'], cv=3, min_propensity=0.01)),


    'x_learner': ( ['model_final'], lambda model_dict, model_final: XLearner(propensity_model= model_dict['model_t']['model_func'],
                                                                             models= model_dict['model_y']['model_func'],
                                                                             cate_models= model_final['model_func'])),

    's_learner_upd': (['model_final'], lambda model_dict, model_final: SLearnerUpdated(overall_model=model_dict['model_y']['model_func'],
                                                                                        final_model=model_final['model_func'])),

    'causal_forest_learner': ([], lambda model_dict: CausalForestDML(model_t=model_dict['model_t']['model_func'],
                                                                         model_y=model_dict['model_y']['model_func'],
                                                                         discrete_treatment=True,
                                                                         cv=3)),

    's_learner': ( [], lambda model_dict: SLearner(overall_model=model_dict['model_y']['model_func'])),
    
    't_learner':( [], lambda model_dict: TLearner(models=model_dict['model_y']['model_func']))
    
}

def create(*args):
    path = '/'.join(a for a in args)
    if not os.path.isdir(path):
        os.makedirs(path)


class Logging:
    def __init__(self, saveroot, filename='log.txt', log_=True):
        self.log_path = os.path.join(saveroot, filename)
        self.log_ = log_

    def info(self, s, print_=True):
        if print_:
            print(f'{datetime.now()} / {s}')
        if self.log_:
            with open(self.log_path, 'a+') as f_log:
                f_log.write(f'{datetime.now()} / {s} \n')


#To facilitate tweaking the heterogenity knobs etc
def sample_real_cause_dataset(gen_model, dataset_name, seed=0, case='train', const_ite=0):

    w, t, (y0, y1) = gen_model.sample(dataset=case, seed=seed, ret_counterfactuals=True)
    y = y1 * t + y0 * (1 - t)

    if case == 'train':
        return w, t, y, None, None

    base_dir = 'datasets/' + dataset_name + '/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    fname = base_dir + 'seed_' + str(seed) + '_' + case + '_ate.npy'
    if os.path.isfile(fname):
        print('Load ATE estimates from a saved file')
        ate = np.load(fname)
    else:
        print('Generate ATE estimates and store them in a file')
        ate = gen_model.ate(w=w, noisy=True)
        np.save(fname, ate)

    fname = base_dir + 'seed_' + str(seed) + '_' + case + '_ite.npy'
    if os.path.isfile(fname):
        print('Load ITE estimates from a saved file')
        ite = np.load(fname)
    else:
        print('Generate ITE estimates and store them in a file')
        ite = gen_model.ite(w=w, noisy=True, seed=seed).squeeze()
        np.save(fname, ite)

    if const_ite:
        y0 = y0 / np.sqrt(np.var(ite))
        y1 = y1 / np.sqrt(np.var(ite))

        y = y1 * t + y0 * (1 - t)
        ite= y1 - y0
        ite= np.reshape(ite, (ite.shape[0]))
        ate= np.mean(ite)


    return w, t, y, ate, ite
                
# def load_dataset_obj(dataset, root_dir):
    res={}
    if dataset in ['twins', 'lalonde_psid1', 'lalonde_cps1']:
        gen_model, _ = load_from_folder(dataset=dataset)
        res['gen_model']= gen_model

    elif 'lbidd' in dataset:
        gen_model = lbidd_main_loader(n='5k', dataroot= root_dir, dataset_idx=int(dataset.split('_')[1]))
        res['gen_model']= gen_model

    elif 'acic_2016' in dataset:
        print('Here')
        gen_model = acic_2016_main_loader(dataset_idx=int(dataset.split('_')[-1]))
        res['gen_model']= gen_model

    elif dataset in ['orthogonal_ml_dgp']:
        gen_model, _, tau_fn, _ = get_data_generator()
        res['gen_model']= gen_model
        res['tau_fn']= tau_fn

    return (dataset, res)


def sample_dataset(dataset_name, dataset_obj, seed=0, case='train'):

    gen_model= dataset_obj['gen_model']

    if case == 'train':
        if dataset_name in ['twins', 'lalonde_psid1', 'lalonde_cps1']:
            train_w, train_t, train_y, _, _= sample_real_cause_dataset(gen_model, dataset_name, seed=seed, case='train')
        elif 'lbidd' in dataset_name or 'acic_2016' in dataset_name:
            train_w, train_t, train_y= gen_model['tr']['w'], gen_model['tr']['t'], gen_model['tr']['y']
        elif dataset_name in ['orthogonal_ml_dgp']:
            train_y, train_t, train_w, _= gen_model()

        return  {'w': train_w, 't': train_t, 'y': train_y}

    elif case == 'eval':
        if dataset_name in ['twins', 'lalonde_psid1', 'lalonde_cps1']:
            eval_w, eval_t, eval_y, ate, ite= sample_real_cause_dataset(gen_model, dataset_name, seed=seed, case='val')

        elif 'lbidd' in dataset_name or 'acic_2016' in dataset_name:
            eval_w, eval_t, eval_y, ate, ite = gen_model['eval']['w'], gen_model['eval']['t'], gen_model['eval']['y'], \
                                               gen_model['eval']['ate'], gen_model['eval']['ites']

        elif dataset_name in ['orthogonal_ml_dgp']:
            tau_fn = dataset_obj['tau_fn']
            eval_y, eval_t, eval_w, _ = gen_model()
            ite = tau_fn(eval_w)
            ate = np.mean(ite)

        return {'w': eval_w, 't': eval_t, 'y': eval_y, 'ate': ate, 'ite': ite}


def get_estimators_list(estimator_name):
    
    hparam_list, estimator = META_EST_GRID[estimator_name]
    return hparam_list, estimator    

def get_nuisance_models_names(nuisance_model):
    
    model_t={'name':'none', 'hparam':'none'}
    model_y={'name':'none', 'hparam':'none'}

    if 'model_t' in nuisance_model.keys():
        model_t['name']= nuisance_model['model_t']['name']
        model_t['hparam']= nuisance_model['model_t']['hparam']
    
    if 'model_y' in nuisance_model.keys():
        model_y['name']= nuisance_model['model_y']['name']
        model_y['hparam']= nuisance_model['model_y']['hparam']
    
    return model_t, model_y


def get_nuisance_models_list():

    # Loop over different hyperparams to construct list: (model_name, model(hparam))
    res={}
    res['outcome_models']= {}
    res['prop_score_models']= {}
    
    for key in res.keys():
        if key == 'outcome_models':
            meta_list= OUTCOME_MODEL_GRID
        elif key == 'prop_score_models':
            meta_list= PROP_SCORE_MODEL_GRID
        
        for sub_key in meta_list.keys():

            for (model_name, model, param_grid) in meta_list[sub_key]:
                if not param_grid:
                    if 'no_hparam' not in res[key].keys():
                        res[key]['no_hparam']= []
                    res[key]['no_hparam'].append( {'name': model_name, 'hparam': 'none', 'model_func': model} )
                else:
                    hparam_list= list(param_grid.values())[0]
                    hparam_name= list(param_grid.keys())[0]
                    if model_name not in res[key].keys():
                        res[key][model_name]= []
                    for hparam in hparam_list:
                        res[key][model_name].append( {'name': model_name, 'hparam': hparam_name + '_' + str(hparam), 'model_func': model(hparam)} )

    # print('No Hparam Case', key, res[key]['no_hparam'])

    return res['outcome_models'], res['prop_score_models']


def stratified_random_sampler(models_dict, grid_size, proportionate_sampler=0, fixed_sampler_size= 10):
    
    sample_size= {}
    population_size=0
    for key in models_dict.keys():
        if key == 'no_hparam':
            continue
        else:
            sample_size[key]= len(models_dict[key])    
            population_size+= len(models_dict[key])

    for key in models_dict.keys():
        if key == 'no_hparam':
            continue
        else:
            sample_size[key]= int(sample_size[key]*grid_size/population_size) 

    print(sample_size)
    
    models_approx_list=[]
    for key in models_dict.keys():
        if key == 'no_hparam':
            models_approx_list += models_dict[key]
        else:
            if proportionate_sampler:
                models_approx_list += random.sample(models_dict[key], sample_size[key])
            else:
                models_approx_list += random.sample(models_dict[key], fixed_sampler_size)
    
    random.shuffle(models_approx_list)

    # #Debugging
    # count_dict={}
    # for item in models_approx_list:
    #     if item['name'] not in count_dict:
    #         count_dict[item['name']]=1
    #     else:
    #         count_dict[item['name']]+=1
    #
    # for key in count_dict.keys():
    #     print(key, count_dict[key])

    return models_approx_list
            
def get_nusiance_models_grid(outcome_models, prop_score_models, approx=False, grid_size=100):
    
    if approx:
        
        models_approx_list= {}
        for model_case in ['outcome', 'prop']:
            if model_case == 'outcome':
                models_approx_list[model_case]= stratified_random_sampler(outcome_models, grid_size)
            elif model_case == 'prop':
                models_approx_list[model_case]= stratified_random_sampler(prop_score_models, grid_size)
#             print(models_approx_list[model_case])

        grid_models= []
        grid_size= max(len(models_approx_list['outcome']), len(models_approx_list['prop']))
        mod_factor= min(len(models_approx_list['outcome']), len(models_approx_list['prop']))

        for idx in range(grid_size):            
            outcome_model= models_approx_list['outcome'][idx % mod_factor]
            prop_model= models_approx_list['prop'][idx % mod_factor]
            grid_models.append( {'model_t': prop_model, 'model_y': outcome_model} )
            
#         grid_models= [ {'model_t': prop_model, 'model_y': outcome_model} for prop_model in models_approx_list['prop'] for outcome_model in models_approx_list['outcome'] ]        
#         grid_models= random.sample(grid_models, grid_size)
    
    else:
        
        grid_models= [ {'model_t': prop_model, 'model_y': outcome_model} for prop_model in prop_score_models for outcome_model in outcome_models]
    
    return grid_models
