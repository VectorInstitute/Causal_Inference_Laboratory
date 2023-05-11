import numpy as np
from sklearn.base import clone

from utils.helpers import *

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