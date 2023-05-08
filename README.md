# Causal Inference
The main goal of this project is to build libraries and models rooted in deep
learning to estimate the causal effects of an intervention on some measurable
outcomes (e.g. the effect of a treatment or procedure on survival).

## Installing dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Performing estimation
While in the base directory of the repo, run:
```
python estimate.py
```
The results of all estimators will be stored in folder ``estimation_results``.

### Available estimators:
- COM/T-Learner: OLS1, RF1, NN1;
- GCOM/S-Learner: OLS2, RF2, NN2;
- Inverse Probability Weighting (IPW);
- Double Machine Learning/R-Learner;
- TARNet
- Dragonnet

where OLS stands for Ordinary Least Squares, RF stands for Random Forest, and
NN stands for Nerual Networks. They represent ML models that are linear,
ensembled, and non-linear respectively.

### Available datasets:
Please see the description of the datasets in the ``data`` folder.
- IHDP-100
- Jobs
- TWINS

### Available metrics:
- Mean absolute error (MAE) of ATE
- Precision in Estimation of Heterogeneous Effect (PEHE)

## Performing nusiance model estimation
The nuisance models are the models used in the paper [Empirical Analysis of Model Selection for Heterogenous Causal Effect Estimation](https://arxiv.org/abs/2211.01939) to estimate the ground truth for PEHE calculation.

While in the base directory of the repo, run:
```
python utils/nuisance_model_selection.py --dataset <dataset_name>
```

This uses AutoML to estimate the best model for each nuisance model. The results are stored in the ``estimation_results/<dataset_name>/models`` folder. The results for Jobs and TWINS are already stored in the repo.

### Available estimators:
- T-Learner 0
- T-Learner 1
- S-Learner
- Double Machine Learning (DML)
- Propensity

### Available datasets:
- Jobs
- TWINS

## Performing evaluation
After performing estimation, run:
```
python evaluate.py
```


# using pre-commit hooks
To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```
