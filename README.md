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
- COM/S-Learner: OLS1, RF1, NN1;
- GCOM/T-Learner: OLS2, RF2, NN2;
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
- IHDP-100
- Jobs
- TWINS

## Performing evaluation
After performing estimation, run:
```
python evaluate.py
```
## Running Notebooks
- Open the notebooks folder and click on the desired notebook.
- At the top of the notebook, click on the ![Open in Colab](https://github.com/VectorInstitute/Causal_Inference_Laboratory/assets/47928320/72fa430a-9e75-4e7d-82fe-080beb58a42d) button.
- Follow the instructions in the Colab to run the code.

For the `Demo_End2End_Causal_Estimation_Pipeline.ipynb` notebook, to ensure that you've initialized it correctly, check the following:
- After running the first code block in Colab, your file directory should look like this:

![image](https://github.com/VectorInstitute/Causal_Inference_Laboratory/assets/47928320/c3fc6d26-369d-4454-8990-3b452b49d86c)

(Note: you can press ![image](https://github.com/VectorInstitute/Causal_Inference_Laboratory/assets/47928320/0eff5ced-6a65-4f22-a8a8-439c48526ba0) to refresh)



# using pre-commit hooks
To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```
