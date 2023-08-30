import os

import numpy as np

import utils.estimators as models
import utils.preprocessing as helper
from utils.preprocessing import sys_config

datasets_folder = sys_config["datasets_folder"]
results_folder = sys_config["results_folder"]
dataset_name = "IHDP-100" # optionas are "Jobs", "IHDP-100", "TWINS", "census_e1", "census_e2
estimator_set = [
    "OLS1",
    "OLS2",
    "NN1",
    "NN2",
    "RF1",
    "RF2",
    "Dragonnet",
    "TARNet",
    "DML"
]

if __name__ == "__main__":
    print(f'{" Estimation ":-^79}')
    if "IHDP" in dataset_name:
        x_all, t_all, yf_all = helper.load_IHDP_observational(
            datasets_folder, dataset_name, details=False
        )
        yf_all = helper.scale_y(yf_all)
        x_test_all, t_test_all, yf_test_all = helper.load_IHDP_out_of_sample(
            datasets_folder, dataset_name, details=False
        )
        yf_test_all = helper.scale_y(yf_test_all)
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
    elif dataset_name == "census_e1" or dataset_name == "census_e2":
        datasets_folder = os.path.join(datasets_folder, "CFA")
        x_all, t_all, yf_all = helper.load_census_observational(
            datasets_folder, dataset_name, details=False
        )
        x_test_all, t_test_all, yf_test_all = helper.load_census_out_of_sample(
            datasets_folder, dataset_name, details=False
        )
    num_realizations = x_all.shape[-1]
    z=9
    for estimator_name in estimator_set:
        y0_in_all, y1_in_all, y0_out_all, y1_out_all = [], [], [], []
        ate_in_all, ate_out_all = [], []
        for i in range(num_realizations):
            text = f" Estimation of realization {i} via {estimator_name}"
            print(f"{text:-^79}")
            x, t, yf = x_all[:, :, i], t_all[:, i], yf_all[:, i]
            x_test = x_test_all[:, :, i]
            # train the estimator and predict for this realization
            (
                y0_in,
                y1_in,
                ate_in,
                y0_out,
                y1_out,
                ate_out,
            ) = models.train_and_evaluate(x, t, yf, x_test, estimator_name, dataset_name, tune_hparams=False)
            y0_in_all.append(y0_in)
            y1_in_all.append(y1_in)
            ate_in_all.append(ate_in)
            y0_out_all.append(y0_out)
            y1_out_all.append(y1_out)
            ate_out_all.append(ate_out)
        # follow the dimension order of the dataset,
        # i.e., realizations are captured by the last index
        y0_in_all = np.squeeze(np.array(y0_in_all).transpose()).reshape((-1, num_realizations))
        y1_in_all = np.squeeze(np.array(y1_in_all).transpose()).reshape((-1, num_realizations))
        y0_out_all = np.squeeze(np.array(y0_out_all).transpose()).reshape((-1, num_realizations))
        y1_out_all = np.squeeze(np.array(y1_out_all).transpose()).reshape((-1, num_realizations))
        ate_in_all = np.array(ate_in_all).reshape((num_realizations,))
        ate_out_all = np.array(ate_out_all).reshape((num_realizations,))

        # save estimation results
        estimation_result_folder = os.path.join(
            results_folder, dataset_name+"_"+str(z+1), estimator_name
        )
        print(f"Saving {estimation_result_folder}.")
        helper.save_in_and_out_results(
            estimation_result_folder,
            y0_in_all,
            y1_in_all,
            ate_in_all,
            y0_out_all,
            y1_out_all,
            ate_out_all,
        )
