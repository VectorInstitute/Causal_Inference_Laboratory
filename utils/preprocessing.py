import os

import numpy as np
import pandas as pd

sys_config = {
    "datasets_folder": "./data",
    "results_folder": "./estimation_results",
}

def load_IHDP(
    datasets_folder="./data",
    dataset_name="IHDP-100",
    split="train",
    details=False,
):
    """
    Load multiple realizations (100 and 1000) of the IHDP dataset
    with split training and test sets
    :param datasets_folder: path to all datasets
    :param dataset_name: name of the dataset
    :param split: train or test split
    :param details: boolean value for whether showing the details
    :return:
    """
    dataset_path = os.path.join(datasets_folder, dataset_name)
    if dataset_name == "IHDP-100":
        dataset_filename = f"ihdp_npci_1-100.{split}.npz"
    elif dataset_name == "IHDP-1000":
        dataset_filename = f"ihdp_npci_1-1000.{split}.npz"
    else:
        print("The given dataset is not supported. Use IHDP-100 by default.")
        dataset_filename = f"ihdp_npci_1-100.{split}.npz"

    data = None
    try:
        data = np.load(os.path.join(dataset_path, dataset_filename))
        if details:
            print(f'{"":-^79}')
            print(
                f"The details of the {split} split of {dataset_name} dataset:"
            )
            print(f'Number of realizations: {data["x"].shape[-1]}')
            for k in data.keys():
                print(k, data[k].shape, data[k].dtype)
            print(f'{"":-^79}')
    except FileNotFoundError:
        print(f"Cannot find the file of the {split} split of {dataset_name}.")
    except Exception:
        print("Loading dataset failed.")
    return data


def load_Jobs(
    datasets_folder="./data",
    dataset_name="Jobs",
    split="train",
    details=False,
):
    """
    Load multiple realizations (10) of the Jobs dataset
    with split training and test sets
    :param datasets_folder: path to all datasets
    :param dataset_name: name of the dataset
    :param split: train or test split
    :param details: boolean value for whether showing the details
    :return:
    """
    dataset_path = os.path.join(datasets_folder, dataset_name)
    dataset_filename = f"jobs_DW_bin.new.10.{split}.npz"

    data = None
    try:
        data = np.load(os.path.join(dataset_path, dataset_filename))
        if details:
            print(f'{"":-^79}')
            print(
                f"The details of the {split} split of {dataset_name} dataset:"
            )
            print(f'Number of realizations: {data["x"].shape[-1]}')
            for k in data.keys():
                print(k, data[k].shape, data[k].dtype)
            print(f'{"":-^79}')
    except FileNotFoundError:
        print(f"Cannot find the file of the {split} split of {dataset_name}.")
    except Exception:
        print("Loading dataset failed.")
    return data

def load_TWINS(
    datasets_folder="./data",
    dataset_name="TWINS",
    split="train",
    details=False,
): 
    """
    Load the TWINS dataset with split training and test sets
    :param datasets_folder: path to all datasets
    :param dataset_name: name of the dataset
    :param split: train or test split
    :param details: boolean value for whether showing the details
    :return:
    """
    dataset_path = os.path.join(datasets_folder, dataset_name)
    dataset_filename = f"twins.{split}.npz"

    data = None
    try:
        data = np.load(os.path.join(dataset_path, dataset_filename))
        if details:
            print(f'{"":-^79}')
            print(
                f"The details of the {split} split of {dataset_name} dataset:"
            )
            print(f'Number of realizations: {data["x"].shape[-1]}')
            for k in data.keys():
                print(k, data[k].shape, data[k].dtype)
            print(f'{"":-^79}')
    except FileNotFoundError:
        print(f"Cannot find the file of the {split} split of {dataset_name}.")
    except Exception:
        print("Loading dataset failed.")
    return data

def load_IHDP_observational(
    datasets_folder="./data", dataset_name="IHDP-100", details=False
):
    """
    Load observational data (x, t, yf), i.e., train split of multiple
    realizations (100 and 1000) of the IHDP dataset
    :param datasets_folder: path to all datasets
    :param dataset_name: name of the dataset
    :param split: train or test split
    :param details: boolean value for whether to show the details or not
    :return:
    """
    data = load_IHDP(
        datasets_folder=datasets_folder,
        dataset_name=dataset_name,
        split="train",
        details=details,
    )
    x = data["x"]
    t = data["t"]
    yf = data["yf"]
    return x, t, yf


def load_Jobs_observational(
    datasets_folder="./data", dataset_name="Jobs", details=False
):
    """
    Load observational data (x, t, yf), i.e., train split of multiple
    realizations (10) of the Jobs dataset
    :param datasets_folder: path to all datasets
    :param dataset_name: name of the dataset
    :param details: boolean value for whether to show the details or not
    :return:
    """
    data = load_Jobs(
        datasets_folder=datasets_folder,
        dataset_name=dataset_name,
        split="train",
        details=details,
    )
    x = data["x"]
    t = data["t"]
    yf = data["yf"]
    return x, t, yf

def load_TWINS_observational(
    datasets_folder="./data",
    dataset_name="TWINS",
    details=False,
):
    """
    Load the TWINS dataset with split training and test sets
    :param datasets_folder: path to all datasets
    :param dataset_name: name of the dataset
    :param details: boolean value for whether showing the details
    :return:
    """
    data = load_TWINS(
        datasets_folder, dataset_name, split="train", details=details
    )
    x_len = data["x"].shape[0]
    x = data["x"].reshape((x_len, -1, 1))
    t = data["t"].reshape((-1, 1))
    yf = data["yf"].reshape((-1, 1))
    return x, t, yf

def load_IHDP_out_of_sample(
    datasets_folder="./data", dataset_name="IHDP-100", details=False
):
    """
    Load out-of-sample (x) data, i.e., test split of multiple realizations
     (100 and 1000) of the IHDP dataset
    :param datasets_folder: path to all datasets
    :param dataset_name: name of the dataset
    :param details: boolean value for whether to show the details or not
    :return:
    """
    data = load_IHDP(
        datasets_folder=datasets_folder,
        dataset_name=dataset_name,
        split="test",
        details=details,
    )
    x_test = data["x"]
    t_test = data["t"]
    yf_test = data["yf"]
    return x_test, t_test, yf_test


def load_Jobs_out_of_sample(
    datasets_folder="./data", dataset_name="Jobs", details=False
):
    """
    Load out-of-sample (x) data, i.e., test split of multiple realizations
     (10) of the Jobs dataset
    :param datasets_folder: path to all datasets
    :param dataset_name: name of the dataset
    :param details: boolean value for whether to show the details or not
    :return:
    """
    data = load_Jobs(
        datasets_folder=datasets_folder,
        dataset_name=dataset_name,
        split="test",
        details=details,
    )
    x_test = data["x"]
    t_test = data["t"]
    yf_test = data["yf"]
    return x_test, t_test, yf_test

def load_TWINS_out_of_sample(
    datasets_folder="./data",
    dataset_name="TWINS",
    details=False,
):
    """
    Load the TWINS dataset with split training and test sets
    :param datasets_folder: path to all datasets
    :param dataset_name: name of the dataset
    :param details: boolean value for whether showing the details
    :return:
    """
    data = load_TWINS(
        datasets_folder, dataset_name, split="test", details=details
    )
    x_len = data["x"].shape[0]
    x_test = data["x"].reshape((x_len, -1, 1))
    t_test = data["t"].reshape((-1, 1))
    yf_test = data["yf"].reshape((-1, 1))

    return x_test, t_test, yf_test

def load_IHDP_ground_truth(
    datasets_folder="./data", dataset_name="IHDP-100", details=False
):
    """
    Load the ground truth of both in-sample and out-of-sample of IHDP
    :param datasets_folder:
    :param dataset_name:
    :param details:
    :return:
    """
    training_data = load_IHDP(
        datasets_folder=datasets_folder,
        dataset_name=dataset_name,
        split="train",
        details=details,
    )
    mu0_in, mu1_in = training_data["mu0"], training_data["mu1"]
    test_data = load_IHDP(
        datasets_folder=datasets_folder,
        dataset_name=dataset_name,
        split="test",
        details=details,
    )
    mu0_out, mu1_out = test_data["mu0"], test_data["mu1"]
    return mu0_in, mu1_in, mu0_out, mu1_out


def load_Jobs_ground_truth(
    datasets_folder="./data", dataset_name="Jobs", details=False
):
    """
    Load the ground truth of both in-sample and out-of-sample of Jobs
    :param datasets_folder:
    :param dataset_name:
    :param details:
    :return:
    """
    training_data = load_Jobs(
        datasets_folder=datasets_folder,
        dataset_name=dataset_name,
        split="train",
        details=details,
    )
    ate_in = training_data["ate"]
    test_data = load_Jobs(
        datasets_folder=datasets_folder,
        dataset_name=dataset_name,
        split="test",
        details=details,
    )
    ate_out = test_data["ate"]
    return ate_in, ate_out

def load_TWINS_ground_truth(
    datasets_folder="./data",
    dataset_name="TWINS",
    details=False,
):
    """
    Load the TWINS dataset with split training and test sets
    :param datasets_folder: path to all datasets
    :param dataset_name: name of the dataset
    :param details: boolean value for whether showing the details
    :return:
    """
    training_data = load_TWINS(
        datasets_folder=datasets_folder,
        dataset_name=dataset_name,
        split="train",
        details=details,
    )
    y0_in, y1_in = training_data["y0"].reshape((-1, 1)), training_data["y1"].reshape((-1, 1))
    test_data = load_TWINS(
        datasets_folder=datasets_folder,
        dataset_name=dataset_name,
        split="test",
        details=details,
    )
    y0_out, y1_out = test_data["y0"].reshape((-1, 1)), test_data["y1"].reshape((-1, 1))
    return y0_in, y1_in, y0_out, y1_out


def save_in_and_out_results(
    estimation_result_folder, y0_in, y1_in, ate_in, y0_out, y1_out, ate_out
):
    os.makedirs(estimation_result_folder, exist_ok=True)
    file_name = os.path.join(
        estimation_result_folder, "in_and_out_results.npy"
    )
    with open(file_name, "wb") as f:
        np.save(f, y0_in)
        np.save(f, y1_in)
        np.save(f, ate_in)
        np.save(f, y0_out)
        np.save(f, y1_out)
        np.save(f, ate_out)


def load_in_and_out_results(estimation_result_folder):
    file_name = os.path.join(
        estimation_result_folder, "in_and_out_results.npy"
    )
    with open(file_name, "rb") as f:
        y0_in = np.load(f, allow_pickle=True)
        y1_in = np.load(f, allow_pickle=True)
        ate_in = np.load(f, allow_pickle=True)
        y0_out = np.load(f, allow_pickle=True)
        y1_out = np.load(f, allow_pickle=True)
        ate_out = np.load(f, allow_pickle=True)

    return y0_in, y1_in, ate_in, y0_out, y1_out, ate_out
