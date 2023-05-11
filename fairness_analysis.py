import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from math import sqrt

from fairness.fairness_cookbook import fairness_cookbook

estimator_names = ["AutoML", "OLS1", "OLS2", "NN1", "NN2", "RF1", "RF2", "IPW", "DML", "Dragonnet", "TARNet"]


def make_numeric(data, data_name):
    num_sample = data.shape[0]
    if data_name == "census":
        data[:, 0] = np.array([0 if data[i, 0] == "male" else 1 for i in range(num_sample)])
        data[:, 2] = np.array([hash(data[i, 2]) % 31 for i in range(num_sample)])
        data[:, 3] = np.array([hash(data[i, 3]) % 15 for i in range(num_sample)])
        data[:, 5] = np.array([hash(data[i, 5]) % 15 for i in range(num_sample)])
        data[:, 6] = np.array([hash(data[i, 6]) % 31 for i in range(num_sample)])
        data[:, 14] = np.array([hash(data[i, 14]) % (2 ** 13 - 1) for i in range(num_sample)])
        data[:, 15] = np.array([hash(data[i, 15]) % 2047 for i in range(num_sample)])
        data[:, 16] = np.array([hash(data[i, 16]) % 63 for i in range(num_sample)])
        pd.DataFrame(data).to_csv("data/CFA/gov_census_numeric.csv")
    elif data_name == "berkeley":
        data[:, 0] = np.array([1 if data[i, 0] == "Admitted" else 0 for i in range(num_sample)])
        data[:, 1] = np.array([0 if data[i, 1] == "Male" else 1 for i in range(num_sample)])
        data[:, 2] = np.array([ord(data[i, 2]) - ord('A') for i in range(num_sample)])
        pd.DataFrame(data).to_csv("data/CFA/berkeley_numeric.csv")
    elif data_name == "compas":
        data[:, 0] = np.array([0 if data[i, 0] == "Male" else 1 for i in range(num_sample)])
        data[:, 2] = np.array([0 if data[i, 2] == "White" else 1 for i in range(num_sample)])
        data[:, 7] = np.array([0 if data[i, 7] == "F" else 1 for i in range(num_sample)])
        pd.DataFrame(data).to_csv("data/CFA/compas_numeric.csv")

def load_data(data_addr):
    df = pd.read_csv(data_addr)
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    data = df.to_numpy()
    # data = data[:20000, :]
    return data

def plot_confidence_intervals(metrics, z=1.96, plot_name="fairness_metrics", plot_dir="fairness/plots/", save_plot=True):
    def plot_confidence_interval(x, values, bar_color='#2187bb', o_color='#f44336', horizontal_line_width=0.25):
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        confidence_interval = z * stdev / sqrt(len(values))

        left = x - horizontal_line_width / 2
        top = mean - confidence_interval
        right = x + horizontal_line_width / 2
        bottom = mean + confidence_interval
        plt.plot([x, x], [top, bottom], color=bar_color)
        plt.plot([left, right], [top, top], color=bar_color)
        plt.plot([left, right], [bottom, bottom], color=bar_color)
        plt.plot(x, mean, 'o', color=o_color)
    
    num_metrics = metrics.shape[1]
    x = [i + 1 for i in range(num_metrics)]
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['figure.figsize'] = [4, 3]

    plt.xticks(x, ["TV", "DE", "IE", "SE"])
    plt.title("Confidence Interval")

    for i in range(num_metrics):
        plot_confidence_interval(i + 1, metrics[:, i])
    if save_plot:
        plt.savefig(plot_dir + plot_name + ".png")


if __name__ == "__main__":

    data_name = "200obs"
    # data_name = "berkeley"
    # data_name = "compas"
    # data_name = "census"

    # if data_name == "census":
    #     data_addr = "data/CFA/gov_census.csv"
    # elif data_name == "berkeley":
    #     data_addr = "data/CFA/berkeley.csv"
    # elif data_name == "compas":
    #     data_addr = "data/CFA/compas.csv"

    # data = load_data(data_addr)
    # make_numeric(data, data_name)

    if data_name == "200obs":
        data_addr = "data/CFA-Synthetic/synthetic_data_200.csv"
    elif data_name == "census":
        data_addr = "data/CFA/gov_census_numeric.csv"
    elif data_name == "berkeley":
        data_addr = "data/CFA/berkeley_numeric.csv"
    elif data_name == "compas":
        data_addr = "data/CFA/compas_numeric.csv"

    data = load_data(data_addr)
    
    if data_name == "200obs":
        X = [1]
        Y = [0]
        W = [2]
        Z = [3]
        x0 = 0
        x1 = 1
    elif data_name == "census":
        X = [0]
        Y = [11]
        W = [1, 2, 3, 4, 5, 16]
        Z = [6, 7, 8, 9, 10, 12, 13, 14, 15]
        x0 = 0
        x1 = 1
    elif data_name == "berkeley":
        X = [1]
        Y = [0]
        W = [2]
        Z = []
        x0 = 0
        x1 = 1
    elif data_name == "compas":
        X = [2]
        Y = [8]
        W = [3, 4, 5, 6, 7]
        Z = [0, 1]
        x0 = 0
        x1 = 1 

    estimator_name = estimator_names[0]
    num_boot = 2

    all_metrics = np.zeros((num_boot, 4))
    for i in range(num_boot):
        print("-" * 15 + " Run " + str(i) + " " + "-" * 15)
    # TODO: Add symmetry for metrics.
        metrics = fairness_cookbook(data, X = X, Z = Z, Y = Y, W = W,
                                    x0 = x0, x1 = x1, estimator_name = estimator_name)
        all_metrics[i][0] = metrics["tv"]
        all_metrics[i][1] = metrics["ctfde"]
        all_metrics[i][2] = metrics["ctfie"]
        all_metrics[i][3] = metrics["ctfse"]
        print("-" * 37)

        
    plot_confidence_intervals(all_metrics)
