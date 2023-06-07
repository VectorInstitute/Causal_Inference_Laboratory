# open berekely_numeric.csv and save as npz files

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 0

csv = pd.read_csv('data/CFA/berkeley_numeric.csv')
X = [1] #column ids of the protected attributes
Y = [0] #column ids of the outcome
W = [2]
Z = []
dataset_name = "berkeley"

# drop first row and first column
csv = csv.drop(csv.columns[0], axis=1)
csv = csv.drop(csv.index[0])

# first col is outcome
yf = csv.iloc[:, Y].values
# second col is treatment
t = csv.iloc[:, X].values
# rest are covariates
x = csv.iloc[:, Z + W].values
x = np.expand_dims(x, axis=-1)

# # convert to numpy arrays
yf = np.array(yf, dtype=np.float32)
t = np.array(t, dtype=np.float32)
x = np.array(x, dtype=np.float32)

# # test train split
x_train, x_test, yf_train, yf_test, t_train, t_test = train_test_split(x, yf, t, test_size=0.2, random_state=seed)

# # save as npz files
np.savez('data/CFA/' + dataset_name + '_e1.train.npz', x=x_train, yf=yf_train, t=t_train)
np.savez('data/CFA/' + dataset_name + '_e1.test.npz', x=x_test, yf=yf_test, t=t_test)


x = csv.iloc[:, Z].values

# # convert to numpy arrays
x = np.array(x, dtype=np.float32)
x = np.expand_dims(x, axis=-1)

# # test train split
x_train, x_test, yf_train, yf_test, t_train, t_test = train_test_split(x, yf, t, test_size=0.2, random_state=seed)

# # save as npz files
np.savez('data/CFA/' + dataset_name + '_e2.train.npz', x=x_train, yf=yf_train, t=t_train)
np.savez('data/CFA/' + dataset_name + '_e2.test.npz', x=x_test, yf=yf_test, t=t_test)