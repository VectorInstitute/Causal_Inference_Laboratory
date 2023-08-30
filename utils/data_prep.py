# open berekely_numeric.csv and save as npz files

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split


csv = pd.read_csv('..//data//CFA//berkeley_numeric.csv')
# drop first row and first column
csv = csv.drop(csv.columns[0], axis=1)
csv = csv.drop(csv.index[0])

# first col is outcome
yf = csv.iloc[:,0].values
# second col is treatment
t = csv.iloc[:,1].values
# rest are covariates
x = csv.iloc[:,2].values

# convert to numpy arrays
yf = np.array(yf, dtype=np.float32)
t = np.array(t, dtype=np.float32)
x = np.array(x, dtype=np.float32)

# test train split
x_train, x_test, yf_train, yf_test, t_train, t_test = train_test_split(x, yf, t, test_size=0.2, random_state=42)

# save as npz files
np.savez('..//data//CFA//berkeley.train.npz', x=x_train, yf=yf_train, t=t_train)
np.savez('..//data//CFA//berkeley.test.npz', x=x_test, yf=yf_test, t=t_test)
