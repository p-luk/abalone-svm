#!/usr/bin/env python
# coding: utf-8



import os
import sys
import pandas as pd
import numpy as np

# clean data
filepath = "./abalone/abalone.data"
data = pd.read_csv(filepath, delimiter=',',header=None)

# convert data to vectors
data.columns = [str(col) for col in data.columns]
y = data['8'].apply(lambda x:1 if x<= 9 else -1).to_numpy(dtype=int)
data = pd.concat([data, pd.get_dummies(data['0'])], axis=1)
data.drop(['0','8'], axis=1, inplace=True)
x = data.values

# apply transformation
x_prime = []
for i in range(len(x)):
    x_prime_vec = []
    for j in range(len(x)):
        x_prime_vec.append(y[i] * (x[i].dot(x[j])))
    x_prime.append(x_prime_vec)

# write to files
with open('./abalone/abalone_train_mod.txt', 'w') as f:
    for i in range(3133):
        label = "+1" if y[i] > 0 else "-1"
        sparse_line = ""
        for j,elem in enumerate(x_prime[i]):
            sparse_line += " {}:{}".format(j,elem)
        f.write(label + sparse_line + "\n")


with open('./abalone/abalone_test_mod.txt', 'w') as f:
    for i in range(3134,4177):
        label = "+1" if y[i] > 0 else "-1"
        sparse_line = ""
        for j,elem in enumerate(x_prime[i]):
            sparse_line += " {}:{}".format(j,elem)
        f.write(label + sparse_line + "\n")


# scale data
os.popen('libsvm-3.24/svm-scale -s abalone/scaled_params_mod.txt abalone/abalone_train_mod.txt > abalone/abalone_train_mod_scaled.txt')
os.popen('libsvm-3.24/svm-scale -r abalone/scaled_params_mod.txt  abalone/abalone_test.txt > abalone/abalone_test_mod_scaled.txt')

