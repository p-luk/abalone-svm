#!/usr/bin/env python
# coding: utf-8


import os
import sys

# clean data
filepath = "./abalone/abalone.data"
sparse_abalone = []

with open(filepath) as abalone:
    sex = {"M":"1", "F":"2", "I": "3"}
    for i,line in enumerate(abalone):
        sparse_segment = ""
        for j,tok in enumerate(line.split(',')):
            # labels are last token of a row; convert to 2-class classification
            if j == len(line.split(','))-1:
                label = "+1" if int(tok) <= 9 else "-1"
            
            # otherwise, treat as normal feature
            elif tok in sex.keys(): # libsvm does not accept nominal data; create indicators
                sparse_segment += " " + sex[tok] + ":"+ "1"
            else: 
                sparse_segment += " " + str(j+3) + ":"+ tok
        
        sparse_line = label + sparse_segment +"\n"
        # construct sparse row format
        sparse_abalone.append(sparse_line)

sparse = open("./abalone/abalone_sparse.txt", "w")
sparse.writelines([str(line) for line in sparse_abalone])
sparse.close()

train = open("./abalone/abalone_train.txt", "w")
train.writelines([str(line) for line in sparse_abalone[:3133]])
train.close()

test = open("./abalone/abalone_test.txt", "w")
test.writelines([str(line) for line in sparse_abalone[3133:]])
test.close()

# scale data
os.popen('libsvm-3.24/svm-scale -s abalone/scaled_params.txt abalone/abalone_train.txt > abalone/abalone_train_scaled.txt')
os.popen('libsvm-3.24/svm-scale -r abalone/scaled_params.txt  abalone/abalone_test.txt > abalone/abalone_test_scaled.txt')



