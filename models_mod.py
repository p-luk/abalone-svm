#!/usr/bin/env python
# coding: utf-8


import os
import sys
import re

# Problem 6d. Cross-Validation Varying both d and k


output_file = 'output/cv_results_mod.out'
train_script = './libsvm-3.24/svm-train -t 1 -d {} -c {} -v 10 ./abalone/abalone_train_mod_scaled.txt'

best = [0,0,1] #d, k, error

print('Initial CV ...')
with open(output_file, 'w') as f:
    f.write('d, k, error\n')
    for d in range(1,5):
        for k in range(-8,9):
            print(d,k)
            c = float(2**k)
            stream = os.popen(train_script.format(d,c,d))
            result = stream.read()
            acc = float(re.findall('(?<= = ).+(?=%)', result)[0])/100
            if (1-acc) <= best[2]: # new best
                best = [d,k,1-acc]
            f.write('{},{},{}\n'.format(d,k, str(1-acc)))

print('Writing best-in-class ...')
with open('output/cv_best_mod.out', 'w') as f:
    f.write('d={}, k={},error={}'.format(d,k, str(1-acc)))

print('CV for best-in-class ...')
# Problem 5. Fixing k = best from above and varying d

k = best[1]
c = float(2**k)

# cross-validation for best-in-class
output_file = 'output/cv_best_mod.out'
cv_script = './libsvm-3.24/svm-train -t 1 -d {} -c {} -v 10 ./abalone/abalone_train_mod_scaled.txt'
train_script = './libsvm-3.24/svm-train -t 1 -d {} -c {} ./abalone/abalone_train_mod_scaled.txt ./output/model_mod_d{}'

with open(output_file, 'w') as f:
    f.write('d, k, error, nsv, nbsv\n')
    for d in range(1,5):
        stream = os.popen(cv_script.format(d,c))
        result = stream.read()
        
        nsv = re.findall('(?<=nSV = ).+(?=,)', result)
        nbsv = re.findall('(?<=nBSV = ).+(?=\n)', result)
        acc = float(re.findall('(?<= = ).+(?=%)', result)[0])/100
        error = 1-acc
        for i in range(len(nsv)):
            f.write('{},{},{},{},{}\n'.format(d,k, error, nsv[i], nbsv[i]))
        
        os.popen(train_script.format(d,c,d))


print('Test for best-in-class ...')
# test for best-in-class
output_file = 'output/test_mod_best.out'
test_script = './libsvm-3.24/svm-predict ./abalone/abalone_test_mod_scaled.txt ./output/model_mod_d{} ./output/test_results_mod_d{}'

with open(output_file, 'w') as f:
    f.write('d, k, error\n')
    for d in range(1,5):
        stream = os.popen(test_script.format(d,c,d))
        result = stream.read()
        acc = float(re.findall('(?<== ).+(?=%)', result)[0])/100
        error = 1-acc
        print(stream)
        f.write('{},{},{}\n'.format(d,k, error))
print('done!')

