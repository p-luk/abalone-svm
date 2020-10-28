#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import math



# Problem 4. Cross-Validation Varying both d and k
cv_df = pd.read_csv('output/cv_results_mod.out', names=['d', 'k', 'error']).apply(pd.to_numeric, errors='coerce')

plt.figure()
fig, ax = plt.subplots(2,2, sharex=True, figsize=(12,8))
ax = ax.flatten()
plt.suptitle("Cross-Validation Error (+/-1 std) by Polynomial Degree", fontsize=14)


for d in range(1,5):
    sub_df = cv_df[cv_df['d'] == d]
    std_up = sub_df['error'].apply(lambda x: x - math.sqrt((x*(1-x))/3133))
    std_down = sub_df['error'].apply(lambda x: x + math.sqrt((x*(1-x))/3133))
    ax[d-1].plot(sub_df['k'], sub_df['error'], label='Mean CV Error')
    ax[d-1].plot(sub_df['k'], std_up, label='Error +1 Std')
    ax[d-1].plot(sub_df['k'], std_down, label='Error -1 Std')
    ax[d-1].set_title('d='+str(d))
    ax[d-1].grid(True)
handles, labels = ax[d-1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.text(0.5, 0.03, '$\mathregular{Log_2C}$', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'Avg. Error', va='center', rotation='vertical', fontsize=12)

plt.savefig("plots/cv_vs_d_mod.png")
plt.close()



# Problem 5. Fixing k = best from above and varying d

cv_best_df = pd.read_csv('output/cv_best_mod.out', names=['d', 'k', 'error', 'nsv', 'nbsv']).apply(pd.to_numeric, errors='coerce')
test_best_df = pd.read_csv('output/test_mod_best.out', names=['d', 'k', 'error']).apply(pd.to_numeric, errors='coerce')

plt.figure()
plt.plot(cv_best_df['d'], cv_best_df['error'], label='Cross Validation')
plt.plot(test_best_df['d'], test_best_df['error'], label='Test')

plt.legend()
plt.grid(True)
plt.xlabel('Degree d')
plt.ylabel('Error (%)')
plt.title('Errors vs. Polynomial Kernel Degree for C=$\mathregular{2^8}$')
plt.savefig("plots/cv_vs_test_mod.png")
plt.close()