
# testing general behavioral characteristics with randomization
# finished editing?

import pandas as pd
import random
import pprint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import string
import re

trialresponses = pd.read_excel('/Users/charlesconnor/Dropbox/grants/social.memory/20250401_ANOVA_and_linear_regression_results/spike_counts_for_all_anova_passed_time_windowed_cells.xlsx')

responsestringarray = trialresponses.values

#nmonkeys = 10 #must match behavior matrix size; subject and source monkeys removed as appropriate in code below
#nmonkeysnotsubject = 9

nmonkeys = 10

affiliationtomatrix =[[0,19,9,38,84,27,13,14,4,9],[15,0,9,41,4,13,19,71,12,0],[18,10,0,21,7,17,49,18,3,1],[38,43,24,0,18,6,31,26,29,4],[90,3,8,8,0,22,8,9,1,18],[23,12,17,1,23,0,23,16,3,10],[17,18,43,34,10,23,0,34,2,9],[11,70,18,17,8,17,33,0,5,3],[3,6,4,31,2,1,2,4,0,9],[8,0,0,1,26,8,2,1,7,0]]

affiliationfrommatrix = []    #frequency[sinkmonkey][sourcemonkey]
for sinkmonkey in range (0, nmonkeys):
    sourcelist = []
    for sourcemonkey in range (0, nmonkeys):
        sourcelist.append(affiliationtomatrix[sourcemonkey][sinkmonkey])
    affiliationfrommatrix.append(sourcelist)

submissiontomatrix = [[0,0,0,1,0,1,0,0,2,0],[4,0,0,13,0,0,0,2,1,0],[9,2,0,10,0,0,4,3,0,0],[7,0,2,0,0,0,0,1,6,0],[11,8,3,5,0,2,4,2,4,18],[90,16,14,43,0,0,10,7,9,6],[29,12,1,17,0,0,0,23,17,1],[41,6,1,28,0,0,2,0,5,0],[16,2,0,8,0,1,0,0,0,2],[7,2,3,1,2,0,5,2,5,0]]

submissionfrommatrix = []    #frequency[sinkmonkey][sourcemonkey]
for sinkmonkey in range (0, nmonkeys):
    sourcelist = []
    for sourcemonkey in range (0, nmonkeys):
        sourcelist.append(submissiontomatrix[sourcemonkey][sinkmonkey])
    submissionfrommatrix.append(sourcelist)

agonismtomatrix = [[0,4,2,7,4,7,3,9,3,2],[0,0,0,2,1,1,8,4,2,0],[0,1,0,0,0,7,0,1,0,2],[1,8,0,0,9,5,6,10,11,2],[0,0,0,0,0,1,0,1,3,10],[0,0,0,0,2,0,0,0,3,1],[0,0,3,2,0,4,0,1,0,6],[0,1,1,0,1,0,7,0,0,1],[0,0,0,2,2,5,2,3,0,6],[0,0,0,0,5,4,1,0,6,0]]

agonismfrommatrix = []    #frequency[sinkmonkey][sourcemonkey]
for sinkmonkey in range (0, nmonkeys):
    sourcelist = []
    for sourcemonkey in range (0, nmonkeys):
        sourcelist.append(agonismtomatrix[sourcemonkey][sinkmonkey])
    agonismfrommatrix.append(sourcelist)

ncells = 74
monkeyname = [ '7124', '69X', '72X', '94B', '110E', '67G', '81G', '143H', '87J', '151J' ]
monkey_column = [14, 2, 12, 13, 16, 10, 36, 23, 15]
monkeyname_notsubject = [ '7124', '69X', '72X', '94B', '110E', '67G', '143H', '87J', '151J' ]

nboots = 1 # > 1 for bootstrap random sampling of one response from each cell; commented out below
subjectmonkey = 6    #81G
    
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
svr_sig = SVR(kernel="sigmoid", C=100, gamma="auto")
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

lw = 2

svrs = [svr_lin]
kernel_label = ["RBF", "Linear", "Polynomial"]
model_color = ["m", "c", "g"]

svr = svr_lin
#svr = svr_sig

nbehaviors = 6
#Rsquared = [[[0.0 for x in range(ncells)] for x in range(nmonkeys)] for x in range (nbehaviors)]
#print('Rsquared = ', Rsquared)

Rsquared = []    #3-dimensional list of Rsquared values above 0.25; Rsquared[behavior][sourcemonkey][cell]
for i in range (0, nbehaviors):
    behavior_list = []    #outer or 1st dimension of list array is behaviors
    for j in range (0, nmonkeys):
        monkey_list = []    #2nd dimension is source monkeys
        for k in range (0, ncells):
            monkey_list.append(0.0)    #3rd dimension is cells
        behavior_list.append(monkey_list)
    Rsquared.append(behavior_list)
    
    

nmonkeys = 9		#leaving out subject monkey

sng_dif = []
for i in range (0, nmonkeys):
    sng_dif.append(0.0)
        
anon_sng_dif = []
for i in range (0, nmonkeys):
    anon_sng_dif.append(0.0)
        
sng_less = []
for i in range (0, nmonkeys):
    sng_less.append(0)

anon_sng_less = []
for i in range (0, nmonkeys):
    anon_sng_less.append(0)
    


nlesstotal = 0
nlesssig = 0
nsig = 0

sum_obs = 0.0
sum_less = 0
aff_sum = 0.0
aff_prd_sum = 0.0


# observed values

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

behsumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for icell in range (0, ncells):
    
    means = []
    for sourcemonkey in range (0, nmonkeys):
        responsestring = responsestringarray[icell, monkey_column[sourcemonkey]]
        responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
        meanresponse = sum(responselist) / len(responselist)
        means.append(meanresponse)
    maxmean = max(means)
    idx_max = means.index(maxmean)
    means[idx_max] = 0.0
    sng_avg = maxmean
    pop_avg = sum(means) / (len(means) - 1.0)
    obs_sng_dif = (sng_avg - pop_avg) / maxmean
    sum_obs += obs_sng_dif
    means[idx_max] = maxmean
    sng_dif[idx_max] += obs_sng_dif
    

#randomization
    

for ir in range (0, 1000):
    
    sum_rnd_obs = 0.0
    
    rnd_dif = []
    for i in range (0, nmonkeys):
        rnd_dif.append(0.0)

    for icell in range (0, ncells):

        means = []
        bucket = []
        mnk_len = []
        for sourcemonkey in range (0, nmonkeys):
            responsestring = responsestringarray[icell, monkey_column[sourcemonkey]]
            responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
            mnk_len.append(len(responselist))
            for irsp in range (0, len(responselist)):
                bucket.append(responselist[irsp])
        random.shuffle(bucket)
        bkt_idx = 0
        for sourcemonkey in range (0, nmonkeys):
            meanresponse = 0.0
            for irsp in range (bkt_idx, bkt_idx + mnk_len[sourcemonkey]):
                meanresponse += bucket[irsp]
            meanresponse /= float(mnk_len[sourcemonkey])
            means.append(meanresponse)
            bkt_idx += mnk_len[sourcemonkey]
            
        maxmean = max(means)
        idx_max = means.index(maxmean)
        means[idx_max] = 0.0
        sng_avg = maxmean
        pop_avg = sum(means) / (len(means) - 1.0)
        rnd_sng_dif = (sng_avg - pop_avg) / maxmean
        sum_rnd_obs += rnd_sng_dif
        means[idx_max] = maxmean
        rnd_dif[idx_max] += rnd_sng_dif


    for i in range (0, nmonkeys):
        if (rnd_dif[i] < sng_dif[i]):
                sng_less[i] += 1
                
    if (sum_rnd_obs < sum_obs):
        sum_less += 1
        
        
print ('sum_less (total):', sum_less)
                        
for i in range (0, nmonkeys):
    print ('for ', monkeyname_notsubject[i])
    print ('sng_less:', sng_less[i])
        
        
        
        
        
        
        
        
        
        