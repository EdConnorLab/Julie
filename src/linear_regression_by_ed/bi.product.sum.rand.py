
# linear regression for behavior patterns and single monkey ID based on mean responses,
# no bootstrapping from individual responses

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
import math

trialresponses = pd.read_excel('/Users/charlesconnor/Dropbox/grants/social.memory/20250401_ANOVA_and_linear_regression_results/spike_counts_for_all_anova_passed_time_windowed_cells.xlsx')

responsestringarray = trialresponses.values

nmonkeys = 10 #must match behavior matrix size; subject and source monkeys removed as appropriate in code below
nmonkeysnotsubject = 9

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

#cells = [0, 2, 7, 10, 13, 15, 17, 19, 21, 22, 24, 25, 30, 32, 34, 44, 46, 48, 50, 53, 55, 57, 59, 67]
#cellnames = ['9/26 1 2 1', '9/26 3 2 1', '10/3 3 13 1', '10/3 4 10 2', '10/4 1 4 1', '10/4 1 19 3', '10/4 2 18 1', '10/4 2 20 2', '10/4 3 2 1', '10/4 3 4 2', '10/4 3 9 1', '10/4 3 9 3', '10/4 4 22 2', '10/4 4 27 1', '10/5 1 4', '10/11 1 2', '10/11 3 2', '10/11 3 13', '10/24 2 2', '10/27 4 11', '10/27 4 20', '10/31 1 5', '10/31 1 20', '11/8 1 7']
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

    
nless = []    # randomized Rsquared values less than observed out of 1000 randomizations
for i in range (0, nbehaviors):
    rand_behavior_list = []    #outer or 1st dimension of list array is behaviors
    for j in range (0, nmonkeys):
        rand_monkey_list = []    #2nd dimension is source monkeys
        for k in range (0, ncells):
            rand_monkey_list.append(0)    #3rd dimension is cells
        rand_behavior_list.append(rand_monkey_list)
    nless.append(rand_behavior_list)
    
#print(Rsquared)
    
behaviornames = ['affil to', 'affil from', 'sub to', 'sub from', 'agon to', 'agon from']

bisquared = []    #1-dimensional list of explained variance for pair coding
for i in range (0, ncells):
    bisquared.append(0.0)
    
maxrsq = []    #-dimensional list of rsquared max for each cell across behaviors and source monkeys
for i in range (0, ncells):
    maxrsq.append(0.0)

maxbeh = []    #-dimensional list of rsquared max for each cell across behaviors and source monkeys
for i in range (0, ncells):
    maxbeh.append(0)

maxsrcmnk = []    #-dimensional list of rsquared max for each cell across behaviors and source monkeys
for i in range (0, ncells):
    maxsrcmnk.append(0)
    
aff_prd_sum = 0.0
sub_prd_sum = 0.0
agn_prd_sum = 0.0

aff_less = 0
sub_less = 0
agn_less = 0


for icell in range (0, ncells):
    lostmonkeys = 0
    nless = 0
    y = []
    ypred = []
                        
    means = []
    for sourcemonkey in range (0, nmonkeys):
        if (sourcemonkey == subjectmonkey):
            lostmonkeys +=1
        else:
            responsestring = responsestringarray[icell, monkey_column[sourcemonkey - lostmonkeys]]
            responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
            meanresponse = sum(responselist) / len(responselist)
            means.append(meanresponse)
    maxmean = max(means)
    idx_max = means.index(maxmean)
    means[idx_max] = 0.0
    nxtmean = max(means)
    idx_nxt = means.index(nxtmean)
    #print('icell: ', icell, 'idx_max: ', idx_max, 'idx_nxt: ', idx_nxt)
    means[idx_nxt] = 0.0
    pair_avg = (maxmean + nxtmean) / 2.0
    pop_avg = sum(means) / (len(means) - 2.0)
    means[idx_max] = maxmean
    means[idx_nxt] = nxtmean

    for i in range(0, len(means)):
        if ((i == idx_max) or (i == idx_nxt)):
            y.append(pair_avg)
        else:
            y.append(pop_avg)
        ypred.append(means[i])

    bisquared[icell] = r2_score(y, ypred)

    if (idx_max < subjectmonkey):
        maxmonkey = idx_max
    else:
        maxmonkey = idx_max +1
    if (idx_nxt < subjectmonkey):
        nxtmonkey = idx_nxt
    else:
        nxtmonkey = idx_nxt +1


    aff = affiliationtomatrix[maxmonkey][nxtmonkey] + affiliationfrommatrix[maxmonkey][nxtmonkey]
    aff_prd = aff * bisquared[icell]
    aff_prd_sum += aff_prd

    sub = submissiontomatrix[maxmonkey][nxtmonkey] + submissionfrommatrix[maxmonkey][nxtmonkey]
    sub_prd = sub * bisquared[icell]
    sub_prd_sum += sub_prd

    agn = agonismtomatrix[maxmonkey][nxtmonkey] + agonismfrommatrix[maxmonkey][nxtmonkey]
    agn_prd = agn * bisquared[icell]
    agn_prd_sum += agn_prd
    
    

for ir in range (0, 1000):
    
    rnd_aff_prd_sum = 0.0
    rnd_sub_prd_sum = 0.0
    rnd_agn_prd_sum = 0.0

    for icell in range (0, ncells):
        lostmonkeys = 0
        nless = 0
        y = []
        ypred = []
                            
        means = []
        for sourcemonkey in range (0, nmonkeys):
            if (sourcemonkey == subjectmonkey):
                lostmonkeys +=1
            else:
                responsestring = responsestringarray[icell, monkey_column[sourcemonkey - lostmonkeys]]
                responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
                meanresponse = sum(responselist) / len(responselist)
                means.append(meanresponse)
        random.shuffle(means)
        maxmean = max(means)
        idx_max = means.index(maxmean)
        means[idx_max] = 0.0
        nxtmean = max(means)
        idx_nxt = means.index(nxtmean)
        #print('icell: ', icell, 'idx_max: ', idx_max, 'idx_nxt: ', idx_nxt)
        means[idx_nxt] = 0.0
        pair_avg = (maxmean + nxtmean) / 2.0
        pop_avg = sum(means) / (len(means) - 2.0)
        means[idx_max] = maxmean
        means[idx_nxt] = nxtmean

        for i in range(0, len(means)):
            if ((i == idx_max) or (i == idx_nxt)):
                y.append(pair_avg)
            else:
                y.append(pop_avg)
            ypred.append(means[i])

        bisquared[icell] = r2_score(y, ypred)

        if (idx_max < subjectmonkey):
            maxmonkey = idx_max
        else:
            maxmonkey = idx_max +1
        if (idx_nxt < subjectmonkey):
            nxtmonkey = idx_nxt
        else:
            nxtmonkey = idx_nxt +1

        aff = affiliationtomatrix[maxmonkey][nxtmonkey] + affiliationfrommatrix[maxmonkey][nxtmonkey]
        aff_prd = aff * bisquared[icell]
        rnd_aff_prd_sum += aff_prd

        sub = submissiontomatrix[maxmonkey][nxtmonkey] + submissionfrommatrix[maxmonkey][nxtmonkey]
        sub_prd = sub * bisquared[icell]
        rnd_sub_prd_sum += sub_prd

        agn = agonismtomatrix[maxmonkey][nxtmonkey] + agonismfrommatrix[maxmonkey][nxtmonkey]
        agn_prd = agn * bisquared[icell]
        rnd_agn_prd_sum += agn_prd
        
    if (rnd_aff_prd_sum < aff_prd_sum):
        aff_less += 1
    if (rnd_sub_prd_sum < sub_prd_sum):
        sub_less += 1
    if (rnd_agn_prd_sum < agn_prd_sum):
        agn_less += 1
        
print('aff_less =', aff_less, 'sub_less = ', sub_less, 'agn_less =', agn_less)

