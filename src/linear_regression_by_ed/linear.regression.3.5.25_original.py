
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

trialresponses = pd.read_excel('/home/connorlab/Documents/JulieData/Cortana/Ed and ANOVA/used_for_R01/zombies_spike_counts_for_all_anova_passed_time_windowed_cells_old--usedforgrant.xlsx')

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
    
#siglist[icell][ibehavior][sourcemonkey] = rsquared
    
siglist = []

for icell in range (0, ncells):
    celladd = []
    for ibehavior in range (0, nbehaviors):
        behadd = []
        for sourcemonkey in range (0, nmonkeys):
            behadd.append(0)
        celladd.append(behadd)
    siglist.append(celladd)
    
behaviornames = ['affil to', 'affil from', 'sub to', 'sub from', 'agon to', 'agon from']

for ibehavior in range (0, 6):
#for ibehavior in range (1, 2):
    sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print('behavior: ',ibehavior)


    for icell in range (0, ncells):
    #for icell in range (8, 9):
        #print('icell: ', icell)
        for sourcemonkey in range (0, nmonkeys):
            #print('sourcemonkey: ', sourcemonkey)
            y = []
            x = []



            lostmonkeys = 0
            for sinkmonkey in range (0, nmonkeys):    #responses don't include subject monkey
                if ((sinkmonkey == sourcemonkey) or (sinkmonkey == subjectmonkey)):
                    if (sinkmonkey == subjectmonkey):
                        lostmonkeys += 1    #skip advancement through responselist per the index subtract below for absence of subject monkey
                else:
                    #print('sinkmonkey: ', sinkmonkey)
                    if (ibehavior == 0):
                        y.append(float(affiliationtomatrix[sourcemonkey][sinkmonkey]))
                    if (ibehavior == 1):
                        y.append(float(affiliationfrommatrix[sourcemonkey][sinkmonkey]))
                    if (ibehavior == 2):
                        y.append(float(submissiontomatrix[sourcemonkey][sinkmonkey]))
                    if (ibehavior == 3):
                        y.append(float(submissionfrommatrix[sourcemonkey][sinkmonkey]))
                    if (ibehavior == 4):
                        y.append(float(agonismtomatrix[sourcemonkey][sinkmonkey]))
                    if (ibehavior == 5):
                        y.append(float(agonismfrommatrix[sourcemonkey][sinkmonkey]))
                    response = []
                    responsestring = responsestringarray[icell,monkey_column[sinkmonkey - lostmonkeys]]
                    responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
                    #print (responsestring)
                    meanresponse = sum(responselist) / len(responselist)
                    x.append(meanresponse)

            n = len(x)

            m_x = sum(x) / len(x)
            SS_xy = 0.0
            SS_xx = 0.0
            m_y = 0.0
            m_x = 0.0
            S_x = 0.0
            SS_yy = 0.0
            S_y = 0.0
            for ixy in range (0, n):
                SS_xy += y[ixy]*x[ixy]
                SS_xx += x[ixy]*x[ixy]
                m_y += y[ixy]
                m_x += x[ixy]
                S_x += x[ixy]
                S_y += y[ixy]
                SS_yy += y[ixy]**2
                
            b_0 = (S_y*SS_xx - S_x*SS_xy) / (len(x)*SS_xx - S_x**2)
            b_1 = (len(x)*SS_xy - S_x*S_y) / (len(x)*SS_xx - S_x**2)
                    
            # predicted response vector
            ypred = []
            for ixy in range (0, len(y)):
                ypred.append(b_0 + b_1*x[ixy])

            r2_score(y, ypred)
            # Rsquared[ibehavior][sourcemonkey][icell] = explained_variance_score(y, ypred)
            Rsquared[ibehavior][sourcemonkey][icell] = r2_score(y, ypred)
            
            if (Rsquared[ibehavior][sourcemonkey][icell] > 0.5):
                sumRsquared[sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])
                if (b_1 < 0.0):
                    pass # print('b_1 = ', b_1)
                #print('r2_score: ', Rsquared[ibehavior][sourcemonkey][icell])
                            
                for ir in range (0, 1000):
                    random.shuffle(x)
                    # calculating cross-deviation and deviation about x
                    SS_xy = 0.0
                    SS_xx = 0.0
                    m_y = 0.0
                    m_x = 0.0
                    S_x = 0.0
                    SS_yy = 0.0
                    S_y = 0.0
                    for ixy in range (0, n):
                        SS_xy += y[ixy]*x[ixy]
                        SS_xx += x[ixy]*x[ixy]
                        m_y += y[ixy]
                        m_x += x[ixy]
                        S_x += x[ixy]
                        S_y += y[ixy]
                    b_0 = (S_y*SS_xx - S_x*SS_xy) / (len(x)*SS_xx - S_x**2)
                    b_1 = (len(x)*SS_xy - S_x*S_y) / (len(x)*SS_xx - S_x**2)

                    # predicted response vector
                    ypred = []
                    for ixy in range (0, len(y)):
                        ypred.append(b_0 + b_1*x[ixy])

                    r2_score(y, ypred)
                    randomRsquared = explained_variance_score(y, ypred)
                    if randomRsquared < Rsquared[ibehavior][sourcemonkey][icell]:
                        nless[ibehavior][sourcemonkey][icell] += 1
                # if (nless[ibehavior][sourcemonkey][icell] > 949):
                #     print('BEH ', ibehavior, 'cell ', icell, 'source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell], 'nless = ', nless[ibehavior][sourcemonkey][icell])
            if (Rsquared[ibehavior][sourcemonkey][icell] > 0.5):
                print('BEH', ibehavior, 'cell', icell, 'source monkey', monkeyname[sourcemonkey],
                      'Rsquared =', Rsquared[ibehavior][sourcemonkey][icell])

#for ibehavior in range (0, 6):
    #for sourcemonkey in range (0, nmonkeys):
        #for icell in range (0, ncells):
            #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.50000):
            #if (nless[ibehavior][sourcemonkey][icell] > 989):
                #print('BEH ', ibehavior, 'cell ', icell, 'source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell], 'nless = ', nless[ibehavior][sourcemonkey][icell])
                
for icell in range (0, ncells):
    for ibehavior in range (0, 6):
        for sourcemonkey in range (0, nmonkeys):
            if (Rsquared[ibehavior][sourcemonkey][icell] > 0.50000):
            #if (nless[ibehavior][sourcemonkey][icell] > 989):
                print('BEH ', ibehavior, 'cell ', icell, 'source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell], 'nless = ', nless[ibehavior][sourcemonkey][icell])

#same_matrix = []
#for imonkey in range (0, nmonkeys):
    #for ibeh in range (0, 6):
        #same_matrix.append(0)
        
#match_matrix = []
#for imonkey in range (0, nmonkeys):
    #for ibeh in range (0, 6):
        #match_matrix.append(same_matrix)

    
match_matrix = []    #4-dimensional list of R2 > 0.5 found in same cells; match_matrix[monkey][behavior][monkey][behavior]
for imnk in range (0, nmonkeys):
    imnk_list = []    #outer or 1st dimension of list array is behaviors
    for ibeh in range (0, 6):
        ibeh_list = []    #2nd dimension is source monkeys
        for jmnk in range (0, nmonkeys):
            jmnk_list = []
            for jbeh in range (0, 6):
                jmnk_list.append(0)
            ibeh_list.append(jmnk_list)
        imnk_list.append(ibeh_list)
    match_matrix.append(imnk_list)
            

        


for icell in range (0, ncells):
    nsame = 0
    beh_lst = []
    mnk_lst = []
    for ibehavior in range (0, 6):
        for sourcemonkey in range (0, nmonkeys):
            if (Rsquared[ibehavior][sourcemonkey][icell] > 0.50000):
                beh_lst.append(ibehavior)
                mnk_lst.append(sourcemonkey)
                nsame += 1
    for isame in range (0, nsame):
        for imatch in range (isame + 1, nsame):
            match_matrix[mnk_lst[isame]][beh_lst[isame]][mnk_lst[imatch]][beh_lst[imatch]] += 1
            
for imnk in range (0, nmonkeys):
    for ibeh in range (0, 6):
        for jmnk in range (0, nmonkeys):
            for jbeh in range (0, 6):
                if (match_matrix[imnk][ibeh][jmnk][jbeh] > 0):
                    print(monkeyname[imnk], ibeh, monkeyname [jmnk], jbeh, match_matrix[imnk][ibeh][jmnk][jbeh])

        
        
        
        
        
        

