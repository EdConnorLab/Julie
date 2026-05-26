
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

Rsquared = []    #2-dimensional list of Rsquared values above 0.25; Rsquared[behavior][cell]
for i in range (0, nbehaviors):
    behavior_list = []    #outer or 1st dimension of list array is behaviors
    for k in range (0, ncells):
        behavior_list.append(0.0)    #2nd dimension is cells
    Rsquared.append(behavior_list)
    
#print(Rsquared)
    
nless = []    # randomized Rsquared values less than observed out of 1000 randomizations
for i in range (0, nbehaviors):
    rand_behavior_list = []    #outer or 1st dimension of list array is behaviors
    for k in range (0, ncells):
        rand_behavior_list.append(0.0)    #2nd dimension is cells
    nless.append(rand_behavior_list)
    
    
#print(Rsquared)
    

nless = [0, 0, 0, 0, 0, 0]
nlesstotal = 0
nlesssig = 0
nsig = 0


# observed values

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

affsumRsquared = 0.0
subsumRsquared = 0.0
agnsumRsquared = 0.0

for ibehavior in range (0, 6):


    for icell in range (0, ncells):
        y = []
        x = []
        lostmonkeys = 0
        for sourcemonkey in range (0, nmonkeys):
            if (sourcemonkey == subjectmonkey):
                lostmonkeys += 1    #skip advancement through responselist per the index subtract below for absence of subject monkey
            else:
                yadd = 0.0
                for sinkmonkey in range (0, nmonkeys):
                    if (ibehavior == 0):
                        yadd += float(affiliationtomatrix[sourcemonkey][sinkmonkey])
                    if (ibehavior == 1):
                        yadd += float(affiliationfrommatrix[sourcemonkey][sinkmonkey])
                    if (ibehavior == 2):
                        yadd += float(submissiontomatrix[sourcemonkey][sinkmonkey])
                    if (ibehavior == 3):
                        yadd += float(submissionfrommatrix[sourcemonkey][sinkmonkey])
                    if (ibehavior == 4):
                        yadd += float(agonismtomatrix[sourcemonkey][sinkmonkey])
                    if (ibehavior == 5):
                        yadd += float(agonismfrommatrix[sourcemonkey][sinkmonkey])
                y.append(yadd)
                response = []
                responsestring = responsestringarray[icell,monkey_column[sourcemonkey - lostmonkeys]]
                responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
                meanresponse = sum(responselist) / len(responselist)
                x.append(meanresponse)
                   
            #print('y = ', y)
            #print('x = ', x)
                
        n = len(x)

        # mean of x and y vector
        m_x = sum(x) / len(x)
        #m_y = sum(y) / len(y)

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
            SS_yy += y[ixy]**2
                            
        # calculating regression coefficients
        b_0 = (S_y*SS_xx - S_x*SS_xy) / (len(x)*SS_xx - S_x**2)
        #print('numerator = ', (S_y*SS_xx - S_x*SS_xy))
        #print('denominator = ', (len(x)*SS_xx - S_x**2))
        b_1 = (len(x)*SS_xy - S_x*S_y) / (len(x)*SS_xx - S_x**2)
    
        # predicted response vector
        ypred = []
        for ixy in range (0, len(y)):
            ypred.append(b_0 + b_1*x[ixy])

        r2_score(y, ypred)
        #Rsquared[ibehavior][sourcemonkey][icell] = explained_variance_score(y, ypred)
        Rsquared[ibehavior][icell] = r2_score(y, ypred)
        if (Rsquared[ibehavior][icell] > 0.5):
            nsig += 1
            sumRsquared[ibehavior] += abs(Rsquared[ibehavior][icell])
            print('for cell ', icell, 'for behavior', ibehavior, 'Rsquared = ', Rsquared[ibehavior][icell])
            if (ibehavior == 0):
                affsumRsquared += Rsquared[ibehavior][icell]
            if (ibehavior == 1):
                affsumRsquared += Rsquared[ibehavior][icell]
            if (ibehavior == 2):
                subsumRsquared += Rsquared[ibehavior][icell]
            if (ibehavior == 3):
                subsumRsquared += Rsquared[ibehavior][icell]
            if (ibehavior == 4):
                agnsumRsquared += Rsquared[ibehavior][icell]
            if (ibehavior == 5):
                agnsumRsquared += Rsquared[ibehavior][icell]
 
sumRsquaredtotal = 0.0
for ibehavior in range (0, 6):
    #print('sumRsquared for behavior', ibehavior,' = ', sumRsquared[ibehavior])
    sumRsquaredtotal += sumRsquared[ibehavior]
print ('sumRsquaredtotal = ', sumRsquaredtotal)
nlessaff = 0
nlesssub = 0
nlessagn = 0
    
                    
 # randomization                        
for ir in range (0, 10000):

    rndRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
    nrndsig = 0
    rndaffsumRsquared = 0.0
    rndsubsumRsquared = 0.0
    rndagnsumRsquared = 0.0
    
    for ibehavior in range (0, 6):

        for icell in range (0, ncells):
            y = []
            x = []
            lostmonkeys = 0

            for sourcemonkey in range (0, nmonkeys):        
                if (sourcemonkey == subjectmonkey):
                    lostmonkeys += 1    #skip advancement through responselist per the index subtract below for absence of subject monkey
                else:
                    yadd = 0.0
                    for sinkmonkey in range (0, nmonkeys):
                        if (ibehavior == 0):
                            yadd += float(affiliationtomatrix[sourcemonkey][sinkmonkey])
                        if (ibehavior == 1):
                            yadd += float(affiliationfrommatrix[sourcemonkey][sinkmonkey])
                        if (ibehavior == 2):
                            yadd += float(submissiontomatrix[sourcemonkey][sinkmonkey])
                        if (ibehavior == 3):
                            yadd += float(submissionfrommatrix[sourcemonkey][sinkmonkey])
                        if (ibehavior == 4):
                            yadd += float(agonismtomatrix[sourcemonkey][sinkmonkey])
                        if (ibehavior == 5):
                            yadd += float(agonismfrommatrix[sourcemonkey][sinkmonkey])
                    y.append(yadd)
                    response = []
                    responsestring = responsestringarray[icell,monkey_column[sourcemonkey - lostmonkeys]]
                    responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
                    meanresponse = sum(responselist) / len(responselist)
                    x.append(meanresponse)
                
            random.shuffle(x)
                
            n = len(x)

            # mean of x and y vector
            m_x = sum(x) / len(x)
            #m_y = sum(y) / len(y)

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
                SS_yy += y[ixy]**2
                                
            # calculating regression coefficients
            b_0 = (S_y*SS_xx - S_x*SS_xy) / (len(x)*SS_xx - S_x**2)
            b_1 = (len(x)*SS_xy - S_x*S_y) / (len(x)*SS_xx - S_x**2)                        
            # predicted response vector
            ypred = []
            for ixy in range (0, len(y)):
                ypred.append(b_0 + b_1*x[ixy])

            r2_score(y, ypred)
            #Rsquared[ibehavior][sourcemonkey][icell] = explained_variance_score(y, ypred)
            Rsquared[ibehavior][icell] = r2_score(y, ypred)            
            if (Rsquared[ibehavior][icell] > 0.5):
                nrndsig += 1
                rndRsquared[ibehavior] += abs(Rsquared[ibehavior][icell])
                # print('for cell ', icell, 'for source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell])
            if (ibehavior == 0):
                rndaffsumRsquared += Rsquared[ibehavior][icell]
            if (ibehavior == 1):
                rndaffsumRsquared += Rsquared[ibehavior][icell]
            if (ibehavior == 2):
                rndsubsumRsquared += Rsquared[ibehavior][icell]
            if (ibehavior == 3):
                rndsubsumRsquared += Rsquared[ibehavior][icell]
            if (ibehavior == 4):
                rndagnsumRsquared += Rsquared[ibehavior][icell]
            if (ibehavior == 5):
                rndagnsumRsquared += Rsquared[ibehavior][icell]

    rndRsquaredtotal = 0.0
    for ibehavior in range (0, 6):
        rndRsquaredtotal += rndRsquared[ibehavior]
        if (rndRsquared[ibehavior] < sumRsquared[ibehavior]):
            nless[ibehavior] += 1
    #print ('rndRsquaredtotal = ', rndRsquaredtotal)
        
    if (rndRsquaredtotal < sumRsquaredtotal):
        nlesstotal += 1
    if (nrndsig < nsig):
        nlesssig += 1
        
    if (rndaffsumRsquared < affsumRsquared):
        nlessaff += 1
    if (rndsubsumRsquared < subsumRsquared):
        nlesssub += 1
    if (rndagnsumRsquared < agnsumRsquared):
        nlessagn += 1

for ibehavior in range (0, 6):
    print('nless for behavior', ibehavior,' = ', nless[ibehavior])
print('nless total = ', nlesstotal)
print('nlessaff = ', nlessaff)
print('nlesssub = ', nlesssub)
print('nlessagn = ', nlessagn)
    
  
 


