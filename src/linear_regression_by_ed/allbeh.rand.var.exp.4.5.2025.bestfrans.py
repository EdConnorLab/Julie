
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

trialresponses = pd.read_excel('/Users/charlesconnor/Dropbox/grants/social.memory/20250401_ANOVA_and_linear_regression_results/bestfrans/bestfrans_spike_counts_for_all_anova_passed_time_windowed_cells.xlsx')

responsestringarray = trialresponses.values

nmonkeys = 5 #must match behavior matrix size; subject and source monkeys removed as appropriate in code below
nmonkeysnotsubject = 5


affiliationtomatrix =[[0,17,10,18,21],[18,0,7,17,30],[12,5,0,20,17],[13,8,22,0,17],[10,30,5,11,0]]

affiliationfrommatrix = []    #frequency[sinkmonkey][sourcemonkey]
for sinkmonkey in range (0, nmonkeys):
    sourcelist = []
    for sourcemonkey in range (0, nmonkeys):
        sourcelist.append(affiliationtomatrix[sourcemonkey][sinkmonkey])
    affiliationfrommatrix.append(sourcelist)

submissiontomatrix = [[0,1,2,0,0],[1,0,1,0,5],[5,8,0,38,10],[22,26,3,0,14],[24,26,12,18,0]]

submissionfrommatrix = []    #frequency[sinkmonkey][sourcemonkey]
for sinkmonkey in range (0, nmonkeys):
    sourcelist = []
    for sourcemonkey in range (0, nmonkeys):
        sourcelist.append(submissiontomatrix[sourcemonkey][sinkmonkey])
    submissionfrommatrix.append(sourcelist)

agonismtomatrix = [[0,1,3,4,22],[0,0,2,10,25],[1,0,0,5,18],[0,1,12,0,21],[0,12,1,3,0]]

agonismfrommatrix = []    #frequency[sinkmonkey][sourcemonkey]
for sinkmonkey in range (0, nmonkeys):
    sourcelist = []
    for sourcemonkey in range (0, nmonkeys):
        sourcelist.append(agonismtomatrix[sourcemonkey][sinkmonkey])
    agonismfrommatrix.append(sourcelist)

ncells = 37
monkeyname = [ 'G701', '14F', '68F', '101G', '19J' ]
monkey_column = [40, 13, 29, 8, 16]

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
    
    
#print(Rsquared)
    

nless = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
nlesstotal = 0
nlesssig = 0
nsig = 0


# observed values

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

behsumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
behmnksumRsquared = [[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0]]

for ibehavior in range (0, 6):


    for icell in range (0, ncells):
        for sourcemonkey in range (0, nmonkeys):
            y = []
            x = []
            lostmonkeys = 0
            for sinkmonkey in range (0, nmonkeys):    #responses don't include subject monkey
                if ((sinkmonkey == sourcemonkey) or (sinkmonkey == subjectmonkey)):
                    if (sinkmonkey == subjectmonkey):
                        lostmonkeys += 1    #skip advancement through responselist per the index subtract below for absence of subject monkey
                else:
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
            #print('m_y = ', m_y)
            #print('m_x = ', m_x)
            #print('SS_xx = ', SS_xx)
            #SS_xy -= n*m_y*m_x
            #SS_xx -= n*m_x*m_x
            #m_y /= float(n)
            #m_x /= float(n)
            #print('SS_xx = ', SS_xx)
                        
                        
            # calculating regression coefficients
            #b_1 = SS_xy / SS_xx
            #b_0 = m_y - b_1*m_x
            b_0 = (S_y*SS_xx - S_x*SS_xy) / (len(x)*SS_xx - S_x**2)
            b_1 = (len(x)*SS_xy - S_x*S_y) / (len(x)*SS_xx - S_x**2)
                    
            # predicted response vector
            ypred = []
            for ixy in range (0, len(y)):
                ypred.append(b_0 + b_1*x[ixy])

            r2_score(y, ypred)
            #Rsquared[ibehavior][sourcemonkey][icell] = explained_variance_score(y, ypred)
            Rsquared[ibehavior][sourcemonkey][icell] = r2_score(y, ypred)
            #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.7):
            if (Rsquared[ibehavior][sourcemonkey][icell] > 0.5):
            #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.25):
                nsig += 1
                sumRsquared[sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])
                behsumRsquared[ibehavior] += abs(Rsquared[ibehavior][sourcemonkey][icell])
                behmnksumRsquared[ibehavior][sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])
                # print('for cell ', icell, 'for source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell])
 
sumRsquaredtotal = 0.0
for sourcemonkey in range (0, nmonkeys):
    print('sumRsquared for ', monkeyname[sourcemonkey],' = ', sumRsquared[sourcemonkey])
    sumRsquaredtotal += sumRsquared[sourcemonkey]
nlessaff = 0
nlesssub = 0
nlessagn = 0
nlessbehavior = [0, 0, 0, 0, 0, 0]
nlessbehmnk = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]




nlesstotal = 0


 # randomization                        
for ir in range (0, 1000):

    rndsumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    behrndsumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rndbehmnksumRsquared = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    nrndsig = 0
    rndaffsumRsquared = 0.0
    rndsubsumRsquared = 0.0
    rndagnsumRsquared = 0.0
    
    for ibehavior in range (0, 6):

        for icell in range (0, ncells):
            for sourcemonkey in range (0, nmonkeys):
                y = []
                x = []
                lostmonkeys = 0
                for sinkmonkey in range (0, nmonkeys):    #responses don't include subject monkey
                    if ((sinkmonkey == sourcemonkey) or (sinkmonkey == subjectmonkey)):
                        if (sinkmonkey == subjectmonkey):
                            lostmonkeys += 1    #skip advancement through responselist per the index subtract below for absence of subject monkey
                    else:
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
                #print('m_y = ', m_y)
                #print('m_x = ', m_x)
                #print('SS_xx = ', SS_xx)
                #SS_xy -= n*m_y*m_x
                #SS_xx -= n*m_x*m_x
                #m_y /= float(n)
                #m_x /= float(n)
                #print('SS_xx = ', SS_xx)
                            
                            
                # calculating regression coefficients
                #b_1 = SS_xy / SS_xx
                #b_0 = m_y - b_1*m_x
                b_0 = (S_y*SS_xx - S_x*SS_xy) / (len(x)*SS_xx - S_x**2)
                b_1 = (len(x)*SS_xy - S_x*S_y) / (len(x)*SS_xx - S_x**2)
                        
                # predicted response vector
                ypred = []
                for ixy in range (0, len(y)):
                    ypred.append(b_0 + b_1*x[ixy])

                r2_score(y, ypred)
                #Rsquared[ibehavior][sourcemonkey][icell] = explained_variance_score(y, ypred)
                Rsquared[ibehavior][sourcemonkey][icell] = r2_score(y, ypred)            
                #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.7):
                if (Rsquared[ibehavior][sourcemonkey][icell] > 0.5):
                #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.25):
                    nrndsig += 1
                    rndsumRsquared[sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])
                    behrndsumRsquared[ibehavior] += abs(Rsquared[ibehavior][sourcemonkey][icell])
                    rndbehmnksumRsquared[ibehavior][sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])
                    # print('for cell ', icell, 'for source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell])

    rndRsquaredtotal = 0.0
    for sourcemonkey in range (0, nmonkeys):
        rndRsquaredtotal += rndsumRsquared[sourcemonkey]
        if (rndsumRsquared[sourcemonkey] < sumRsquared[sourcemonkey]):
            nless[sourcemonkey] += 1
    for ibehavior in range (0, 6):
        if (behrndsumRsquared[ibehavior] < behsumRsquared[ibehavior]):
            nlessbehavior[ibehavior] += 1
        for sourcemonkey in range (0, nmonkeys):
            if (rndbehmnksumRsquared[ibehavior][sourcemonkey] < behmnksumRsquared[ibehavior][sourcemonkey]):
                nlessbehmnk[ibehavior][sourcemonkey] += 1
        
        #print( 'monkey = ', monkeyname[sourcemonkey], 'rndRsquared = ', rndRsquared[sourcemonkey], 'sumRsquared = ', sumRsquared[sourcemonkey])
    if (rndRsquaredtotal < sumRsquaredtotal):
        nlesstotal += 1
    if (nrndsig < nsig):
        nlesssig += 1

        
for sourcemonkey in range (0, nmonkeys):
    print('nless for ', monkeyname[sourcemonkey],' = ', nless[sourcemonkey])
for ibehavior in range (0, 6):
    print('nless for behavior', ibehavior, ' = ', nlessbehavior[ibehavior])
    for sourcemonkey in range (0, nmonkeys):
        print('nless behavior ', ibehavior, ' monkey ', sourcemonkey, ' = ', nlessbehmnk[ibehavior][sourcemonkey])
print('nless total = ', nlesstotal)
print('nlesssig = ', nlesssig)
    
  
 


