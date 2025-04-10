


# linear regression for behavior patterns and single monkey ID based on mean responses,
# no bootstrapping from individual responses

import pandas as pd
import random
import pprint
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import string
import re
import math

trialresponses = pd.read_excel('/Users/charlesconnor/Dropbox/grants/social.memory/selected_cells_time_windowed/BestFrans/combined_bestfrans_spike_count_time_windowed.xlsx')

responsestringarray = trialresponses.values

nmonkeys = 5 #must match behavior matrix size; subject and source monkeys removed as appropriate in code below
nmonkeysnotsubject = 5

affiliationtomatrix =[[0, 17, 8, 17, 15],[18, 0, 7, 17, 27],[10, 5, 0, 20, 13],[13, 6, 22, 0, 13],[10, 28, 5, 11, 0]]

affiliationfrommatrix = []    #frequency[sinkmonkey][sourcemonkey]
for sinkmonkey in range (0, nmonkeys):
    sourcelist = []
    for sourcemonkey in range (0, nmonkeys):
        sourcelist.append(affiliationtomatrix[sourcemonkey][sinkmonkey])
    affiliationfrommatrix.append(sourcelist)

submissiontomatrix = [[0, 1, 2, 0, 0],[1, 0, 1, 0, 4],[5, 8, 0, 33, 10],[19, 25, 3, 0, 11],[17, 24, 6, 13, 0]]

submissionfrommatrix = []    #frequency[sinkmonkey][sourcemonkey]
for sinkmonkey in range (0, nmonkeys):
    sourcelist = []
    for sourcemonkey in range (0, nmonkeys):
        sourcelist.append(submissiontomatrix[sourcemonkey][sinkmonkey])
    submissionfrommatrix.append(sourcelist)

agonismtomatrix = [[0, 1, 2, 3, 16],[0, 0, 2, 10, 23],[1, 0, 0, 5, 13],[0, 0, 12, 0, 17],[0, 12, 1, 3, 0]]

agonismfrommatrix = []    #frequency[sinkmonkey][sourcemonkey]
for sinkmonkey in range (0, nmonkeys):
    sourcelist = []
    for sourcemonkey in range (0, nmonkeys):
        sourcelist.append(agonismtomatrix[sourcemonkey][sinkmonkey])
    agonismfrommatrix.append(sourcelist)

cells = [73, 60, 55, 53, 34, 29, 16, 15, 10, 105, 110, 115, 119, 122, 127, 129]
cellnames = ['11/8 3 1 0 0–250', '10/31 1 22 0–750', '10/27 4 20 0–750', '10/27 4 11 0–750', '10/5 1 4 0–2000', '10/4 4 21 2 0–700', '10/4 2 9 1 100–500', '10/4 1 19 3 100–500', '10/3 4 10 2 200–600', '9/28 1 25 2', '10/4 4 7', '10/31 2 5 250–750', '11/22 3 28', '11/25 2 1', '11/27 3 2', '11/27 3 17']
ncells = 16
monkeyname = [ 'G701', '14F', '68F', '101G', '19J' ]

nboots = 1 # > 1 for bootstrap random sampling of one response from each cell; commented out below
subjectmonkey = 6    #81G

maxabscorrelation = []    #only reporting results based on max absolute correlation for each cell
for icell in cells:
    maxabscorrelation.append(0.0)
    
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
    
#siglist[icell][ibehavior][sourcemonkey] = rsquared
    
siglist = []

for icell in range (0, ncells):
    celladd = []
    for ibehavior in range (0, nbehaviors):
        behadd = []
        for sourcemonkey in range (0, nmonkeys):
            behadd.append(0.0)
        celladd.append(behadd)
    siglist.append(celladd)
    
behaviornames = ['affil to', 'affil from', 'sub to', 'sub from', 'agon to', 'agon from']
    
print('AFFILIATION TO ANALYSIS')
ibehavior = 0

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for icell in range (0, ncells):
    for sourcemonkey in range (0, nmonkeys):
        y = []
        x = []
        for sinkmonkey in range (0, nmonkeys):    #responses don't include subject monkey
            if (sinkmonkey != sourcemonkey):
                y.append(float(affiliationtomatrix[sourcemonkey][sinkmonkey]))
                response = []
                responsestring = responsestringarray[cells[icell],sinkmonkey + 1]
                responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
                meanresponse = sum(responselist) / len(responselist)
                #if ((sourcemonkey == 6) and (icell == 13)):
                    #print('responsestring for sinkmonkey ', sinkmonkey, ' = ', responsestring)
                    #print('responselist = ', responselist)
                    #print('meanresponse = ', meanresponse)
                x.append(meanresponse)
                
        if ((icell == 14) and (sourcemonkey == 1)):
            print('responses for cell 14 and sourcemonkey 14F: ', x)
            print('affiliation from frequencies for 14F: ', y)
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
        for ixy in range (0, n):
            SS_xy += y[ixy]*x[ixy]
            SS_xx += x[ixy]*x[ixy]
            m_y += y[ixy]
            m_x += x[ixy]
        #print('m_y = ', m_y)
        #print('m_x = ', m_x)
        #print('SS_xx = ', SS_xx)
        SS_xy -= n*m_y*m_x
        SS_xx -= n*m_x*m_x
        m_y /= float(n)
        m_x /= float(n)
        #print('SS_xx = ', SS_xx)
                    
                    
        # calculating regression coefficients
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1*m_x
                
        # predicted response vector
        ypred = []
        for ixy in range (0, len(y)):
            ypred.append(b_0 + b_1*x[ixy])

        r2_score(y, ypred)
        Rsquared[ibehavior][sourcemonkey][icell] = explained_variance_score(y, ypred)
        if (Rsquared[ibehavior][sourcemonkey][icell] > 0.25):
        #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.395):
            sumRsquared[sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])
            print('for cell ', icell, ' = ', cells[icell], ' in response array for source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell])
            siglist[icell][ibehavior][sourcemonkey] = Rsquared[ibehavior][sourcemonkey][icell]

for sourcemonkey in range (0, nmonkeys):
    print('sumRsquared for ', monkeyname[sourcemonkey],' = ', sumRsquared[sourcemonkey])
    
                    
# plotting the actual points as scatter plot
  #plt.scatter(x, y, color = "m",
        #marker = "o", s = 30)

  # predicted response vector
  #y_pred = b[0] + b[1]*x

  # plotting the regression line
  #plt.plot(x, y_pred, color = "g")

  # putting labels
  #plt.xlabel('x')
  #plt.ylabel('y')
            
            
            
   
                        
print('AFFILIATION FROM ANALYSIS')
ibehavior = 1

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for icell in range (0, ncells):
    for sourcemonkey in range (0, nmonkeys):
        y = []
        x = []
        for sinkmonkey in range (0, nmonkeys):    #responses don't include subject monkey
            if (sinkmonkey != sourcemonkey):
                y.append(float(affiliationfrommatrix[sourcemonkey][sinkmonkey]))
                response = []
                responsestring = responsestringarray[cells[icell],sinkmonkey + 1]
                responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
                meanresponse = sum(responselist) / len(responselist)
                #if ((sourcemonkey == 6) and (icell == 13)):
                    #print('responsestring for sinkmonkey ', sinkmonkey, ' = ', responsestring)
                    #print('responselist = ', responselist)
                    #print('meanresponse = ', meanresponse)
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
        for ixy in range (0, n):
            SS_xy += y[ixy]*x[ixy]
            SS_xx += x[ixy]*x[ixy]
            m_y += y[ixy]
            m_x += x[ixy]
        #print('m_y = ', m_y)
        #print('m_x = ', m_x)
        #print('SS_xx = ', SS_xx)
        SS_xy -= n*m_y*m_x
        SS_xx -= n*m_x*m_x
        m_y /= float(n)
        m_x /= float(n)
        #print('SS_xx = ', SS_xx)
                     
                    
        # calculating regression coefficients
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1*m_x
                
        # predicted response vector
        ypred = []
        for ixy in range (0, len(y)):
            ypred.append(b_0 + b_1*x[ixy])

        r2_score(y, ypred)
        Rsquared[ibehavior][sourcemonkey][icell] = explained_variance_score(y, ypred)
        if (Rsquared[ibehavior][sourcemonkey][icell] > 0.25):
        #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.395):
            sumRsquared[sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])
            print('for cell ', icell, ' = ', cells[icell], ' in response array for source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell])
            siglist[icell][ibehavior][sourcemonkey] = Rsquared[ibehavior][sourcemonkey][icell]


            #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.25):
            #plotting the actual points as scatter plot
            #plt.scatter(x, y, color = "m",
            #marker = "o", s = 30)

            #predicted response vector
            #y_pred = b_0 + b_1*x

            #plotting the regression line
            #plt.plot(x, ypred, color = "g")

            # putting labels
            #plt.xlabel('x')
            #plt.ylabel('y')
            #plt.show()
 
for sourcemonkey in range (0, nmonkeys):
    print('sumRsquared for ', monkeyname[sourcemonkey],' = ', sumRsquared[sourcemonkey])
    
                    
   
 
print('SUBMISSION TO ANALYSIS')
ibehavior = 2

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for icell in range (0, ncells):
    for sourcemonkey in range (0, nmonkeys):
        y = []
        x = []
        for sinkmonkey in range (0, nmonkeys):    #responses don't include subject monkey
            if (sinkmonkey != sourcemonkey):
                y.append(float(submissiontomatrix[sourcemonkey][sinkmonkey]))
                response = []
                responsestring = responsestringarray[cells[icell],sinkmonkey + 1]
                responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
                meanresponse = sum(responselist) / len(responselist)
                #if ((sourcemonkey == 6) and (icell == 13)):
                    #print('responsestring for sinkmonkey ', sinkmonkey, ' = ', responsestring)
                    #print('responselist = ', responselist)
                    #print('meanresponse = ', meanresponse)
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
        for ixy in range (0, n):
            SS_xy += y[ixy]*x[ixy]
            SS_xx += x[ixy]*x[ixy]
            m_y += y[ixy]
            m_x += x[ixy]
        #print('m_y = ', m_y)
        #print('m_x = ', m_x)
        #print('SS_xx = ', SS_xx)
        SS_xy -= n*m_y*m_x
        SS_xx -= n*m_x*m_x
        m_y /= float(n)
        m_x /= float(n)
        #print('SS_xx = ', SS_xx)
                     
                    
        # calculating regression coefficients
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1*m_x
                
        # predicted response vector
        ypred = []
        for ixy in range (0, len(y)):
            ypred.append(b_0 + b_1*x[ixy])

        r2_score(y, ypred)
        Rsquared[ibehavior][sourcemonkey][icell] = explained_variance_score(y, ypred)
        if (Rsquared[ibehavior][sourcemonkey][icell] > 0.25):
        #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.395):
            sumRsquared[sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])
            print('for cell ', icell, ' = ', cells[icell], ' in response array for source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell])
            siglist[icell][ibehavior][sourcemonkey] = Rsquared[ibehavior][sourcemonkey][icell]

for sourcemonkey in range (0, nmonkeys):
    print('sumRsquared for ', monkeyname[sourcemonkey],' = ', sumRsquared[sourcemonkey])
    
                    
                         
print('SUBMISSION FROM ANALYSIS')
ibehavior = 3

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for icell in range (0, ncells):
    for sourcemonkey in range (0, nmonkeys):
        y = []
        x = []
        for sinkmonkey in range (0, nmonkeys):    #responses don't include subject monkey
            if (sinkmonkey != sourcemonkey):
                y.append(float(submissionfrommatrix[sourcemonkey][sinkmonkey]))
                response = []
                responsestring = responsestringarray[cells[icell],sinkmonkey + 1]
                responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
                meanresponse = sum(responselist) / len(responselist)
                #if ((sourcemonkey == 6) and (icell == 13)):
                    #print('responsestring for sinkmonkey ', sinkmonkey, ' = ', responsestring)
                    #print('responselist = ', responselist)
                    #print('meanresponse = ', meanresponse)
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
        for ixy in range (0, n):
            SS_xy += y[ixy]*x[ixy]
            SS_xx += x[ixy]*x[ixy]
            m_y += y[ixy]
            m_x += x[ixy]
        #print('m_y = ', m_y)
        #print('m_x = ', m_x)
        #print('SS_xx = ', SS_xx)
        SS_xy -= n*m_y*m_x
        SS_xx -= n*m_x*m_x
        m_y /= float(n)
        m_x /= float(n)
        #print('SS_xx = ', SS_xx)
                     
                    
        # calculating regression coefficients
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1*m_x
                
        # predicted response vector
        ypred = []
        for ixy in range (0, len(y)):
            ypred.append(b_0 + b_1*x[ixy])

        r2_score(y, ypred)
        Rsquared[ibehavior][sourcemonkey][icell] = explained_variance_score(y, ypred)
        if (Rsquared[ibehavior][sourcemonkey][icell] > 0.25):
        #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.395):
            sumRsquared[sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])
            print('for cell ', icell, ' = ', cells[icell], ' in response array for source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell])
            siglist[icell][ibehavior][sourcemonkey] = Rsquared[ibehavior][sourcemonkey][icell]

        #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.53):
            #plotting the actual points as scatter plot
            #plt.scatter(x, y, color = "m",
            #marker = "o", s = 30)

            #predicted response vector
            #y_pred = b_0 + b_1*x

            #plotting the regression line
            #plt.plot(x, ypred, color = "g")

            # putting labels
            #plt.xlabel('x')
            #plt.ylabel('y')
            #plt.show()

for sourcemonkey in range (0, nmonkeys):
    print('sumRsquared for ', monkeyname[sourcemonkey],' = ', sumRsquared[sourcemonkey])
    
                    




print('AGONISM TO ANALYSIS')
ibehavior = 4

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]



for icell in range (0, ncells):
    for sourcemonkey in range (0, nmonkeys):
        y = []
        x = []
        for sinkmonkey in range (0, nmonkeys):    #responses don't include subject monkey
            if (sinkmonkey != sourcemonkey):
                y.append(float(agonismtomatrix[sourcemonkey][sinkmonkey]))
                response = []
                responsestring = responsestringarray[cells[icell],sinkmonkey + 1]
                responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
                meanresponse = sum(responselist) / len(responselist)
                #if ((sourcemonkey == 6) and (icell == 13)):
                    #print('responsestring for sinkmonkey ', sinkmonkey, ' = ', responsestring)
                    #print('responselist = ', responselist)
                    #print('meanresponse = ', meanresponse)
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
        for ixy in range (0, n):
            SS_xy += y[ixy]*x[ixy]
            SS_xx += x[ixy]*x[ixy]
            m_y += y[ixy]
            m_x += x[ixy]
        #print('m_y = ', m_y)
        #print('m_x = ', m_x)
        #print('SS_xx = ', SS_xx)
        SS_xy -= n*m_y*m_x
        SS_xx -= n*m_x*m_x
        m_y /= float(n)
        m_x /= float(n)
        #print('SS_xx = ', SS_xx)
                     
                    
        # calculating regression coefficients
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1*m_x
                
        # predicted response vector
        ypred = []
        for ixy in range (0, len(y)):
            ypred.append(b_0 + b_1*x[ixy])

        r2_score(y, ypred)
        Rsquared[ibehavior][sourcemonkey][icell] = explained_variance_score(y, ypred)
        if (Rsquared[ibehavior][sourcemonkey][icell] > 0.25):
        #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.395):
            sumRsquared[sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])
            print('for cell ', icell, ' = ', cells[icell], ' in response array for source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell])
            siglist[icell][ibehavior][sourcemonkey] = Rsquared[ibehavior][sourcemonkey][icell]

for sourcemonkey in range (0, nmonkeys):
    print('sumRsquared for ', monkeyname[sourcemonkey],' = ', sumRsquared[sourcemonkey])
    
                    
                         
print('AGONISM FROM ANALYSIS')
ibehavior = 5

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


for icell in range (0, ncells):
    for sourcemonkey in range (0, nmonkeys):
        y = []
        x = []
        for sinkmonkey in range (0, nmonkeys):    #responses don't include subject monkey
            if (sinkmonkey != sourcemonkey):
                y.append(float(agonismfrommatrix[sourcemonkey][sinkmonkey]))
                response = []
                responsestring = responsestringarray[cells[icell],sinkmonkey + 1]
                responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
                meanresponse = sum(responselist) / len(responselist)
                #if ((sourcemonkey == 6) and (icell == 13)):
                    #print('responsestring for sinkmonkey ', sinkmonkey, ' = ', responsestring)
                    #print('responselist = ', responselist)
                    #print('meanresponse = ', meanresponse)
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
        for ixy in range (0, n):
            SS_xy += y[ixy]*x[ixy]
            SS_xx += x[ixy]*x[ixy]
            m_y += y[ixy]
            m_x += x[ixy]
        #print('m_y = ', m_y)
        #print('m_x = ', m_x)
        #print('SS_xx = ', SS_xx)
        SS_xy -= n*m_y*m_x
        SS_xx -= n*m_x*m_x
        m_y /= float(n)
        m_x /= float(n)
        #print('SS_xx = ', SS_xx)
                     
                    
        # calculating regression coefficients
        b_1 = SS_xy / SS_xx
        b_0 = m_y - b_1*m_x
        
        # predicted response vector
        ypred = []
        for ixy in range (0, len(y)):
            ypred.append(b_0 + b_1*x[ixy])

        r2_score(y, ypred)
        Rsquared[ibehavior][sourcemonkey][icell] = explained_variance_score(y, ypred)
        if (Rsquared[ibehavior][sourcemonkey][icell] > 0.25):
        #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.395):
            sumRsquared[sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])
            print('for cell ', icell, ' = ', cells[icell], ' in response array for source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell])
            siglist[icell][ibehavior][sourcemonkey] = Rsquared[ibehavior][sourcemonkey][icell]
        #if (Rsquared[ibehavior][sourcemonkey][icell] > 0.48):
            #plotting the actual points as scatter plot
            #plt.scatter(x, y, color = "m",
            #marker = "o", s = 30)

            #predicted response vector
            #y_pred = b_0 + b_1*x

            #plotting the regression line
            #plt.plot(x, ypred, color = "g")

            # putting labels
            #plt.xlabel('x')
            #plt.ylabel('y')
            #plt.show()
 

for sourcemonkey in range (0, nmonkeys):
    print('sumRsquared for ', monkeyname[sourcemonkey],' = ', sumRsquared[sourcemonkey])
    



#correct by counting only max behavioral correlation for any give cell
print('SUMMED FOR EACH MONKEY')

nbehaviors = 6
behaviorname = ['AFFILIATION TO', 'AFFILIATION FROM', 'SUBMISSION TO', 'SUBMISSION FROM', 'AGONISM TO', 'AGONISM FROM']

Rsquared_total = []    #2-dimensional list of total Rsquared values above 0.25; Rsquared[behavior][monkey]
for i in range (0, nbehaviors):
    behavior_list = []    #outer or 1st dimension of list array is behaviors
    for j in range (0, nmonkeys):
        behavior_list.append(0.0)    #2nd dimension is monkeys
    Rsquared_total.append(behavior_list)

#print(Rsquared_total)

for ibehavior in range (0, nbehaviors):
    for icell in range (0, ncells):
        for imonkey in range (0, nmonkeys):
            if (Rsquared[ibehavior][imonkey][icell] > 0.25):
                Rsquared_total[ibehavior][imonkey] += Rsquared[ibehavior][imonkey][icell]
             
for imonkey in range (0, nmonkeys):
    print('MONKEY ', imonkey, monkeyname[imonkey])
    total_Rsquared = 0.0
    for ibehavior in range (0, nbehaviors):
        print('behavior ', ibehavior, behaviorname[ibehavior], 'Rsquared_total = ', Rsquared_total[ibehavior][imonkey])
        total_Rsquared += Rsquared_total[ibehavior][imonkey]
    print('TOTAL RSQUARED = ', total_Rsquared)
    
print('BY CELL BY BEHAVIOR')

for icell in range (0, ncells):
    for ibehavior in range (0, nbehaviors):
        for sourcemonkey in range (0, nmonkeys):
            if (siglist[icell][ibehavior][sourcemonkey] > 0):
                print('icell = ', icell, cellnames[icell], behaviornames[ibehavior], monkeyname[sourcemonkey], siglist[icell][ibehavior][sourcemonkey])

print('BY CELL BY MONKEY')

for icell in range (0, ncells):
    for sourcemonkey in range (0, nmonkeys):
        for ibehavior in range (0, nbehaviors):
            if (siglist[icell][ibehavior][sourcemonkey] > 0):
                print('icell = ', icell, cellnames[icell], behaviornames[ibehavior], monkeyname[sourcemonkey], siglist[icell][ibehavior][sourcemonkey])

print('BY MONKEY')

for sourcemonkey in range (0, nmonkeys):
    for icell in range (0, ncells):
        for ibehavior in range (0, nbehaviors):
            if (siglist[icell][ibehavior][sourcemonkey] > 0):
                print('icell = ', icell, cellnames[icell], behaviornames[ibehavior], monkeyname[sourcemonkey], siglist[icell][ibehavior][sourcemonkey])

print('COMBINED TUNING WITHIN CELLS')
print('max products of Rsquared values summed across cells')

comblist = []     # sum of combination maxRsquared values in same cells summed across cells [sourcemonkey][sinkmonkey]

for sourcemonkey in range (0, nmonkeys):
    sourceadd = []
    for sinkmonkey in range (0, nmonkeys):
        sourceadd.append(0.0)
    comblist.append(sourceadd)
    
cellcomblist = []     # sum of combination maxRsquared values in current cell [sourcemonkey][sinkmonkey]

for sourcemonkey in range (0, nmonkeys):
    sourceadd = []
    for pairmonkey in range (0, nmonkeys):
        sourceadd.append(0.0)
    cellcomblist.append(sourceadd) 

maxmonkey = []      # max Rsquared for each monkey for current cell

for i in range (0, nmonkeys):
    maxmonkey.append(0.0)

for icell in range (0, ncells):
    for sourcemonkey in range (0, nmonkeys):
        for pairmonkey in range (0, nmonkeys):
            cellcomblist[sourcemonkey][pairmonkey] = 0.0
    for sourcemonkey in range (0, nmonkeys):
        maxmonkey[sourcemonkey] = 0.0
        for ibehavior in range (0, nbehaviors):
            if (siglist[icell][ibehavior][sourcemonkey] > maxmonkey[sourcemonkey]):
                maxmonkey[sourcemonkey] = siglist[icell][ibehavior][sourcemonkey]
    for sourcemonkey in range (0, nmonkeys):
        for pairmonkey in range (0, nmonkeys):
            cellcomblist[sourcemonkey][pairmonkey] = maxmonkey[sourcemonkey] * maxmonkey[pairmonkey]
            comblist[sourcemonkey][pairmonkey] += cellcomblist[sourcemonkey][pairmonkey]
            
for sourcemonkey in range (0, nmonkeys):
    for pairmonkey in range (0, nmonkeys):
        print('sourcemonkey ', monkeyname[sourcemonkey], 'pairmonkey ', monkeyname[pairmonkey], 'sum max Rsquared products', comblist[sourcemonkey][pairmonkey])
            

















 

