
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

trialresponses = pd.read_excel('/Users/charlesconnor/Dropbox/grants/social.memory/selected_cells_time_windowed/spike_count_for_each_trial_windowed.xlsx')

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

cells = [0, 2, 7, 10, 13, 15, 17, 19, 21, 22, 24, 25, 30, 32, 34, 44, 46, 48, 50, 53, 55, 57, 59, 67]
ncells = 24
monkeyname = [ '7124', '69X', '72X', '94B', '110E', '67G', '81G', '143H', '87J', '151J' ]

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

Rsquared = []    #2-dimensional list of Rsquared values above 0.25; Rsquared[behavior][cell]
for i in range (0, nbehaviors):
    behavior_list = []    #outer or 1st dimension of list array is behaviors
    for k in range (0, ncells):
        behavior_list.append(0.0)    #2nd dimension is cells
    Rsquared.append(behavior_list)
    
#print(Rsquared)

print('AFFILIATION TO ANALYSIS')
ibehavior = 0

sumRsquared = 0.0

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
                yadd += affiliationtomatrix[sourcemonkey][sinkmonkey]
            y.append(yadd)
            response = []
            responsestring = responsestringarray[cells[icell],sourcemonkey + 4 - lostmonkeys]
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
    SS_xy = 0
    SS_xx = 0
    m_y = 0
    m_x = 0
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
    m_y /+ n
    m_x /= n
    #print('SS_xx = ', SS_xx)
                                        
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
                
    # predicted response vector
    ypred = []
    for ixy in range (0, len(y)):
        ypred.append(b_0 + b_1*x[ixy])

    r2_score(y, ypred)
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    #if (Rsquared[ibehavior][icell] > 0.25):
    if (Rsquared[ibehavior][icell] > 0.395):
        print('for cell ', icell, 'Rsquared = ', Rsquared[ibehavior][icell])
        #print('absolute Rsquared = ', abs(svr.score(X, y)))
        # could do significance test here, probably bootstrap based on subsampling of responses for each monkey stimulus,
        # since permutation across monkeys would confound ANOVA with regression by destroying all response differences between monkeys,
        # not just the linear relationship to behavior, making the null hypothesis uninterpretable
        sumRsquared += Rsquared[ibehavior][icell]

print('sumRsquared = ', sumRsquared)
                        
                    
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

sumRsquared = 0.0

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
                yadd += affiliationfrommatrix[sourcemonkey][sinkmonkey]
            y.append(yadd)
            response = []
            responsestring = responsestringarray[cells[icell],sourcemonkey + 4 - lostmonkeys]
            responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
            meanresponse = sum(responselist) / len(responselist)
            x.append(meanresponse)

        if ((icell == 16) and (sourcemonkey == 3)):
            print('responses for cell 16 and sourcemonkey 3: ', x)
    #print('y = ', y)
    #print('x = ', x)
            
    n = len(x)

    # mean of x and y vector
    m_x = sum(x) / len(x)
    #m_y = sum(y) / len(y)

    # calculating cross-deviation and deviation about x
    SS_xy = 0
    SS_xx = 0
    m_y = 0
    m_x = 0
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
    m_y /+ n
    m_x /= n
    #print('SS_xx = ', SS_xx)
                                        
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
                
    # predicted response vector
    ypred = []
    for ixy in range (0, len(y)):
        ypred.append(b_0 + b_1*x[ixy])

    r2_score(y, ypred)
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    #if (Rsquared[ibehavior][icell] > 0.25):
    if (Rsquared[ibehavior][icell] > 0.395):
        print('for cell ', icell, 'Rsquared = ', Rsquared[ibehavior][icell])
        #print('absolute Rsquared = ', abs(svr.score(X, y)))
        # could do significance test here, probably bootstrap based on subsampling of responses for each monkey stimulus,
        # since permutation across monkeys would confound ANOVA with regression by destroying all response differences between monkeys,
        # not just the linear relationship to behavior, making the null hypothesis uninterpretable
        sumRsquared += Rsquared[ibehavior][icell]

print('sumRsquared = ', sumRsquared)
                        
                    
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
            
            
                    
   
 
print('SUBMISSION TO ANALYSIS')
ibehavior = 2

sumRsquared = 0.0

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
                yadd += submissiontomatrix[sourcemonkey][sinkmonkey]
            y.append(yadd)
            response = []
            responsestring = responsestringarray[cells[icell],sourcemonkey + 4 - lostmonkeys]
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
    SS_xy = 0
    SS_xx = 0
    m_y = 0
    m_x = 0
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
    m_y /+ n
    m_x /= n
    #print('SS_xx = ', SS_xx)
                                        
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
                
    # predicted response vector
    ypred = []
    for ixy in range (0, len(y)):
        ypred.append(b_0 + b_1*x[ixy])

    r2_score(y, ypred)
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    #if (Rsquared[ibehavior][icell] > 0.25):
    if (Rsquared[ibehavior][icell] > 0.395):
        print('for cell ', icell, 'Rsquared = ', Rsquared[ibehavior][icell])
        #print('absolute Rsquared = ', abs(svr.score(X, y)))
        # could do significance test here, probably bootstrap based on subsampling of responses for each monkey stimulus,
        # since permutation across monkeys would confound ANOVA with regression by destroying all response differences between monkeys,
        # not just the linear relationship to behavior, making the null hypothesis uninterpretable
        sumRsquared += Rsquared[ibehavior][icell]

print('sumRsquared = ', sumRsquared)
                        
                    
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
            
            
                    
                         
print('SUBMISSION FROM ANALYSIS')
ibehavior = 3

sumRsquared = 0.0

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
                yadd += submissionfrommatrix[sourcemonkey][sinkmonkey]
            y.append(yadd)
            response = []
            responsestring = responsestringarray[cells[icell],sourcemonkey + 4 - lostmonkeys]
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
    SS_xy = 0
    SS_xx = 0
    m_y = 0
    m_x = 0
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
    m_y /+ n
    m_x /= n
    #print('SS_xx = ', SS_xx)
                                        
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
                
    # predicted response vector
    ypred = []
    for ixy in range (0, len(y)):
        ypred.append(b_0 + b_1*x[ixy])

    r2_score(y, ypred)
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    #if (Rsquared[ibehavior][icell] > 0.25):
    if (Rsquared[ibehavior][icell] > 0.395):
        print('for cell ', icell, 'Rsquared = ', Rsquared[ibehavior][icell])
        #print('absolute Rsquared = ', abs(svr.score(X, y)))
        # could do significance test here, probably bootstrap based on subsampling of responses for each monkey stimulus,
        # since permutation across monkeys would confound ANOVA with regression by destroying all response differences between monkeys,
        # not just the linear relationship to behavior, making the null hypothesis uninterpretable
        sumRsquared += Rsquared[ibehavior][icell]

print('sumRsquared = ', sumRsquared)
                        
                    
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
            
            




print('AGONISM TO ANALYSIS')
ibehavior = 4
sumRsquared = 0.0

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
                yadd += agonismtomatrix[sourcemonkey][sinkmonkey]
            y.append(yadd)
            response = []
            responsestring = responsestringarray[cells[icell],sourcemonkey + 4 - lostmonkeys]
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
    SS_xy = 0
    SS_xx = 0
    m_y = 0
    m_x = 0
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
    m_y /+ n
    m_x /= n
    #print('SS_xx = ', SS_xx)
                                        
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
                
    # predicted response vector
    ypred = []
    for ixy in range (0, len(y)):
        ypred.append(b_0 + b_1*x[ixy])

    r2_score(y, ypred)
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    #if (Rsquared[ibehavior][icell] > 0.25):
    if (Rsquared[ibehavior][icell] > 0.395):
        print('for cell ', icell, 'Rsquared = ', Rsquared[ibehavior][icell])
        #print('absolute Rsquared = ', abs(svr.score(X, y)))
        # could do significance test here, probably bootstrap based on subsampling of responses for each monkey stimulus,
        # since permutation across monkeys would confound ANOVA with regression by destroying all response differences between monkeys,
        # not just the linear relationship to behavior, making the null hypothesis uninterpretable
        sumRsquared += Rsquared[ibehavior][icell]

print('sumRsquared = ', sumRsquared)
                        
                    
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
            
            
                    
                         
print('AGONISM FROM ANALYSIS')
ibehavior = 5

sumRsquared = 0.0

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
                yadd += agonismfrommatrix[sourcemonkey][sinkmonkey]
            y.append(yadd)
            response = []
            responsestring = responsestringarray[cells[icell],sourcemonkey + 4 - lostmonkeys]
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
    SS_xy = 0
    SS_xx = 0
    m_y = 0
    m_x = 0
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
    m_y /+ n
    m_x /= n
    #print('SS_xx = ', SS_xx)
                                        
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
                
    # predicted response vector
    ypred = []
    for ixy in range (0, len(y)):
        ypred.append(b_0 + b_1*x[ixy])

    r2_score(y, ypred)
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    #if (Rsquared[ibehavior][icell] > 0.25):
    if (Rsquared[ibehavior][icell] > 0.395):
        print('for cell ', icell, 'Rsquared = ', Rsquared[ibehavior][icell])
        #print('absolute Rsquared = ', abs(svr.score(X, y)))
        # could do significance test here, probably bootstrap based on subsampling of responses for each monkey stimulus,
        # since permutation across monkeys would confound ANOVA with regression by destroying all response differences between monkeys,
        # not just the linear relationship to behavior, making the null hypothesis uninterpretable
        sumRsquared += Rsquared[ibehavior][icell]

print('sumRsquared = ', sumRsquared)
                        
                    
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
            
            

