
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


# trial_responses = pd.read_excel('/Users/charlesconnor/Dropbox/grants/social.memory/selected_cells_time_windowed/BestFrans/combined_bestfrans_spike_count_time_windowed.xlsx')
trial_responses = pd.read_excel('/home/connorlab/Documents/GitHub/Julie/files_for_lin_reg_analysis_by_ed/combined_bestfrans_spike_count_time_windowed.xlsx')

response_string_array = trial_responses.values

nmonkeys = 5 #must match behavior matrix size; subject and source monkeys removed as appropriate in code below
nmonkeysnotsubject = 5

affiliation_to_matrix =np.array([[0, 17, 8, 17, 15], [18, 0, 7, 17, 27], [10, 5, 0, 20, 13], [13, 6, 22, 0, 13], [10, 28, 5, 11, 0]])
affiliation_from_matrix = affiliation_to_matrix.T

submission_to_matrix = np.array([[0, 1, 2, 0, 0], [1, 0, 1, 0, 4], [5, 8, 0, 33, 10], [19, 25, 3, 0, 11], [17, 24, 6, 13, 0]])
submission_from_matrix = submission_to_matrix.T

agonism_to_matrix = np.array([[0, 1, 2, 3, 16], [0, 0, 2, 10, 23], [1, 0, 0, 5, 13], [0, 0, 12, 0, 17], [0, 12, 1, 3, 0]])
agonism_from_matrix = agonism_to_matrix.T

cells = [73, 60, 55, 53, 34, 29, 16, 15, 10, 105, 110, 115, 119, 122, 127, 129]
cellnames = ['11/8 3 1 0 0–250', '10/31 1 22 0–750', '10/27 4 20 0–750', '10/27 4 11 0–750',
             '10/5 1 4 0–2000', '10/4 4 21 2 0–700', '10/4 2 9 1 100–500', '10/4 1 19 3 100–500',
             '10/3 4 10 2 200–600', '9/28 1 25 2', '10/4 4 7', '10/31 2 5 250–750',
             '11/22 3 28', '11/25 2 1', '11/27 3 2', '11/27 3 17']
ncells = len(cells)
monkeyname = [ 'G701', '14F', '68F', '101G', '19J' ]

nboots = 1 # > 1 for bootstrap random sampling of one response from each cell; commented out below
subjectmonkey = 6    #81G


maxabscorrelation = []    # only reporting results based on max absolute correlation for each cell
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

    for sourcemonkey in range (0, nmonkeys):
        yadd = 0.0
        for sinkmonkey in range (0, nmonkeys):
            yadd += affiliation_to_matrix[sourcemonkey][sinkmonkey]
        y.append(yadd)
        response = []
        responsestring = response_string_array[cells[icell], sourcemonkey + 1]
        responselist = [int(s) for s in re.findall(r'\b\d+\b', responsestring)]
        meanresponse = sum(responselist) / len(responselist)
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
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    if (Rsquared[ibehavior][icell] > 0.25):
    #if (Rsquared[ibehavior][icell] > 0.395):
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

    for sourcemonkey in range (0, nmonkeys):
        yadd = 0.0
        for sinkmonkey in range (0, nmonkeys):
            yadd += affiliation_to_matrix[sourcemonkey][sinkmonkey]
        y.append(yadd)
        response = []
        responsestring = response_string_array[cells[icell], sourcemonkey + 1]
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
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    if (Rsquared[ibehavior][icell] > 0.25):
    #if (Rsquared[ibehavior][icell] > 0.395):
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

    for sourcemonkey in range (0, nmonkeys):
        yadd = 0.0
        for sinkmonkey in range (0, nmonkeys):
            yadd += affiliation_to_matrix[sourcemonkey][sinkmonkey]
        y.append(yadd)
        response = []
        responsestring = response_string_array[cells[icell], sourcemonkey + 1]
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
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    if (Rsquared[ibehavior][icell] > 0.25):
    #if (Rsquared[ibehavior][icell] > 0.395):
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

    for sourcemonkey in range (0, nmonkeys):
        yadd = 0.0
        for sinkmonkey in range (0, nmonkeys):
            yadd += affiliation_to_matrix[sourcemonkey][sinkmonkey]
        y.append(yadd)
        response = []
        responsestring = response_string_array[cells[icell], sourcemonkey + 1]
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
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    if (Rsquared[ibehavior][icell] > 0.25):
    #if (Rsquared[ibehavior][icell] > 0.395):
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

    for sourcemonkey in range (0, nmonkeys):
        yadd = 0.0
        for sinkmonkey in range (0, nmonkeys):
            yadd += affiliation_to_matrix[sourcemonkey][sinkmonkey]
        y.append(yadd)
        response = []
        responsestring = response_string_array[cells[icell], sourcemonkey + 1]
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
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    if (Rsquared[ibehavior][icell] > 0.25):
    #if (Rsquared[ibehavior][icell] > 0.395):
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

    for sourcemonkey in range (0, nmonkeys):
        yadd = 0.0
        for sinkmonkey in range (0, nmonkeys):
            yadd += affiliation_to_matrix[sourcemonkey][sinkmonkey]
        y.append(yadd)
        response = []
        responsestring = response_string_array[cells[icell], sourcemonkey + 1]
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
    Rsquared[ibehavior][icell] = explained_variance_score(y, ypred)
    if (Rsquared[ibehavior][icell] > 0.25):
    #if (Rsquared[ibehavior][icell] > 0.395):
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
            
      