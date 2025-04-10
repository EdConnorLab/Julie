
# linear regression for behavior patterns and single monkey ID based on mean responses,
# no bootstrapping from individual responses

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import re
import math

# trial_responses = pd.read_excel('/Users/charlesconnor/Dropbox/grants/social.memory/selected_cells_time_windowed/spike_count_for_each_trial_windowed.xlsx')
trial_responses = pd.read_excel('/home/connorlab/Documents/GitHub/Julie/Cortana/files_for_lin_reg_analysis_by_ed/spike_count_for_each_trial_windowed.xlsx')
response_string_array = trial_responses.values

affiliation_to_matrix =np.array([
    [0, 19, 9, 38, 84, 27, 13, 14, 4, 9], [15, 0, 9, 41, 4, 13, 19, 71, 12, 0],
    [18, 10, 0, 21, 7, 17, 49, 18, 3, 1], [38, 43, 24, 0, 18, 6, 31, 26, 29, 4],
    [90, 3, 8, 8, 0, 22, 8, 9, 1, 18], [23, 12, 17, 1, 23, 0, 23, 16, 3, 10],
    [17, 18, 43, 34, 10, 23, 0, 34, 2, 9], [11, 70, 18, 17, 8, 17, 33, 0, 5, 3],
    [3, 6, 4, 31, 2, 1, 2, 4, 0, 9], [8, 0, 0, 1, 26, 8, 2, 1, 7, 0]])

affiliation_from_matrix = affiliation_to_matrix.T

submission_to_matrix = np.array([
    [0, 0, 0, 1, 0, 1, 0, 0, 2, 0], [4, 0, 0, 13, 0, 0, 0, 2, 1, 0],
    [9, 2, 0, 10, 0, 0, 4, 3, 0, 0], [7, 0, 2, 0, 0, 0, 0, 1, 6, 0],
    [11, 8, 3, 5, 0, 2, 4, 2, 4, 18], [90, 16, 14, 43, 0, 0, 10, 7, 9, 6],
    [29, 12, 1, 17, 0, 0, 0, 23, 17, 1], [41, 6, 1, 28, 0, 0, 2, 0, 5, 0],
    [16, 2, 0, 8, 0, 1, 0, 0, 0, 2], [7, 2, 3, 1, 2, 0, 5, 2, 5, 0]])

submission_from_matrix = submission_to_matrix.T

agonism_to_matrix = np.array([
    [0, 4, 2, 7, 4, 7, 3, 9, 3, 2], [0, 0, 0, 2, 1, 1, 8, 4, 2, 0],
    [0, 1, 0, 0, 0, 7, 0, 1, 0, 2], [1, 8, 0, 0, 9, 5, 6, 10, 11, 2],
    [0, 0, 0, 0, 0, 1, 0, 1, 3, 10], [0, 0, 0, 0, 2, 0, 0, 0, 3, 1],
    [0, 0, 3, 2, 0, 4, 0, 1, 0, 6], [0, 1, 1, 0, 1, 0, 7, 0, 0, 1],
    [0, 0, 0, 2, 2, 5, 2, 3, 0, 6], [0, 0, 0, 0, 5, 4, 1, 0, 6, 0]])

agonism_from_matrix = agonism_to_matrix.T

# Cells
cells = [0, 2, 7, 10, 13, 15, 17, 19, 21, 22, 24, 25, 30, 32, 34, 44, 46, 48, 50, 53, 55, 57, 59, 67]
cell_names = ['9/26 1 2 1', '9/26 3 2 1', '10/3 3 13 1', '10/3 4 10 2', '10/4 1 4 1', '10/4 1 19 3',
              '10/4 2 18 1', '10/4 2 20 2', '10/4 3 2 1', '10/4 3 4 2', '10/4 3 9 1', '10/4 3 9 3',
              '10/4 4 22 2', '10/4 4 27 1', '10/5 1 4', '10/11 1 2', '10/11 3 2', '10/11 3 13',
              '10/24 2 2', '10/27 4 11', '10/27 4 20', '10/31 1 5', '10/31 1 20', '11/8 1 7']
ncells = len(cells)

# Monkeys
monkey_name = ['7124', '69X', '72X', '94B', '110E', '67G', '81G', '143H', '87J', '151J']
nmonkeys = len(monkey_name)
nmonkeysnotsubject = 9
subjectmonkey = 6    # 81G

# Behaviors
behavior_names = ['AFFILIATION TO', 'AFFILIATION FROM', 'SUBMISSION TO', 'SUBMISSION FROM', 'AGONISM TO', 'AGONISM FROM']
nbehaviors = len(behavior_names)

Rsquared = np.zeros((nbehaviors, nmonkeys, ncells))
behavior_list = np.zeros((nmonkeys, ncells))
monkey_list = np.zeros(ncells)
sumRsquared = np.zeros(10)
    
beh_add = np.zeros(nmonkeys)
cell_add = np.zeros((nbehaviors, nmonkeys))
siglist = np.zeros((ncells, nbehaviors, nmonkeys))

    
print('AFFILIATION TO ANALYSIS')
ibehavior = 0

for icell in range (0, ncells):
    for source_monkey in range (0, nmonkeys):
        y = []
        x = []
        lostmonkeys = 0
        for sink_monkey in range (0, nmonkeys):    #responses don't include subject monkey
            if ((sink_monkey == source_monkey) or (sink_monkey == subjectmonkey)):
                if (sink_monkey == subjectmonkey):
                    lostmonkeys += 1    #skip advancement through responselist per the index subtract below for absence of subject monkey
            else:
                y.append(float(affiliation_to_matrix[source_monkey][sink_monkey]))
                response = []
                response_string = response_string_array[cells[icell], sink_monkey + 4 - lostmonkeys] # find the cell (row) and get corresponding monkey spike rate (column)
                response_list = [int(s) for s in re.findall(r'\b\d+\b', response_string)] # change string to a list
                mean_response = sum(response_list) / len(response_list)
                x.append(mean_response)

        #print('y = ', y)
        #print('x = ', x)
            
        n = len(x) # number of data points

        # calculating cross-deviation and deviation about x
        SS_xy = 0.0
        SS_xx = 0.0
        m_y = 0.0
        m_x = 0.0
        for ixy in range (0, n):
            SS_xy += y[ixy]*x[ixy] # sum of products of corresponding x and y values (for covar calculations)
            SS_xx += x[ixy]*x[ixy] # sum of squares of x values (for var calculations)
            # the sums of y and x values
            m_y += y[ixy]
            m_x += x[ixy]
        #print('m_y = ', m_y)
        #print('m_x = ', m_x)
        #print('SS_xx = ', SS_xx)
        SS_xy -= n*m_y*m_x # for covariance
        SS_xx -= n*m_x*m_x # for variance
        # compute mean of x and y
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
        Rsquared[ibehavior][source_monkey][icell] = explained_variance_score(y, ypred)
        # if ((icell == 0) and (source_monkey == 4)):
        #     print('rsquare icell 15 sourcemonkey 4: ', Rsquared[ibehavior][source_monkey][icell])
        #     print('affiliation from frequencies for 94B: ', y)
        if (source_monkey == 6):
            print(f'rsquare icell {icell} sourcemonkey 6: ', Rsquared[ibehavior][source_monkey][icell])
        if (Rsquared[ibehavior][source_monkey][icell] > 0.25):
            sumRsquared[source_monkey] += abs(Rsquared[ibehavior][source_monkey][icell])
            print('for cell', icell, 'for source monkey', monkey_name[source_monkey], 'Rsquared = ', Rsquared[ibehavior][source_monkey][icell])
            siglist[icell][ibehavior][source_monkey] = Rsquared[ibehavior][source_monkey][icell]

for source_monkey in range (0, nmonkeys):
    print('sumRsquared for', monkey_name[source_monkey], ' = ', sumRsquared[source_monkey])
    

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
    for source_monkey in range (0, nmonkeys):
        y = []
        x = []
        lostmonkeys = 0
        for sink_monkey in range (0, nmonkeys):    #responses don't include subject monkey
            if ((sink_monkey == source_monkey) or (sink_monkey == subjectmonkey)):
                if (sink_monkey == subjectmonkey):
                    lostmonkeys += 1    #skip advancement through responselist per the index subtract below for absence of subject monkey
            else:
                y.append(float(affiliation_from_matrix[source_monkey][sink_monkey]))
                response = []
                response_string = response_string_array[cells[icell], sink_monkey + 4 - lostmonkeys]
                response_list = [int(s) for s in re.findall(r'\b\d+\b', response_string)]
                mean_response = sum(response_list) / len(response_list)
                x.append(mean_response)
                
        if ((icell == 16) and (source_monkey == 3)):
            print('responses for cell 16 and sourcemonkey 3: ', x)
            print('affiliation from frequencies for sourcemonkey 3: ', y)
                
        #print('y = ', y)
        #print('x = ', x)
            
        n = len(x)

        # mean of x and y vector
        # m_x = sum(x) / len(x)
        # m_y = sum(y) / len(y)

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
        Rsquared[ibehavior][source_monkey][icell] = explained_variance_score(y, ypred)
        if (Rsquared[ibehavior][source_monkey][icell] > 0.25):
            sumRsquared[source_monkey] += abs(Rsquared[ibehavior][source_monkey][icell])
            print('for cell ', icell, 'for source monkey', monkey_name[source_monkey], 'Rsquared = ', Rsquared[ibehavior][source_monkey][icell])
            siglist[icell][ibehavior][source_monkey] = Rsquared[ibehavior][source_monkey][icell]
        if (Rsquared[ibehavior][source_monkey][icell] > 0.25):
            #plotting the actual points as scatter plot
            plt.scatter(x, y, color = "m",
            marker = "o", s = 30)

            #predicted response vector
            #y_pred = b_0 + b_1*x

            #plotting the regression line
            plt.plot(x, ypred, color = "g")

            # putting labels
            plt.xlabel('x')
            plt.ylabel('y')
            #plt.show()
 
for source_monkey in range (0, nmonkeys):
    print('sumRsquared for ', monkey_name[source_monkey], ' = ', sumRsquared[source_monkey])
    
                    
   
 
print('SUBMISSION TO ANALYSIS')
ibehavior = 2

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for icell in range (0, ncells):
    for source_monkey in range (0, nmonkeys):
        y = []
        x = []
        lostmonkeys = 0
        for sink_monkey in range (0, nmonkeys):    #responses don't include subject monkey
            if ((sink_monkey == source_monkey) or (sink_monkey == subjectmonkey)):
                if (sink_monkey == subjectmonkey):
                    lostmonkeys += 1    #skip advancement through responselist per the index subtract below for absence of subject monkey
            else:
                y.append(float(submission_to_matrix[source_monkey][sink_monkey]))
                response = []
                response_string = response_string_array[cells[icell], sink_monkey + 4 - lostmonkeys]
                response_list = [int(s) for s in re.findall(r'\b\d+\b', response_string)]
                mean_response = sum(response_list) / len(response_list)
                x.append(mean_response)
                
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
        Rsquared[ibehavior][source_monkey][icell] = explained_variance_score(y, ypred)
        if (Rsquared[ibehavior][source_monkey][icell] > 0.25):
            sumRsquared[source_monkey] += abs(Rsquared[ibehavior][source_monkey][icell])
            print('for cell ', icell, 'for source monkey', monkey_name[source_monkey], 'Rsquared = ', Rsquared[ibehavior][source_monkey][icell])
            siglist[icell][ibehavior][source_monkey] = Rsquared[ibehavior][source_monkey][icell]

for source_monkey in range (0, nmonkeys):
    print('sumRsquared for ', monkey_name[source_monkey], ' = ', sumRsquared[source_monkey])
    
                    
                         
print('SUBMISSION FROM ANALYSIS')
ibehavior = 3

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for icell in range (0, ncells):
    for source_monkey in range (0, nmonkeys):
        y = []
        x = []
        lostmonkeys = 0
        for sink_monkey in range (0, nmonkeys):    #responses don't include subject monkey
            if ((sink_monkey == source_monkey) or (sink_monkey == subjectmonkey)):
                if (sink_monkey == subjectmonkey):
                    lostmonkeys += 1    #skip advancement through responselist per the index subtract below for absence of subject monkey
            else:
                y.append(float(submission_from_matrix[source_monkey][sink_monkey]))
                response = []
                response_string = response_string_array[cells[icell], sink_monkey + 4 - lostmonkeys]
                response_list = [int(s) for s in re.findall(r'\b\d+\b', response_string)]
                mean_response = sum(response_list) / len(response_list)
                x.append(mean_response)
                
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
        Rsquared[ibehavior][source_monkey][icell] = explained_variance_score(y, ypred)
        if (Rsquared[ibehavior][source_monkey][icell] > 0.25):
            sumRsquared[source_monkey] += abs(Rsquared[ibehavior][source_monkey][icell])
            print('for cell ', icell, 'for source monkey', monkey_name[source_monkey], 'Rsquared = ', Rsquared[ibehavior][source_monkey][icell])
            siglist[icell][ibehavior][source_monkey] = Rsquared[ibehavior][source_monkey][icell]
        if (Rsquared[ibehavior][source_monkey][icell] > 0.53):
            #plotting the actual points as scatter plot
            plt.scatter(x, y, color = "m",
            marker = "o", s = 30)

            #predicted response vector
            #y_pred = b_0 + b_1*x

            #plotting the regression line
            plt.plot(x, ypred, color = "g")

            # putting labels
            plt.xlabel('x')
            plt.ylabel('y')
            #plt.show()

for source_monkey in range (0, nmonkeys):
    print('sumRsquared for ', monkey_name[source_monkey], ' = ', sumRsquared[source_monkey])
    
                    




print('AGONISM TO ANALYSIS')
ibehavior = 4

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]



for icell in range (0, ncells):
    for source_monkey in range (0, nmonkeys):
        y = []
        x = []
        lostmonkeys = 0
        for sink_monkey in range (0, nmonkeys):    #responses don't include subject monkey
            if ((sink_monkey == source_monkey) or (sink_monkey == subjectmonkey)):
                if (sink_monkey == subjectmonkey):
                    lostmonkeys += 1    #skip advancement through responselist per the index subtract below for absence of subject monkey
            else:
                y.append(float(agonism_to_matrix[source_monkey][sink_monkey]))
                response = []
                response_string = response_string_array[cells[icell], sink_monkey + 4 - lostmonkeys]
                response_list = [int(s) for s in re.findall(r'\b\d+\b', response_string)]
                mean_response = sum(response_list) / len(response_list)
                x.append(mean_response)
                
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
        Rsquared[ibehavior][source_monkey][icell] = explained_variance_score(y, ypred)
        if (Rsquared[ibehavior][source_monkey][icell] > 0.25):
            sumRsquared[source_monkey] += abs(Rsquared[ibehavior][source_monkey][icell])
            print('for cell ', icell, 'for source monkey', monkey_name[source_monkey], 'Rsquared = ', Rsquared[ibehavior][source_monkey][icell])
            siglist[icell][ibehavior][source_monkey] = Rsquared[ibehavior][source_monkey][icell]

for source_monkey in range (0, nmonkeys):
    print('sumRsquared for ', monkey_name[source_monkey], ' = ', sumRsquared[source_monkey])
    
                    
                         
print('AGONISM FROM ANALYSIS')
ibehavior = 5

sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


for icell in range (0, ncells):
    for source_monkey in range (0, nmonkeys):
        y = []
        x = []
        lostmonkeys = 0
        for sink_monkey in range (0, nmonkeys):    #responses don't include subject monkey
            if ((sink_monkey == source_monkey) or (sink_monkey == subjectmonkey)):
                if (sink_monkey == subjectmonkey):
                    lostmonkeys += 1    #skip advancement through responselist per the index subtract below for absence of subject monkey
            else:
                y.append(float(agonism_from_matrix[source_monkey][sink_monkey]))
                response = []
                response_string = response_string_array[cells[icell], sink_monkey + 4 - lostmonkeys]
                response_list = [int(s) for s in re.findall(r'\b\d+\b', response_string)]
                mean_response = sum(response_list) / len(response_list)
                x.append(mean_response)
                
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
        Rsquared[ibehavior][source_monkey][icell] = explained_variance_score(y, ypred)
        if (Rsquared[ibehavior][source_monkey][icell] > 0.25):
            sumRsquared[source_monkey] += abs(Rsquared[ibehavior][source_monkey][icell])
            print('for cell ', icell, 'for source monkey', monkey_name[source_monkey], 'Rsquared = ', Rsquared[ibehavior][source_monkey][icell])
            siglist[icell][ibehavior][source_monkey] = Rsquared[ibehavior][source_monkey][icell]
        if (Rsquared[ibehavior][source_monkey][icell] > 0.48):
            #plotting the actual points as scatter plot
            plt.scatter(x, y, color = "m",
            marker = "o", s = 30)

            #predicted response vector
            #y_pred = b_0 + b_1*x

            #plotting the regression line
            plt.plot(x, ypred, color = "g")

            # putting labels
            plt.xlabel('x')
            plt.ylabel('y')
            #plt.show()
 

for source_monkey in range (0, nmonkeys):
    print('sumRsquared for ', monkey_name[source_monkey], ' = ', sumRsquared[source_monkey])
    

########################################################################################################################

#correct by counting only max behavioral correlation for any given cell
print('SUMMED FOR EACH MONKEY')

nbehaviors = 6

behavior_list = np.zeros(nmonkeys)
Rsquared_total = np.zeros((nbehaviors, nmonkeys))
#print(Rsquared_total)

for ibehavior in range (0, nbehaviors):
    for icell in range (0, ncells):
        for imonkey in range (0, nmonkeys):
            if (Rsquared[ibehavior][imonkey][icell] > 0.25):
                Rsquared_total[ibehavior][imonkey] += Rsquared[ibehavior][imonkey][icell]
             
for imonkey in range (0, nmonkeys):
    print('MONKEY ', imonkey, monkey_name[imonkey])
    total_Rsquared = 0.0
    for ibehavior in range (0, nbehaviors):
        print('behavior ', ibehavior, behavior_names[ibehavior], 'Rsquared_total = ', Rsquared_total[ibehavior][imonkey])
        total_Rsquared += Rsquared_total[ibehavior][imonkey]
    print('TOTAL RSQUARED = ', total_Rsquared)

print('BY CELL BY BEHAVIOR')

for icell in range (0, ncells):
    for ibehavior in range (0, nbehaviors):
        for source_monkey in range (0, nmonkeys):
            if (siglist[icell][ibehavior][source_monkey] > 0):
                print('icell = ', icell, cell_names[icell], behavior_names[ibehavior], monkey_name[source_monkey], siglist[icell][ibehavior][source_monkey])

print('BY CELL BY MONKEY')

for icell in range (0, ncells):
    for source_monkey in range (0, nmonkeys):
        for ibehavior in range (0, nbehaviors):
            if (siglist[icell][ibehavior][source_monkey] > 0):
                print('icell = ', icell, cell_names[icell], behavior_names[ibehavior], monkey_name[source_monkey], siglist[icell][ibehavior][source_monkey])


print('BY MONKEY')

for source_monkey in range (0, nmonkeys):
    for icell in range (0, ncells):
        for ibehavior in range (0, nbehaviors):
            if (siglist[icell][ibehavior][source_monkey] > 0):
                print('icell = ', icell, cell_names[icell], behavior_names[ibehavior], monkey_name[source_monkey], siglist[icell][ibehavior][source_monkey])

print('COMBINED TUNING WITHIN CELLS')
print('"tuning r" in the figure')
print('max products of Rsquared values summed across cells')

comblist = np.zeros((10,10))    # sum of combination maxRsquared values in same cells summed across cells [sourcemonkey][sinkmonkey]
cellcomblist = np.zeros((10,10))   # sum of combination maxRsquared values in current cell [sourcemonkey][sinkmonkey]
maxmonkey = np.zeros(10)  # max Rsquared for each monkey for current cell

for icell in range(ncells):
    maxmonkey.fill(0)
    cellcomblist.fill(0)

    for source_monkey in range(nmonkeys):
        maxmonkey[source_monkey] = np.max(siglist[icell, :, source_monkey])

    for source_monkey in range(nmonkeys):
        for pairmonkey in range(nmonkeys):
            cellcomblist[source_monkey][pairmonkey] = maxmonkey[source_monkey] * maxmonkey[pairmonkey]
            comblist[source_monkey][pairmonkey] += cellcomblist[source_monkey][pairmonkey]
            
for source_monkey in range(nmonkeys):
    for pairmonkey in range(nmonkeys):
        print('sourcemonkey ', monkey_name[source_monkey], 'pairmonkey ', monkey_name[pairmonkey], 'sum max Rsquared products', comblist[source_monkey][pairmonkey])

'''
print('CLUSTERING BY ANATOMY')

#0 '9/26 1 2 1', 10.0 ML, 15.8 AP, 8.4 DV
#1 '9/26 3 2 1', 9.9 ML, 15.8 AP, 7.7 DV
#2 '10/3 3 13 1', 11.1 ML, 15.8 AP, 8.0 DV
#3 '10/3 4 10 2', 11.1 ML, 15.8 AP, 8.0 DV
#4 '10/4 1 4 1', 9.2 ML, 15.9 AP, 8.2 DV
#5 '10/4 1 19 3', 9.2 ML, 15.9 AP, 8.2 DV
#6 '10/4 2 18 1', 9.1 ML, 15.9 AP, 7.8 DV
#7 '10/4 2 20 2', 9.1 ML, 15.9 AP, 7.8 DV
#8 '10/4 3 2 1', 9.1 ML, 15.9 AP, 7.6 DV
#9 '10/4 3 4 2', 9.1 ML, 15.9 AP, 7.6 DV
#10 '10/4 3 9 1', 9.1 ML, 15.9 AP, 7.6 DV
#11 '10/4 3 9 3', 9.1 ML, 15.9 AP, 7.6 DV
#12 '10/4 4 22 2', 9.0 ML, 15.9 AP, 7.3 DV
#13 '10/4 4 27 1', 9.0 ML, 15.9 AP, 7.3 DV
#14 '10/5 1 4', 9.7 ML, 18.2 AP, 4.3 DV
#15 '10/11 1 2', 11.3 ML, 18.7 AP, 6.1 DV
#16 '10/11 3 2', 11.1 ML, 18.9 AP, 4.3 DV
#17 '10/11 3 13', 11.1 ML, 18.9 AP, 4.3 DV
#18 '10/24 2 2', 11 ML, 18.1 AP, 2.8 DV
#19 '10/27 4 11', 9.2 ML, 18.7 AP, 1.9 DV
#20 '10/27 4 20', 9.2 ML, 18.7 AP, 1.9 DV
#21 '10/31 1 5', 8.1 ML, 16.8 AP, 3.8 DV
#22 '10/31 1 20', 8.1 ML, 16.8 AP, 3.8 DV
#23 '11/8 1 7'], 9.3 ML, 16.4 AP, 5 DV

cellML = [10.0, 9.9, 11.1, 11.1, 9.2, 9.2, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.0, 9.0, 9.7, 11.3, 11.1, 11.1, 11.0, 9.2, 9.2, 8.1, 8.1, 9.3]


cellAP = [15.8, 15.8, 15.8, 15.8, 15.9, 15.9, 15.9, 15.9, 15.9, 15.9, 15.9, 15.9, 15.9, 15.9, 18.2, 18.7, 18.9, 18.9, 18.1, 18.7, 18.7, 16.8, 16.8, 16.4]


cellDV = [8.4, 7.7, 8.0, 8.0, 8.2, 8.2, 7.8, 7.8, 7.6, 7.6, 7.6, 7.6, 7.3, 7.3, 4.3, 6.1, 4.3, 4.3, 2.8, 1.9, 1.9, 3.8, 3.8, 5.0]


celldistance = []    # distance [icell][paircell]
for icell in range (0, ncells):
    pairdistance = []
    pairzero = icell + 1
    for paircell in range (pairzero, ncells):
        ssdistance = (cellML[icell] - cellML[paircell])**2 + (cellAP[icell] - cellAP[paircell])**2 + (cellDV[icell] - cellDV[paircell])**2
        if (ssdistance > 0.0):
            distance = math.sqrt(ssdistance)
        else:
            distance = 0.0
        pairdistance.append(distance)
    #print('length of pairdistance = ', len(pairdistance))
    celldistance.append(pairdistance)


cellsum = []    # summed correlations [icell][sourcemonkey]
for icell in range (0, ncells):
    monkeysum = []
    for source_monkey in range (0, nmonkeys):
        monkeytotal = 0.0
        for ibehavior in range (0, nbehaviors):
            monkeytotal += siglist[icell][ibehavior][source_monkey]
        monkeysum.append(monkeytotal)
    #print('length of monkeysum = ', len(monkeysum))
    cellsum.append(monkeysum)


same_product_sum = 0.0
nsame = 0
different_product_sum = 0.0
ndifferent = 0

cellcorrelation = []    # products of cellsums summed over monkeys [icell][paircell]
for icell in range (0, ncells):
    paircorrelation = []
    pairzero = icell + 1
    for paircell in range (pairzero, ncells):
        pairsum = 0.0
        for source_monkey in range (0, nmonkeys):
            #print('icell = ', icell, 'paircell = ', paircell)
            pairsum += cellsum[icell][source_monkey] * cellsum[paircell][source_monkey]
        paircorrelation.append(pairsum)
        #print(' icell ', icell, 'paircell ', paircell, 'summed correlation is ', pairsum)
    cellcorrelation.append(paircorrelation)

for icell in range (0, ncells):
    pairlength = ncells - icell -1
    for paircell in range (0, pairlength):
            if (celldistance[icell][paircell] < 0.000000001):
                same_product_sum += cellcorrelation[icell][paircell]
                nsame += 1
            else:
                different_product_sum += cellcorrelation[icell][paircell]
                ndifferent += 1

    
print ('same penetration product sum = ', same_product_sum, 'nsame = ', nsame, 'average = ', same_product_sum / float(nsame))
print ('different penetration product sum = ', different_product_sum, 'ndifferent = ', ndifferent, 'average = ', different_product_sum / float(ndifferent))
            


y = []
x = []
        
for icell in range (0, ncells):
    pairzero = icell + 1
    for paircell in range (pairzero, ncells):
        y.append(cellcorrelation[icell][paircell])
        x.append(celldistance[icell][paircell])
                
print('y = ', y)
print('x = ', x)
            
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
print('SS_xx = ', SS_xx)
                     
                    
# calculating regression coefficients
b_1 = SS_xy / SS_xx
b_0 = m_y - b_1*m_x
                
# predicted response vector
ypred = []
for ixy in range (0, len(y)):
    ypred.append(b_0 + b_1*x[ixy])

r2_score(y, ypred)
R2d_distance_tuning = explained_variance_score(y, ypred)
print('R2d_distance_tuning =', R2d_distance_tuning)
print('r2_score =', r2_score)
        
#plotting the actual points as scatter plot
plt.scatter(x, y, color = "m",
marker = "o", s = 30)

#predicted response vector
#y_pred = b_0 + b_1*x

#plotting the regression line
plt.plot(x, ypred, color = "g")

# putting labels
plt.xlabel('x')
plt.ylabel('y')
#plt.show()

'''