import os
import pandas as pd
import random
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import re
from sklearn.linear_model import LinearRegression
# linear regression for behavior patterns and single monkey ID based on mean responses,
# no bootstrapping from individual responses

from analyses.enums.monkey_names import get_monkeys_by_default_order

trialresponses = pd.read_excel('/home/connorlab/Documents/JulieData/Cortana/old/Ed and ANOVA/used_for_R01/zombies_spike_counts_for_all_anova_passed_time_windowed_cells_old--usedforgrant.xlsx')

responsestringarray = trialresponses.values

nmonkeys = 10
nmonkeysnotsubject = 9

base_dir = '/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/'

# Load and assign matrices explicitly
affiliationtomatrix = pd.read_excel(os.path.join(base_dir, "zombies_feature_df_affiliation.xlsx")).iloc[:, 1:].to_numpy()
submissiontomatrix = pd.read_excel(os.path.join(base_dir, "zombies_feature_df_submission.xlsx")).iloc[:, 1:].to_numpy()
agonismtomatrix = pd.read_excel(os.path.join(base_dir, "zombies_feature_df_agonism.xlsx")).iloc[:, 1:].to_numpy()

affiliationfrommatrix = affiliationtomatrix.T
submissionfrommatrix = submissiontomatrix.T
agonismfrommatrix = agonismtomatrix.T

ncells = 74
monkeyname = get_monkeys_by_default_order("Zombies")
monkeyname_notsubject = [m for m in monkeyname if m != "81G"]
monkey_column = [14, 2, 12, 13, 16, 10, 36, 23, 15]
subjectmonkey = 6
nbehaviors = 6
behaviornames = ['affil to', 'affil from', 'sub to', 'sub from', 'agon to', 'agon from']


# Use NumPy for clean and fast initialization
Rsquared = np.zeros((nbehaviors, nmonkeys, ncells))
nless = np.zeros((nbehaviors, nmonkeys, ncells))
siglist = np.zeros((ncells, nbehaviors, nmonkeys))

for ibehavior in range (0, 6):
    sumRsquared = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print('behavior: ',ibehavior)

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

            n = len(x)
            X = np.array(x).reshape(-1, 1)  # shape (n_samples, 1)
            y = np.array(y)
            model = LinearRegression()
            model.fit(X, y)
            ypred = model.predict(X)
            Rsquared[ibehavior][sourcemonkey][icell] = r2_score(y, ypred)

            if (Rsquared[ibehavior][sourcemonkey][icell] > 0.5):
                sumRsquared[sourcemonkey] += abs(Rsquared[ibehavior][sourcemonkey][icell])

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

# for ibehavior in range (0, 6):
#     for sourcemonkey in range (0, nmonkeys):
#         for icell in range (0, ncells):
#             if (Rsquared[ibehavior][sourcemonkey][icell] > 0.50000):
#             #if (nless[ibehavior][sourcemonkey][icell] > 989):
#                 print('BEH ', ibehavior, 'cell ', icell, 'source monkey', monkeyname[sourcemonkey], 'Rsquared = ', Rsquared[ibehavior][sourcemonkey][icell], 'nless = ', nless[ibehavior][sourcemonkey][icell])
#


# same_matrix = []
# for imonkey in range (0, nmonkeys):
# for ibeh in range (0, 6):
# same_matrix.append(0)

# match_matrix = []
# for imonkey in range (0, nmonkeys):
# for ibeh in range (0, 6):
# match_matrix.append(same_matrix)

match_matrix = []  # 4-dimensional list of R2 > 0.5 found in same cells; match_matrix[monkey][behavior][monkey][behavior]
for imnk in range(0, nmonkeys):
    imnk_list = []  # outer or 1st dimension of list array is behaviors
    for ibeh in range(0, 6):
        ibeh_list = []  # 2nd dimension is source monkeys
        for jmnk in range(0, nmonkeys):
            jmnk_list = []
            for jbeh in range(0, 6):
                jmnk_list.append(0)
            ibeh_list.append(jmnk_list)
        imnk_list.append(ibeh_list)
    match_matrix.append(imnk_list)



        





