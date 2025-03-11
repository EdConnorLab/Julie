import numpy as np
import pandas as pd

from monkey_names import Zombies, BestFrans
import statsmodels.api as sm
import re

def perform_regression(x, y):
    x = sm.add_constant(x)  # Adds a constant term to the predictor
    model = sm.OLS(y, x)  # OLS model
    results = model.fit()  # Fit model
    return results.rsquared, results.params  # Return R-squared and parameters


# Spike data
trial_responses = pd.read_excel('/home/connorlab/Documents/GitHub/Julie/Cortana/files_for_lin_reg_analysis_by_ed/spike_count_for_each_trial_windowed.xlsx')
response_string_array = trial_responses.values

# Zombies
zombies = [member.value for name, member in Zombies.__members__.items()]
del zombies[-1]

base_dir = '/home/connorlab/Documents/GitHub/Julie/behaviors/'
zombies_affiliation_file_name = 'zombies_feature_df_affiliation.xlsx'
zombies_submission_file_name = 'zombies_feature_df_submission.xlsx'
zombies_agonism_file_name = 'zombies_feature_df_agonism.xlsx'

zombies_affiliation_df = pd.read_excel(base_dir + zombies_affiliation_file_name)
zombies_submission_df = pd.read_excel(base_dir + zombies_submission_file_name)
zombies_agonism_df = pd.read_excel(base_dir + zombies_agonism_file_name)

zombies_affiliation_to = zombies_affiliation_df.iloc[:, 1:].to_numpy()
zombies_affiliation_from = zombies_affiliation_to.T

zombies_submission_to = zombies_submission_df.iloc[:, 1:].to_numpy()
zombies_submission_from = zombies_submission_to.T

zombies_agonism_to = zombies_agonism_df.iloc[:, 1:].to_numpy()
zombies_agonism_from = zombies_agonism_to.T

# Cells
cells = [0, 2, 7, 10, 13, 15, 17, 19, 21, 22, 24, 25, 30, 32, 34, 44, 46, 48, 50, 53, 55, 57, 59, 67]
cell_names = ['9/26 1 2 1', '9/26 3 2 1', '10/3 3 13 1', '10/3 4 10 2', '10/4 1 4 1', '10/4 1 19 3',
              '10/4 2 18 1', '10/4 2 20 2', '10/4 3 2 1', '10/4 3 4 2', '10/4 3 9 1', '10/4 3 9 3',
              '10/4 4 22 2', '10/4 4 27 1', '10/5 1 4', '10/11 1 2', '10/11 3 2', '10/11 3 13',
              '10/24 2 2', '10/27 4 11', '10/27 4 20', '10/31 1 5', '10/31 1 20', '11/8 1 7']
ncells = len(cells)
nmonkeys = len(zombies)
subjectmonkey = 6
nbehaviors = 6
Rsquared = np.zeros((nbehaviors, nmonkeys, ncells))
sumRsquared = np.zeros(10)
siglist = np.zeros((ncells, nbehaviors, nmonkeys))
print('AFFILIATION TO ANALYSIS')

################# CURRENTLY NOT WORKING
ibehavior = 0  # Example for one behavior, extendable for others

for icell in range(ncells):
    for source_monkey in range(nmonkeys):
        y = []
        x = []
        lostmonkeys = 0
        for sink_monkey in range(nmonkeys):  # responses don't include subject monkey
            if sink_monkey == source_monkey or sink_monkey == subjectmonkey:
                if sink_monkey == subjectmonkey:
                    lostmonkeys += 1  # skip for absence of subject monkey
            else:
                y.append(float(zombies_affiliation_to[source_monkey][sink_monkey]))
                response_string = response_string_array[cells[icell], sink_monkey + 4 - lostmonkeys]
                response_list = [int(s) for s in re.findall(r'\b\d+\b', response_string)]
                mean_response = sum(response_list) / len(response_list) if response_list else 0
                x.append(mean_response)

        if len(x) > 0 and len(y) > 0:  # Ensure there are data points to fit
            rsquared, params = perform_regression(x, y)
            Rsquared[ibehavior][source_monkey][icell] = rsquared
            b_0, b_1 = params  # intercept and slope
            print(
                f'Cell {icell}, Source Monkey {zombies[source_monkey]}: R^2 = {rsquared:.3f}, Intercept = {b_0:.3f}, Slope = {b_1:.3f}')

            if rsquared > 0.25:
                sumRsquared[source_monkey] += rsquared
                siglist[icell][ibehavior][source_monkey] = rsquared
                print(f'Significant R^2 for cell {icell}, source monkey {zombies[source_monkey]}: {rsquared:.3f}')

