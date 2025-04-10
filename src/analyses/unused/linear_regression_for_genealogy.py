import pandas as pd
import numpy as np

from excel_data_reader import ExcelDataReader
from data_readers.recording_metadata_reader import RecordingMetadataReader
from statsmodels.regression.linear_model import OLS

from monkey_names import Monkey
import spike_rate_analysis

if __name__ == '__main__':

    # Get genealogy matrix
    excel_data_reader = ExcelDataReader(file_name='../../../social_data/genealogy_matrix.xlsx')
    genealogy_data = excel_data_reader.get_first_sheet()
    print(genealogy_data)
    genealogy_data['Focal Name'] = genealogy_data['Focal Name'].astype(str)

    zombies = [member.value for name, member in Monkey.__members__.items() if name.startswith('Z_')]

    zombies_df = pd.DataFrame({'Focal Name': zombies})
    temp = pd.DataFrame({'Focal Name': zombies})

    feature_df = genealogy_data.copy(deep=True)

    # Get only numbers
    numeric_columns = feature_df.iloc[:, 1:]
    # Convert selected numeric columns to a NumPy array
    numeric_array = numeric_columns.values
    # Append a column of 1s to the array to represent the last column
    numeric_array_with_ones = np.hstack((numeric_array, np.ones((numeric_array.shape[0], 1))))
    X = numeric_array_with_ones
    ''' PCA
    pca = decomposition.PCA(n_components=X.shape[0])
    pca.fit(X.T)
    L = pca.explained_variance_  # eigenvalues
    E = pca.components_.T  # eigvenvectors

    A = pca.transform(X.T).T  # A is transformed data
    # Sort the eigenvalues and eigenvectors in descending order
    L = L[::1]
    E = E[:, ::1]
    '''
    # Get Spike Rate -- Y
    reader = RecordingMetadataReader()
    recording_metadata = reader.get_metadata()
    ER_data = recording_metadata[recording_metadata['Location'] == 'ER']
    AMG_data = recording_metadata[recording_metadata['Location'] == 'Amygdala']

    for index, row in ER_data.iterrows():
        date = row['Date'].strftime('%Y-%m-%d')
        round = row['Round No.']
        spike_rates = spike_rate_computation.get_average_spike_rates_for_each_monkey(date, round)
        zombies_names = zombies_df['Focal Name'].tolist()
        zombies_spike_rates = spike_rates.reindex(columns=zombies_names, fill_value=0)
        # Extract values from a column as a NumPy array
        for index, row in zombies_spike_rates.iterrows():
            Y = row.values
            # lr = LinearRegression(fit_intercept=False)
            # lr.fit(numeric_array, Y)
            #
            # print('coeff')
            # print(lr.coef_)
            print(f"Data on {date} Round No. {round} -- Neuron {index}")
            model = OLS(Y, X)
            results = model.fit()
            print(results.summary())
