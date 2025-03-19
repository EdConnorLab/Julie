from sklearn.metrics import cohen_kappa_score
import pandas as pd

from excel_data_reader import ExcelDataReader

#Cohen's Kappa Scores:
#<0: Poor agreement
#0.00 – 0.20: Slight agreement
#0.21 – 0.40: Fair agreement
#0.41 – 0.60: Moderate agreement
#0.61 – 0.80: Substantial agreement
#0.81 – 1.00: Almost perfect agreement
#for more info see here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/
excel_data_reader = ExcelDataReader(file_name='/home/connorlab/Documents/GitHub/Julie/reliability_test/Instigators_ReliabilityTest.xlsx')
df = excel_data_reader.get_sheet_by_name('social')
# df = pd.read_csv("/home/connorlab/Documents/GitHub/Julie/reliability_test/juliejerrysocialior.csv")
unique_labels = pd.unique(df[['Ty', 'Julie']].values.ravel())
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

# Map the numeric labels back to the DataFrame
df['Rater1_numeric'] = df['Ty'].map(label_mapping)
df['Rater2_numeric'] = df['Julie'].map(label_mapping)
print(df)

# Cohen's Kappa Score
kappa = cohen_kappa_score(df['Rater1_numeric'], df['Rater2_numeric'])
print("Cohen's Kappa Score:", kappa)

# Percent Agreement
agreements = (df['Rater1_numeric'] == df['Rater2_numeric']).sum()
total_entries = len(df)
percent_agreement = (agreements / total_entries) * 100
print(f"Percentage Agreement: {percent_agreement:.2f}%")
