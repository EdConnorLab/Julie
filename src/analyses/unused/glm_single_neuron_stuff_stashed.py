import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm


def run_multi_predictor_glm(df, formula, neuron_col="NeuronID"):
    results = []
    unique_neurons = df[neuron_col].unique()

    for neuron in tqdm(unique_neurons, desc="Running multi-GLM per neuron"):
        neuron_df = df[df[neuron_col] == neuron]
        if neuron_df['SpikeCount'].sum() == 0:
            continue
        try:
            model = smf.glm(formula=formula, data=neuron_df, family=sm.families.Poisson()).fit()
            summary = model.summary2().tables[1].reset_index()
            summary['NeuronID'] = neuron
            results.append(summary)
        except Exception as e:
            print(f"Error processing neuron {neuron}: {e}")
            continue

    if results:
        results_df = pd.concat(results, ignore_index=True)
        significant_df = results_df[results_df['P>|z|'] < 0.05]
        # print("\nMulti-GLM Significant neurons:")
        # print(significant_df.head())
        return results_df, significant_df
    else:
        print("No valid neurons found.")
        return pd.DataFrame(), pd.DataFrame()



def run_glm(df, formula="SpikeCount ~ C(MonkeyName)", neuron_col="NeuronID"):
    results = []
    unique_neurons = df[neuron_col].unique()

    for neuron in tqdm(unique_neurons, desc="Running GLM per neuron"):
        neuron_df = df[df[neuron_col] == neuron]

        # Skip neurons with too few spikes
        if neuron_df['SpikeCount'].sum() == 0:
            continue
        try:
            model = smf.glm(formula=formula, data=neuron_df, family=sm.families.Poisson()).fit()
            summary = model.summary2().tables[1].reset_index()
            summary['NeuronID'] = neuron
            results.append(summary)
        except Exception as e:
            print(f"Error processing neuron {neuron}: {e}")
            continue

    # Combine results
    if results:
        results_df = pd.concat(results, ignore_index=True)
        # print("\nGLM Results:")
        # print(results_df)
        significant_df = results_df[results_df['P>|z|'] < 0.05]
        print("\nGLM Significant neurons:")
        print(significant_df.head())
        return results_df, significant_df
    else:
        print("No valid neurons found for GLM.")
        return pd.DataFrame(),pd.DataFrame()

def plot_glm_coefficients(glm_results):
    coef_df = glm_results[glm_results['index'] != '(Intercept)'].copy()

    # Clean predictor names
    coef_df['Predictor'] = coef_df['index'].str.replace('C\\(MonkeyName\\)\\[T\\.', '', regex=True)
    coef_df['Predictor'] = coef_df['Predictor'].str.replace(']', '')

    # Pivot for heatmap
    pivot_df = coef_df.pivot(index='NeuronID', columns='Predictor', values='Coef.')

    # Plot
    plt.figure(figsize=(12, max(6, len(pivot_df) * 0.3)))
    sns.heatmap(pivot_df, cmap='coolwarm', center=0, annot=False, cbar_kws={'label': 'Coefficient'})
    plt.title('GLM Coefficients (Stimulus Identity Effect per Neuron)')
    plt.xlabel('Stimulus Identity (MonkeyName)')
    plt.ylabel('Neuron ID')
    plt.tight_layout()
    plt.show()