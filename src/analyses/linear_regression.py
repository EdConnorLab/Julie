import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import zscore
from statsmodels.formula.api import mixedlm
import statsmodels.api as sm

from analyses.spike_count import prepare_exploded_spike_data
from analyses.spike_rate import compute_mean_spike_rate_table


##### ---- Single Neuron Level Analysis

def run_neuron_wise_spike_rate_regression(exploded_df, social_df, social_col='AffiliationTo_z', model_type='ols'):
    # Step 1: Compute mean spike rate (per neuron per monkey identity)
    mean_df = compute_mean_spike_rate_table(exploded_df)
    # Step 2: Merge with social score
    merged_df = pd.merge(mean_df, social_df[['MonkeyName', social_col]], on='MonkeyName')

    # Step 3: Per-neuron regression
    results = []
    for neuron in merged_df['NeuronID'].unique():
        neuron_df = merged_df[merged_df['NeuronID'] == neuron]
        if len(neuron_df) < 3:
            continue
        if model_type == 'ols':
            model = smf.ols(f"MeanSpikeRate ~ {social_col}", data=neuron_df).fit()
        elif model_type == 'glm':
            model = smf.glm(f"MeanSpikeRate ~ {social_col}", data=neuron_df,
                            family=sm.families.Poisson()).fit()
        else:
            raise ValueError("model_type must be 'ols' or 'glm'")

        results.append({
            'NeuronID': neuron,
            'R_squared': model.rsquared if model_type == 'ols' else None,
            'coef': model.params.get(social_col),
            'p_value': model.pvalues.get(social_col)
        })

    return pd.DataFrame(results)

# NOTE: Trial-level regression was removed.
# Justification: social predictors are constant across trials,
# causing collinearity and weak interpretability.
# Preferred alternatives:
#   - run_mean_rate_regression()
#   - run_population_glmm()

##### ---- Population Level Analysis

def run_stimulus_level_glmm(exploded_df, social_df, social_col='AffiliationTo_z'):
    # Step 1: Prepare spike count data
    df = exploded_df.copy()
    df['SpikeCount'] = df['SpikeTimes'].apply(len)

    # Collapse to one spike count per trial
    trial_spike_count_df = (
        df.groupby(['NeuronID', 'MonkeyName', 'TaskField'])['SpikeCount']
        .sum()
        .reset_index()
    )

    # Step 2: Merge with social score
    merged_df = pd.merge(trial_spike_count_df, social_df[['MonkeyName', social_col]], on='MonkeyName')
    model = mixedlm(f"SpikeCount ~ {social_col}", data=merged_df, groups=merged_df['MonkeyName'])
    result = model.fit()
    return result.summary()

def run_population_glmm(exploded_df, social_df, social_col='AffiliationTo_z'):
    # Step 1: Prepare spike count data
    df = exploded_df.copy()
    df['SpikeCount'] = df['SpikeTimes'].apply(len)

    # Collapse to one spike count per trial
    trial_spike_count_df = (
        df.groupby(['NeuronID', 'MonkeyName', 'TaskField'])['SpikeCount']
        .sum()
        .reset_index()
    )

    # Step 2: Merge with social score
    merged_df = pd.merge(trial_spike_count_df, social_df[['MonkeyName', social_col]], on='MonkeyName')
    model = mixedlm(f"SpikeCount ~ {social_col}", data=merged_df, groups=merged_df['NeuronID'])
    result = model.fit()
    return result.summary()

if __name__ == "__main__":
    date = '2023-09-26'
    round_no = 1
    exploded_df = prepare_exploded_spike_data(date, round_no, True, use_sorted=False)
    # trial_df = compute_trial_level_spike_rate(exploded_df)
    # mean_df = compute_mean_spike_rate(trial_df)
    # print(mean_df)
    file_name = '../social_data/zombies_social_data/zombies_feature_df_affiliation.xlsx'
    affiliation_df = pd.read_excel(file_name)

    # base_dir = '../social_data/zombies_social_data/'
    # zombies_affiliation_from_file_name = 'zombies_affiliation_from.csv'
    # zombies_submission_from_file_name = 'zombies_submission_from.csv'
    # zombies_agonism_from_file_name = 'zombies_agonism_from.csv'
    # zombies_affiliation_to_file_name = 'zombies_affiliation_to.csv'
    # zombies_submission_to_file_name = 'zombies_submission_to.csv'
    # zombies_agonism_to_file_name = 'zombies_agonism_to.csv'
    # zombies_aff_from = pd.read_csv(base_dir + zombies_affiliation_from_file_name)
    # zombies_aff_to = pd.read_csv(base_dir + zombies_affiliation_to_file_name)
    # zombies_sub_from = pd.read_csv(base_dir + zombies_submission_from_file_name)
    # zombies_sub_to = pd.read_csv(base_dir + zombies_submission_to_file_name)
    # zombies_agon_from = pd.read_csv(base_dir + zombies_agonism_from_file_name)
    # zombies_agon_to = pd.read_csv(base_dir + zombies_agonism_to_file_name)
    #
    #
    # df = run_trial_level_regression(
    #     date='2023-09-26',
    #     round_no=1,
    #     social_df=zombies_aff_to,
    #     social_col='AffiliationTo_z',
    #     model_type='glm'  # or 'glm'
    #
    # )
    # print(df)