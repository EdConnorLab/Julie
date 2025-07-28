import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tqdm import tqdm
from analyses.spike_rate import compute_mean_spike_rate_for_windows
import matplotlib.pyplot as plt
import seaborn as sns

'''
Linear Regression on using all types of behaviors at once (affiliation, agonism, submission)
-- Simple Linear Regression on entire social data 
-- Principal Component Regression (PCR)
'''

def combine_all_three_behavioral_matrices(monkeys_to_remove=[]):
    parent_dir = "/home/connorlab/Documents/GitHub/Julie/social_data/zombies_social_data/"
    aff_df = pd.read_excel(parent_dir + "zombies_feature_df_affiliation.xlsx", index_col=0)
    sub_df = pd.read_excel(parent_dir + "zombies_feature_df_submission.xlsx", index_col=0)
    ago_df = pd.read_excel(parent_dir + "zombies_feature_df_agonism.xlsx", index_col=0)

    # Clean column names: "Behavior Towards 94B" → "94B"
    def clean_col(colname):
        return colname.split()[-1]

    aff_df.columns = [clean_col(c) for c in aff_df.columns]
    sub_df.columns = [clean_col(c) for c in sub_df.columns]
    ago_df.columns = [clean_col(c) for c in ago_df.columns]

    # Remove specific monkeys (e.g., '7124') from both rows and columns
    for m in monkeys_to_remove:
        if m in aff_df.index:
            aff_df = aff_df.drop(index=m)
        if m in aff_df.columns:
            aff_df = aff_df.drop(columns=m)

        if m in sub_df.index:
            sub_df = sub_df.drop(index=m)
        if m in sub_df.columns:
            sub_df = sub_df.drop(columns=m)

        if m in ago_df.index:
            ago_df = ago_df.drop(index=m)
        if m in ago_df.columns:
            ago_df = ago_df.drop(columns=m)

    # Get common monkeys across all matrices
    monkeys = aff_df.index.intersection(sub_df.index).intersection(ago_df.index)

    # Build concatenated behavioral vectors
    X = []
    for m in monkeys:
        v = np.concatenate([
            aff_df.loc[m].values,
            sub_df.loc[m].values,
            ago_df.loc[m].values
        ])
        X.append(v)
    X = np.array(X)


    return X, monkeys


def run_multiple_regression(spike_df, feature_df, regression_type='ols', alpha=1.0, n_components=None, standardize=False, l1_ratio=0.5):

    results = []

    if n_components is None or regression_type == 'ols':
        X_full = feature_df.copy()
    elif regression_type == 'pcr':
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(feature_df.values)
        X_full = pd.DataFrame(X_pca, index=feature_df.index, columns=[f'PC{i+1}' for i in range(n_components)])
    else:
        X_full = feature_df.iloc[:, :n_components]

    if regression_type in ['lasso', 'ridge', 'elasticnet'] or standardize:
        scaler = StandardScaler()
        X_full = pd.DataFrame(
            scaler.fit_transform(X_full),
            index=X_full.index,
            columns=X_full.columns
        )

    grouped = spike_df.groupby(['NeuronID', 'WindowStart_ms', 'WindowEnd_ms'])

    for (neuron_id, w_start, w_end), group in tqdm(grouped, desc=f'Running {regression_type.upper()} regressions'):
        group = group.set_index('MonkeyName')
        monkeys_in_window = group.index.intersection(X_full.index)

        if len(monkeys_in_window) < (n_components or feature_df.shape[1]) + 2:
            print(
                f"[SKIP] Neuron {neuron_id} | Window {w_start}-{w_end} | "
                f"#Monkeys = {len(monkeys_in_window)},"
                f" #Predictors = {(n_components or feature_df.shape[1])}")
            continue

        y = group.loc[monkeys_in_window, 'MeanSpikeRate'].values
        X = X_full.loc[monkeys_in_window].values

        try:
            if regression_type == 'ols' or regression_type == 'pcr':
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                results.append({
                    'NeuronID': neuron_id,
                    'WindowStart_ms': w_start,
                    'WindowEnd_ms': w_end,
                    'R_squared': model.rsquared,
                    'p_values': model.pvalues.tolist(),
                    'coefficients': model.params.tolist()
                })
            elif regression_type == 'lasso':
                model = Lasso(alpha=alpha, max_iter=10000)
                model.fit(X, y)
                y_pred = model.predict(X)
                results.append({
                    'NeuronID': neuron_id,
                    'WindowStart_ms': w_start,
                    'WindowEnd_ms': w_end,
                    'R_squared': r2_score(y, y_pred),
                    'coefficients': model.coef_.tolist(),
                    'intercept': model.intercept_
                })
            elif regression_type == 'ridge':
                model = Ridge(alpha=alpha, max_iter=10000)
                model.fit(X, y)
                y_pred = model.predict(X)
                results.append({
                    'NeuronID': neuron_id,
                    'WindowStart_ms': w_start,
                    'WindowEnd_ms': w_end,
                    'R_squared': r2_score(y, y_pred),
                    'coefficients': model.coef_.tolist(),
                    'intercept': model.intercept_
                })
            elif regression_type == 'elasticnet':
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)  # l1_ratio can be tuned
                model.fit(X, y)
                y_pred = model.predict(X)
                results.append({
                    'NeuronID': neuron_id,
                    'WindowStart_ms': w_start,
                    'WindowEnd_ms': w_end,
                    'R_squared': r2_score(y, y_pred),
                    'coefficients': model.coef_.tolist(),
                    'intercept': model.intercept_
                })
        except Exception as e:
            print(f"{neuron_id} ({w_start}-{w_end}) failed: {e}")
            continue

    return pd.DataFrame(results)

def run_simple_regression(spike_df, feature_df, n_components=None, standardize=False):
    return run_multiple_regression(spike_df, feature_df, 'ols', n_components=n_components, standardize=standardize)


def run_lasso_regression(spike_df, feature_df, alpha=1.0, n_components=None):
    return run_multiple_regression(spike_df, feature_df, 'lasso', alpha=alpha, n_components=n_components)


def run_ridge_regression(spike_df, feature_df, alpha=1.0, n_components=None):
    return run_multiple_regression(spike_df, feature_df, 'ridge', alpha=alpha, n_components=n_components)


def run_pc_regression(spike_df, feature_df, n_components=3):
    return run_multiple_regression(spike_df, feature_df, 'pcr', n_components=n_components)


def run_elasticnet_regression(spike_df, feature_df, alpha=1.0, n_components=None, l1_ratio=0.5):
    return run_multiple_regression(spike_df, feature_df,
                                   regression_type='elasticnet',
                                   alpha=alpha,
                                   n_components=n_components,
                                   l1_ratio=l1_ratio)


if __name__ == '__main__':
    X, monkey_list = combine_all_three_behavioral_matrices()
    monkey_group = "Zombies"
    subject_monkey_index = 6
    sig_windows = pd.read_pickle(
        f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_cache/{monkey_group}_significant_windows_pANOVAorGLM_passed.pkl')
    ed_sig_windows = pd.read_pickle(
        '/home/connorlab/Documents/GitHub/Julie/Cortana/Ed and ANOVA/used_for_R01/R01_ed_list_duplicate_removed_without_full_window.pkl')
    mean_spike_rate_windows = compute_mean_spike_rate_for_windows(ed_sig_windows)
    spike_rate = mean_spike_rate_windows[mean_spike_rate_windows['MonkeyGroup'] == 'Zombies']
    X_df = pd.DataFrame(X, index=monkey_list)
    X_df_clean = X_df.drop(index="81G")
    results_df = run_lasso_regression(spike_rate, X_df_clean)
    # print(results_df)

