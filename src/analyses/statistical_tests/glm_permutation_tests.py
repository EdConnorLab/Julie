import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm  # progress bar
from fpdf import FPDF
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import os

from spike_count import prepare_combined_spike_data


class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Neuron Analysis Report', ln=True, align='C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 10, title, ln=True)

    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 8, text)
        self.ln()

    def add_image(self, image_path):
        self.image(image_path, w=180)
        self.ln()

def generate_neuron_reports(analysis_df, glm_results, perm_results, output_dir='neuron_pdf_reports', time_bin_size=0.05):
    os.makedirs(output_dir, exist_ok=True)

    # Merge glm + perm results
    summary_report = merge_glm_and_permutation(glm_results, perm_results)

    sig_neurons = summary_report[
        (summary_report['GLM_significant'] == True) | (summary_report['Permutation_significant'] == True)
    ]['NeuronID'].unique()

    for neuron in tqdm(sig_neurons, desc="Generating PDF reports"):
        neuron_df = analysis_df[analysis_df['NeuronID'] == neuron]

        # Prepare PSTH plot and save as PNG
        fig_path = os.path.join(output_dir, f'{neuron}_PSTH.png')
        plt.figure(figsize=(8, 4))
        sns.lineplot(
            data=neuron_df,
            x=neuron_df['TimeBinIndex'] * time_bin_size,
            y='SpikeCount',
            hue='MonkeyGroup',
            estimator='mean',
            ci='sd'
        )
        plt.title(f'Neuron: {neuron} PSTH')
        plt.xlabel('Time (s)')
        plt.ylabel('Spike Count')
        plt.legend(title='Stimulus Group')
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

        # Prepare GLM summary
        neuron_glm = glm_results[glm_results['NeuronID'] == neuron]
        glm_text = ""
        if not neuron_glm.empty:
            for _, row in neuron_glm.iterrows():
                glm_text += f"Predictor: {row['index']}, Coef: {row['Coef.']:.3f}, P: {row['P>|z|']:.3f}\n"
        else:
            glm_text = "No GLM results."

        # Prepare permutation summary
        neuron_perm = perm_results[perm_results['NeuronID'] == neuron]
        if not neuron_perm.empty:
            perm_row = neuron_perm.iloc[0]
            perm_text = f"Observed Difference: {perm_row['ObservedDifference']:.3f}, P-value: {perm_row['P-value']:.3f}"
        else:
            perm_text = "No permutation results."

        # Create PDF
        pdf = PDFReport()
        pdf.add_page()

        pdf.chapter_title(f'Neuron ID: {neuron}')
        pdf.chapter_title('GLM Summary:')
        pdf.chapter_body(glm_text)

        pdf.chapter_title('Permutation Test Summary:')
        pdf.chapter_body(perm_text)

        pdf.chapter_title('PSTH Plot:')
        pdf.add_image(fig_path)

        # Save PDF
        pdf.output(os.path.join(output_dir, f'{neuron}_Report.pdf'))

def run_glm(df, formula="SpikeCount ~ C(MonkeyName)", neuron_col="NeuronID"):
    results = []

    # unique neuron list
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
        return results_df
    else:
        print("No valid neurons found for GLM.")
        return pd.DataFrame()

def permutation_test(df, n_permutations=1000, neuron_col='NeuronID', group_col='MonkeyGroup',
                     count_col='SpikeCount'):
    results = []

    unique_neurons = df[neuron_col].unique()

    for neuron in tqdm(unique_neurons, desc="Running permutation test per neuron"):
        neuron_df = df[df[neuron_col] == neuron]

        # Get actual group means
        group_means = neuron_df.groupby(group_col)[count_col].mean()
        if group_means.shape[0] < 2:
            continue  # Skip if not enough groups

        # Actual observed difference (max group - min group mean)
        observed_diff = group_means.max() - group_means.min()

        # Permutation distribution
        perm_diffs = []
        for _ in range(n_permutations):
            shuffled = neuron_df.copy()
            shuffled[group_col] = np.random.permutation(shuffled[group_col].values)
            perm_group_means = shuffled.groupby(group_col)[count_col].mean()
            perm_diff = perm_group_means.max() - perm_group_means.min()
            perm_diffs.append(perm_diff)

        # Calculate p-value
        perm_diffs = np.array(perm_diffs)
        p_value = np.mean(perm_diffs >= observed_diff)

        results.append({
            'NeuronID': neuron,
            'ObservedDifference': observed_diff,
            'P-value': p_value
        })

    return pd.DataFrame(results)

def plot_permutation_results(perm_results):
    plt.figure(figsize=(12, 5))

    # Plot p-value distribution
    plt.subplot(1, 2, 1)
    sns.histplot(perm_results['P-value'], bins=20, kde=False)
    plt.title('Permutation Test P-value Distribution')
    plt.xlabel('P-value')
    plt.ylabel('Neuron Count')

    # Plot observed difference distribution
    plt.subplot(1, 2, 2)
    sns.histplot(perm_results['ObservedDifference'], bins=20, kde=False)
    plt.title('Observed Differences in Group Means')
    plt.xlabel('Observed Difference (Max group - Min group)')
    plt.ylabel('Neuron Count')

    plt.tight_layout()
    plt.show()


def merge_glm_and_permutation(glm_results, perm_results):
    # GLM 결과 neuron 별로 p-value 정리
    glm_summary = glm_results.groupby('NeuronID')['P>|z|'].min().reset_index()
    glm_summary.rename(columns={'P>|z|': 'GLM_P_value'}, inplace=True)

    # permutation 결과는 이미 neuron 별로 되어 있음
    merged = pd.merge(glm_summary, perm_results, on='NeuronID', how='outer')

    # Multiple comparison correction (optional)
    from statsmodels.stats.multitest import multipletests

    # GLM
    if not merged['GLM_P_value'].isnull().all():
        reject_glm, glm_pvals_corrected, _, _ = multipletests(merged['GLM_P_value'].fillna(1), method='fdr_bh')
        merged['GLM_pval_corrected'] = glm_pvals_corrected
        merged['GLM_significant'] = reject_glm
    else:
        merged['GLM_pval_corrected'] = None
        merged['GLM_significant'] = None

    # Permutation
    if not merged['P-value'].isnull().all():
        reject_perm, perm_pvals_corrected, _, _ = multipletests(merged['P-value'].fillna(1), method='fdr_bh')
        merged['Permutation_pval_corrected'] = perm_pvals_corrected
        merged['Permutation_significant'] = reject_perm
    else:
        merged['Permutation_pval_corrected'] = None
        merged['Permutation_significant'] = None

    return merged


def plot_glm_coefficients(glm_results):
    # Prepare data
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

def plot_significant_neuron_psth(analysis_df, summary_report, time_bin_size=0.05):
    # Filter significant neurons (either GLM or permutation)
    sig_neurons = summary_report[
        (summary_report['GLM_significant'] == True) | (summary_report['Permutation_significant'] == True)][
        'NeuronID'].unique()

    for neuron in sig_neurons:
        neuron_df = analysis_df[analysis_df['NeuronID'] == neuron]

        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=neuron_df,
            x=neuron_df['TimeBinIndex'] * time_bin_size,
            y='SpikeCount',
            hue='MonkeyGroup',
            estimator='mean',
            errorbar='sd'
        )

        plt.title(f'Neuron: {neuron} PSTH (Significant)')
        plt.xlabel('Time (s)')
        plt.ylabel('Spike Count')
        plt.legend(title='Stimulus Group')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    date = '2023-09-26'
    round_no = 1

    analysis_df = prepare_combined_spike_data(
        None, date, round_no, 0.05
    )
    # formula = "SpikeCount ~ C(MonkeyName)"  # Stimulus identity
    formula = "SpikeCount ~ C(MonkeyGroup)"  # Stimulus group 간 차이

    glm_results = run_glm(analysis_df, formula=formula)

    print(glm_results.head())
    # ---- multiple comparison correction ---
    # glm_results 에서 P-value 컬럼 선택
    pvals = glm_results['P>|z|']

    # correction 방법: 'fdr_bh' (False Discovery Rate, Benjamini/Hochberg)
    reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')

    # glm_results 에 corrected p-value 와 reject 여부 추가
    glm_results['pval_corrected'] = pvals_corrected
    glm_results['significant'] = reject
    glm_results.to_excel('glm_results.xlsx')
    plot_glm_coefficients(glm_results)

    # ---- permutation test
    perm_results = permutation_test(
        analysis_df,
        n_permutations=1000,
        neuron_col='NeuronID',
        group_col='MonkeyGroup',  # stimulus group
        count_col='SpikeCount'
    )

    print(perm_results.head())

    # ---- multiple comparison correction ---
    reject, pvals_corrected, _, _ = multipletests(perm_results['P-value'], method='fdr_bh')
    perm_results['pval_corrected'] = pvals_corrected
    perm_results['significant'] = reject
    plot_permutation_results(perm_results)

    # --- merge glm and permutation ---
    summary_report = merge_glm_and_permutation(glm_results, perm_results)
    plot_significant_neuron_psth(analysis_df, summary_report)



    ### Report

    generate_neuron_reports(
        analysis_df=analysis_df,
        glm_results=glm_results,
        perm_results=perm_results,
        output_dir='neuron_pdf_reports'
    )