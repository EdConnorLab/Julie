import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import seaborn as sns





def get_specific_monkey_behavior_pair_results_for_monkey_group(monkey_group: str, monkey_of_interest: str, behavior: str) -> pd.DataFrame:
    results = pd.read_pickle(
        f'/home/connorlab/Documents/GitHub/Julie/Cortana/analysis_results/{monkey_group}_dir_ols_on_single_neurons_with_xy_values.pkl')
    monkey_beh_pair_subset = results[(results['Behavior'] == behavior) & (results['Source_Monkey'] == monkey_of_interest)]
    return monkey_beh_pair_subset


def plot_specific_monkey_behavior_pair_for_specific_monkey_group(monkey_group: str, monkey_of_interest: str, behavior: str):
    subset = get_specific_monkey_behavior_pair_results_for_monkey_group(monkey_group, monkey_of_interest, behavior)
    # assuming each cell has arrays/lists in 'x' and 'y' columns
    X_all = pd.concat([pd.DataFrame(xi) for xi in subset['Design_Matrix']], ignore_index=True)
    y_all = pd.concat([pd.Series(yi) for yi in subset['Neural_Response']], ignore_index=True)

    # Fit OLS (x already includes constant)
    model = sm.OLS(y_all, X_all)
    fit = model.fit()
    # Predicted values
    y_pred = fit.predict(X_all)

    # Scatter + line
    plt.figure(figsize=(6, 5))
    plt.scatter(y_all, y_pred, alpha=0.6, s=25, label='Data')
    plt.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()],
             'r--', lw=2, label='Ideal fit (y=x)')
    plt.xlabel("Observed y (Neural response)")
    plt.ylabel("Predicted y (Model fit)")
    plt.title("OLS fit across neurons (AffiliationTo, 69X)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_results_filtered_based_on_r(mon_beh_subset: pd.DataFrame, r_cutoff: float):
    unique_beh = mon_beh_subset['Behavior'].unique()
    behavior = unique_beh[0] if len(unique_beh) > 0 else None
    unique_mon = mon_beh_subset['Source_Monkey'].unique()
    monkey_of_interest = unique_mon[0] if len(unique_mon) > 0 else None
    filtered_based_on_r = mon_beh_subset[mon_beh_subset['R-squared'] > r_cutoff]
    # assuming each cell has arrays/lists in 'x' and 'y' columns
    X_all = pd.concat([pd.DataFrame(xi) for xi in filtered_based_on_r['Design_Matrix']], ignore_index=True)
    y_all = pd.concat([pd.Series(yi) for yi in filtered_based_on_r['Neural_Response']], ignore_index=True)

    # Assuming x_all has one predictor column besides the constant
    x_vals = X_all.iloc[:, 1]
    y_vals = y_all
    # re-fit it with only the selected cells
    model = sm.OLS(y_all, X_all)
    fit = model.fit()

    # Predicted line
    x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
    X_plot = sm.add_constant(x_range) if 'const' not in X_all.columns else pd.DataFrame(
        {'const': 1, X_all.columns[1]: x_range})
    y_pred_line = fit.predict(X_plot)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(x_vals, y_vals, alpha=0.6, s=25, label='Data')
    plt.plot(x_range, y_pred_line, 'r-', lw=2, label='Fitted line')
    plt.xlabel(f"Behavior (Predictor)")
    plt.ylabel("y (Neural response)")
    plt.title(f"OLS fit across neurons ({behavior}, {monkey_of_interest})")
    # plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ols_fits_per_neuron(subset: pd.DataFrame, r_cutoff: float):
    filtered_based_on_r = subset[subset['R-squared'] > r_cutoff]
    # Create color palette (one color per neuron)
    colors = sns.color_palette("husl", len(filtered_based_on_r))

    plt.figure(figsize=(7, 6))

    for i, (x, y, r2, neuron) in enumerate(zip(filtered_based_on_r['Design_Matrix'],
                                               filtered_based_on_r['Neural_Response'],
                                               filtered_based_on_r['R-squared'],
                                               filtered_based_on_r['NeuronID'])):
        # Fit OLS for this neuron (x already includes constant)
        y_norm = (y - np.mean(y)) / np.std(y)
        model = sm.OLS(y_norm, x)
        fit = model.fit()

        # Predicted line (sorted by x to make the line smooth)
        order = np.argsort(x[:, 1])  # assumes first col = constant, second = predictor
        x_sorted = x[order, 1]
        y_pred = fit.predict(x[order])

        # Plot scatter and line
        plt.scatter(x[:, 1], y_norm, color=colors[i], alpha=0.4, s=20)
        plt.plot(x_sorted, y_pred, color=colors[i], label=f"{neuron}, r-sq: {r2:.3f}", lw=1.5)

    plt.xlabel("Predictor (e.g., AffiliationTo)")
    plt.ylabel("Neural response")
    plt.title("OLS fits per neuron")
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_global_ols_fit(subset: pd.DataFrame, r_cutoff: float):
    filtered_based_on_r = subset[subset['R-squared'] > r_cutoff]
    plt.figure(figsize=(7, 6))
    colors = sns.color_palette("husl", len(filtered_based_on_r))

    all_x = []
    all_y = []

    for i, (x, y, r2, neuron) in enumerate(zip(filtered_based_on_r['Design_Matrix'],
                                               filtered_based_on_r['Neural_Response'],
                                               filtered_based_on_r['R-squared'], filtered_based_on_r['NeuronID'])):
        # Normalize y per neuron (z-score)
        y_norm = (y - np.mean(y)) / np.std(y)
        # y_norm = y/float(np.max(y))
        # print(f"x: {x[:, 1]}")
        # print(f"y: {y_norm}")
        # Plot scatter for each neuron
        plt.scatter(x[:, 1], y_norm, color=colors[i], alpha=0.2, label=f"{neuron}: R² = {r2:.3f}")

        # Accumulate for global fit
        all_x.append(x)
        all_y.append(y_norm)

    # Combine all neurons into single regression
    X_all = np.vstack(all_x)
    y_all = np.concatenate(all_y)
    # print(y_all)
    # print(X_all)
    # Fit one global OLS (x already includes constant)
    model = sm.OLS(y_all, X_all)
    fit = model.fit()

    # Generate prediction line
    x_vals = np.linspace(X_all[:, 1].min(), X_all[:, 1].max(), 100)
    X_pred = np.column_stack([np.ones(len(x_vals)), x_vals])  # constant + predictor
    y_pred = fit.predict(X_pred)

    # Plot global fit
    plt.plot(x_vals, y_pred, 'k-', lw=2, label=f'R-sq: {fit.rsquared:.3f} p-val: {fit.pvalues[1]:.3f}')

    plt.xlabel("Behavioral Observation")
    plt.ylabel("Normalized Neural Response (z-score)")
    plt.title(f"Global OLS fit across neurons")
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    monkey_group = "Zombies"
    monkey_of_interest = "69X"
    behavior = "AffiliationTo"
    specific_monkey_pair_subset = get_specific_monkey_behavior_pair_results_for_monkey_group(monkey_group, monkey_of_interest, behavior)
    plot_specific_monkey_behavior_pair_for_specific_monkey_group(monkey_group, monkey_of_interest, behavior)
    plot_ols_fits_per_neuron(specific_monkey_pair_subset, 0.15)
    plot_results_filtered_based_on_r(specific_monkey_pair_subset, 0.15)
    plot_global_ols_fit(specific_monkey_pair_subset, 0.15)
