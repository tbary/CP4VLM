import pandas as pd
import numpy as np
import os
import argparse
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from setup import *

def find_closest_file(folder_path):
    pattern = re.compile(r"results_(?:[^_]*)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv\.gz")
    current_time = datetime.now()
    closest_file = None
    closest_time_diff = float("inf")

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            file_datetime = datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
            time_diff = abs((file_datetime - current_time).total_seconds())
            if time_diff < closest_time_diff:
                closest_time_diff = time_diff
                closest_file = filename

    return closest_file

def save_figure(fig_number:int):
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(os.path.join(PATH_TO_FIGURES, f'figure_{fig_number}_{timestamp}.pdf'))
    if args.show:
        plt.show()

def compute_min_quantiles(results_dataframe:pd.DataFrame):
    df_avg = results_dataframe.groupby(["temperature", "alpha"], as_index=False)[["quantile", "set_sizes"]].mean()
    data_min_quantiles = df_avg.loc[df_avg.groupby("alpha")["quantile"].idxmin(), ["alpha", "quantile", "temperature"]]
    return data_min_quantiles

def add_shaded_area(ax, q, line_idx, color):
    y_values = ax.lines[line_idx].get_ydata()  # Extract KDE values from the plotted line
    x_kde = ax.lines[line_idx].get_xdata()  # Corresponding x values

    # Mask to keep only the right tail
    mask = x_kde >= q

    # Fill the tail region
    ax.fill_between(x_kde[mask], y_values[mask], alpha=0.3, color=color)

def generate_fig_1(results_dataframe:pd.DataFrame):
    alpha_keep = [0.01, 0.03]
    baseline_temp = 100
    optimized_temps = compute_min_quantiles(results_dataframe)

    fig, axs = plt.subplots(nrows=2, figsize = (5,7), sharex=True)
    cmap = sns.color_palette("flare", as_cmap=True)
    colors = cmap(np.linspace(0,1,len(alpha_keep)))

    for a, alpha in enumerate(alpha_keep):
        opt_temp = optimized_temps.loc[optimized_temps["alpha"]==alpha, "temperature"].iat[0]
        baseline_distr = results_dataframe.query("alpha == @alpha and temperature == @baseline_temp")["set_sizes"].to_numpy()
        opt_distr = results_dataframe.query("alpha == @alpha and temperature == @opt_temp")["set_sizes"].to_numpy()

        sns.kdeplot(baseline_distr, fill=False, bw_adjust=2, color=colors[a], clip=(0, np.inf), ax=axs[0], label=rf'$\alpha = {alpha}$', linewidth=1)
        q = np.quantile(baseline_distr,0.9)
        add_shaded_area(axs[0], q, a, colors[a])
        
        sns.kdeplot(opt_distr, fill=False, bw_adjust=2, color=colors[a], clip=(0, np.inf), ax=axs[1], linewidth=1)
        q = np.quantile(opt_distr,0.9)
        add_shaded_area(axs[1], q, a, colors[a])
    axs[0].grid()
    axs[1].grid()
    axs[0].legend()
    axs[0].set_ylabel("Baseline\nDensity")
    axs[1].set_ylabel("Temperature tuning\nDensity")
    axs[1].set_xlabel("Number of classes in the conformal set")
    save_figure(1)

def generate_fig_2(results_dataframe:pd.DataFrame):
    alpha_range = np.sort(results_dataframe['alpha'].unique())
    
    data_min_quantiles = compute_min_quantiles(results_dataframe)
    
    fig, ax = plt.subplots()

    # Create a colormap for the 'alpha' variable
    cmap = sns.color_palette("crest", as_cmap=True)
    norm = mcolors.Normalize(vmin=np.min(alpha_range), vmax=np.max(alpha_range))

    sns.lineplot(data=results_dataframe, x="temperature", y="quantile", hue="alpha", ax=ax,  legend=None, palette=cmap, errorbar=None)
    sns.scatterplot(data=data_min_quantiles, x="temperature", y="quantile", ax=ax, s=20, color='red', zorder=10, label='Minimum')
    
    # Add the colormap to the plot
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable; data isn't needed here
    cbar_ax = inset_axes(
        ax,
        width="3%",  # Change this to adjust the width (percentage of axis width)
        height="30%",  # Change this to adjust the height (percentage of axis height)
        loc='lower left',  # Position: 'upper right', 'lower left', etc.
        borderpad=0.75  # Adjust spacing between plot and colorbar
    )
    cbar = plt.colorbar(sm, ax=ax, cax=cbar_ax, orientation='vertical')
    cbar.set_label(r'$\alpha$', rotation=0, labelpad = -29)

    cbar.set_ticks([np.min(alpha_range), np.max(alpha_range)])
    cbar.set_ticklabels([np.min(alpha_range), np.max(alpha_range)], fontsize = 8)
    cbar.ax.get_yaxis().label.set_position((0, 1.15))

    ax.grid()
    ax.set_title(r"Evolution of $\hat{q}$, the value of the $(1-\alpha)$ quantile with $1/\tau$")
    ax.set_xlabel(r"$1/\tau$")
    ax.set_ylabel(r"$\hat{q}$")
    ax.legend(loc='lower right')

    save_figure(2)

def generate_fig_3(results_dataframe:pd.DataFrame):
    alpha_range = np.sort(results_dataframe['alpha'].unique())
    filtered_bins = [4,5,6]
    quantiles = [0.9, 0.95, 0.975]
    filt_results_dataframe = results_dataframe[~results_dataframe['alpha'].isin(alpha_range[filtered_bins])]
    data_tail_set_sizes = filt_results_dataframe.groupby(["temperature", "alpha", "fold"])["set_sizes"].quantile(quantiles).unstack().reset_index()

    estimated_min_temps = compute_min_quantiles(filt_results_dataframe)
    estimated_minmax = pd.merge(data_tail_set_sizes, estimated_min_temps, on=['temperature', 'alpha']).groupby(['temperature', 'alpha']).mean().reset_index()
    real_minmax = pd.DataFrame()
    mean_data_tail_set_sizes = data_tail_set_sizes.groupby(["alpha", "temperature"]).mean().reset_index()
    for q in quantiles:
        real_minmax[q] = mean_data_tail_set_sizes.groupby("alpha")[q].min()
        real_minmax[f'temp_{q}'] = mean_data_tail_set_sizes.loc[mean_data_tail_set_sizes.groupby("alpha")[q].idxmin(), ["temperature"]].to_numpy()

    cmap = sns.color_palette("flare", as_cmap=True)

    fig, axs = plt.subplots(ncols=len(quantiles), figsize=(len(quantiles)*4,3), sharey=True)

    for q, quant in enumerate(quantiles):
        sns.lineplot(data=data_tail_set_sizes, x="temperature", y=quant, hue="alpha", legend=None, ax=axs[q])
        sns.scatterplot(data=estimated_minmax, x="temperature", y=quant, s=20, color='red', zorder=12, ax=axs[q], label=r'$1/\tau_{*}$' if q==0 else "")
        sns.scatterplot(data=real_minmax,  x=f"temp_{quant}", y=quant, s=20, color='green', zorder=10, ax=axs[q], label=r'$1/\tau_{opt}$' if q==0 else "")
        axs[q].grid()
        axs[q].set_xlabel(r'$1/\tau$', fontsize=12)
        axs[q].set_ylabel('Tail set size', fontsize=12)
        axs[q].set_title(f"{quant}-quantile set size")
    
    axs[0].legend(loc='upper right')

    norm = mcolors.Normalize(vmin=np.min(alpha_range), vmax=np.max(alpha_range))

    # Add the colormap to the plot
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable; data isn't needed here
    cbar_ax = inset_axes(
        axs[0],
        width="4%",  # Change this to adjust the width (percentage of axis width)
        height="30%",  # Change this to adjust the height (percentage of axis height)
        loc='upper left',  # Position: 'upper right', 'lower left', etc.
        borderpad=1.75  # Adjust spacing between plot and colorbar
    )
    cbar = plt.colorbar(sm, ax=axs[0], cax=cbar_ax, orientation='vertical')
    cbar.set_label(r'$\alpha$', rotation=0, labelpad = -30)
    cbar.set_ticks([np.min(alpha_range), np.max(alpha_range)])
    cbar.set_ticklabels([np.min(alpha_range), np.max(alpha_range)], fontsize = 8)
    cbar.ax.get_yaxis().label.set_position((0, 1.25))

    save_figure(3)

def generate_fig_5(results_dataframe:pd.DataFrame):
    alpha_range = np.sort(results_dataframe['alpha'].unique())
    filtered_bins = [4,5,6]
    filt_results_dataframe = results_dataframe[~results_dataframe['alpha'].isin(alpha_range[filtered_bins])]
    data_tail_set_sizes = filt_results_dataframe.groupby(["temperature", "alpha", "fold"])["set_sizes"].mean().reset_index()

    estimated_min_temps = compute_min_quantiles(filt_results_dataframe)
    estimated_minmax = pd.merge(data_tail_set_sizes, estimated_min_temps, on=['temperature', 'alpha']).groupby(['temperature', 'alpha']).mean().reset_index()

    cmap = sns.color_palette("flare", as_cmap=True) 
    fig, ax = plt.subplots(figsize=(5,4))

    sns.lineplot(data=data_tail_set_sizes, x="temperature", y="set_sizes", hue="alpha", legend=None, ax=ax)
    sns.scatterplot(data=estimated_minmax, x="temperature", y="set_sizes", s=20, color='red', zorder=12, ax=ax, label=r'$1/\tau_{*}$')

    ax.grid()
    ax.legend(loc='upper right')
    ax.set_xlabel(r'$1/\tau$', fontsize=12)
    ax.set_ylabel('Average set size', fontsize=12)

    norm = mcolors.Normalize(vmin=np.min(alpha_range), vmax=np.max(alpha_range))

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable; data isn't needed here
    cbar_ax = inset_axes(
        ax,
        width="4%",  # Change this to adjust the width (percentage of axis width)
        height="20%",  # Change this to adjust the height (percentage of axis height)
        loc='upper center',  # Position: 'upper right', 'lower left', etc.
        borderpad=1.25  # Adjust spacing between plot and colorbar
    )
    cbar = plt.colorbar(sm, ax=ax, cax=cbar_ax, orientation='vertical')
    cbar.set_label(r'$\alpha$', rotation=0, labelpad = -30)
    cbar.set_ticks([np.min(alpha_range), np.max(alpha_range)])
    cbar.set_ticklabels([np.min(alpha_range), np.max(alpha_range)], fontsize = 8)
    cbar.ax.get_yaxis().label.set_position((0, 1.2))

    save_figure(5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replicating the paper figures.")
    parser.add_argument("--results_file", default="last", type=str, help="The name of the result file. 'last' will fetch the last registered file in the results folder.")
    parser.add_argument("--show", default=True, type=bool, help="Whether to show the figures before saving them. Default to True.")
    parser.add_argument("--figures", default=[1,2,3,5], nargs='+', type=int, help="The figures from the article to generate, adapted to one dataset. Only from figures 1, 2, 4, and 5. Other figures are irrelevant with a single dataset or model.")
    args = parser.parse_args()

    if args.results_file=="last":
        args.results_file = find_closest_file(PATH_TO_RESULTS)

    dtypes = {
        'fold': 'int32',
        'temperature': 'int32',
        'alpha': 'float32',
        'quantile': 'float32',
        'set_sizes': 'int32',
    }

    print("Loading experiment CSV...", end="", flush=True)
    results_dataframe = pd.read_csv(os.path.join(PATH_TO_RESULTS, args.results_file), compression='gzip', index_col=0, dtype=dtypes)
    print(" Done!")

    if 1 in args.figures:
        print("Generating Fig. 1...", end="", flush=True)
        generate_fig_1(results_dataframe)
        print(" Done!")
    if 2 in args.figures:
        print("Generating Fig. 2...", end="", flush=True)
        generate_fig_2(results_dataframe)
        print(" Done!")
    if 3 in args.figures:
        print("Generating Fig. 3...", end="", flush=True)
        generate_fig_3(results_dataframe)
        print(" Done!")
    if 5 in args.figures:
        print("Generating Fig. 5...", end="", flush=True)
        generate_fig_5(results_dataframe)
        print(" Done!")
