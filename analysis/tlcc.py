import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .common import parse_vectors


# computing Time-Lagged Cross-Correlation (TLCC) using numpy
def compute_tlcc_using_numpy(seq1, seq2, max_lag=100, separate_dimensions=False):
    """
    Compute Time-Lagged Cross-Correlation (TLCC) using numpy's correlate.

    Parameters:
    - separate_dimensions: bool, if True, calculates for each dimension separately (X, Y, Z).
    """
    num_lags = 2 * max_lag + 1
    lags = np.arange(-max_lag, max_lag + 1)

    if separate_dimensions:
        # calculate each dimension separately (X, Y, Z)
        correlations = np.zeros((num_lags, 3))  # For 3 dimensions

        for dim in range(3):
            full_corr = np.correlate(seq1[:, dim], seq2[:, dim], mode='full')
            mid_point = len(full_corr) // 2

            # Check if full_corr can accommodate the desired number of lags
            if len(full_corr) < num_lags:
                print(f"compute_tlcc_using_numpy: sequence too short for the given max_lag. full_corr length: {len(full_corr)}")
                return None, None

            correlations[:, dim] = full_corr[mid_point - max_lag:mid_point + max_lag + 1] / np.max(full_corr)

    else:
        # unified 3D vector
        seq1_flattened = np.linalg.norm(seq1, axis=1)
        seq2_flattened = np.linalg.norm(seq2, axis=1)
        full_corr = np.correlate(seq1_flattened, seq2_flattened, mode='full')
        mid_point = len(full_corr) // 2

        # Check if full_corr can accommodate the desired number of lags
        if len(full_corr) < num_lags:
            print(f"compute_tlcc_using_numpy: sequence too short for the given max_lag. full_corr length: {len(full_corr)}")
            return None, None

        correlations = full_corr[mid_point - max_lag:mid_point + max_lag + 1] / np.max(full_corr)
        correlations = correlations[:, np.newaxis]  # Make 2D array for compatibility

    return lags, correlations


# TLCC analysis
def tlcc_analyze(df_target1, df_target2, body_part, max_lag=100, normalize=False, separate_dimensions=False,
                 verbose_results=False):
    """
    Analyzes two sequences using Time-Lagged Cross-Correlation (TLCC).
    Returns the lags, correlation values, and the full data arrays.

    Parameters:
    - separate_dimensions: bool, if True, calculates for each dimension separately (X, Y, Z).
    """

    if df_target1 is None or df_target1.empty or df_target2 is None or df_target2.empty:
        print("tlcc_analyze: provided target DataFrame is None or empty.")
        return None, None, None, None

    target1_data = parse_vectors(df_target1, body_part, normalize=normalize)
    target2_data = parse_vectors(df_target2, body_part, normalize=normalize)

    target1_segments = target1_data[body_part]
    target2_segments = target2_data[body_part]

    # TLCC compute
    lags, correlations = compute_tlcc_using_numpy(target1_segments, target2_segments, max_lag=max_lag,
                                                  separate_dimensions=separate_dimensions)

    if verbose_results:
        print(f'Lags: {lags}')
        print(f'Correlation: {correlations}')

    return lags, correlations, target1_data, target2_data


# visualization function
def tlcc_visualize(lags, correlations, body_part, separate_dimensions=False,
                   title='Time-Lagged Cross-Correlation (TLCC)',
                   show_visualization=False, save_visualization=False, path_to_save=None, prefix=''):
    """
    Visualizes the Time-Lagged Cross-Correlation (TLCC).

    If separate_dimensions is True, plots each dimension (X, Y, Z) separately.
    If False, plots the unified vector correlation.
    """
    # checking parameters
    if lags is None or correlations is None:
        print(f'tlcc_visualize: no lags or no correlations to plot. parameter value is None')
        return
    # checking if the length of lags matches the number of correlations
    if separate_dimensions:
        if correlations.shape[0] != len(lags):
            print(f"tlcc_visualize: the length of lags ({len(lags)}) does not match the number of correlation points ({correlations.shape[0]}) for separate dimensions.")
            return
    else:
        if len(correlations) != len(lags):
            print(f"tlcc_visualize: the length of lags ({len(lags)}) does not match the number of correlation points ({len(correlations)}) for unified vector.")
            return

    if separate_dimensions:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        dimensions = ['X', 'Y', 'Z']
        colors = ['blue', 'orange', 'green']

        for dim in range(3):
            ax = axes[dim]
            correlation = correlations[:, dim]
            peak_idx = np.argmax(correlation)
            offset = lags[peak_idx]

            # plot correlation for the current dimension
            ax.plot(lags, correlation, color=colors[dim], label=f'Dimension {dimensions[dim]}')
            ax.axvline(0, color='black', linestyle='--', label='Center (Lag 0)')
            ax.axvline(lags[peak_idx], color='red', linestyle='--', label=f'Peak synchrony: Offset = {offset}')
            ax.set_title(f'TLCC for Dimension {dimensions[dim]}')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Pearson r')
            ax.grid(True)
            ax.legend()

        plt.suptitle(f'{title}\nTime-Lagged Cross-Correlation for {body_part}', fontsize=18)
    else:
        # unified vector correlation
        fig, ax = plt.subplots(figsize=(12, 5))
        correlation = correlations[:, 0]  # first (and only) column for unified correlation
        peak_idx = np.argmax(correlation)
        offset = lags[peak_idx]

        # plot correlation for the unified vector
        ax.plot(lags, correlation, color='blue', label='Unified Vector')
        ax.axvline(0, color='black', linestyle='--', label='Center (Lag 0)')
        ax.axvline(lags[peak_idx], color='red', linestyle='--', label=f'Peak synchrony: Offset = {offset}')
        ax.set_title(f'{title}\nUnified Vector Correlation for {body_part}')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Pearson r')
        ax.grid(True)
        ax.legend()

    if save_visualization and path_to_save is not None:
        plt.savefig(f'{path_to_save}/tlcc_visualization_{prefix}.png')

    if show_visualization:
        plt.show()

    plt.close()


# if __name__ == "__main__":
#     df_target1 = pd.read_excel('./images/work_files/pair_target_id_0_vectorframe_data.xlsx')
#     df_target2 = pd.read_excel('./images/work_files/pair_target_id_1_vectorframe_data.xlsx')

#     body_part = 'ARM_RIGHT'
#
#     lags, correlations, target1_data, target2_data = tlcc_analyze(df_target1, df_target2, body_part, max_lag=5,
#                                                                   normalize=False, verbose_results=False)
#
#     tlcc_visualize(lags, correlations, 'ARM_RIGHT', separate_dimensions=False)
#
#     lags, correlations, target1_data, target2_data = tlcc_analyze(
#         df_target1, df_target2, body_part='ARM_RIGHT', max_lag=5, normalize=False, separate_dimensions=True
#     )
#     tlcc_visualize(lags, correlations, 'ARM_RIGHT', separate_dimensions=True)


