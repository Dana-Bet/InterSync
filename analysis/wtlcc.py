import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .common import parse_vectors


# Windowed TLCC for 3 dimensions (X, Y, Z) or unified vector
def compute_windowed_tlcc(target1_segments, target2_segments, max_lag, no_splits, separate_dimensions=False):
    """
    Computes Windowed Time-Lagged Cross-Correlation (TLCC) for each axis (X, Y, Z) or for the unified vector.

    Parameters:
    - target1_segments: np.ndarray
        The vector data for target 1.
    - target2_segments: np.ndarray
        The vector data for target 2.
    - max_lag: int
        The maximum lag (in vector segments) to compute cross-correlation.
    - no_splits: int
        Number of windows or splits to divide the data into.
    - separate_dimensions: bool
        If True, compute TLCC separately for each dimension (X, Y, Z). If False, compute for the unified vector.

    Returns:
    - lags: np.ndarray
        Array of lags (from -max_lag to +max_lag).
    - correlations: np.ndarray
        Array of correlations with shape (no_splits, num_lags, 3) for each axis (X, Y, Z) if `separate_dimensions=True`.
        Array of correlations with shape (no_splits, num_lags) for unified vector if `separate_dimensions=False`.
    """

    # check if the input arrays are valid
    if target1_segments is None or target2_segments is None:
        print("compute_windowed_tlcc: target1_segments or target2_segments is None.")
        return None, None

    if len(target1_segments) == 0 or len(target2_segments) == 0:
        print("compute_windowed_tlcc: target1_segments or target2_segments is empty.")
        return None, None

    # check if no_splits is greater than the number of samples
    segment_length = len(target1_segments)
    if no_splits > segment_length:
        print(f"compute_windowed_tlcc: no_splits ({no_splits}) exceeds segment length ({segment_length}).")
        return None, None

    segment_length = len(target1_segments)
    samples_per_split = segment_length // no_splits

    if separate_dimensions:
        correlations = np.zeros((no_splits, 2 * max_lag + 1, 3))  # array for X, Y, Z
    else:
        correlations = np.zeros((no_splits, 2 * max_lag + 1))  # array for unified vector

    for t in range(no_splits):
        start_idx = t * samples_per_split
        end_idx = (t + 1) * samples_per_split

        # Make sure the segment is not empty
        if start_idx >= len(target1_segments) or start_idx >= len(target2_segments):
            print(f"compute_windowed_tlcc: Skipping window {t} due to out-of-bounds indices.")
            continue

        d1 = np.linalg.norm(target1_segments[start_idx:end_idx], axis=1)
        d2 = np.linalg.norm(target2_segments[start_idx:end_idx], axis=1)

        # Check if d1 or d2 is empty or has mismatched dimensions
        if len(d1) == 0 or len(d2) == 0 or len(d1) != len(d2):
            print(f"compute_windowed_tlcc: Skipping window {t} due to mismatched or empty arrays.")
            continue

        # TLCC for the unified vector (Euclidean norm)
        rs = [np.corrcoef(np.roll(d1, lag), d2)[0, 1] for lag in range(-max_lag, max_lag + 1)]
        correlations[t, :] = rs

    lags = np.arange(-max_lag, max_lag + 1)
    return lags, correlations


# rolling window TLCC for 3 dimensions (X, Y, Z) or unified vector
def compute_rolling_window_tlcc(target1_segments, target2_segments, max_lag, window_size, step_size,
                                separate_dimensions=False):
    """
    Computes Rolling Window Time-Lagged Cross-Correlation (TLCC) for each axis (X, Y, Z) or for the unified vector.

    Parameters:
    - target1_segments: np.ndarray
        The vector data for target 1.
    - target2_segments: np.ndarray
        The vector data for target 2.
    - max_lag: int
        The maximum lag (in vector segments) to compute cross-correlation.
    - window_size: int
        The size of each rolling window in terms of number of vector segments.
    - step_size: int
        The step size to shift the rolling window.
    - separate_dimensions: bool
        If True, compute TLCC separately for each dimension (X, Y, Z). If False, compute for the unified vector.

    Returns:
    - lags: np.ndarray
        Array of lags (from -max_lag to +max_lag).
    - correlations: np.ndarray
        Array of correlations with shape (num_windows, num_lags, 3) for each axis (X, Y, Z) if `separate_dimensions=True`.
        Array of correlations with shape (num_windows, num_lags) for unified vector if `separate_dimensions=False`.
    """
    # check if the input arrays are valid
    if target1_segments is None or target2_segments is None:
        print("compute_rolling_window_tlcc: target1_segments or target2_segments is None.")
        return None, None

    if len(target1_segments) == 0 or len(target2_segments) == 0:
        print("compute_rolling_window_tlcc: target1_segments or target2_segments is empty.")
        return None, None

    # check if window size and step size are valid
    if window_size > len(target1_segments) or window_size > len(target2_segments):
        print(f"compute_rolling_window_tlcc: window_size ({window_size}) exceeds segment length.")
        return None, None

    if step_size <= 0:
        print(f"compute_rolling_window_tlcc: step_size ({step_size}) must be greater than 0.")
        return None, None

    t_start = 0
    t_end = window_size
    num_windows = (len(target1_segments) - window_size) // step_size + 1

    if num_windows <= 0:
        print(f"compute_rolling_window_tlcc: the number of windows ({num_windows}) is less than or equal to 0. check the window_size and step_size.")
        return None, None

    if separate_dimensions:
        correlations = np.zeros((num_windows, 2 * max_lag + 1, 3))  # array for X, Y, Z
    else:
        correlations = np.zeros((num_windows, 2 * max_lag + 1))  # array for unified vector

    for idx in range(num_windows):
        if separate_dimensions:
            for dim in range(3):
                d1 = target1_segments[t_start:t_end, dim]
                d2 = target2_segments[t_start:t_end, dim]

                if len(d1) == 0 or len(d2) == 0 or np.isnan(d1).any() or np.isnan(d2).any():
                    print(
                        f"compute_rolling_window_tlcc: skipping window {idx} due to NaN or empty data in dimension {dim}.")
                    continue

                rs = []
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        slice_d1, slice_d2 = d1[:lag], d2[-lag:]
                    else:
                        slice_d1, slice_d2 = d1[lag:], d2[:len(d2) - lag]

                    # Ensure both slices are of equal length and have enough data points
                    min_len = min(len(slice_d1), len(slice_d2))
                    if min_len > 1:
                        rs.append(np.corrcoef(slice_d1[:min_len], slice_d2[:min_len])[0, 1])
                    else:
                        rs.append(np.nan)

                correlations[idx, :, dim] = rs
        else:
            # TLCC for the unified vector (Euclidean norm)
            d1 = np.linalg.norm(target1_segments[t_start:t_end], axis=1)
            d2 = np.linalg.norm(target2_segments[t_start:t_end], axis=1)

            if len(d1) == 0 or len(d2) == 0 or np.isnan(d1).any() or np.isnan(d2).any():
                print(f"compute_rolling_window_tlcc: skipping window {idx} due to NaN or empty data in unified vector.")
                continue

            rs = []
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    slice_d1, slice_d2 = d1[:lag], d2[-lag:]
                else:
                    slice_d1, slice_d2 = d1[lag:], d2[:len(d2) - lag]

                # Ensure both slices are of equal length and have enough data points
                min_len = min(len(slice_d1), len(slice_d2))
                if min_len > 1:
                    rs.append(np.corrcoef(slice_d1[:min_len], slice_d2[:min_len])[0, 1])
                else:
                    rs.append(np.nan)

            correlations[idx, :] = rs

        # Move window
        t_start += step_size
        t_end += step_size

    lags = np.arange(-max_lag, max_lag + 1)

    # Check if all correlation values are NaN
    if np.isnan(correlations).all():
        print("compute_rolling_window_tlcc: all calculated correlations are NaN.")
        return None, None

    return lags, correlations


# visualization function for Windowed TLCC
def wtlcc_visualize_windowed_tlcc(lags, correlations, no_splits, separate_dimensions=False, title="Windowed TLCC",
                                  show_visualization=False, save_visualization=False, path_to_save=None, prefix=''):
    """
    Visualizes the Windowed TLCC result using a heatmap for each axis (X, Y, Z) or for the unified vector.

    Parameters:
    - lags: np.ndarray
        Array of lags (from -max_lag to +max_lag).
    - correlations: np.ndarray
        The array containing the windowed TLCC results.
    - no_splits: int
        Number of windows (splits) in the data.
    - separate_dimensions: bool
        If True, plot TLCC for each axis (X, Y, Z). If False, plot the unified vector correlation.
    """

    if lags is None or correlations is None:
        print(f'wtlcc_visualize_windowed_tlcc: no lags or no correlations to plot. parameter value is None')
        return

    # check for empty or all NaN correlations
    if correlations.size == 0 or np.isnan(correlations).all():
        print(f'wtlcc_visualize_windowed_tlcc: correlation matrix is empty or all NaN')
        return

    if separate_dimensions:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        dimensions = ['X', 'Y', 'Z']
        for dim in range(3):
            ax = axes[dim]
            sns.heatmap(correlations[:, :, dim], cmap='RdBu_r', ax=ax)
            ax.set(title=f'{title} - Dimension {dimensions[dim]}', xlabel='Lag', ylabel='Window Segments')
            ax.set_xticks(np.linspace(0, 2 * max(lags), num=7))
            ax.set_xticklabels(np.linspace(-max(lags), max(lags), num=7).astype(int))
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(correlations, cmap='RdBu_r', ax=ax)
        ax.set(title=f'{title} - Unified Vector', xlabel='Lag', ylabel='Window Segments')
        ax.set_xticks(np.linspace(0, 2 * max(lags), num=7))
        ax.set_xticklabels(np.linspace(-max(lags), max(lags), num=7).astype(int))

    if save_visualization and path_to_save is not None:
        plt.savefig(f'{path_to_save}/window_tlcc_{no_splits}splits_visualization_{prefix}.png')

    if show_visualization:
        plt.show()


# visualization function for Rolling Window TLCC
def wtlcc_visualize_rolling_window_tlcc(lags, correlations, separate_dimensions=False, title="Rolling Window TLCC",
                                        show_visualization=False, save_visualization=False, path_to_save=None, prefix=''):
    """
    Visualizes the Rolling Window TLCC result using a heatmap for each axis (X, Y, Z) or for the unified vector.

    Parameters:
    - lags: np.ndarray
        Array of lags
    - correlations: np.ndarray
        The array containing the rolling window TLCC results.
    - separate_dimensions: bool
        If True, plot TLCC for each axis (X, Y, Z). If False, plot the unified vector correlation.
    - title: str
        Title for the plot.
    """

    if lags is None or correlations is None:
        print(f'wtlcc_visualize_rolling_window_tlcc: no lags or no correlations to plot. parameter value is None')
        return

    if correlations.size == 0 or np.isnan(correlations).all():
        print(f'wtlcc_visualize_rolling_window_tlcc: correlation matrix is empty or all NaN')
        return

    if separate_dimensions:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        dimensions = ['X', 'Y', 'Z']
        for dim in range(3):
            ax = axes[dim]
            sns.heatmap(correlations[:, :, dim], cmap='RdBu_r', ax=ax)
            ax.set(title=f'{title} - Dimension {dimensions[dim]}', xlabel='Lag', ylabel='Window Segments')
            ax.set_xticks(np.linspace(0, 2 * max(lags), num=7))
            ax.set_xticklabels(np.linspace(-max(lags), max(lags), num=7).astype(int))
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(correlations, cmap='RdBu_r', ax=ax)
        ax.set(title=f'{title} - Unified Vector', xlabel='Lag', ylabel='Window Segments')
        ax.set_xticks(np.linspace(0, 2 * max(lags), num=7))
        ax.set_xticklabels(np.linspace(-max(lags), max(lags), num=7).astype(int))

    if save_visualization and path_to_save is not None:
        plt.savefig(f'{path_to_save}/rolling_window_tlcc_visualization_{prefix}.png')

    if show_visualization:
        plt.show()

    plt.close()


def wtlcc_analyze(df_target1, df_target2, body_part, normalize=False, max_lag=10, windowed=True, separate_dimensions=False,
                  no_splits=10, window_size=20, step_size=5, verbose_results=False):
    """
    Unifying method to compute either Windowed or Rolling Window Time-Lagged Cross-Correlation (TLCC).

    Parameters:
    - max_lag: int, optional
        The maximum lag (in vector segments) to compute cross-correlation (default is 100).
    - windowed: bool, optional
        If True, perform windowed TLCC. If False, perform rolling window TLCC (default is True).
    - separate_dimensions: bool, optional
        If True, compute TLCC separately for each dimension (X, Y, Z). If False, compute for unified vector (default is False).
    - no_splits: int, optional
        Number of windows for the windowed TLCC (only used if windowed=True).
    - window_size: int, optional
        Size of each rolling window for the rolling window TLCC (only used if windowed=False).
    - step_size: int, optional
        Step size for the rolling window TLCC (only used if windowed=False).
    - verbose_results: bool, optional
        If True, prints the lag and correlation results (default is False).

    Returns:
    - lags: np.ndarray
        Array of lags (from -max_lag to +max_lag).
    - correlations: np.ndarray
        The computed TLCC correlations.
    """

    if df_target1 is None or df_target1.empty or df_target2 is None or df_target2.empty:
        print("wtlcc_analyze: provided target DataFrame is None or empty.")
        return None, None

    target1_data = parse_vectors(df_target1, body_part, normalize=normalize)
    target2_data = parse_vectors(df_target2, body_part, normalize=normalize)

    # check if the input arrays are valid
    if target1_data is None or target2_data is None:
        print("wtlcc_analyze: target1_data or target2_data is None.")
        return None, None

    target1_segments = target1_data[body_part]
    target2_segments = target2_data[body_part]

    if windowed:
        # windowed TLCC
        lags, correlations = compute_windowed_tlcc(target1_segments, target2_segments, max_lag=max_lag,
                                                   no_splits=no_splits, separate_dimensions=separate_dimensions)
        if verbose_results:
            print(f'Windowed TLCC results:\nLags: {lags}\nCorrelations: {correlations}')

    else:
        # rolling window TLCC
        lags, correlations = compute_rolling_window_tlcc(target1_segments, target2_segments, max_lag=max_lag,
                                                         window_size=window_size, step_size=step_size,
                                                         separate_dimensions=separate_dimensions)
        if verbose_results:
            print(f'Rolling Window TLCC results:\nLags: {lags}\nCorrelations: {correlations}')

    return lags, correlations


# if __name__ == "__main__":
#     df_target1 = pd.read_excel('./images/work_files/pair_target_id_0_vectorframe_data.xlsx')
#     df_target2 = pd.read_excel('./images/work_files/pair_target_id_1_vectorframe_data.xlsx')
#
#     # Specify the column that contains vector data
#     body_part = 'ARM_RIGHT'  # Replace with the correct body part or data column name
#
#     # lags, correlations = wtlcc_analyze(
#     #     df_target1, df_target2, body_part, max_lag=5, windowed=True, separate_dimensions=True, no_splits=10
#     # )
#     # wtlcc_visualize_windowed_tlcc(lags, correlations, no_splits=10, separate_dimensions=True)
#     #
#     lags, correlations = wtlcc_analyze(
#         df_target1, df_target2, body_part, max_lag=10, windowed=True, separate_dimensions=False, no_splits=20
#     )
#     wtlcc_visualize_windowed_tlcc(lags, correlations, no_splits=10, separate_dimensions=False)
#     #
#     # lags, correlations = wtlcc_analyze(
#     #     df_target1, df_target2, body_part, max_lag=5, windowed=False, separate_dimensions=True, window_size=5, step_size=10
#     # )
#     # wtlcc_visualize_rolling_window_tlcc(lags, correlations, separate_dimensions=True)
#
#     lags, correlations = wtlcc_analyze(
#         df_target1, df_target2, body_part, max_lag=10, windowed=False, separate_dimensions=False, window_size=10,
#         step_size=5
#     )
#     wtlcc_visualize_rolling_window_tlcc(lags, correlations, separate_dimensions=False)


