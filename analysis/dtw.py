import pandas as pd
import numpy as np
from fastdtw import fastdtw
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean, cosine
from .common import METRIC_COSINE, METRIC_EUCLIDEAN, parse_vectors


def dtw_analyze_fastdtw(df_target1: pd.DataFrame, df_target2: pd.DataFrame, body_part, radius=1, normalize=False, smooth=False,
                        smooth_window_size=3, metric=METRIC_COSINE, verbose_results=False):
    """
    Analyzes two sequences using Dynamic Time Warping (DTW).
    Returns the DTW path, distance, and the full data arrays including timestamps.
    """

    if df_target1 is None or df_target1.empty or df_target2 is None or df_target2.empty:
        print("similarity_analyze_similar_movements: provided target DataFrame is None or empty.")
        return None, None, None, None

    target1_data = parse_vectors(df_target1, body_part, normalize=normalize)
    target2_data = parse_vectors(df_target2, body_part, normalize=normalize)

    target1_segments = target1_data[body_part]
    target2_segments = target2_data[body_part]

    if smooth:
        def smooth_func(data, window_size=smooth_window_size):
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        target1_segments = np.apply_along_axis(smooth_func, axis=0, arr=target1_segments)
        target2_segments = np.apply_along_axis(smooth_func, axis=0, arr=target2_segments)

    if metric == METRIC_EUCLIDEAN:
        distance, path = fastdtw(target1_segments, target2_segments, dist=euclidean, radius=radius)
    elif metric == METRIC_COSINE:
        distance, path = fastdtw(target1_segments, target2_segments, dist=cosine, radius=radius)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if verbose_results:
        print("DTW Path:", path)
        print("DTW Distance:", distance)

    return path, distance, target1_data, target2_data  # Return the full data with timestamps


def dtw_calculate_acc_cost_matrix(target1_data_from_analyze, target2_data_from_analyze, body_part, metric=METRIC_COSINE):
    """
    Calculates the accumulated cost matrix for two sequences using the specified distance metric.

    Parameters:
    - target1_segments: np.ndarray, sequence of vectors for target 1.
    - target2_segments: np.ndarray, sequence of vectors for target 2.
    - metric: str, distance metric to use ('euclidean' or 'cosine').

    Returns:
    - acc_cost_matrix: np.ndarray, accumulated cost matrix.
    """

    if target1_data_from_analyze is None or target2_data_from_analyze is None:
        print('dtw_calculate_acc_cost_matrix: error in dtw_analyze_fastdtw - returned target1_data or target2_data is None')
        return None

    target1_segments = target1_data_from_analyze[body_part]
    target2_segments = target2_data_from_analyze[body_part]

    acc_cost_matrix = np.zeros((len(target1_segments), len(target2_segments)))

    for i in range(len(target1_segments)):
        for j in range(len(target2_segments)):
            if metric == 'euclidean':
                acc_cost_matrix[i, j] = euclidean(target1_segments[i], target2_segments[j])
            elif metric == 'cosine':
                acc_cost_matrix[i, j] = cosine(target1_segments[i], target2_segments[j])
            else:
                raise ValueError(f"Unsupported metric: {metric}")

    return acc_cost_matrix


def dtw_visualize_alignment_separate_axis(target1_data, target2_data, path,
                                          distance,
                                          body_part,
                                          title='DTW Alignment Visualization',
                                          smooth=False, smooth_window_size=3,
                                          show_visualization=False, save_visualization=False,
                                          path_to_save=None, prefix=''):
    """
    Visualizes the DTW alignment for 3 dimensions separately and displays the distance in the plot.
    X-axis uses the 't_s' (time in milliseconds) from the original data.
    Arguments:
        target1_data: structured numpy array of sequence 1 with vectors and timestamps.
        target2_data: structured numpy array of sequence 2 with vectors and timestamps.
        path: DTW path showing the alignment between the two sequences.
        distance: The DTW distance between the two sequences.
        title: Title of the plot.
        show_visualization: Whether to display the plot.
        save_visualization: Whether to save the plot to a file.
        path_to_save: Directory path to save the plot.
        prefix: Prefix for the saved file name.
    """

    if target1_data is None or target2_data is None:
        print('dtw_visualize_alignment_separate_axis: target1_data is None or target2_data is None')
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))  # Increased the size for clarity

    dimensions = ['X', 'Y', 'Z']  # assuming 3D vectors
    colors = ['blue', 'orange']  # colors for the sequences
    markers = ['o', 'x']  # different markers for each sequence

    # extract vector data and timestamps from target1 and target2 data
    target1_segments = target1_data[body_part]
    target2_segments = target2_data[body_part]

    time_target1 = target1_data['t_s']  # time in milliseconds for Sequence 1
    time_target2 = target2_data['t_s']  # time in milliseconds for Sequence 2

    if smooth:
        def smooth_func(data, window_size=smooth_window_size):
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        def truncate_to_match(arr1, arr2):
            min_len = min(len(arr1), len(arr2))
            return arr1[:min_len], arr2[:min_len]

        target1_segments = np.apply_along_axis(smooth_func, axis=0, arr=target1_segments)
        target2_segments = np.apply_along_axis(smooth_func, axis=0, arr=target2_segments)
        time_target1, target1_segments = truncate_to_match(time_target1, target1_segments)
        time_target2, target2_segments = truncate_to_match(time_target2, target2_segments)

    for dim in range(3):
        ax = axes[dim]

        # plot for given dimension with time on x-axis
        ax.plot(time_target1, target1_segments[:, dim], label=f'Target 1 (Dimension {dimensions[dim]})',
                marker=markers[0], color=colors[0], markersize=5)
        ax.plot(time_target2, target2_segments[:, dim], label=f'Target 2 (Dimension {dimensions[dim]})',
                marker=markers[1], color=colors[1], markersize=5)

        # plot DTW alignment path
        for (i, j) in path:
            if i < len(time_target1) and j < len(time_target2):  # indices within bounds check
                ax.plot([time_target1[i], time_target2[j]], [target1_segments[i, dim], target2_segments[j, dim]],
                        color='gray', linestyle='--', linewidth=2, alpha=0.7)

        ax.set_title(f'Dimension {dimensions[dim]} Alignment')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel(f'Value (Dimension {dimensions[dim]})')
        ax.legend()
        ax.grid(True)

    plt.suptitle(f'{title} - DTW Distance: {distance:.3f}', fontsize=18)

    if save_visualization and path_to_save is not None:
        plt.savefig(f'{path_to_save}/dtw_visualization_{prefix}.png')

    if show_visualization:
        plt.show()

    plt.close()


def dtw_visualize_acc_cost_matrix_with_path(path, acc_cost_matrix, distance,
                                            body_part,
                                            title='DTW Minimum Path with Accumulated Cost Matrix',
                                            show_visualization=False, save_visualization=False,
                                            path_to_save=None, prefix=''):
    """
    Visualizes the DTW minimum path overlaid on the accumulated cost matrix.

    Arguments:
    - path: DTW path showing the alignment between the two sequences.
    - acc_cost_matrix: Accumulated cost matrix from DTW analysis.
    - distance: The DTW distance between the two sequences.
    - metric: Distance metric used ('euclidean' or 'cosine').
    - title: Title of the plot.
    - show_visualization: Whether to display the plot.
    - save_visualization: Whether to save the plot to a file.
    - path_to_save: Directory path to save the plot.
    - prefix: Prefix for the saved file name.
    """

    if acc_cost_matrix is None:
        print('dtw_visualize_acc_cost_matrix_with_path: error in dtw_calculate_acc_cost_matrix - returned acc_cost_matrix is None')
        return

    plt.figure(figsize=(8, 8))

    # visualize accumulated cost matrix
    plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')

    # overlay of the DTW path
    plt.plot([p[0] for p in path], [p[1] for p in path], 'w', linewidth=2)  # Plot the path in white

    # plot labels and title
    plt.xlabel('Subject 1 Index')
    plt.ylabel('Subject 2 Index')
    plt.title(f'{title} \n (Part: {body_part}, Distance: {distance:.2f})')

    if save_visualization and path_to_save is not None:
        plt.savefig(f'{path_to_save}/dtw_acc_cost_matrix_{prefix}.png')

    if show_visualization:
        plt.show()

    plt.close()


def dtw_visualize_alignment_unified(target1_data, target2_data, path, distance, body_part,
                                    title='DTW Unified Axis Alignment Visualization',
                                    smooth=False, smooth_window_size=3, show_visualization=False, save_visualization=False,
                                    path_to_save=None, prefix=''):
    """
    Visualizes the DTW alignment for a unified value (Euclidean norm) across all dimensions.
    X-axis uses the 't_s' (time in milliseconds) from the original data.

    Parameters:
    - target1_data: structured numpy array of sequence 1 with vectors and timestamps.
    - target2_data: structured numpy array of sequence 2 with vectors and timestamps.
    - path: DTW path showing the alignment between the two sequences.
    - distance: The DTW distance between the two sequences.
    - body_part: The body part to be visualized (used to extract data from structured arrays).
    - title: Title of the plot (default: 'DTW Unified Path Visualization').
    - smooth: Apply smoothing to the data (default: False).
    - smooth_window_size: Size of the smoothing window if smoothing is applied (default: 3).
    - show_visualization: Whether to display the plot (default: False).
    - save_visualization: Whether to save the plot to a file (default: False).
    - path_to_save: Directory path to save the plot (default: None).
    - prefix: Prefix for the saved file name (default: '').
    """

    if target1_data is None or target2_data is None:
        print('dtw_visualize_alignment_unified: target1_data is None or target2_data is None')
        return

    def smooth_func(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # extracting vector data and timestamps from target1 and target2 data
    target1_segments = target1_data[body_part]
    target2_segments = target2_data[body_part]

    time_target1 = target1_data['t_s']
    time_target2 = target2_data['t_s']

    # euclidean norm (unified value) for both sequences
    unified_target1 = np.linalg.norm(target1_segments, axis=1)
    unified_target2 = np.linalg.norm(target2_segments, axis=1)

    # smoothing if enabled
    if smooth:
        unified_target1 = smooth_func(unified_target1, smooth_window_size)
        unified_target2 = smooth_func(unified_target2, smooth_window_size)

        # adjusting time sequences to match the lengths after smoothing
        time_target1 = time_target1[:len(unified_target1)]
        time_target2 = time_target2[:len(unified_target2)]

    plt.figure(figsize=(12, 6))

    # unified values for both sequences
    plt.plot(time_target1, unified_target1, label='Sequence 1 (Unified)', marker='o', color='blue', markersize=4)
    plt.plot(time_target2, unified_target2, label='Sequence 2 (Unified)', marker='x', color='orange', markersize=4)

    # DTW alignment path
    for (i, j) in path:
        if i < len(time_target1) and j < len(time_target2):
            plt.plot([time_target1[i], time_target2[j]], [unified_target1[i], unified_target2[j]],
                     color='gray', linestyle='--', linewidth=1, alpha=0.6)

    plt.title(f'{title} - DTW Distance: {distance:.3f} \n Unified using Euclidean norm for X, Y, Z', fontsize=14)
    plt.xlabel('Time (ms)')
    plt.ylabel('Unified Value (Across Dimensions)')
    plt.legend()
    plt.grid(True)

    if save_visualization and path_to_save is not None:
        plt.savefig(f'{path_to_save}/dtw_unified_{prefix}.png')

    if show_visualization:
        plt.show()

    plt.close()


# if __name__ == "__main__":
#     import pandas as pd
#
#     df_target1 = pd.read_excel('./images/work_files/pair_target_id_0_vectorframe_data.xlsx')
#     df_target2 = pd.read_excel('./images/work_files/pair_target_id_1_vectorframe_data.xlsx')
#
#     # Specify the column that contains vector data (you might need to adjust this)
#     body_part = 'FOREARM_RIGHT'  #
#
#     path, distance, target1_data, target2_data = dtw_analyze_fastdtw(df_target1, df_target2, body_part, normalize=False, metric=METRIC_COSINE)
#
#     # Visualize the DTW path for all 3 dimensions using the segments already returned
#     dtw_visualize_alignment_separate_axis(target1_data, target2_data, path,
#                                           distance, body_part,  # Pass the DTW distance as a parameter
#                                           show_visualization=True)
#
#     dtw_visualize_alignment_unified(target1_data, target2_data, path, distance, body_part, show_visualization=True)
#
#     acc_cost_matrix = dtw_calculate_acc_cost_matrix(target1_data, target2_data, body_part)
#     dtw_visualize_acc_cost_matrix_with_path(path, acc_cost_matrix, distance, body_part, show_visualization=True)
