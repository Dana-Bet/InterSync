import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.spatial.distance import euclidean, cosine
from .common import parse_vectors, METRIC_COSINE, METRIC_EUCLIDEAN


def similarity_visualize_similarities(similarities, title="Movement Similarities Between Targets", show_visualization=False, save_visualization=False, path_to_save=None, prefix=''):
    if not similarities:  # check if the similarities list is empty
        print("No similarities found to plot.")
        return

    plt.figure(figsize=(10, 6))

    # flag for repeating legend labels
    target1_label_added = False
    target2_label_added = False

    # plot similarity pairs of timeframes
    for similarity in similarities:
        target1_start, target1_end = similarity['Target 1 Timeframe']
        for target2_start, target2_end in similarity['Target 2 Timeframes']:
            # timeframe for Target 1
            if not target1_label_added:
                plt.plot([target1_start, target1_end], [1, 1], color='blue', label='Target 1')
                target1_label_added = True
            else:
                plt.plot([target1_start, target1_end], [1, 1], color='blue')

            # timeframe for Target 2
            if not target2_label_added:
                plt.plot([target2_start, target2_end], [2, 2], color='red', label='Target 2')
                target2_label_added = True
            else:
                plt.plot([target2_start, target2_end], [2, 2], color='red')

            # dashed line connecting similar timeframes
            plt.plot([target1_end, target2_start], [1, 2], color='gray', linestyle='--')

    plt.yticks([1, 2], ['Target 1', 'Target 2'])
    plt.xlabel('Timeframe')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.grid(True)
    if save_visualization and path_to_save is not None:
        plt.savefig(f'{path_to_save}/similarity_visualization_{prefix}')
    if show_visualization:
        plt.show()
    plt.close()


# function to compare a single segment from target1 with all segments from target2
def compare_single_segment(seg1, segments2, threshold, metric=METRIC_COSINE):
    similar_pairs = []
    for j, seg2 in enumerate(segments2):
        # Calculate similarity based on the chosen metric
        if metric == METRIC_COSINE:
            similarity = 1 - cosine(seg1, seg2)
            if not np.isnan(similarity) and (1 - similarity) <= threshold:
                similar_pairs.append(j)
        elif metric == METRIC_EUCLIDEAN:
            distance = euclidean(seg1, seg2)
            if not np.isnan(distance) and distance < threshold:
                similar_pairs.append(j)
        else:
            raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")

    return similar_pairs


# function to compare all single body part segments from target1 with target2 using parallel processing
def compare_all_segments(target1_segments, target2_segments, threshold, metric=METRIC_COSINE):
    similar_segments = Parallel(n_jobs=-1, verbose=10)(
        delayed(compare_single_segment)(seg1, target2_segments, threshold, metric)
        for seg1 in target1_segments
    )
    return similar_segments


# function to perform the analysis
def similarity_analyze_similar_movements(df_target1, df_target2, body_part, threshold=0.1, normalize=False, metric=METRIC_COSINE):

    if df_target1 is None or df_target1.empty or df_target2 is None or df_target2.empty:
        print("similarity_analyze_similar_movements: provided target DataFrame is None or empty.")
        return None

    # Segment the movements based on timeframes
    target1_data = parse_vectors(df_target1, body_part, normalize=normalize)
    target2_data = parse_vectors(df_target2, body_part, normalize=normalize)

    target1_segments = target1_data[body_part]
    target2_segments = target2_data[body_part]

    # compare segments
    similar_segments = compare_all_segments(target1_segments, target2_segments, threshold, metric)

    # prepare output
    results = []
    for i, similar_list in enumerate(similar_segments):
        if similar_list:
            target1_timeframe = (df_target1['t_s'].iloc[i], df_target1['t_e'].iloc[i])
            target2_timeframes = [(df_target2['t_s'].iloc[j], df_target2['t_e'].iloc[j]) for j in similar_list]
            results.append({
                'Target 1 Timeframe': target1_timeframe,
                'Target 2 Timeframes': target2_timeframes
            })

    return results
