import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cosine
from .common import parse_vectors, METRIC_COSINE, METRIC_EUCLIDEAN

# Scoring constants
MATCH_SCORE = 4  # 2
MISMATCH_PENALTY = -1
GAP_PENALTY = -0.5  # -2
THRESHOLD = 0.003


def smith_waterman_score(seq1, seq2, match_score=MATCH_SCORE, mismatch_penalty=MISMATCH_PENALTY,
                         gap_penalty=GAP_PENALTY, threshold=THRESHOLD, metric=METRIC_EUCLIDEAN):
    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m + 1, n + 1))
    max_score = 0
    max_pos = None

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # similarity metric
            if metric == METRIC_EUCLIDEAN:
                distance = euclidean(seq1[i - 1], seq2[j - 1])
                is_similar = not np.isnan(distance) and distance <= threshold
            elif metric == METRIC_COSINE:
                similarity = 1 - cosine(seq1[i - 1], seq2[j - 1])
                is_similar = not np.isnan(similarity) and (1 - similarity) <= threshold

            # similarity score based on the chosen metric
            if is_similar:
                similarity_score = match_score
            else:
                similarity_score = mismatch_penalty

            match = score_matrix[i - 1, j - 1] + similarity_score
            delete = score_matrix[i - 1, j] + gap_penalty
            insert = score_matrix[i, j - 1] + gap_penalty
            score_matrix[i, j] = max(0, match, delete, insert)

            if score_matrix[i, j] > max_score:
                max_score = score_matrix[i, j]
                max_pos = (i, j)

    return max_score, max_pos, score_matrix


def smith_waterman_traceback(seq1, seq2, score_matrix, max_pos, match_score=MATCH_SCORE,
                             mismatch_penalty=MISMATCH_PENALTY, gap_penalty=GAP_PENALTY, threshold=THRESHOLD,
                             metric=METRIC_EUCLIDEAN):
    i, j = max_pos
    aligned_seq1 = []
    aligned_seq2 = []

    while score_matrix[i, j] > 0:
        current_score = score_matrix[i, j]
        if i > 0 and j > 0:
            # similarity metric for traceback
            if metric == METRIC_EUCLIDEAN:
                distance = euclidean(seq1[i - 1], seq2[j - 1])
                is_similar = distance < threshold
            elif metric == METRIC_COSINE:
                similarity = 1 - cosine(seq1[i - 1], seq2[j - 1])
                is_similar = not np.isnan(similarity) and (1 - similarity) <= threshold

            if is_similar:
                similarity_score = match_score
            else:
                similarity_score = mismatch_penalty

            if score_matrix[i - 1, j - 1] + similarity_score == current_score:
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append(seq2[j - 1])
                i -= 1
                j -= 1
                continue

        if i > 0 and score_matrix[i - 1, j] + gap_penalty == current_score:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(None)  # represents a gap
            i -= 1
            continue

        if j > 0 and score_matrix[i, j - 1] + gap_penalty == current_score:
            aligned_seq1.append(None)  # represents a gap
            aligned_seq2.append(seq2[j - 1])
            j -= 1
            continue

    aligned_seq1.reverse()
    aligned_seq2.reverse()

    return aligned_seq1, aligned_seq2


def smith_waterman_visualize_alignment(aligned_data1, aligned_data2, title='Aligned Timeframes with Gaps',
                                       show_visualization=False, save_visualization=False, path_to_save=None,
                                       prefix=''):
    if aligned_data1 is None or aligned_data2 is None:
        print('smith_waterman_visualize_alignment: no data to visualize')
        return

    plt.figure(figsize=(20, 10))
    ax = plt.gca()

    y_indices = np.arange(len(aligned_data1)) * 3

    for idx, y in enumerate(y_indices):
        vec1, t_s1 = aligned_data1['vector'][idx], aligned_data1['t_s'][idx]
        vec2, t_s2 = aligned_data2['vector'][idx], aligned_data2['t_s'][idx]

        # lines for sequences
        ax.plot([2, 3], [y, y], 'k-', lw=1, alpha=0.5)
        ax.plot([4, 5], [y, y], 'k-', lw=1, alpha=0.5)

        # matching segments and gaps
        if not np.isnan(vec1).all() and not np.isnan(vec2).all():
            distance = euclidean(vec1, vec2)  # Calculate Euclidean distance
            ax.plot([3, 4], [y, y], 'g-', lw=4)
            ax.text(7, y, f'Vec1: {np.round(vec1, 3)}', verticalalignment='center', fontsize=7, color='blue')
            ax.text(8, y, f'Vec2: {np.round(vec2, 3)}', verticalalignment='center', fontsize=7, color='green')
            ax.text(9, y, f'Dist: {distance:.2f}', verticalalignment='center', fontsize=7,
                    color='red')
        else:
            ax.plot([3, 4], [y, y], 'r:', lw=2)
            ax.text(3.5, y, 'GAP', verticalalignment='center', fontsize=8, color='gray')

        # timestamps only if meaningful
        if t_s1 != -1:
            ax.text(1.5, y, f't_s1: {t_s1}', verticalalignment='center', fontsize=7, color='darkblue')
        if t_s2 != -1:
            ax.text(5, y, f't_s2: {t_s2}', verticalalignment='center', fontsize=7, color='darkgreen')

    plt.xlim(0, 10)
    plt.ylim(-1, max(y_indices) + 1)
    plt.xticks([])  # remove x-axis ticks
    plt.yticks([])  # remove y-axis ticks
    plt.title(title, fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if save_visualization and path_to_save is not None:
        plt.savefig(f'{path_to_save}/smith_waterman_visualization_{prefix}.png')
    if show_visualization:
        plt.show()
    plt.close()


def create_aligned_data(original_data, aligned_sequence):
    dt = np.dtype([
        ('vector', np.float64, (3,)),
        ('t_s', np.float64),
        ('t_e', np.float64)
    ])

    aligned_data = np.zeros(len(aligned_sequence), dtype=dt)
    index = 0
    for i, vec in enumerate(aligned_sequence):
        if vec is not None:
            aligned_data[i]['vector'] = vec
            aligned_data[i]['t_s'] = original_data['t_s'][index]
            aligned_data[i]['t_e'] = original_data['t_e'][index]
            index += 1
        else:
            aligned_data[i]['vector'] = np.array([np.nan, np.nan, np.nan],
                                                 dtype=np.float64)  # filling the vector with NaNs for gaps
            aligned_data[i]['t_s'] = -1
            aligned_data[i]['t_e'] = -1

    return aligned_data


def smith_waterman_analyze(df_target1, df_target2, body_part, match_score=MATCH_SCORE, gap_penalty=GAP_PENALTY,
                           mismatch_penalty=MISMATCH_PENALTY,
                           threshold=THRESHOLD, normalize=False, verbose_results=False, metric=METRIC_EUCLIDEAN):

    if df_target1 is None or df_target1.empty or df_target2 is None or df_target2.empty:
        print("smith_waterman_analyze: provided target DataFrame is None or empty.")
        return None, None, None, None

    target1_data = parse_vectors(df_target1, body_part, normalize=normalize)
    target2_data = parse_vectors(df_target2, body_part, normalize=normalize)

    target1_segments = target1_data[body_part]
    target2_segments = target2_data[body_part]

    score, max_pos, score_matrix = smith_waterman_score(
        target1_segments, target2_segments, match_score=match_score, gap_penalty=gap_penalty,
        mismatch_penalty=mismatch_penalty, threshold=threshold, metric=metric
    )

    if max_pos is None:
        print(
            f'smith_waterman_analyze: no positive alignment score between the sequences for {body_part} - no meaningful alignment was found')
        return None, None, 0, 0

    # Perform traceback to get the aligned sequences
    aligned_seq1, aligned_seq2 = smith_waterman_traceback(
        target1_segments, target2_segments, score_matrix, max_pos, match_score=match_score, gap_penalty=gap_penalty,
        mismatch_penalty=mismatch_penalty, threshold=threshold, metric=metric
    )

    # calculating the length of the aligned region (excluding gaps)
    aligned_region_length = sum(
        1 for vec1, vec2 in zip(aligned_seq1, aligned_seq2) if vec1 is not None and vec2 is not None)

    if verbose_results:
        print(f'Maximum Alignment Score for {body_part}: {score}')
        print(f"Alignment Position: {max_pos}")
        print(f"Aligned Sequence 1: {aligned_seq1}")
        print(f"Aligned Sequence 2: {aligned_seq2}")
        print(f"Length of Aligned Region (excluding gaps): {aligned_region_length}")

    # full alignment datasets
    aligned_data1 = create_aligned_data(target1_data, aligned_seq1)
    aligned_data2 = create_aligned_data(target2_data, aligned_seq2)

    return aligned_data1, aligned_data2, aligned_region_length, score
