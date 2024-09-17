import numpy as np
import pandas as pd


def dataprep_append_summaries(df, *summaries):
    combined_summary = []
    for summary in summaries:
        combined_summary.extend(summary)
    new_df = pd.DataFrame(combined_summary)
    if df is None:
        return new_df
    else:
        return pd.concat([df, new_df], ignore_index=True)


def dataprep_merge_summaries(*summaries):
    combined_summary = []
    for summary in summaries:
        combined_summary.extend(summary)
    df = pd.DataFrame(combined_summary)
    return df


def dataprep_summarize_similarity_analysis(results, body_part, threshold, normalize, metric):
    summary = []
    for result in results:
        target1_timeframe = result['Target 1 Timeframe']
        target2_timeframes = [(int(start), int(end)) for start, end in result['Target 2 Timeframes']]
        summary.append({
            'Body Part': body_part,
            'Threshold': threshold,
            'Normalized': normalize,
            'Metric': metric,
            'Target 1 Start Time': target1_timeframe[0],
            'Target 1 End Time': target1_timeframe[1],
            'Number of Similar Segments': len(target2_timeframes),
            'Target 2 Timeframes': target2_timeframes
        })
    return summary


def dataprep_summarize_similarity_analysis_short(results, body_part, threshold, normalize, metric):
    total_similar_segments = 0
    for result in results:
        target2_timeframes = result['Target 2 Timeframes']
        total_similar_segments += len(target2_timeframes)
    summary = {
        'Body Part': body_part,
        'Threshold': threshold,
        'Normalized': normalize,
        'Metric': metric,
        'Total of Similar Segments': total_similar_segments
    }
    return summary


def dataprep_summarize_dtw_analysis(path, distance, body_part, radius, normalize, smooth, smooth_window_size, metric,
                                    acc_cost_matrix):
    dtw_path_length = len(path) if path is not None else 0
    if acc_cost_matrix is not None:
        min_cost = np.min(acc_cost_matrix)
        max_cost = np.max(acc_cost_matrix)
        mean_cost = np.mean(acc_cost_matrix)
        cost_matrix_shape = acc_cost_matrix.shape
    else:
        min_cost = max_cost = mean_cost = None
        cost_matrix_shape = None
    summary = {
        'Body Part': body_part,
        'Metric': metric,
        'Radius': radius,
        'Normalized': normalize,
        'Smooth': smooth,
        'Smooth Window': smooth_window_size if smooth else None,
        'Distance': round(distance, 4) if distance is not None else None,
        'Path Length': dtw_path_length,
        'Cost Matrix Shape': cost_matrix_shape,
        'Min Cost': round(min_cost, 4) if min_cost is not None else None,
        'Max Cost': round(max_cost, 4) if max_cost is not None else None,
        'Mean Cost': round(mean_cost, 4) if mean_cost is not None else None
    }
    return summary


def dataprep_summarize_smith_waterman_analysis(body_part, metric, match_score, mismatch_penalty, gap_penalty, threshold,
                                               alignment_score, aligned_region_length):
    summary = {
        'Body Part': body_part,
        'Metric': metric,
        'Match \nScore': match_score,
        'Mismatch \nPenalty': mismatch_penalty,
        'Gap \nPenalty': gap_penalty,
        'Threshold': threshold,
        'Alignment \nScore': round(alignment_score, 4) if alignment_score is not None else None,
        'Aligned \nSegments': aligned_region_length
    }
    return summary


def dataprep_summarize_tlcc_analysis(body_part, max_lag, separate_dimensions, lags, correlations):
    if separate_dimensions:
        # peak correlation and lag for each dimension (X, Y, Z)
        peak_correlation_x = np.max(correlations[:, 0])
        peak_lag_x = lags[np.argmax(correlations[:, 0])]

        peak_correlation_y = np.max(correlations[:, 1])
        peak_lag_y = lags[np.argmax(correlations[:, 1])]

        peak_correlation_z = np.max(correlations[:, 2])
        peak_lag_z = lags[np.argmax(correlations[:, 2])]

        # average correlation for each dimension
        avg_correlation_x = np.mean(correlations[:, 0])
        avg_correlation_y = np.mean(correlations[:, 1])
        avg_correlation_z = np.mean(correlations[:, 2])

        # overall average correlation and standard deviation
        overall_avg_correlation = np.mean(correlations)
        corr_std_dev = np.std(correlations)

        summary = {
            'Body Part': body_part,
            # 'Max \nLag': max_lag,
            # 'Dimensions': True,
            'Peak \nCorr\n X': round(peak_correlation_x, 4),
            'Peak \nLag\n X': peak_lag_x,
            'Peak \nCorr\n Y': round(peak_correlation_y, 4),
            'Peak \nLag\n Y': peak_lag_y,
            'Peak \nCorr\n Z': round(peak_correlation_z, 4),
            'Peak \nLag\n Z': peak_lag_z,
            'Avg \nCorr\n X': round(avg_correlation_x, 4),
            'Avg \nCorr\n Y': round(avg_correlation_y, 4),
            'Avg \nCorr\n Z': round(avg_correlation_z, 4),
            'Overall \nAvg\n Corr': round(overall_avg_correlation, 4),
            'Corr \nStd\n Dev': round(corr_std_dev, 4)
        }

    else:
        # Peak correlation and lag for unified vector
        peak_idx = np.argmax(correlations[:, 0])
        peak_correlation = correlations[peak_idx, 0]
        peak_lag = lags[peak_idx]

        # Average and standard deviation of correlations
        avg_correlation = np.mean(correlations[:, 0])
        corr_std_dev = np.std(correlations[:, 0])

        # Create summary row for unified vector analysis
        summary = {
            'Body Part': body_part,
            'Max Lag': max_lag,
            # 'Dimensions': False,
            'Peak Corr': round(peak_correlation, 4),
            'Peak Lag': peak_lag,
            'Average Corr': round(avg_correlation, 4),
            'Corr Std Dev': round(corr_std_dev, 4)
        }

    return summary
