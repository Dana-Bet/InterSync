import os
from datetime import datetime

import cv2

import analysis
import utility as test_util

from data import Target, excel_export, limbs_dict
import frames
import mediapipe as mp

from detector import Detector, target_detector


def add_text_to_top(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=2):
    """
    Adds text to the top of the image.

    :param image: Input image
    :param text: Text to add
    :param font: Font type (default is cv2.FONT_HERSHEY_SIMPLEX)
    :param font_scale: Font scale (default is 1)
    :param color: Text color (default is white)
    :param thickness: Text thickness (default is 2)
    :return: Image with text added
    """
    # Calculate the position for the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    text_x = (image.shape[1] - text_width) // 2  # Center the text horizontally
    text_y = text_height + 10  # Position text at the top with some padding

    # Add text to the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)
    return image


def convert_cv2_to_mp_image(cv2_image):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe Image object from the RGB image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    return mp_image


# video analysis and update of targets accordingly
# creates and returns an array of updated images with movement vectors drawn on them for video
# shows a slideshow of future video frames
# TODO: needs to be split into 3 methods
def video_analysis_run(frame_array):
    for index, kf in enumerate(frame_array):
        image = kf[0]
        timestamp = kf[1]
        print(f'video_analysis_run: frame {index}, timestamp {timestamp}')
        detect_result = __detector_instance.run_detect_on_image_object(
            image=convert_cv2_to_mp_image(image))
        tuo = target_detector.create_target_update_by_center_position(detect_result, __target1, __target2)
        number_of_targets_for_update = tuo.get_number_of_targets()
        if number_of_targets_for_update == 0:
            print(f'video_analysis_run: skipped frame {index}, timestamp {timestamp}, 0 targets for update')
            continue
        if number_of_targets_for_update == 2:
            __target1.update_target_with_normalized_landmarks_list(
                tuo.get_update_values_norm_landmarks_by_id(__target1.get_id()),
                tuo.get_update_values_center_by_id(__target1.get_id()), timestamp)
            __target2.update_target_with_normalized_landmarks_list(
                tuo.get_update_values_norm_landmarks_by_id(__target2.get_id()),
                tuo.get_update_values_center_by_id(__target2.get_id()), timestamp)
        if number_of_targets_for_update == 1:
            if tuo.is_target_id_present(__target1.get_id()):
                __target1.update_target_with_normalized_landmarks_list(
                    tuo.get_update_values_norm_landmarks_by_id(__target1.get_id()),
                    tuo.get_update_values_center_by_id(__target1.get_id()), timestamp)
            else:
                __target2.update_target_with_normalized_landmarks_list(
                    tuo.get_update_values_norm_landmarks_by_id(__target2.get_id()),
                    tuo.get_update_values_center_by_id(__target2.get_id()), timestamp)

    ret_array = []
    # fix: mp_image -> image, already cv2 format

    t1_fdts = __target1.first_detected_timestamp()
    t2_fdts = __target2.first_detected_timestamp()

    for index, kf in enumerate(frame_array):
        image = kf[0]
        timestamp = kf[1]
        if timestamp < max([t1_fdts, t2_fdts]):
            print(f'waiting for first detection of both targets, skipping frame at t-{timestamp}')
            continue
        tvf1 = __target1.get_vector_frame_by_time(time_included=timestamp)
        tvf2 = __target2.get_vector_frame_by_time(time_included=timestamp)
        if tvf1 is None or tvf2 is None:
            continue

        image = test_util.draw_landmarks_on_image(image, tvf1.get_movement_vectors_origin_points())
        image = test_util.draw_landmarks_on_image(image, tvf2.get_movement_vectors_origin_points())

        image = test_util.draw_motion_vectors_on_img(image_object=image,
                                                     motion_vectors=tvf1.get_transformed_limb_movement_vectors(),
                                                     initial_position_landmarks=None,
                                                     initial_position_points=tvf1.get_limb_movement_vectors_origin_points(),
                                                     scale=100)
        image = test_util.draw_motion_vectors_on_img(image_object=image,
                                                     motion_vectors=tvf2.get_transformed_limb_movement_vectors(),
                                                     initial_position_landmarks=None,
                                                     initial_position_points=tvf2.get_limb_movement_vectors_origin_points(),
                                                     scale=100)
        image = add_text_to_top(image, f't-{timestamp}')
        ret_array.append(image)

    # for x in ret_array:
    #     cv2.imshow("window_tile", x)
    #     key = cv2.waitKey(1000)
    #     if key == ord('q'):
    #         break
    # cv2.destroyAllWindows()

    return ret_array


if __name__ == '__main__':
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    dest_path = f'./images/test/{date_time_str}'
    __detector_instance = Detector()
    __target1 = Target()
    __target2 = Target()

    video_output_path = os.path.join(dest_path, date_time_str + ' test_vid.mp4')
    loaded_file_path = './images/work_files/pair_test2.mp4'

    detected_frames = frames.keyframe_detection(detection_threshold=0.35, video_source=loaded_file_path,
                                                destination=dest_path,
                                                save_images=True, verbose=True, save_plots=True)

    # detected_frames = frames.frame_detection(video_source=loaded_file_path, destination=dest_path, save_images=True,
    # verbose=True, frame_skip_low=5, frame_skip_high=5)

    analysed_frames = video_analysis_run(detected_frames)

    height, width, layers = detected_frames[0][0].shape

    # Define the codec and create VideoWriter object
    fps = 3  # 500 ms - frames per second
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For saving as .mp4 file
    video = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    for frame in analysed_frames:
        video.write(frame)

    # Release the video writer object
    video.release()
    print(f"Video saved as {video_output_path}")

    excel_export.target_vectorframe_data_export_to_excel(target_for_export=__target1, export_path=dest_path)
    excel_export.target_vectorframe_data_export_to_excel(target_for_export=__target2, export_path=dest_path)

    # bp = 'ARM_RIGHT'
    # tr = 0.01
    # image_save_path = os.path.join(dest_path, f'threshold_COS')
    # test_util.ensure_folder_exists(image_save_path)
    # similar_movements = analysis.similarity_analyze_similar_movements(df_target1=__target1.get_vector_data_as_dataframe(),
    #                                                                   df_target2=__target2.get_vector_data_as_dataframe(),
    #                                                                   body_part=bp,
    #                                                                   threshold=tr,
    #                                                                   normalize=False,
    #                                                                   metric=analysis.similarity.METRIC_COSINE)
    # analysis.similarity_visualize_similarities(similar_movements, title=f'Movement_correlation_{bp}_tr:{tr}',
    #                                            save_visualization=True,
    #                                            path_to_save=image_save_path,
    #                                            prefix=f'COS_{bp}')

    ###

    similarity_analysis_summary_df = None
    for bp in limbs_dict.limbs_connect_dict_arr_index_description.values():
        tr = 0.02
        image_save_path = os.path.join(dest_path, f'similarity_threshold_{int(tr * 1000)}')
        test_util.ensure_folder_exists(image_save_path)
        similar_movements = analysis.similarity_analyze_similar_movements(
            df_target1=__target1.get_vector_data_as_dataframe(),
            df_target2=__target2.get_vector_data_as_dataframe(),
            body_part=bp,
            threshold=tr,
            metric=analysis.METRIC_EUCLIDEAN)

        sim_an_res = analysis.dataprep.dataprep_summarize_similarity_analysis_short(results=similar_movements,
                                                                                    body_part=bp,
                                                                                    threshold=tr, normalize=False,
                                                                                    metric=analysis.METRIC_EUCLIDEAN)

        similarity_analysis_summary_df = analysis.dataprep.dataprep_append_summaries(similarity_analysis_summary_df,
                                                                                     [sim_an_res])

        analysis.similarity_visualize_similarities(similar_movements,
                                                   title=f'Movement vector similarity {bp}, thr:{tr:.4f}',
                                                   save_visualization=True,
                                                   path_to_save=image_save_path,
                                                   prefix=f'{int(tr * 1000)}_{bp}')
    output_file = os.path.join(dest_path, 'similarity_report_summary.xlsx')
    similarity_analysis_summary_df.to_excel(output_file, index=False)

    # for bp in detector_dict.landmark_description.values():
    #     for tr in numpy.arange(0.08, 0.22, 0.02):
    #         image_save_path = os.path.join(dest_path, f'threshold_{int(tr * 1000)}')
    #         test_util.ensure_folder_exists(image_save_path)
    #         similar_movements = analysis.analyze_similar_movements(df_target1=__target1.get_vector_data_as_dataframe(),
    #                                                                df_target2=__target2.get_vector_data_as_dataframe(),
    #                                                                body_part=bp,
    #                                                                # threshold=0.005,
    #                                                                threshold=tr,
    #                                                                normalize=True,
    #                                                                metric=analysis.METRIC_EUCLIDEAN)
    #         analysis.visualize_correlations(similar_movements, title=f'Movement_correlation_{bp}_tr:{tr}',
    #                                         save_visualization=True,
    #                                         path_to_save=image_save_path,
    #                                         prefix=f'{int(tr * 1000)}_{bp}')

    # for bp in detector_dict.landmark_description.values():
    #     for tr in numpy.arange(0.08, 0.22, 0.02):
    #         image_save_path = os.path.join(dest_path, f'threshold_{int(tr * 1000)}')
    #         test_util.ensure_folder_exists(image_save_path)
    #         similar_movements = analysis.analyze_similar_movements(df_target1=__target1.get_vector_data_as_dataframe(),
    #                                                                df_target2=__target2.get_vector_data_as_dataframe(),
    #                                                                body_part=bp,
    #                                                                # threshold=0.005,
    #                                                                threshold=tr,
    #                                                                normalize=True,
    #                                                                metric=analysis.METRIC_EUCLIDEAN)
    #         analysis.visualize_correlations(similar_movements, title=f'Movement_correlation_{bp}_tr:{tr}',
    #                                         save_visualization=True,
    #                                         path_to_save=image_save_path,
    #                                         prefix=f'{int(tr * 1000)}_{bp}')

    ###
    smith_waterman_analysis_summary_df = None
    image_save_path = os.path.join(dest_path, f'smith_waterman')
    test_util.ensure_folder_exists(image_save_path)
    for bp in limbs_dict.limbs_connect_dict_arr_index_description.values():
        aligned_movements_t1, aligned_movements_t2, aligned_region_length, score = analysis.smith_waterman_analyze(
            df_target1=__target1.get_vector_data_as_dataframe(),
            df_target2=__target2.get_vector_data_as_dataframe(),
            body_part=bp,
            verbose_results=False, match_score=4, mismatch_penalty=-1, gap_penalty=-0.5, threshold=0.02,
            metric=analysis.METRIC_EUCLIDEAN)
        analysis.smith_waterman_visualize_alignment(aligned_movements_t1, aligned_movements_t2,
                                                    save_visualization=True,
                                                    title=f'Aligned Timeframes with Gaps {bp}',
                                                    path_to_save=image_save_path,
                                                    prefix=bp)
        smith_w_an_res = analysis.dataprep.dataprep_summarize_smith_waterman_analysis(body_part=bp,
                                                                                      metric=analysis.METRIC_EUCLIDEAN,
                                                                                      match_score=4,
                                                                                      mismatch_penalty=-1,
                                                                                      gap_penalty=-0.5,
                                                                                      threshold=0.02,
                                                                                      alignment_score=score,
                                                                                      aligned_region_length=aligned_region_length)
        smith_waterman_analysis_summary_df = analysis.dataprep.dataprep_append_summaries(smith_waterman_analysis_summary_df,
                                                                                     [smith_w_an_res])
    output_file = os.path.join(dest_path, 'smith_waterman_report_summary.xlsx')
    smith_waterman_analysis_summary_df.to_excel(output_file, index=False)

    # # for bp in detector_dict.landmark_description.values():
    # #     aligned_movements_t1, aligned_movements_t2, _, _ = smith_waterman.smith_waterman_analyze(
    # #         df_target1=__target1.get_vector_data_as_dataframe(),
    # #         df_target2=__target2.get_vector_data_as_dataframe(),
    # #         body_part=bp,
    # #         normalize=True, verbose_results=False, match_score=4, mismatch_penalty=-1, gap_penalty=-0.5, threshold=0.15)
    # #     smith_waterman.smith_waterman_visualize_alignment(aligned_movements_t1, aligned_movements_t2,
    # #                                                  save_visualization=True,
    # #                                                  title=f'Aligned Timeframes with Gaps {bp}',
    # #                                                  path_to_save=image_save_path,
    # #                                                  prefix=bp)

    ###
    dtw_analysis_summary_df = None
    image_save_path = os.path.join(dest_path, f'dtw')
    test_util.ensure_folder_exists(image_save_path)
    for bp in limbs_dict.limbs_connect_dict_arr_index_description.values():
        path, distance, target1_data, target2_data = analysis.dtw_analyze_fastdtw(
            df_target1=__target1.get_vector_data_as_dataframe(), df_target2=__target2.get_vector_data_as_dataframe(),
            body_part=bp, normalize=False, smooth=False, radius=3, metric=analysis.METRIC_EUCLIDEAN)

        analysis.dtw_visualize_alignment_separate_axis(target1_data, target2_data, path,
                                                       distance, bp, save_visualization=True, smooth=True,
                                                       title=f'DTW Path Visualization {bp}',
                                                       path_to_save=image_save_path,
                                                       prefix=bp)
        analysis.dtw_visualize_alignment_unified(target1_data, target2_data, path, distance,
                                                 body_part=bp, save_visualization=True,
                                                 path_to_save=image_save_path, prefix=bp)
        acc_cost_matrix = analysis.dtw_calculate_acc_cost_matrix(target1_data, target2_data,
                                                                 body_part=bp, metric=analysis.METRIC_EUCLIDEAN)
        analysis.dtw_visualize_acc_cost_matrix_with_path(path, acc_cost_matrix, distance,
                                                         body_part=bp, save_visualization=True,
                                                         path_to_save=image_save_path, prefix=bp)

        dtw_ans = analysis.dataprep.dataprep_summarize_dtw_analysis(path=path, distance=distance, body_part=bp,
                                                                    radius=3, normalize=False, smooth=False,
                                                                    smooth_window_size=3,
                                                                    metric=analysis.METRIC_EUCLIDEAN,
                                                                    acc_cost_matrix=acc_cost_matrix)
        dtw_analysis_summary_df = analysis.dataprep.dataprep_append_summaries(dtw_analysis_summary_df, [dtw_ans])

    output_file = os.path.join(dest_path, 'dtw_report_summary.xlsx')
    dtw_analysis_summary_df.to_excel(output_file, index=False)

    # # for bp in detector_dict.landmark_description.values():
    # #     path, distance, target1_data, target2_data = dtw.dtw_analyze_fastdtw(
    # #         df_target1=__target1.get_vector_data_as_dataframe(), df_target2=__target2.get_vector_data_as_dataframe(),
    # #         body_part=bp, normalize=True, smooth=True)
    # #     dtw.dtw_visualize_alignment_separate_axis(target1_data, target2_data, path,
    # #                                         distance, bp, save_visualization=True,
    # #                                         title=f'DTW Path Visualization {bp}',
    # #                                         path_to_save=image_save_path,
    # #                                         prefix=bp)

    ###
    tlcc_analysis_summary_unified_df = None
    tlcc_analysis_summary_separate_df = None
    image_save_path = os.path.join(dest_path, f'tlcc')
    test_util.ensure_folder_exists(image_save_path)
    for bp in limbs_dict.limbs_connect_dict_arr_index_description.values():
        lags, correlations, target1_data, target2_data = analysis.tlcc_analyze(
            df_target1=__target1.get_vector_data_as_dataframe(), df_target2=__target2.get_vector_data_as_dataframe(),
            body_part=bp, max_lag=5,
            normalize=False, verbose_results=False)
        analysis.tlcc_visualize(lags, correlations, bp, separate_dimensions=False, save_visualization=True,
                                path_to_save=image_save_path, prefix=bp)
        tlcc_ans_unified = analysis.dataprep.dataprep_summarize_tlcc_analysis(body_part=bp, max_lag=5,
                                                                               separate_dimensions=False, lags=lags,
                                                                               correlations=correlations)
        tlcc_analysis_summary_unified_df = analysis.dataprep.dataprep_append_summaries(tlcc_analysis_summary_unified_df,
                                                                                       [tlcc_ans_unified])

        lags, correlations, target1_data, target2_data = analysis.tlcc_analyze(
            df_target1=__target1.get_vector_data_as_dataframe(), df_target2=__target2.get_vector_data_as_dataframe(),
            body_part=bp, max_lag=5, normalize=False, separate_dimensions=True
        )
        analysis.tlcc_visualize(lags, correlations, bp, separate_dimensions=True, save_visualization=True,
                                path_to_save=image_save_path, prefix=f'{bp}_separate_dim')
        tlcc_ans_separate = analysis.dataprep.dataprep_summarize_tlcc_analysis(body_part=bp, max_lag=5,
                                                                               separate_dimensions=True, lags=lags,
                                                                               correlations=correlations)
        tlcc_analysis_summary_separate_df = analysis.dataprep.dataprep_append_summaries(tlcc_analysis_summary_separate_df,
                                                                                       [tlcc_ans_separate])
        output_file = os.path.join(dest_path, 'tlcc_unified_report_summary.xlsx')
        tlcc_analysis_summary_unified_df.to_excel(output_file, index=False)
        output_file = os.path.join(dest_path, 'tlcc_separate_report_summary.xlsx')
        tlcc_analysis_summary_separate_df.to_excel(output_file, index=False)

    #
    # ###
    # image_save_path = os.path.join(dest_path, f'wtlcc')
    # test_util.ensure_folder_exists(image_save_path)
    # for bp in limbs_dict.limbs_connect_dict_arr_index_description.values():
    #     lags, correlations = analysis.wtlcc_analyze(
    #         df_target1=__target1.get_vector_data_as_dataframe(), df_target2=__target2.get_vector_data_as_dataframe(),
    #         body_part=bp, max_lag=10, windowed=True, separate_dimensions=False, no_splits=20
    #     )
    #     analysis.wtlcc_visualize_windowed_tlcc(lags, correlations, no_splits=20, separate_dimensions=False,
    #                                            save_visualization=True, path_to_save=image_save_path, prefix=bp)
    #     lags, correlations = analysis.wtlcc_analyze(
    #         df_target1=__target1.get_vector_data_as_dataframe(), df_target2=__target2.get_vector_data_as_dataframe(),
    #         body_part=bp, max_lag=10, windowed=False, separate_dimensions=False, window_size=10,
    #         step_size=5
    #     )
    #     analysis.wtlcc_visualize_rolling_window_tlcc(lags, correlations, separate_dimensions=False,
    #                                                  save_visualization=True, path_to_save=image_save_path, prefix=bp)
