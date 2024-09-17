import glob

import cv2
import os
import mediapipe as mp

import utility
from detector import Detector, target_detector
from data.target import Target

# file name pattern
pattern = 'p002*.jpg'
# directory path
path = './images/p002'


# read images in a folder, return an image array
def read_image_files_from_disk(directory_path=path, image_file_name_pattern=pattern):
    # using glob to find all matching files
    matching_image_files = glob.glob(os.path.join(directory_path, pattern))

    frame_array = []

    for image_path in matching_image_files:
        # checking if the file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"The specified image file does not exist: {image_path}")

        # Attempt to read the image
        img = mp.Image.create_from_file(image_path)

        # checking if the image was successfully opened
        if img is None:
            raise IOError(f"Failed to open the image file: {image_path}")

        frame_array.append(img)
    return frame_array


def run_and_show_on_frames_array(frame_array, output_video_path):
    # Initialize video writer
    height, width, layers = frame_array[0].numpy_view().shape
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

    for index, image in enumerate(frame_array):
        detect_result = det.run_detect_on_image_object(image=image)
        tuo = target_detector.create_target_update_by_center_position(detect_result, t1, t2)
        number_of_targets_for_update = tuo.get_number_of_targets()
        if number_of_targets_for_update == 2:
            t1.update_target_with_normalized_landmarks_list(tuo.get_update_values_norm_landmarks_by_id(t1.get_id()),
                                                            tuo.get_update_values_center_by_id(t1.get_id()), index)
            t2.update_target_with_normalized_landmarks_list(tuo.get_update_values_norm_landmarks_by_id(t2.get_id()),
                                                            tuo.get_update_values_center_by_id(t2.get_id()), index)
        if number_of_targets_for_update == 1:
            if tuo.is_target_id_present(t1.get_id()):
                t1.update_target_with_normalized_landmarks_list(tuo.get_update_values_norm_landmarks_by_id(t1.get_id()),
                                                                tuo.get_update_values_center_by_id(t1.get_id()), index)
            else:
                t2.update_target_with_normalized_landmarks_list(tuo.get_update_values_norm_landmarks_by_id(t2.get_id()),
                                                                tuo.get_update_values_center_by_id(t2.get_id()), index)

    for index, mp_image in enumerate(frame_array):
        image = cv2.cvtColor(mp_image.numpy_view(), cv2.COLOR_RGB2BGR)
        tvf1 = t1.get_vector_frame_by_index(index)
        tvf2 = t2.get_vector_frame_by_index(index)
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
        # Write the processed frame to the video file
        video_writer.write(image)

    # Release the video writer
    video_writer.release()


if __name__ == "__main__":
    t1 = Target()
    t2 = Target()
    det = Detector()
    frames = read_image_files_from_disk()
    output_video_path = './postAnalysisVideos'
    run_and_show_on_frames_array(frame_array=frames , output_video_path)



