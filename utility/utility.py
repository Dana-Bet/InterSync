# test util module
import os

import numpy as np
import cv2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import solutions


def draw_motion_vectors_on_img(image_object, motion_vectors, initial_position_landmarks, initial_position_points=None,
                               scale=1, color=(0, 255, 0), show_image=False):
    # Calculate image dimensions and scale landmarks
    image_height, image_width = image_object.shape[:2]
    movement_vectors_2d = motion_vectors[:, :2]
    initial_positions = None
    if initial_position_points is None:
        initial_positions = np.array([(lm.x * image_width, lm.y * image_height) for lm in initial_position_landmarks])
    else:
        initial_positions = initial_position_points[:, :2] * np.array([image_width, image_height])
    final_positions = initial_positions + movement_vectors_2d * scale  # Scale vectors for visibility

    # Draw vectors on the image
    for start, end in zip(initial_positions, final_positions):
        start = tuple(start.astype(int))
        end = tuple(end.astype(int))
        cv2.arrowedLine(image_object, start, end, color, 2, tipLength=0.15)
    if show_image:
        cv2.imshow('Landmark Vectors', image_object)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image_object


def draw_landmarks_on_image(image_object, normalized_landmarks):
    rgb_image_object = image_object
    # pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image_object)
    #
    # # Loop through the detected poses to visualize.
    # for idx in range(len(pose_landmarks_list)):
    #     pose_landmarks = pose_landmarks_list[idx]
    pose_landmarks = normalized_landmarks

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


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


def ensure_folder_exists(path):
    # Check if the folder exists
    if not os.path.exists(path):
        # If the folder does not exist, create it
        os.makedirs(path)
        print(f"Folder created at: {path}")
    else:
        print(f"Folder already exists at: {path}")


def remove_folder(path):
    if os.path.exists(path):
        if not os.listdir(path):
            os.rmdir(path)
            print(f"Folder removed at: {path}")
        else:
            print(f"Folder not empty at: {path}")
    else:
        print(f"Folder does not exist at: {path}")
