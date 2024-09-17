# detector class

# Standard library imports

# Third-party imports
import mediapipe as mp
import numpy as np
import cv2

# Library specific imports
# mediapipe
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# pathlib
from pathlib import Path

# project imports
from . import config


class Detector:
    __model_path = None
    __detector_instance = None

    def __init__(self, model_path=config.DETECTOR_MODEL_PATH, detection_targets=config.MP_DEFAULT_TARGETS):
        if detection_targets not in [1, 2]:
            print('Detector class __init__(): detector_targets wrong value')
            return None
        self.__model_path = model_path
        self.__create_detector_instance(detection_targets=detection_targets)

    def __create_detector_instance(self, detection_targets=config.MP_DEFAULT_TARGETS):
        base_options = python.BaseOptions(model_asset_path=self.__model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=detection_targets,
            output_segmentation_masks=True)
        self.__detector_instance = vision.PoseLandmarker.create_from_options(options)

    # def get_detector_instance(self):
    #     if self.__detector_instance is None:
    #         self.__create_detector_instance()
    #     return self.__detector_instance

    def __create_img_object_from_img_path(self, image_path):
        image_obj = mp.Image.create_from_file(image_path)
        return image_obj

    def __test_show_image_object_in_window(self, image_obj, window_title='test window'):
        cv2.imshow(window_title, image_obj)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __test_add_landmarks_onto_image(self, img_object, detection_result):
        rgb_image_object = img_object.numpy_view()
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image_object)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

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
        return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    def run_detect_on_image_object(self, image, show_test_window=False):
        # print("run_detect_on_image_object")
        detection_result = self.__detector_instance.detect(image)
        if show_test_window is True:
            annotated_image = self.__test_add_landmarks_onto_image(image, detection_result)
            self.__test_show_image_object_in_window(annotated_image)
        return detection_result

    def run_detect_on_image_path(self, image_path=None, show_test_window=False):
        # print("run_detect_on_image_path")

        file_path = Path(image_path)
        if not file_path.exists():
            print("File does not exist.")
            return None
        image_obj = mp.Image.create_from_file(image_path)
        return self.run_detect_on_image_object(image_obj, show_test_window)

    def get_normalized_landmarks_all_arrays_from_image_path(self, image_path=None):
        detection_result = self.run_detect_on_image_path(image_path=image_path)
        return detection_result.pose_landmarks

    def get_normalized_landmarks_target_array_from_image_path(self, image_path=None, target=0):
        arrays = self.get_normalized_landmarks_all_arrays_from_image_path(image_path=image_path)
        return arrays[target]


# if __name__ == "__main__":
#     print("detector class running as main")
#     # show_test_image()
#     detector = Detector()
#     run_result = detector.run_detect_on_image_path(image_path='./images/p002/p002_keyframe12.jpg', show_test_window=True)
#     print(run_result)
#     print("detector class running end")
