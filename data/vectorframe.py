# VectorFrame class
import numpy as np
from mediapipe.tasks.python.components.containers import NormalizedLandmark

from . import config
import data.detector_dict
from data.limbs_dict import limbs_connect_dict_arr


# calculate the unit vector of the numpy vector
def unit_vector(vector):
    return vector / np.linalg.norm(vector)


# calculate new vector axis as follows:
# X - from body center to head
# Y - from left shoulder to right shoulder
# Z - from chest forward
def calculate_basis_vectors(shoulder_left, shoulder_right, hip_left, hip_right):
    # midpoints
    shoulder_mid = (shoulder_left + shoulder_right) / 2
    hip_mid = (hip_left + hip_right) / 2

    # x-axis: vector from hip midpoint to shoulder midpoint
    x_vector = unit_vector(shoulder_mid - hip_mid)

    # y-axis: vector from one shoulder to the other
    y_vector = unit_vector(shoulder_right - shoulder_left)

    # z-axis: cross product of X and Y axes
    z_vector = unit_vector(np.cross(x_vector, y_vector))

    return np.array([x_vector, y_vector, z_vector])


# vector transformation to new vector axis coordinates
# should be used to move the vectors from image-based coordinates to target-based ones
def transform_single_vector(movement_vector, basis_vectors):
    x_vector = basis_vectors[0]
    y_vector = basis_vectors[1]
    z_vector = basis_vectors[2]
    x = np.dot(movement_vector, x_vector)
    y = np.dot(movement_vector, y_vector)
    z = np.dot(movement_vector, z_vector)
    return np.array([x, y, z])


def transform_vectors_array(movement_vectors, basis_vectors):
    transformed_vectors = np.dot(movement_vectors, np.transpose(basis_vectors))
    return transformed_vectors


current_global_vectorframe_id = 0


def reset_global_vectorframe_id():
    global current_global_vectorframe_id
    current_global_vectorframe_id = 0


class VectorFrame:

    __vector_id = None
    __target_id = None

    __vector_origin_points = None
    __vector_existing = None
    __movement_vectors = None
    __limb_movement_vectors = None

    __movement_vectors_transformed = None
    __limb_movement_vectors_transformed = None

    __vector_frame_start_time = config.VF_DEFAULT_START_TIME
    __vector_frame_end_time = config.VF_DEFAULT_END_TIME

    def __init__(self, start_landmarks: list[NormalizedLandmark], end_landmarks: list[NormalizedLandmark],
                 frame_start_time=config.VF_DEFAULT_START_TIME, frame_end_time=config.VF_DEFAULT_END_TIME, target_id=0):
        global current_global_vectorframe_id

        self.__vector_id = current_global_vectorframe_id
        current_global_vectorframe_id += 1

        self.__target_id = target_id
        self.__vector_origin_points = start_landmarks
        self.__vector_frame_start_time = frame_start_time
        self.__vector_frame_end_time = frame_end_time
        ar1 = np.array([(lm.x, lm.y, lm.z) for lm in start_landmarks])
        ar2 = np.array([(lm.x, lm.y, lm.z) for lm in end_landmarks])
        # landmark movement vectors
        self.__movement_vectors = ar2 - ar1
        # landmark 'existence' values
        self.__vector_existing = np.array([[lm_start.visibility > config.VF_DEFAULT_VISIBLE_VALUE
                                            and lm_start.presence > config.VF_DEFAULT_PRESENCE_VALUE
                                            and lm_end.visibility > config.VF_DEFAULT_VISIBLE_VALUE
                                            and lm_end.presence > config.VF_DEFAULT_PRESENCE_VALUE,
                                            lm_start.visibility, lm_start.presence, lm_end.visibility, lm_end.presence]
                                           for lm_start, lm_end in zip(start_landmarks, end_landmarks)])
        # limb movement vectors
        limb_centerpoints1 = np.array([((start_landmarks[cn[0]].x + start_landmarks[cn[1]].x) / 2,
                                        (start_landmarks[cn[0]].y + start_landmarks[cn[1]].y) / 2,
                                        (start_landmarks[cn[0]].z + start_landmarks[cn[1]].z) / 2
                                        ) for cn in limbs_connect_dict_arr])

        limb_centerpoints2 = np.array([((end_landmarks[cn[0]].x + end_landmarks[cn[1]].x) / 2,
                                        (end_landmarks[cn[0]].y + end_landmarks[cn[1]].y) / 2,
                                        (end_landmarks[cn[0]].z + end_landmarks[cn[1]].z) / 2
                                        ) for cn in limbs_connect_dict_arr])

        self.__limb_movement_vectors = limb_centerpoints2 - limb_centerpoints1

        new_basis_vectors = calculate_basis_vectors(shoulder_left=ar1[data.detector_dict.LEFT_SHOULDER],
                                                    shoulder_right=ar1[data.detector_dict.RIGHT_SHOULDER],
                                                    hip_left=ar1[data.detector_dict.LEFT_HIP],
                                                    hip_right=ar1[data.detector_dict.RIGHT_HIP])

        self.__limb_movement_vectors_transformed = transform_vectors_array(movement_vectors=self.__limb_movement_vectors,
                                                                           basis_vectors=new_basis_vectors)
        self.__movement_vectors_transformed = transform_vectors_array(movement_vectors=self.__movement_vectors,
                                                                      basis_vectors=new_basis_vectors)

    def __str__(self):
        return f'VectorFrame:(vector id:{self.__vector_id}, target id:{self.__target_id}, ' \
               f'start:{self.__vector_frame_start_time}, end:{self.__vector_frame_end_time}'

    def __repr__(self):
        return self.__str__()

    def get_vectorframe_id(self):
        return self.__vector_id

    def get_movement_vectors(self):
        return self.__movement_vectors

    def get_movement_vectors_origin_points(self):
        return self.__vector_origin_points

    def get_limb_movement_vectors(self):
        return self.__limb_movement_vectors

    def get_limb_movement_vectors_origin_points(self):
        return np.array([((self.__vector_origin_points[cn[0]].x + self.__vector_origin_points[cn[1]].x) / 2,
                          (self.__vector_origin_points[cn[0]].y + self.__vector_origin_points[cn[1]].y) / 2,
                          (self.__vector_origin_points[cn[0]].z + self.__vector_origin_points[cn[1]].z) / 2
                          ) for cn in limbs_connect_dict_arr])

    def get_transformed_movement_vectors(self):
        return self.__movement_vectors_transformed

    def get_transformed_limb_movement_vectors(self):
        return self.__limb_movement_vectors_transformed

    def get_vector_existing_values(self):
        return self.__vector_existing[:, 1:]

    def get_vector_existing_boolean(self):
        return self.__vector_existing[:, :1]

    def get_time_interval(self):
        return self.__vector_frame_start_time, self.__vector_frame_end_time

    def is_merge_possible(self, compared_to, epsilon: float):
        pass


# from detector import Detector
#
# if __name__ == "__main__":
#     IMG_PATH1 = './images/p001_keyframe20.jpg'
#     IMG_PATH2 = './images/p001_keyframe21.jpg'
#
#     det = Detector()
#     ldmr1 = det.get_normalized_landmarks_target_array_from_image_path(IMG_PATH1)
#     ldmr2 = det.get_normalized_landmarks_target_array_from_image_path(IMG_PATH2)
#
#     vf = VectorFrame(ldmr1, ldmr2)
#     # print(vf.get_movement_vectors())
#     # print(vf.get_vector_existing_values())
#     # print(vf.get_vector_existing_boolean())
