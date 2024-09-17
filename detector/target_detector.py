# target detection logic
# do we need to wrap it in a class?
import math

import numpy as np
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

from data import detector_dict
from data.target import Target, TargetUpdateObject


def calculate_center_from_world_landmarks_list(world_landmark_list: list[NormalizedLandmark]):
    lsn = world_landmark_list[detector_dict.LEFT_SHOULDER]  # Left Shoulder Node
    rsn = world_landmark_list[detector_dict.RIGHT_SHOULDER]
    left_shoulder_loc = np.array([lsn.x, lsn.y, lsn.z])
    right_shoulder_loc = np.array([rsn.x, rsn.y, rsn.z])
    result = (right_shoulder_loc + left_shoulder_loc) / 2
    return tuple(float(num) for num in result)


def calculate_center_distance_between_two_centers(target_one_center: [tuple[float, float, float]],
                                                  target_two_center: [tuple[float, float, float]]):
    t1_x, t1_y, t1_z = target_one_center
    t2_x, t2_y, t2_z = target_two_center
    return math.sqrt((t2_x - t1_x) ** 2 + (t2_y - t1_y) ** 2 + (t2_z - t1_z) ** 2)


# here currently sits logic for detecting which target should get which update frame
# will be probably wrapped in an 'update object' class of some sorts
def create_target_update_by_center_position(detection_result: PoseLandmarkerResult,
                                            target_one: Target, target_two: Target, closeness_epsilon=0.01):
    normalized_lm = detection_result.pose_landmarks
    id1 = target_one.get_id()
    id2 = target_two.get_id()
    trg1_center = target_one.get_current_center()
    trg2_center = target_two.get_current_center()
    single_target_detected_flag = False

    if len(normalized_lm) == 0:
        print(f'update_targets_by_center_position(): detection_result does not contain data of any targets')
        return_object = TargetUpdateObject()
        return return_object

    if len(normalized_lm) != 2:
        print(f'update_targets_by_center_position(): detection_result does not contain data of exactly two targets')
        single_target_detected_flag = True

    if trg1_center == (0, 0, 0) and trg2_center == (0, 0, 0):
        t1_center = calculate_center_from_world_landmarks_list(normalized_lm[0])
        return_object = TargetUpdateObject(id1, t1_center, normalized_lm[0])
        if not single_target_detected_flag:
            t2_center = calculate_center_from_world_landmarks_list(normalized_lm[1])
            return_object.add_target_update(id2, t2_center, normalized_lm[1])
        return return_object

    if trg1_center != (0, 0, 0) or trg2_center != (0, 0, 0):
        return_object = TargetUpdateObject()

        # first we solve the 'singe target detected' case
        if single_target_detected_flag:
            # detected center
            lm0_center = calculate_center_from_world_landmarks_list(normalized_lm[0])
            # distances to existing target centers
            lm0_to_old_trg1 = calculate_center_distance_between_two_centers(lm0_center, trg1_center)
            lm0_to_old_trg2 = calculate_center_distance_between_two_centers(lm0_center, trg2_center)
            # preparing the 'return_object'
            if lm0_to_old_trg1 < lm0_to_old_trg2:
                # lm0 is closer to trg1
                return_object.add_target_update(id1, lm0_center, normalized_lm[0])
            else:
                return_object.add_target_update(id2, lm0_center, normalized_lm[0])
            return return_object

        # we calculate where new centers would be, still not knowing which target they belong to
        lm0_center = calculate_center_from_world_landmarks_list(normalized_lm[0])
        lm1_center = calculate_center_from_world_landmarks_list(normalized_lm[1])

        lm0_to_old_trg1 = calculate_center_distance_between_two_centers(lm0_center, trg1_center)
        lm0_to_old_trg2 = calculate_center_distance_between_two_centers(lm0_center, trg2_center)
        lm1_to_old_trg1 = calculate_center_distance_between_two_centers(lm1_center, trg1_center)
        lm1_to_old_trg2 = calculate_center_distance_between_two_centers(lm1_center, trg2_center)

        distances = [lm0_to_old_trg1, lm0_to_old_trg2, lm1_to_old_trg1, lm1_to_old_trg2]
        minimal_distance = min(distances, key=abs)

        # check if there is a new center that is under-epsilon-close to one of the old ones
        # or if it is a minimal distance found, and assign based on that
        if ((lm0_to_old_trg1 < closeness_epsilon or lm0_to_old_trg1 == minimal_distance)
                or (lm1_to_old_trg2 < closeness_epsilon or lm1_to_old_trg2 == minimal_distance)):
            return_object.add_target_update(id1, lm0_center, normalized_lm[0])
            return_object.add_target_update(id2, lm1_center, normalized_lm[1])
            return return_object

        if ((lm0_to_old_trg2 < closeness_epsilon or lm0_to_old_trg2 == minimal_distance)
                or (lm1_to_old_trg1 < closeness_epsilon or lm1_to_old_trg1 == minimal_distance)):
            return_object.add_target_update(id2, lm0_center, normalized_lm[0])
            return_object.add_target_update(id1, lm1_center, normalized_lm[1])
            return return_object

