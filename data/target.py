# Target class
# should serve as our main data structure for detection target

from typing import NewType, Optional

from mediapipe.tasks.python.components.containers import NormalizedLandmark
import pandas as pd

import data.detector_dict
import data.limbs_dict
from data import detector_dict, limbs_dict
from data.vectorframe import VectorFrame

TargetId = NewType('TargetId', int)

current_global_target_id = TargetId(0)


class Target:
    __target_id: Optional[TargetId] = None
    # center for id purposes is calculated as a position between right and left shoulder
    __current_center: Optional[tuple[float, float, float]] = None
    __vector_frames: Optional[list[VectorFrame]] = None
    __current_normalized_landmarks: Optional[list[NormalizedLandmark]] = None
    __current_timestamp = 0

    def __init__(self, target_id: TargetId = None, vector_frame=None,
                 current_center: [tuple[float, float, float]] = (0, 0, 0)):

        global current_global_target_id

        if target_id is None:
            self.__target_id = current_global_target_id
            current_global_target_id += 1
        else:
            self.__target_id = target_id

        self.__vector_frames = []

        if vector_frame is None:
            self.__current_center = current_center
        else:
            self.__vector_frames.append(vector_frame)
            x, time_e = vector_frame.get_time_interval()
            self.__current_timestamp = time_e

    def __str__(self):
        return f'Target(id:{self.__target_id}, center:{self.__current_center}, timestamp:{self.__current_timestamp} )' \
            + self.__vector_frames.__str__() + self.__current_normalized_landmarks.__str__()

    def __debug_print(self):
        return f'Target(id:{self.__target_id}, center:{self.__current_center}, timestamp:{self.__current_timestamp})'

    def update_target_with_vector_frame(self, vector_frame: VectorFrame, new_center: [tuple[float, float, float]]):
        self.__vector_frames.append(vector_frame)
        self.__current_normalized_landmarks = None
        self.__current_center = new_center
        time_s, time_e = vector_frame.get_time_interval()
        self.__current_timestamp = time_e

    def update_target_with_normalized_landmarks_list(self, normalized_landmarks: list[NormalizedLandmark],
                                                     new_center: [tuple[float, float, float]], time_stamp):
        # if there is no current landmark list waiting to be combined into VectorFrame
        if self.__current_normalized_landmarks is None:
            self.__current_normalized_landmarks = normalized_landmarks
            self.__current_center = new_center
            self.__current_timestamp = time_stamp
            return
        # otherwise - there is a landmark list waiting - let's make a VectorFrame, and update our Target
        vf = VectorFrame(self.__current_normalized_landmarks, normalized_landmarks,
                         self.__current_timestamp, time_stamp, self.__target_id)
        self.update_target_with_vector_frame(vf, new_center)
        # save the incoming data as a start point for next vectorframe
        self.__current_normalized_landmarks = normalized_landmarks
        self.__current_center = new_center
        self.__current_timestamp = time_stamp

    def get_id(self) -> TargetId:
        return self.__target_id

    def get_current_center(self) -> [tuple[float, float, float]]:
        return self.__current_center

    def get_last_normalized_landmarks(self):
        return self.__current_normalized_landmarks

    def get_vector_frame_by_id(self, vectorframe_id: int = None):
        if vectorframe_id is not None:
            for vf in self.__vector_frames:
                if vf.get_vectorframe_id() == vectorframe_id:
                    return vf
            print(f'get_vector_frame - no id {vectorframe_id} found')
            return None
        return self.__vector_frames[-1]

    def get_vector_frame_by_index(self, vectorframe_index: int = None):
        if vectorframe_index is not None and vectorframe_index < len(self.__vector_frames):
            return self.__vector_frames[vectorframe_index]
        # test for out-of-bounds issue
        if vectorframe_index is not None and vectorframe_index >= len(self.__vector_frames):
            print(f'get_vector_frame_by_index: out-of-bounds error looking for a frame at index {vectorframe_index}, '
                  f'target_id: {self.__target_id}')
            return None
        return self.__vector_frames[-1]

    def get_vector_frame_by_time(self, start_time: Optional[float] = None, end_time: Optional[float] = None,
                                 time_included: Optional[float] = None, exact_time_match=True, time_match_epsilon_ms=10):
        if start_time is None and end_time is None and time_included is None:
            print('get_vector_frame_by_time: no time parameter provided')
            return None
        for vf in self.__vector_frames:
            vs_time, ve_time = vf.get_time_interval()
            if exact_time_match:
                if start_time is not None and start_time == vs_time:
                    return vf
                if end_time is not None and end_time == ve_time:
                    return vf
                if time_included is not None and vs_time <= time_included < ve_time:    # '<' instead of '<=' ve_time
                    return vf
            else:
                if start_time is not None and (vs_time - time_match_epsilon_ms) <= start_time <= (vs_time + time_match_epsilon_ms):
                    return vf
                if end_time is not None and (ve_time - time_match_epsilon_ms) <= end_time <= (ve_time + time_match_epsilon_ms):
                    return vf
                if time_included is not None and (vs_time - time_match_epsilon_ms) <= time_included < (ve_time + time_match_epsilon_ms):
                    return vf
        # error handling?
        msg = f'get_vector_frame_by_time: no timestamp found, target_id {self.__target_id}'
        if start_time is not None:
            msg += f' t_s-{start_time}'
        if end_time is not None:
            msg += f' t_e-{end_time}'
        if time_included is not None:
            msg += f' t_in-{time_included}'
        print(msg)
        return None

    def first_detected_timestamp(self):
        if len(self.__vector_frames) == 0:
            return 0
        s_t, _ = self.__vector_frames[0].get_time_interval()
        return s_t

    def get_vectorframes_array(self):
        return self.__vector_frames

    def get_vector_data_as_dataframe(self):
        data = []
        for vf in self.__vector_frames:
            vf_data = [vf.get_vectorframe_id()]  # VF ID
            vf_s, vf_e = vf.get_time_interval()
            vf_data.extend([vf_s, vf_e])  # VF start and end time
            vf_data.extend(vf.get_transformed_movement_vectors())  # landmark movement vectors
            vf_data.extend(vf.get_transformed_limb_movement_vectors())  # limb movement vectors
            data.append(vf_data)

        # checking data
        if len(data) == 0:
            print(f"get_vector_data_as_dataframe: no data available for target {self.get_id()}")
            return pd.DataFrame()  # return empty DataFrame

        # create column names
        column_names = ['vf_id', 't_s', 't_e']
        column_names.extend(list(detector_dict.landmark_description.values()))
        column_names.extend(list(limbs_dict.limbs_connect_dict_arr_index_description.values()))

        # checking if number of columns matches the data
        if len(data[0]) != len(column_names):
            print(
                f"get_vector_data_as_dataframe: mismatch in number of columns. expected {len(column_names)}, got {len(data[0])}.")
            return pd.DataFrame()  # return empty DataFrame

        df = pd.DataFrame(data)
        df.columns = column_names
        return df

    # def merge_target_frame_array(self):
    #     # if nothing to merge - return
    #     if self.__vector_frames is None or len(self.__vector_frames) == 0:
    #         return
    #     merged_list = [self.__vector_frames[0]]


class TargetUpdateObject:
    __tuo = None

    def __init__(self, target_id: TargetId = None, center: [tuple[float, float, float]] = None,
                 normalized_landmarks_array: [list[NormalizedLandmark]] = None):
        if target_id is None or center is None or normalized_landmarks_array is None:
            self.__tuo = {}
            return
        self.__tuo = {target_id: [center, normalized_landmarks_array]}

    def __str__(self):
        return self.__tuo.__str__()

    def add_target_update(self, target_id: TargetId, center: [tuple[float, float, float]],
                          normalized_landmarks_array: [list[NormalizedLandmark]]):
        self.__tuo[target_id] = [center, normalized_landmarks_array]

    def get_update_values_array_by_id(self, target_id: TargetId):
        return self.__tuo[target_id]

    def get_update_values_center_by_id(self, target_id: TargetId):
        return self.__tuo[target_id][0]

    def get_update_values_norm_landmarks_by_id(self, target_id: TargetId):
        return self.__tuo[target_id][1]

    def get_number_of_targets(self):
        return len(self.__tuo)

    def is_target_id_present(self, target_id: TargetId):
        if target_id in self.__tuo:
            return True
        return False
