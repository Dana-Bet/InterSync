import os
from typing import Optional

import cv2


def create_directories(flowframe_path):
    # create directories if they don't exist
    os.makedirs(flowframe_path, exist_ok=True)


def save_frames(frames, frame_path, timestamp_name=True):
    # save keyframe images to specified path
    for index, kf in enumerate(frames, start=1):
        image = kf[0]
        timestamp = str(kf[1])
        filename = f'frame_{index}_t-{timestamp}.jpg' if timestamp_name else f'frame_{index}.jpg'
        cv2.imwrite(os.path.join(frame_path, filename), image)


def frame_detection(video_source: str, destination: Optional[str] = None, frame_skip_low=5, frame_skip_high=10,
                    verbose=False, high_frame_rate_threshold=10, start_time_ms=None, end_time_ms=None,
                    save_images=False):
    if save_images and destination is not None:
        flowframe_path = destination
        create_directories(flowframe_path)

    # set up video capture and parameters
    video_capture = cv2.VideoCapture(video_source)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_time_ms is not None:
        video_capture.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_skip = frame_skip_low
    if frame_rate > high_frame_rate_threshold:
        frame_skip = frame_skip_high

    if verbose:
        print(f'frame rate: {frame_rate}, frame skip: {frame_skip}')

    if not video_capture.isOpened():
        print("error opening video file")
        return []

    frame_numbers, time_stamps, original_images = [], [], []

    # process each frame in the video
    for i in range(frame_count):
        if i % frame_skip != 0:
            video_capture.grab()  # skip this frame
            continue

        ret, frame = video_capture.read()
        if not ret:
            break

        frame_number = video_capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
        frame_numbers.append(frame_number)

        original_images.append(frame)

        # timestamps from video start
        timestamp = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        time_stamps.append(timestamp)
        if end_time_ms is not None and timestamp > end_time_ms:
            break

    video_capture.release()

    # fix:
    time_stamps[0] = 0

    # store frames and timestamps
    frames = list(zip(original_images, time_stamps))

    # save frames if enabled
    if save_images and flowframe_path is not None:
        save_frames(frames, flowframe_path)

    # verbose output if enabled
    if verbose:
        for index, (image, timestamp) in enumerate(frames, start=1):
            print(f'frame {index} at {timestamp:.2f} ms.')

    return frames


if __name__ == '__main__':
    f = frame_detection(video_source='./images/work_files/couple-walking-in-park-cut.mp4', frame_skip_high=3,
                        destination='./images/test_flowframes/', save_images=True, verbose=True)
