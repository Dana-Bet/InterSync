import os
from typing import Optional

import cv2
import numpy as np
import peakutils
import matplotlib.pyplot as plt


def convert_to_grayscale_and_blur(frame):
    # convert frame to grayscale and blur it
    if frame is not None:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0.0)
        return gray_image, blurred_image
    return None, None


def create_directories(keyframe_path):
    # create directories if they don't exist
    os.makedirs(keyframe_path, exist_ok=True)


def display_metrics(indices, frame_numbers, difference_magnitudes, show_metrics=False, save_metrics=False,
                    save_metrics_path=None):
    # making sure both lists have the same length by trimming the longer list
    if len(frame_numbers) > len(difference_magnitudes):
        frame_numbers = frame_numbers[1:]  # trim the first frame number
    elif len(difference_magnitudes) > len(frame_numbers):
        difference_magnitudes = difference_magnitudes[1:]  # trim the first difference magnitude

    # difference magnitudes and peaks
    plt.plot(indices, np.array(difference_magnitudes)[indices], "x")
    plt.plot(frame_numbers, difference_magnitudes, 'r-')
    plt.xlabel('frames')
    plt.ylabel('pixel difference')
    plt.title("frame-to-frame pixel differences and peaks")

    # save plot
    if save_metrics and save_metrics_path:
        plt.savefig(os.path.join(save_metrics_path, 'frame_pixel_difference.png'), bbox_inches='tight')

    # show plot
    if show_metrics:
        plt.show()
    plt.close()


def save_keyframes(keyframes, keyframe_path, timestamp_name=True):
    # save keyframe images to specified path
    for index, kf in enumerate(keyframes, start=1):
        image = kf[0]
        timestamp = str(kf[1])
        filename = f'keyframe_{index}_t-{timestamp}.jpg' if timestamp_name else f'keyframe_{index}.jpg'
        cv2.imwrite(os.path.join(keyframe_path, filename), image)


def keyframe_detection(video_source: str, destination: Optional[str] = None, detection_threshold=0.3, verbose=False,
                       optimize_frame_processing=True, high_frame_rate_threshold=30,
                       start_time_ms: Optional[int] = None,
                       end_time_ms: Optional[int] = None, save_images=False, display_plots=False, save_plots=False):
    if (save_images or save_plots) and destination is not None:
        keyframe_path = destination
        create_directories(keyframe_path)

    # set up video capture and parameters
    video_capture = cv2.VideoCapture(video_source)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_time_ms is not None:
        video_capture.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_skip = 1

    # skip frames if optimize_frame_processing is enabled
    if optimize_frame_processing and frame_rate > high_frame_rate_threshold:
        frame_skip = int(frame_rate / high_frame_rate_threshold)

    if not video_capture.isOpened():
        print("error opening video file")
        return []

    frame_numbers, difference_magnitudes, time_stamps, original_images = [], [], [], []
    last_frame = None

    # process each frame in the video
    for i in range(frame_count):
        if i % frame_skip != 0:
            video_capture.grab()  # skip this frame
            continue

        ret, frame = video_capture.read()
        if not ret:
            break

        # convert frame to grayscale and blur
        grayscale_frame, blurred_frame = convert_to_grayscale_and_blur(frame)
        frame_number = video_capture.get(cv2.CAP_PROP_POS_FRAMES) - 1
        frame_numbers.append(frame_number)
        original_images.append(frame)

        # for the first frame
        if frame_number == 0 or last_frame is None:
            last_frame = blurred_frame
            continue

        # calculate difference magnitude between frames / fix for cv2.error?
        if blurred_frame.shape == last_frame.shape:
            frame_difference = cv2.absdiff(blurred_frame, last_frame)
            difference_magnitude = cv2.countNonZero(frame_difference)
            difference_magnitudes.append(difference_magnitude)
        else:
            print(f"keyframe_detection: frame size mismatch at frame {i}, skipping this frame.")
            continue

        last_frame = blurred_frame

        # timestamps from video start
        timestamp = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        time_stamps.append(timestamp)

        if end_time_ms is not None and timestamp > end_time_ms:
            break

    video_capture.release()
    # find peaks in the difference magnitudes
    difference_magnitude_array = np.array(difference_magnitudes)
    baseline = peakutils.baseline(difference_magnitude_array, 2)
    peak_indices = peakutils.indexes(difference_magnitude_array - baseline, detection_threshold, min_dist=1)

    # store keyframes and timestamps
    keyframes = [[original_images[x], time_stamps[x]] for x in peak_indices]

    # display metrics if enabled
    if display_plots or save_plots:
        display_metrics(peak_indices, frame_numbers, difference_magnitudes, show_metrics=display_plots,
                        save_metrics=save_plots, save_metrics_path=keyframe_path)

    # save keyframes if enabled
    if save_images:
        save_keyframes(keyframes, keyframe_path)

    # verbose output if enabled
    if verbose:
        for index, (image, timestamp) in enumerate(keyframes, start=1):
            print(f'keyframe {index} at {timestamp:.2f} ms.')

    cv2.destroyAllWindows()
    return keyframes


# if __name__ == '__main__':
#     kf = keyframe_detection(video_source='../images/work_files/pair_test2.mp4',
#                             destination='../images/test_keyframes/', optimize_frame_processing=True, save_images=True,
#                             verbose=True, display_plots=False, save_plots=True, start_time_ms=10000)
