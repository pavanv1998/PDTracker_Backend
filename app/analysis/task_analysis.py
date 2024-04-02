import math
import numpy as np
import mediapipe as mp
import cv2

import app.analysis.constants.mp_handlandmarks as MP_HAND_LANDMARKS
import app.analysis.constants.yolo_landmarks as YOLO_LANDMARKS


def get_essential_landmarks(current_frame, current_frame_idx, task, bounding_box, detector):
    is_left = False
    if "left" in str.lower(task):
        is_left = True

    if "hand movement" in str.lower(task):
        return get_hand_movement_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_landmarks(bounding_box, detector, current_frame, is_left)
    # elif "toe tapping" in str.lower(task):
    #     return mp_pose()


def get_signal(display_landmarks, task):
    if "hand movement" in str.lower(task):
        return get_hand_movement_signal(display_landmarks)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_signal(display_landmarks)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_signal(display_landmarks)


def get_normalisation_factor(landmarks, task):
    if "hand movement" in str.lower(task):
        return get_hand_movement_nf(landmarks)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_nf(landmarks)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_nf(landmarks)


def get_display_landmarks(landmarks, task):
    if "hand movement" in str.lower(task):
        return get_hand_movement_display_landmarks(landmarks)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_display_landmarks(landmarks)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_display_landmarks(landmarks)


def get_leg_agility_landmarks(bounding_box, detector, current_frame, is_left):
    [x1, y1, x2, y2] = get_boundaries(bounding_box)
    roi = current_frame[y1:y2, x1:x2]

    # Convert the ROI to RGB since many models expect input in this format
    results = detector.predict(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), conf=0.7)
    landmarks = results.prediction.poses[0]

    knee_idx = YOLO_LANDMARKS.LEFT_KNEE if is_left else YOLO_LANDMARKS.RIGHT_KNEE

    left_shoulder = landmarks[YOLO_LANDMARKS.LEFT_SHOULDER]
    right_shoulder = landmarks[YOLO_LANDMARKS.RIGHT_SHOULDER]
    knee_landmark = landmarks[knee_idx]
    left_hip = landmarks[YOLO_LANDMARKS.LEFT_HIP]
    right_hip = landmarks[YOLO_LANDMARKS.RIGHT_HIP]

    return [left_shoulder, right_shoulder, knee_landmark, left_hip, right_hip]


def get_leg_agility_signal(landmarks_list):
    signal = []
    for landmarks in landmarks_list:
        [left_shoulder, right_shoulder, knee_landmark] = landmarks
        shoulder_midpoint = (np.array(left_shoulder[:2]) + np.array(right_shoulder[:2])) / 2
        distance = math.dist(knee_landmark[:2], shoulder_midpoint)
        signal.append(distance)
    return signal


def get_leg_agility_nf(landmarks_list):
    values = []
    for landmarks in landmarks_list:
        [left_shoulder, right_shoulder, _, left_hip, right_hip] = landmarks
        shoulder_midpoint = (np.array(left_shoulder[:2]) + np.array(right_shoulder[:2])) / 2
        hip_midpoint = (np.array(left_hip[:2]) + np.array(right_hip[:2])) / 2
        distance = math.dist(shoulder_midpoint, hip_midpoint)
        values.append(distance)
    return np.mean(values)


def get_leg_agility_display_landmarks(landmarks_list):
    display_landmarks = []
    for landmarks in landmarks_list:
        [left_shoulder, right_shoulder, knee_landmark, _, _] = landmarks
        display_landmarks.append([left_shoulder, right_shoulder, knee_landmark])
    return display_landmarks


def get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left):
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    # crop frame to the bounding box
    [x1, y1, x2, y2] = get_boundaries(bounding_box)

    image_data = current_frame[y1:y2, x1:x2, :].astype(np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)

    detection_result = detector.detect_for_video(image, current_frame_idx)
    current_frame_idx += 1

    hand_index = get_hand_index(detection_result, is_left)

    if hand_index == -1:
        return []  # skip frame if hand is not detected

    return detection_result.hand_landmarks[hand_index]


def get_hand_movement_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector):
    hand_landmarks = get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left)
    if not hand_landmarks:
        return []
    bounds = get_boundaries(bounding_box)
    index_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], bounds)
    middle_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP], bounds)
    ring_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.RING_FINGER_TIP], bounds)
    wrist = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.WRIST], bounds)

    return [index_finger, middle_finger, ring_finger, wrist]


def get_hand_movement_signal(landmarks_list):
    signal = []
    for landmarks in landmarks_list:
        [index_finger, middle_finger, ring_finger, wrist] = landmarks
        distance = (math.dist(index_finger, wrist) + math.dist(middle_finger, wrist) + math.dist(ring_finger, wrist)) / 3
        signal.append(distance)
    return signal


def get_hand_movement_nf(landmarks_list):
    values = []
    for landmarks in landmarks_list:
        [_, middle_finger, _, wrist] = landmarks
        distance = math.dist(middle_finger, wrist)
        values.append(distance)
    return np.max(values)


def get_hand_movement_display_landmarks(landmarks_list):
    display_landmarks = []
    for landmarks in landmarks_list:
        [index_finger, middle_finger, ring_finger, wrist] = landmarks
        display_landmarks.append([index_finger, middle_finger, ring_finger, wrist])
    return display_landmarks


def get_finger_tap_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector):
    hand_landmarks = get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left)
    if not hand_landmarks:
        return []

    bounds = get_boundaries(bounding_box)
    thumb_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.THUMB_TIP], bounds)
    index_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], bounds)
    middle_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP], bounds)
    wrist = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.WRIST], bounds)

    return [thumb_finger, index_finger, middle_finger, wrist]


def get_finger_tap_signal(landmarks_list):
    signal = []
    for landmarks in landmarks_list:
        [thumb_finger, index_finger] = landmarks
        distance = math.dist(thumb_finger, index_finger)
        signal.append(distance)
    return signal


def get_finger_tap_nf(landmarks_list):
    values = []
    for landmarks in landmarks_list:
        [_, _, middle_finger, wrist] = landmarks
        distance = math.dist(middle_finger, wrist)
        values.append(distance)
    return np.max(values)


def get_finger_tap_display_landmarks(landmarks_list):
    display_landmarks = []
    for landmarks in landmarks_list:
        [thumb_finger, index_finger, _, _] = landmarks
        display_landmarks.append([thumb_finger, index_finger])
    return display_landmarks


def get_hand_index(detection_result, is_left):
    direction = "Left" if is_left else "Right"

    handedness = detection_result.handedness

    for idx in range(0, len(handedness)):
        if handedness[idx][0].category_name == direction:
            return idx

    return -1


def get_landmark_coords(landmark, bounds):
    [x1, y1, x2, y2] = bounds

    return [landmark.x * (x2 - x1), landmark.y * (y2 - y1)]


def get_boundaries(bounding_box):
    x1 = int(bounding_box['x'])
    y1 = int(bounding_box['y'])
    x2 = x1 + int(bounding_box['width'])
    y2 = y1 + int(bounding_box['height'])

    return [x1, y1, x2, y2]
