import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import math
import json
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# %%
import scipy.signal as signal
import scipy.interpolate as interpolate
import scipy.stats as stats
import time

import torch
from super_gradients.training import models

from app.finderPeaksSignal import peakFinder

from app.hand_analysis import finger_tap, hand_analysis


def analysis(bounding_box, start_time, end_time, input_video):
    vision_running_mode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='app/models/pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision_running_mode.VIDEO
    )

    detector = vision.PoseLandmarker.create_from_options(options)
    video = cv2.VideoCapture(input_video)

    fps = cv2.CAP_PROP_FPS
    start_frame = round(fps*start_time)
    end_frame = round(fps*end_time)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame



