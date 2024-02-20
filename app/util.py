import cv2
import datetime


def analyse_video(path=None):
    # create video capture object
    if path is None:
        return 0, 0

    data = cv2.VideoCapture(path)

    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)

    # calculate duration of the video
    seconds = round(frames / fps)
    video_time = datetime.timedelta(seconds=seconds)
    print(f"duration in seconds: {seconds}")
    print(f"video time: {video_time}")

    return seconds, video_time
