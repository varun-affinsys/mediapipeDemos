# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture(0)
# # Define the codec and create VideoWriter object
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# text_start = int(width / 4), (height - 25)
# text = "Press Q when Done"
#
# out = cv.VideoWriter('output1.avi', fourcc, 30.0, (width,  height))
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     frame_1 = cv.flip(frame, 1)
#     # write the flipped frame
#     out.write(frame)
#     cv.putText(frame_1, text, text_start, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#     cv.imshow('frame', frame_1)
#     if cv.waitKey(1) == ord('q'):
#         break
# # Release everything if job is finished
# cap.release()
# out.release()
# cv.destroyAllWindows()

import cv2
from datetime import datetime
import time


def videorecorder(instructions):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    cap = cv2.VideoCapture(0)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    text_start = int(width / 4), (height - 10)
    text = "Press Q when Done"
    inst_start = int(width / 4), (height - 450)
    path = 'recorded_videos/'
    video_path = str(path + "output_" + dt_string + ".avi")

    out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_1 = cv2.flip(frame, 1)
        # write the flipped frame
        out.write(frame)
        cv2.putText(frame_1, text, text_start, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame_1, instructions, inst_start, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow('frame', frame_1)
        if cv2.waitKey(1) == ord('q'):
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return video_path


instructions = "Show Your Left Hand"
print(videorecorder(instructions))
