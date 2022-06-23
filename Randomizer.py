from Horizontal_Head_Nod import HorizontalHeadNod
from Vertical_Head_Nod import VerticalHeadNod
from Open_Mouth import OpenMouth
from Finger_Counter import FingerCount
from Right_Hand_Show import RightShow
from Left_Hand_Show import LeftShow
import random
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
    text_start = int(width / 4), (height - 25)
    inst_start = int(width / 4), (height - 450)
    text = "Press Q when Done"
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


def horizontal_head_nod():
    print("Horizontal Head Nod Called")
    instructions = "Do Horizontal Head Nod"
    videopath = videorecorder(instructions)
    time.sleep(5)
    hnod_obj = HorizontalHeadNod(videopath)
    hnod_var = hnod_obj.liveliness_check()
    print("Horizontal Head Nod Over")
    return hnod_var


def vertical_head_nod():
    print("Vertical Head Nod Called")
    instructions = "Do Vertical Head Nod"
    videopath = videorecorder(instructions)
    time.sleep(5)
    vnod_obj = VerticalHeadNod(videopath)
    vnod_var = vnod_obj.liveliness_check()
    print("Vertical Head Nod Over")
    return vnod_var


def finger_count():
    print("Randomised Finger Counter Called")
    x_fingers = random.randint(1, 5)
    instructions = f"Show {x_fingers} Fingers to Screen"
    videopath = videorecorder(instructions)
    time.sleep(5)
    fcount_obj = FingerCount(videopath, x_fingers)
    fcount_var = fcount_obj.liveliness_check()
    print("Randomised Finger Counter Over")
    return fcount_var


def open_mouth():
    print("Randomised Open Mouth Called")
    x_opens = random.randint(1, 3)
    instructions = f"Open-Close Mouth {x_opens} Times"
    videopath = videorecorder(instructions)
    time.sleep(5)
    omouth_obj = OpenMouth(videopath, x_opens)
    omouth_var = omouth_obj.liveliness_check()
    print("Randomised Open Mouth Over")
    return omouth_var


def right_hand_show():
    print("Right Hand Show Called")
    instructions = "Show Right Hand"
    videopath = videorecorder(instructions)
    time.sleep(5)
    rhand_obj = RightShow(videopath)
    rhand_var = rhand_obj.liveliness_check()
    print("Right Hand Show Over")
    return rhand_var


def left_hand_show():
    print("Left Hand Show Called")
    instructions = "Show Left Hand"
    videopath = videorecorder(instructions)
    time.sleep(5)
    lhand_obj = LeftShow(videopath)
    lhand_var = lhand_obj.liveliness_check()
    print("Left Hand Show Over")
    return lhand_var


my_list = [horizontal_head_nod, vertical_head_nod, finger_count, open_mouth, right_hand_show, left_hand_show]

random_actions = random.sample(my_list, 2)
for random_action in random_actions:
    random_action()

# random.choice(my_list)()
# print(var)
# var[0]()
# random.choices(my_list, k=2)()
