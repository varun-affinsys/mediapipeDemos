import cv2
from datetime import datetime
from pathlib import Path

# Importing necessary packages for Left Palm Mediapipe
import mediapipe as mp
import pandas as pd


class LeftShow:

    def __init__(self, videopath):

        # Video file location
        self.videopath = videopath

        # FPS of saving video
        self.VID_FPS = 30.0

        # Setting up the initial parameters for left hand detection
        # Initial gesture is set to false
        self.GESTURE = False

        # Hand Flag
        self.SIDE = None

        # Correct Hand
        self.CORRECT_HAND = 0

        # Showing Count
        self.COUNT_SHOW = 0

        # number of frames a gesture is displayed (shown)
        self.GESTURE_SHOW = 15

        # storing path of the output files
        Path('videoliveliness_files_left_hand').mkdir(exist_ok=True, parents=True)
        self.PATH = "videoliveliness_files_left_hand/"

        # Initial status for status_flag
        self.STATUS_FLAG = "FAIL"

        # # Saving the record time
        # now = datetime.now()
        # self.dt_string = now.strftime("_%d-%m-%Y-%H-%M-%S")

        # Initial left_hand_count status
        self.left_hand_count = {'status_flag': self.STATUS_FLAG, 'Left_hand_count': self.COUNT_SHOW}

    def liveliness_check(self):

        print("Left Hand check started")

        # Video Capture
        cap = cv2.VideoCapture(self.videopath)

        # Saving the VideoFrame
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        video_save = cv2.VideoWriter(str(self.PATH + self.videopath.rsplit("/")[-1].split('.')[0] +
                                         '_video_viz.avi'), cv2.VideoWriter_fourcc(*'MJPG'), int(self.VID_FPS), size)

        # Creating the DataFrame
        dataframe = pd.DataFrame(
            columns=["t1_x", "t1_y", "t2_x", "t2_y", "t3_x", "t3_y", "t4_x", "t4_y", "t5_x", "t5_y", "Side", "Fingers",
                     "Gesture_Count"])

        # Mediapipe Utils
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
                model_complexity=0, max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            while cap.isOpened():
                ret, image = cap.read()

                # This condition prevents from infinite looping incase video ends.
                if not ret:
                    print("Ignoring empty camera frame.")
                    break

                # To improve performance, optionally mark the image as not writeable to pass by reference.
                image.flags.writeable = False

                # Converting frame from BGR to RGB for Mediapipe, flipping the frame to get exact location wrt actual
                # and captures frame, Right is Right, Left is Left
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.flip(image, 1)
                results = hands.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True

                # Converting color of the frame back to BGR from mediapipe
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # list of all landmarks of the tips of fingers
                lmlist = []
                tipids = [4, 8, 12, 16, 20]

                if results.multi_hand_landmarks:
                    for handlms in results.multi_hand_landmarks:
                        fingercount = 0
                        for id, lm in enumerate(handlms.landmark):
                            h, w, c = image.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            lmlist.append([id, cx, cy])
                            if len(lmlist) != 0 and len(lmlist) == 21:
                                fingerlist = []

                                # thumb and dealing with flipping of hands
                                if lmlist[12][1] > lmlist[20][1]:
                                    if lmlist[tipids[0]][1] > lmlist[tipids[0] - 1][1]:
                                        fingerlist.append(1)
                                    else:
                                        fingerlist.append(0)
                                else:
                                    if lmlist[tipids[0]][1] < lmlist[tipids[0] - 1][1]:
                                        fingerlist.append(1)
                                    else:
                                        fingerlist.append(0)

                                # others
                                for id in range(1, 5):
                                    if lmlist[tipids[id]][2] < lmlist[tipids[id] - 2][2]:
                                        fingerlist.append(1)
                                    else:
                                        fingerlist.append(0)

                                if len(fingerlist) != 0:
                                    fingercount = fingerlist.count(1)

                        # Getting the side of the hand
                        self.SIDE = "Left" if "Left" in str(results.multi_handedness[0]) else "InCorrect"
                        if fingercount == 5 and self.SIDE == "Left":
                            self.GESTURE = True
                            self.COUNT_SHOW += 1
                        # when the person passes the gesture_show thres and distance thres,
                        # then only flag is incremented
                        else:
                            if self.COUNT_SHOW >= self.GESTURE_SHOW:
                                self.CORRECT_HAND += 1
                            self.COUNT_SHOW = 0

                        if self.GESTURE and self.GESTURE_SHOW > 0:
                            self.GESTURE_SHOW -= 1

                        # Reset the parameters if gestures (movements) are not found
                        if self.GESTURE_SHOW == 0:
                            self.GESTURE = False
                            # number of frames a gesture is shown
                            self.GESTURE_SHOW = 15

                        # Storing attributes from Frame 0
                        dataframe = dataframe.append({"t1_x": int(lmlist[4][1]),
                                                      "t1_y": int(lmlist[4][2]),
                                                      "t2_x": int(lmlist[8][1]),
                                                      "t2_y": int(lmlist[8][1]),
                                                      "t3_x": int(lmlist[12][1]),
                                                      "t3_y": int(lmlist[12][1]),
                                                      "t4_x": int(lmlist[16][1]),
                                                      "t4_y": int(lmlist[16][1]),
                                                      "t5_x": int(lmlist[20][1]),
                                                      "t5_y": int(lmlist[20][1]),
                                                      "Side": self.SIDE,
                                                      "Fingers": fingercount,
                                                      "Gesture_Count": self.CORRECT_HAND,
                                                      }, ignore_index=True)

                    # change color of points and lines
                    mp_drawing.draw_landmarks(image, handlms, mp_hands.HAND_CONNECTIONS,
                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                  mp_drawing_styles.get_default_hand_connections_style())

                # Saving the videofile
                video_save.write(image)

                # Breaking the frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if self.CORRECT_HAND >= 1:
                    self.STATUS_FLAG = "PASS"

                # Exit condition
                self.left_hand_count = {'status_flag': self.STATUS_FLAG, 'Left_Hand_Show': self.CORRECT_HAND}

                # Saving all the results into a final csv
                frame_list = [i for i in range(len(dataframe))]
                final_dataframe = dataframe
                final_dataframe['FrameNo'] = frame_list
                # removed self.dt_string from filepath
                outfile = str(self.PATH + self.videopath.rsplit("/")[-1].split('.')[0] + '_outfile_csv.csv')
                final_dataframe.to_csv(outfile, index=False)

        # Exit screen
        cap.release()
        video_save.release()
        cv2.destroyAllWindows()
        print(f"Left Hand Count: {self.left_hand_count}")
        print("Left Hand Count check completed")
        return self.left_hand_count


if __name__ == "__main__":
    # Input file location of the video
    test = LeftShow(videopath="left_hand.mp4")
    # test = LeftShow(videopath=0)
    test.liveliness_check()
    print("Left Hand Count Check Done")
