import cv2
from datetime import datetime
from pathlib import Path
import random

# Importing necessary packages for Headmove
import mediapipe as mp
import pandas as pd


class FingerCount:

    def __init__(self, videopath):

        # Video file location
        self.videopath = videopath

        # FPS of saving video
        self.VID_FPS = 30.0

        # Setting up the initial parameters for left and right movement detection
        # Initial gesture is set to false
        self.GESTURE = False

        # Fingers count
        self.fingers_count = 0

        # Showing Count
        self.COUNT_SHOW = 0

        # Show Random Number
        self.SHOW_NUMBER = 5
        # self.SHOW_NUMBER = random.randint(1, 5)

        # number of frames a gesture is displayed (shown)
        self.GESTURE_SHOW = 15

        # storing path of the output files
        Path('videoliveliness_files_fingers').mkdir(exist_ok=True, parents=True)
        self.PATH = "videoliveliness_files_fingers/"

        # Initial status for status_flag
        self.STATUS_FLAG = "FAIL"

        # Saving the record time
        now = datetime.now()
        self.dt_string = now.strftime("_%d-%m-%Y-%H-%M-%S")

        # Initial left_hand_count status
        self.finger_count = {'status_flag': self.STATUS_FLAG, 'fingers_count': self.fingers_count}

    def liveliness_check(self):

        print("Finger counter check started")

        # Video Capture
        cap = cv2.VideoCapture(self.videopath)

        # Saving the VideoFrame
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        video_save = cv2.VideoWriter(str(self.PATH + self.videopath.rsplit(".", 1)[0] +
                                         '_video_viz.avi'), cv2.VideoWriter_fourcc(*'MJPG'), int(self.VID_FPS), size)

        # Creating the DataFrame
        dataframe = pd.DataFrame(
            columns=["t1_x", "t1_y", "t2_x", "t2_y", "t3_x", "t3_y", "t4_x", "t4_y", "t5_x", "t5_y", "Count",
                     "Gesture_Count"])

        # Mediapipe Utils
        medhands = mp.solutions.hands
        hands = medhands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        draw = mp.solutions.drawing_utils

        while cap.isOpened():
            ret, frame = cap.read()
            # This condition prevents from infinite looping incase video ends.
            if not ret:
                break

            # Converting frame from BGR to RGB for Mediapipe, flipping the frame to get exact location wrt actual
            # and captures frame, Right is Right, Left is Left
            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            results = hands.process(image_rgb)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True

            # Converting color of the frame back to BGR from mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # list of all landmarks of the tips of fingers
            lmlist = []
            tipids = [4, 8, 12, 16, 20]

            cv2.rectangle(image, (20, 350), (90, 440), (0, 255, 204), cv2.FILLED)
            cv2.rectangle(image, (20, 350), (90, 440), (0, 0, 0), 5)

            if results.multi_hand_landmarks:
                for handlms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handlms.landmark):

                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmlist.append([id, cx, cy])
                        if len(lmlist) != 0 and len(lmlist) == 21:
                            fingerlist = []
                            fingercount = 0

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

                            cv2.putText(image, str(fingercount), (25, 430), cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 0), 5)

                            if fingercount == self.SHOW_NUMBER:
                                self.GESTURE = True
                                self.COUNT_SHOW += 1
                            # when the person passes the gesture_show thres and distance thres,
                            # then only flag is incremented
                            else:
                                if self.COUNT_SHOW >= self.GESTURE_SHOW:
                                    self.fingers_count += 1
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
                                                          "Count": fingercount,
                                                          "Gesture_Count": self.fingers_count,
                                                          }, ignore_index=True)

                        # change color of points and lines
                        draw.draw_landmarks(image, handlms, medhands.HAND_CONNECTIONS,
                                            draw.DrawingSpec(color=(0, 255, 204), thickness=2, circle_radius=2),
                                            draw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=3))

            # Saving the videofile
            # video_save.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            video_save.write(image)

            # Breaking the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if self.fingers_count >= 1:
                self.STATUS_FLAG = "PASS"

            # Exit condition
            self.finger_count = {'status_flag': self.STATUS_FLAG, 'fingers_count': self.fingers_count}

            # Saving all the results into a final csv text_output_file
            frame_list = [i for i in range(len(dataframe))]
            final_dataframe = dataframe
            final_dataframe['FrameNo'] = frame_list
            # final_dataframe['Status'] = final_dataframe.apply(lambda x: self.status_func(x['Distance']), axis=1)
            outfile = str(self.PATH + self.videopath.rsplit(".", 1)[0] + self.dt_string + '_outfile_csv.csv')
            final_dataframe.to_csv(outfile, index=False)

        # Exit screen
        cap.release()
        video_save.release()
        cv2.destroyAllWindows()
        print(f"Total_Fingers: {self.finger_count}")
        print("Fingers Count check completed")
        return self.finger_count


if __name__ == "__main__":
    # Input file location of the video
    test = FingerCount(videopath="5.mp4")
    # test = FingerCount(videopath="error.mp4")
    # test = RightShow(videopath=0)
    test.liveliness_check()
    print("Fingers Count Check Done")