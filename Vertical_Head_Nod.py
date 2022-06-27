import cv2
import os
import traceback
from timeit import default_timer as timer
from datetime import datetime
from pathlib import Path
import logging

# Importing necessary packages for Headmove
import numpy as np
import mediapipe as mp
import pandas as pd

logger = logging.getLogger(__name__)


class VerticalHeadNod:
    """
    Liveliness Test based on Horizontal Head Movement.
    Preferred Sequence : Center --> Left --> Center --> Right.
    Left and Right rotations in the above sequence can be changed according to the user convenience.
    Optical Flow + Mediapipe + Nose Coordinates + Break Condition + CSV Saving + File saving outputs
    Working and tested with 6 Major Resolutions
    - 576 X 432
    - 768 X 576
    - 992 X 744
    - 1200 X 900
    - 1500 X 1125
    - 1920 X 1080
    """

    # Class Attributes

    # Parameters and thresholds
    # is_facever=False

    def __init__(self, videopath):
        """
                Misc:
                    - for first time when the Headmove class is triggered, it acquires resources and
                    displays the following message. "Created TensorFlow Lite XNNPACK delegate for CPU."
                    - this is a system generated message and can be ignored.

                Args:
                    videopath (str): video file location.

                Return:
                    liveliness (dict) : Over-all liveliness measure which consists of four attributes.
                        'left_head_move': total number of times user looked left,
                        'right_head_move': total number of times user looked right,
                        'status_flag': PASS/FAIL of the liveliness flow ,
                        'v_head_nod_count': total number of times user has completed the horizontal head nod.
        """

        # Video file location
        self.videopath = videopath

        # Developer Resolution
        self.ORG_WIDTH = 1920
        self.ORG_HEIGHT = 1080

        # Lucas_kanade Parameters
        self.WIN_SIZE = (45, 45)
        self.MAX_LEVEL = 3
        self.MAX_ITER = 15
        self.EPSILON = 0.05

        # FPS of saving video
        self.VID_FPS = 30.0

        # Mediapipe Parameters
        self.MIN_DETECTION_CONFIDENCE = 0.5
        self.MIN_TRACKING_CONFIDENCE = 0.5

        # Setting up the initial parameters for left and right movement detection
        # Initial gesture is set to false
        self.GESTURE = False

        # For Right
        self.COUNTER_UP = 0
        self.FLAG_UP = 0

        # For Left
        self.COUNTER_DOWN = 0
        self.FLAG_DOWN = 0

        # Head Nod count
        self.v_head_nod_count = 0

        # number of frames a gesture is displayed (shown)
        self.GESTURE_SHOW = 15

        # storing path of the output files
        Path('videoliveliness_files_vnod').mkdir(exist_ok=True, parents=True)
        self.PATH = "videoliveliness_files_vnod/"

        # Initial status for status_flag
        self.STATUS_FLAG = "FAIL"

        # Saving the record time
        # now = datetime.now()
        # self.dt_string = now.strftime("_%d-%m-%Y-%H-%M-%S")

        # Initial liveliness status
        self.liveness = {'Down_head_move': self.FLAG_DOWN, 'UP_head_move': self.FLAG_UP,
                         'status_flag': self.STATUS_FLAG, 'v_head_nod_count': self.v_head_nod_count}

    def status_func(self, distance):
        if self.THRES_UP > distance > self.THRES_DOWN:
            status = "Normal"
        elif distance > self.THRES_UP:
            status = "Looking UP"
        elif distance < self.THRES_DOWN:
            status = "Looking DOWN"
        else:
            status = "Normal"
        return status

    def liveliness_check(self):

        # logger.debug("Liveliness check started")
        print("Liveliness check started")
        # Creating a text text_output_file
        filepath = str(self.PATH + self.videopath.rsplit("/")[-1].rpartition('.')[0] + "_video_to_text" + ".txt")
        text_output_file = open(filepath, "w+")

        # Head pose essentials
        from custom.face_geometry import (  # isort:skip
            PCF,
            get_metric_landmarks,
            procrustes_landmark_basis,
        )

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)

        points_idx = [33, 263, 61, 291, 199]
        points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
        points_idx = list(set(points_idx))
        points_idx.sort()

        # uncomment next line to use all points for PnP algorithm
        # points_idx = list(range(0,468)); points_idx[0:2] = points_idx[0:2:-1];

        frame_height, frame_width, channels = (720, 1280, 3)
        # frame_height, frame_width, channels = (1080, 1920, 3)

        # pseudo camera internals
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        dist_coeff = np.zeros((4, 1))

        # PCF value
        pcf = PCF(
            near=1,
            far=10000,
            frame_height=frame_height,
            frame_width=frame_width,
            fy=camera_matrix[1, 1],
        )

        # Video Capture
        cap = cv2.VideoCapture(self.videopath)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=self.WIN_SIZE, maxLevel=self.MAX_LEVEL,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.MAX_ITER, self.EPSILON))

        # Creating the DataFrame
        dataframe = pd.DataFrame(
            columns=["p1_x", "p1_y", "p2_x", "p2_y", "Distance", "Flag_DOWN", "Flag_UP", "OF_Status", "OF_Error"])

        # Take first frame and convert it to grayscale for Optical flow
        ret, old_frame = cap.read()

        # Scale Factor Scaling Threshold
        size1 = old_frame[0].shape[:2]
        FRAME_WIDTH = size1[0]
        SCALE_FACTOR = self.ORG_WIDTH / FRAME_WIDTH
        # convert it to int may be a problem, because you can only have 10 values 1 - 10
        self.THRES_UP = int(120 / int(SCALE_FACTOR))
        self.THRES_DOWN = int(-100 / int(SCALE_FACTOR))

        # converting previous frame to gray
        previous_gray_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # # Mediapipe Utils
        # mp_face_mesh = mp.solutions.face_mesh

        # Saving the VideoFrame
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter(str(self.PATH + self.videopath.rsplit("/")[-1].rpartition('.')[0] +
                                     '_video_viz.avi'), cv2.VideoWriter_fourcc(*'MJPG'), self.VID_FPS, size)

        # Finding the first frame with mediapipe and getting the coordinates for nose tip and bottom nose center
        with mp_face_mesh.FaceMesh(
                min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE) as face_mesh:

            # Converting frame from BGR to RGB for Mediapipe, flipping the frame to get exact location wrt actual
            # and captures frame, Right is Right, Left is Left
            image = cv2.cvtColor(cv2.flip(previous_gray_frame, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            results = face_mesh.process(old_frame)
            # logger.debug(f"Results: {results}")

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True

            # Converting color of the frame back to BGR from mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Steps for getting cooridnates from Mediapipe
            if results.multi_face_landmarks:
                # logger.debug(f"Facial landmarks: {results.multi_face_landmarks}")
                for face_landmarks in results.multi_face_landmarks:
                    # Finding the coordinates of nose center and nose bottom center
                    x1 = face_landmarks.landmark[0].x
                    y1 = face_landmarks.landmark[0].y
                    x2 = face_landmarks.landmark[1].x
                    y2 = face_landmarks.landmark[1].y
                    shape = image.shape
                    relative_x1 = int(x1 * shape[1])
                    relative_y1 = int(y1 * shape[0])
                    relative_x2 = int(x2 * shape[1])
                    relative_y2 = int(y2 * shape[0])

                    # Taking nose tip as the face center and marking the coordinates for face_center
                    lip_center = relative_x1, relative_y1
                    text_output_file.write("Lip Center: " + str(lip_center) + "\n")
                    nose_center = relative_x2, relative_y2
                    text_output_file.write("Nose Center: " + str(nose_center) + "\n")

                    # converting the center coordinates to 2D vector for Optical flow algorithm
                    optical_points_old = np.array([[lip_center], [nose_center]], np.float32)
                    text_output_file.write(
                        "Lip Center and Nose Center Array Point: " + str(optical_points_old) + "\n")

                    # Reshaping optical_points_old for storing the points coordinate values in the dataframe
                    optical_points_old_reshape = optical_points_old.reshape(2, 2)

                    # Storing attributes from Frame 0
                    dataframe = dataframe.append({"p1_x": int(optical_points_old_reshape[0][0]),
                                                  "p1_y": int(optical_points_old_reshape[0][1]),
                                                  "p2_x": int(optical_points_old_reshape[1][0]),
                                                  "p2_y": int(optical_points_old_reshape[1][1]), "Distance": 0,
                                                  "Flag_DOWN": 0, "Flag_UP": 0, "OF_Status": 0, "OF_Error": 0},
                                                 ignore_index=True)

                    # Reading the frames
                    while cap.isOpened():
                        ret, frame = cap.read()
                        # logger.debug("Frame reading started")
                        # This condition prevents from infinite looping incase video ends.
                        if not ret:
                            break

                        # Frame conversion for optical flow
                        new_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # calculate optical flow
                        optical_points_new, st, err = cv2.calcOpticalFlowPyrLK(previous_gray_frame, new_frame_gray,
                                                                               optical_points_old, None, **lk_params)
                        text_output_file.write(
                            "Status: " + str(st.flatten()) + "\n" + " Error: " + str(err.flatten()) + "\n")

                        # Select good points --> Nose Tip and Nose bottom center in our case
                        points_new = optical_points_new[st == 1]
                        text_output_file.write("Points New: " + str(points_new) + "\n")
                        points_old = optical_points_old[st == 1]
                        text_output_file.write("Points Old: " + str(points_old) + "\n")

                        # Mediapipe for points detection
                        with mp_face_mesh.FaceMesh(
                                min_detection_confidence=self.MIN_DETECTION_CONFIDENCE,
                                min_tracking_confidence=self.MIN_TRACKING_CONFIDENCE) as face_mesh:

                            # Flipping the image for better orientation
                            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

                            # Creating the copy of the frame
                            image1 = image

                            # To improve performance, optionally mark the image as not writeable to pass by reference.
                            image.flags.writeable = False
                            results = face_mesh.process(image)
                            multi_face_landmarks = results.multi_face_landmarks

                            # Draw the face mesh annotations on the image.
                            image.flags.writeable = True
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                            # Finding the coordinates again
                            if multi_face_landmarks:
                                for face_landmarks in multi_face_landmarks:
                                    x1 = face_landmarks.landmark[2].x
                                    y1 = face_landmarks.landmark[2].y
                                    x2 = face_landmarks.landmark[1].x
                                    y2 = face_landmarks.landmark[1].y
                                    shape = image.shape
                                    relative_x1 = int(x1 * shape[1])
                                    relative_y1 = int(y1 * shape[0])
                                    relative_x2 = int(x2 * shape[1])
                                    relative_y2 = int(y2 * shape[0])

                            # Getting the extended landmarks
                            if multi_face_landmarks:
                                face_landmarks = multi_face_landmarks[0]
                                landmarks = np.array(
                                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                                )
                                # print(landmarks.shape)
                                landmarks = landmarks.T

                                metric_landmarks, pose_transform_mat = get_metric_landmarks(
                                    landmarks.copy(), pcf
                                )

                                image_points = (
                                        landmarks[0:2, points_idx].T
                                        * np.array([frame_width, frame_height])[None, :]
                                )
                                model_points = metric_landmarks[0:3, points_idx].T

                                # see here:
                                # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
                                pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]
                                mp_rotation_vector, _ = cv2.Rodrigues(pose_transform_mat[:3, :3])
                                mp_translation_vector = pose_transform_mat[:3, 3, None]

                                if False:
                                    # sanity check
                                    # get same result with solvePnP

                                    success, rotation_vector, translation_vector = cv2.solvePnP(
                                        model_points,
                                        image_points,
                                        camera_matrix,
                                        dist_coeff,
                                        flags=cv2.cv2.SOLVEPNP_ITERATIVE,
                                    )

                                    np.testing.assert_almost_equal(mp_rotation_vector, rotation_vector)
                                    np.testing.assert_almost_equal(
                                        mp_translation_vector, translation_vector
                                    )

                                for face_landmarks in multi_face_landmarks:
                                    mp_drawing.draw_landmarks(
                                        image=frame,
                                        landmark_list=face_landmarks,
                                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                                        landmark_drawing_spec=drawing_spec,
                                        connection_drawing_spec=drawing_spec,
                                    )

                                nose_tip = model_points[0]
                                nose_tip_extended = 2.5 * model_points[0]
                                (nose_pointer2D, jacobian) = cv2.projectPoints(
                                    np.array([nose_tip, nose_tip_extended]),
                                    mp_rotation_vector,
                                    mp_translation_vector,
                                    camera_matrix,
                                    dist_coeff,
                                )

                                nose_tip_2D, nose_tip_2D_extended = nose_pointer2D.squeeze().astype(int)
                                image = cv2.line(
                                    image1, tuple(nose_tip_2D), tuple(nose_tip_2D_extended), (255, 0, 0), 2
                                )
                                # print(f"Nose_tip: {tuple(nose_tip_2D)}")
                                # print(f"Nose_tip_extended: {tuple(nose_tip_2D_extended)}")
                                p1 = tuple(nose_tip_2D)[1]
                                p2 = tuple(nose_tip_2D_extended)[1]

                                # Distance function to find the distance between points
                                dist = int(p1 - p2)

                                # Counter for Right side detection, if threshold is passed, counter increases
                                if dist > self.THRES_UP:
                                    self.GESTURE = 'Looking UP'
                                    self.COUNTER_UP += 1
                                # when the person passes the gesture_show thres and distance thres,
                                # then only flag is incremented
                                else:
                                    if self.COUNTER_UP >= self.GESTURE_SHOW:
                                        self.FLAG_UP += 1
                                    self.COUNTER_UP = 0

                                # Counter for Light side detection, if threshold is passed, counter increases
                                if dist < self.THRES_DOWN:
                                    self.GESTURE = 'Looking DOWN'
                                    self.COUNTER_DOWN += 1
                                # when the person passes the gesture_show thres and distance thres,
                                # then only flag is incremented.
                                else:
                                    if self.COUNTER_DOWN >= self.GESTURE_SHOW:
                                        self.FLAG_DOWN += 1
                                    self.COUNTER_DOWN = 0

                                if self.GESTURE and self.GESTURE_SHOW > 0:
                                    cv2.putText(image1, 'Gesture Detected: ' + self.GESTURE, (40, 250),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                (0, 0, 255), 2)
                                    self.GESTURE_SHOW -= 1

                                # Reset the parameters if gestures (movements) are not found
                                if self.GESTURE_SHOW == 0:
                                    self.GESTURE = False

                                    # number of frames a gesture is shown
                                    self.GESTURE_SHOW = 15

                                    # Marking the points on the image
                                    cv2.putText(image1, 'Flag_UP:' + str(self.FLAG_DOWN), (40, 150),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                (0, 0, 255), 2)
                                    cv2.putText(image1, 'Flag_DOWN:' + str(self.FLAG_UP), (40, 200),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                (0, 0, 255), 2)

                                # Storing the frame-wise attributes of the video.
                                dataframe = dataframe.append(
                                    {"p1_x": int(relative_x1), "p1_y": int(relative_y1), "p2_x": int(relative_x2),
                                     "p2_y": int(relative_y2), "Distance": int(dist), "Flag_DOWN": int(self.FLAG_DOWN),
                                     "Flag_UP": int(self.FLAG_UP), "OF_Status": st.flatten(),
                                     "OF_Error": err.flatten()}, ignore_index=True)

                        # Saving the videofile
                        result.write(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

                        # Breaking the frame
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        # Updating Previous frame and points and checkpoint for optical flow
                        previous_gray_frame = new_frame_gray.copy()
                        if points_new.size != 0 or points_old.size != 0 and np.array_equal(points_new,
                                                                                           points_old) == False:
                            optical_points_old = points_new.reshape(-1, 1, 2)
                            continue
                        else:
                            print("Liveliness failed due to discontinuity")
                            break

                    # Head Nod condition
                    if self.FLAG_DOWN >= self.FLAG_UP:
                        self.v_head_nod_count = self.FLAG_UP
                    elif self.FLAG_UP >= self.FLAG_DOWN:
                        self.v_head_nod_count = self.FLAG_DOWN

                    if self.v_head_nod_count >= 1:
                        self.STATUS_FLAG = "PASS"

                    # Exit condition
                    self.liveness = {"DOWN_head_move": int(self.FLAG_DOWN), "UP_head_move": int(self.FLAG_UP),
                                     "status_flag": self.STATUS_FLAG, "v_head_nod_count": self.v_head_nod_count}

                    # Saving the Optical flow attributes and other final information in a text file
                    text_output_file.write(str(self.liveness) + "\n")

                    # Saving the Optical flow attributes in a text text_output_file
                    text_output_file.close()

                    # Saving all the results into a final csv text_output_file
                    frame_list = [i for i in range(len(dataframe))]
                    final_dataframe = dataframe
                    final_dataframe['FrameNo'] = frame_list
                    final_dataframe['Status'] = final_dataframe.apply(lambda x: self.status_func(x['Distance']), axis=1)
                    outfile = str(self.PATH + self.videopath.rsplit("/")[-1].rpartition('.')[0] + '_outfile_csv.csv')
                    final_dataframe.to_csv(outfile, index=False)

            # Exit screen
            cap.release()
            result.release()
            cv2.destroyAllWindows()
            print(f"Liveliness: {self.liveness}")
            print("Liveliness check completed")
            return self.liveness


if __name__ == "__main__":
    # Input file location of the video
    # test = VerticalHeadNod(videopath="speed_test_all/vhn_2.mp4")
    test = VerticalHeadNod(videopath="resolution_test/vhn_1500_1125.mp4")
    # test = VerticalHeadNod(videopath="Flow_cut.mp4")
    test.liveliness_check()
    print("Videoliveliness Done")
